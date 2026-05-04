/*
    Copyright (c) 2012-2018 Christopher A. Taylor.  All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of Wirehair nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
*/

#include <wirehair/wirehair.h>
#include <wirehair/wirehair_cuda.h>
#include "WirehairCodec.h"
#include "WirehairCudaDispatch.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <new> // std::nothrow
#include <thread>
#include <vector>

static std::atomic<bool> m_init(false);

namespace {

static uint32_t NormalizeThreadCount(uint32_t requested)
{
    if (requested > 0) {
        return requested;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? static_cast<uint32_t>(hc) : 1U;
}

static bool ShouldUsePipelineCudaRuntime()
{
    WirehairCudaConfig cfg = {};
    cfg.struct_bytes = sizeof(WirehairCudaConfig);
    if (wirehair_cuda_get_config(&cfg) != Wirehair_Success) {
        return false;
    }
    return cfg.enable_pipeline_cuda_runtime != 0U &&
        cfg.backend_mode != WirehairCudaBackend_CpuOnly;
}

static size_t PipelineCoalesceLimit(size_t pendingQueue, uint32_t symbolsOrBlocks)
{
    WirehairCudaConfig cfg = {};
    cfg.struct_bytes = sizeof(WirehairCudaConfig);
    if (wirehair_cuda_get_config(&cfg) != Wirehair_Success) {
        return 4;
    }
    size_t limit = 4;
    if (cfg.enable_pipeline_cuda_runtime != 0U && cfg.backend_mode != WirehairCudaBackend_CpuOnly) {
        limit = cfg.enable_cuda_graphs != 0U ? 12 : 8;
        if (cfg.enable_single_api_microbatch != 0U) {
            const size_t microbatch = static_cast<size_t>(cfg.single_api_microbatch_size > 0 ? cfg.single_api_microbatch_size : 1U);
            limit = std::max<size_t>(limit, std::min<size_t>(16, microbatch));
        }
    }
    limit = std::max<size_t>(2, std::min<size_t>(16, limit));
    if (symbolsOrBlocks <= 128U) {
        limit = std::min<size_t>(limit, 4);
    } else if (symbolsOrBlocks <= 256U) {
        limit = std::min<size_t>(limit, 8);
    }
    // Keep latency bounded when queue is shallow.
    if (pendingQueue <= 2) {
        limit = std::min<size_t>(limit, 3);
    }
    return limit;
}

static bool EncodeShapeCompatible(
    const WirehairEncodeBatchRequest& a,
    const WirehairEncodeBatchRequest& b)
{
    return a.message_bytes == b.message_bytes &&
        a.block_bytes == b.block_bytes &&
        a.block_stride_bytes == b.block_stride_bytes;
}

static bool DecodeShapeCompatible(
    const WirehairDecodeBatchRequest& a,
    const WirehairDecodeBatchRequest& b)
{
    return a.message_bytes == b.message_bytes &&
        a.block_bytes == b.block_bytes &&
        a.block_stride_bytes == b.block_stride_bytes;
}

class PipelineRuntime
{
public:
    explicit PipelineRuntime(const WirehairPipelineConfig& config)
        : QueueCapacity(config.queue_capacity > 0 ? config.queue_capacity : 64)
    {
        const uint32_t encodeThreads = NormalizeThreadCount(config.encode_threads);
        const uint32_t decodeThreads = NormalizeThreadCount(config.decode_threads);
        for (uint32_t i = 0; i < encodeThreads; ++i) {
            EncodeThreads.emplace_back([this]() { EncodeWorkerLoop(); });
        }
        for (uint32_t i = 0; i < decodeThreads; ++i) {
            DecodeThreads.emplace_back([this]() { DecodeWorkerLoop(); });
        }
    }

    ~PipelineRuntime()
    {
        {
            std::lock_guard<std::mutex> lock(QueueMutex);
            Stopping.store(true, std::memory_order_release);
        }
        EncodeQueueCv.notify_all();
        DecodeQueueCv.notify_all();
        CompletionCv.notify_all();
        for (std::thread& t : EncodeThreads) {
            if (t.joinable()) {
                t.join();
            }
        }
        for (std::thread& t : DecodeThreads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    WirehairResult SubmitEncode(const WirehairEncodeBatchRequest& req)
    {
        if (!req.message || req.message_bytes < 1 || req.block_bytes < 1 || req.block_count < 1 ||
            !req.block_data_out || req.block_stride_bytes < req.block_bytes || !req.bytes_out)
        {
            return Wirehair_InvalidInput;
        }
        {
            std::lock_guard<std::mutex> lock(QueueMutex);
            if (Stopping.load(std::memory_order_acquire) || EncodeQueue.size() >= QueueCapacity) {
                return Wirehair_Error;
            }
            EncodeQueue.push_back(req);
        }
        EncodeQueueCv.notify_one();
        return Wirehair_Success;
    }

    WirehairResult SubmitDecode(const WirehairDecodeBatchRequest& req)
    {
        if (req.message_bytes < 1 || req.block_bytes < 1 || req.symbol_count < 1 || !req.block_ids ||
            !req.block_data || !req.block_data_bytes || !req.message_out || req.block_stride_bytes < req.block_bytes)
        {
            return Wirehair_InvalidInput;
        }
        {
            std::lock_guard<std::mutex> lock(QueueMutex);
            if (Stopping.load(std::memory_order_acquire) || DecodeQueue.size() >= QueueCapacity) {
                return Wirehair_Error;
            }
            DecodeQueue.push_back(req);
        }
        DecodeQueueCv.notify_one();
        return Wirehair_Success;
    }

    WirehairResult Poll(uint32_t timeoutMsec, WirehairPipelineEvent* eventOut)
    {
        if (!eventOut) {
            return Wirehair_InvalidInput;
        }
        std::unique_lock<std::mutex> lock(CompletionMutex);
        if (Completions.empty())
        {
            if (timeoutMsec == 0) {
                return Wirehair_NeedMore;
            }
            const bool ready = CompletionCv.wait_for(
                lock,
                std::chrono::milliseconds(timeoutMsec),
                [this]() {
                    return !Completions.empty() || Stopping.load(std::memory_order_acquire);
                });
            if (!ready || Completions.empty()) {
                return Wirehair_NeedMore;
            }
        }
        *eventOut = Completions.front();
        Completions.pop_front();
        return Wirehair_Success;
    }

private:
    void PushCompletion(const WirehairPipelineEvent& event)
    {
        {
            std::lock_guard<std::mutex> lock(CompletionMutex);
            Completions.push_back(event);
        }
        CompletionCv.notify_one();
    }

    void EncodeWorkerLoop()
    {
        for (;;)
        {
            WirehairEncodeBatchRequest req = {};
            std::vector<WirehairEncodeBatchRequest> coalesced;
            bool useCudaRuntime = false;
            size_t coalesceMax = 4;
            {
                std::unique_lock<std::mutex> lock(QueueMutex);
                EncodeQueueCv.wait(lock, [this]() {
                    return Stopping.load(std::memory_order_acquire) || !EncodeQueue.empty();
                });
                if (Stopping.load(std::memory_order_acquire) && EncodeQueue.empty()) {
                    return;
                }
                req = EncodeQueue.front();
                EncodeQueue.pop_front();
                useCudaRuntime = ShouldUsePipelineCudaRuntime();
                if (useCudaRuntime)
                {
                    coalesceMax = PipelineCoalesceLimit(EncodeQueue.size(), req.block_count);
                    while (!EncodeQueue.empty() && coalesced.size() + 1 < coalesceMax)
                    {
                        if (!EncodeShapeCompatible(req, EncodeQueue.front())) {
                            break;
                        }
                        coalesced.push_back(EncodeQueue.front());
                        EncodeQueue.pop_front();
                    }
                }
            }

            if (useCudaRuntime)
            {
                std::vector<WirehairEncodeBatchRequest> batch;
                batch.reserve(coalesced.size() + 1);
                batch.push_back(req);
                for (const WirehairEncodeBatchRequest& item : coalesced) {
                    batch.push_back(item);
                }
                for (const WirehairEncodeBatchRequest& job : batch)
                {
                    WirehairPipelineEvent completion = {};
                    completion.request_id = job.request_id;
                    completion.job_type = WirehairPipelineJob_EncodeBatch;
                    completion.result = wirehair_cuda_encode_batch(&job);
                    completion.produced_count = (completion.result == Wirehair_Success) ? job.block_count : 0;
                    PushCompletion(completion);
                }
                continue;
            }

            WirehairPipelineEvent completion = {};
            completion.request_id = req.request_id;
            completion.job_type = WirehairPipelineJob_EncodeBatch;
            completion.result = Wirehair_Error;
            completion.produced_count = 0;
            WirehairCodec codec = wirehair_encoder_create(nullptr, req.message, req.message_bytes, req.block_bytes);
            if (codec)
            {
                completion.result = Wirehair_Success;
                for (uint32_t i = 0; i < req.block_count; ++i)
                {
                    uint8_t* out = reinterpret_cast<uint8_t*>(req.block_data_out) + static_cast<size_t>(i) * req.block_stride_bytes;
                    uint32_t bytesOut = 0;
                    const WirehairResult enc = wirehair_encode(codec, req.start_block_id + i, out, req.block_bytes, &bytesOut);
                    req.bytes_out[i] = bytesOut;
                    if (enc != Wirehair_Success) {
                        completion.result = enc;
                        break;
                    }
                    completion.produced_count = i + 1;
                }
                wirehair_free(codec);
            }

            PushCompletion(completion);
        }
    }

    void DecodeWorkerLoop()
    {
        for (;;)
        {
            WirehairDecodeBatchRequest req = {};
            std::vector<WirehairDecodeBatchRequest> coalesced;
            bool useCudaRuntime = false;
            size_t coalesceMax = 4;
            {
                std::unique_lock<std::mutex> lock(QueueMutex);
                DecodeQueueCv.wait(lock, [this]() {
                    return Stopping.load(std::memory_order_acquire) || !DecodeQueue.empty();
                });
                if (Stopping.load(std::memory_order_acquire) && DecodeQueue.empty()) {
                    return;
                }
                req = DecodeQueue.front();
                DecodeQueue.pop_front();
                useCudaRuntime = ShouldUsePipelineCudaRuntime();
                if (useCudaRuntime)
                {
                    coalesceMax = PipelineCoalesceLimit(DecodeQueue.size(), req.symbol_count);
                    while (!DecodeQueue.empty() && coalesced.size() + 1 < coalesceMax)
                    {
                        if (!DecodeShapeCompatible(req, DecodeQueue.front())) {
                            break;
                        }
                        coalesced.push_back(DecodeQueue.front());
                        DecodeQueue.pop_front();
                    }
                }
            }

            if (useCudaRuntime)
            {
                std::vector<WirehairDecodeBatchRequest> batch;
                batch.reserve(coalesced.size() + 1);
                batch.push_back(req);
                for (const WirehairDecodeBatchRequest& item : coalesced) {
                    batch.push_back(item);
                }
                for (const WirehairDecodeBatchRequest& job : batch)
                {
                    WirehairPipelineEvent completion = {};
                    completion.request_id = job.request_id;
                    completion.job_type = WirehairPipelineJob_DecodeBatch;
                    completion.result = wirehair_cuda_decode_batch(&job);
                    completion.produced_count =
                        (completion.result == Wirehair_Success || completion.result == Wirehair_NeedMore)
                        ? job.symbol_count : 0;
                    PushCompletion(completion);
                }
                continue;
            }

            WirehairPipelineEvent completion = {};
            completion.request_id = req.request_id;
            completion.job_type = WirehairPipelineJob_DecodeBatch;
            completion.result = Wirehair_Error;
            completion.produced_count = 0;
            WirehairCodec decoder = wirehair_decoder_create(nullptr, req.message_bytes, req.block_bytes);
            if (decoder)
            {
                completion.result = Wirehair_NeedMore;
                const uint8_t* blockData = reinterpret_cast<const uint8_t*>(req.block_data);
                for (uint32_t i = 0; i < req.symbol_count; ++i)
                {
                    const uint8_t* symbol = blockData + static_cast<size_t>(i) * req.block_stride_bytes;
                    const WirehairResult dec = wirehair_decode(decoder, req.block_ids[i], symbol, req.block_data_bytes[i]);
                    completion.produced_count = i + 1;
                    if (dec == Wirehair_Success) {
                        completion.result = wirehair_recover(decoder, req.message_out, req.message_bytes);
                        break;
                    }
                    if (dec != Wirehair_NeedMore) {
                        completion.result = dec;
                        break;
                    }
                }
                wirehair_free(decoder);
            }

            PushCompletion(completion);
        }
    }

    const size_t QueueCapacity;
    std::atomic<bool> Stopping{false};
    std::mutex QueueMutex;
    std::condition_variable EncodeQueueCv;
    std::condition_variable DecodeQueueCv;
    std::deque<WirehairEncodeBatchRequest> EncodeQueue;
    std::deque<WirehairDecodeBatchRequest> DecodeQueue;
    std::vector<std::thread> EncodeThreads;
    std::vector<std::thread> DecodeThreads;

    std::mutex CompletionMutex;
    std::condition_variable CompletionCv;
    std::deque<WirehairPipelineEvent> Completions;
};

} // namespace


extern "C" {


//-----------------------------------------------------------------------------
// Wirehair API

WIREHAIR_EXPORT const char *wirehair_result_string(
    WirehairResult result ///< Result code to convert to string
)
{
    static_assert(WirehairResult_Count == 11, "Update this switch too");

    switch (result)
    {
    case Wirehair_Success:           return "Wirehair_Success";
    case Wirehair_NeedMore:    return "Wirehair_NeedMore";
    case Wirehair_BadDenseSeed:      return "Wirehair_BadDenseSeed";
    case Wirehair_BadPeelSeed:       return "Wirehair_BadPeelSeed";
    case Wirehair_BadInput_SmallN:   return "Wirehair_BadInput_SmallN";
    case Wirehair_BadInput_LargeN:   return "Wirehair_BadInput_LargeN";
    case Wirehair_ExtraInsufficient: return "Wirehair_ExtraInsufficient";
    case Wirehair_InvalidInput:      return "Wirehair_InvalidInput";
    case Wirehair_Error:             return "Wirehair_Error";
    case Wirehair_OOM:               return "Wirehair_OOM";
    case Wirehair_UnsupportedPlatform: return "Wirehair_UnsupportedPlatform";
    default:
        break;
    }

    return "Unknown";
}

WIREHAIR_EXPORT WirehairResult wirehair_init_(int expected_version)
{
    // If version does not match:
    if (expected_version != WIREHAIR_VERSION) {
        return Wirehair_InvalidInput;
    }

    const int gfInitResult = gf256_init();

    // If gf256 init failed:
    if (gfInitResult != 0) {
        return Wirehair_UnsupportedPlatform;
    }

    m_init.store(true, std::memory_order_release);
    return Wirehair_Success;
}

WIREHAIR_EXPORT WirehairCodec wirehair_encoder_create(
    WirehairCodec reuseOpt, ///< [Optional] Pointer to prior codec object
    const void*    message, ///< Pointer to message
    uint64_t  messageBytes, ///< Bytes in the message
    uint32_t    blockBytes  ///< Bytes in an output block
)
{
    // If input is invalid:
    if (!m_init.load(std::memory_order_acquire) || !message || messageBytes < 1 || blockBytes < 1) {
        return nullptr;
    }

    wirehair::Codec* codec = reinterpret_cast<wirehair::Codec*>(reuseOpt);

    // Allocate a new Codec object
    if (!codec) {
        codec = new (std::nothrow) wirehair::Codec;
    }

    // Initialize codec
    WirehairResult result = codec->InitializeEncoder(messageBytes, blockBytes);

    // If initialization succeeded:
    if (result == Wirehair_Success) {
        // Feed message to codec
        result = codec->EncodeFeed(message);
    }

    // If either function failed:
    if (result != Wirehair_Success)
    {
        // Note this will also release the reuse parameter
        delete codec;
        codec = nullptr;
    }

    return reinterpret_cast<WirehairCodec>(codec);
}

WIREHAIR_EXPORT WirehairResult wirehair_encode(
    WirehairCodec    codec, ///< Pointer to codec from wirehair_encoder_init()
    unsigned       blockId, ///< Identifier of block to generate
    void*     blockDataOut, ///< Pointer to output block data
    uint32_t      outBytes, ///< Bytes in the output buffer
    uint32_t* dataBytesOut  ///< Number of bytes written <= blockBytes
)
{
    if (!codec || !blockDataOut || !dataBytesOut) {
        return Wirehair_InvalidInput;
    }

    wirehair::Codec* session = reinterpret_cast<wirehair::Codec*>(codec);

    return WirehairCudaDispatchEncode(session, blockId, blockDataOut, outBytes, dataBytesOut);
}

WIREHAIR_EXPORT WirehairCodec wirehair_decoder_create(
    WirehairCodec reuseOpt, ///< Codec object to reuse
    uint64_t  messageBytes, ///< Bytes in the message to decode
    uint32_t    blockBytes  ///< Bytes in each encoded block
)
{
    // If input is invalid:
    if (!m_init.load(std::memory_order_acquire) || messageBytes < 1 || blockBytes < 1) {
        return nullptr;
    }

    wirehair::Codec* codec = reinterpret_cast<wirehair::Codec*>(reuseOpt);

    // Allocate a new Codec object
    if (!codec) {
        codec = new (std::nothrow) wirehair::Codec;
    }

    // Allocate memory for decoding
    WirehairResult result = codec->InitializeDecoder(messageBytes, blockBytes);

    // If either function failed:
    if (result != Wirehair_Success)
    {
        // Note this will also release the reuse parameter
        delete codec;
        codec = nullptr;
    }

    return reinterpret_cast<WirehairCodec>(codec);
}

WIREHAIR_EXPORT WirehairResult wirehair_decode(
    WirehairCodec   codec, ///< Codec object
    unsigned      blockId, ///< ID number of received block
    const void* blockData, ///< Pointer to block data
    uint32_t    dataBytes  ///< Number of bytes in the data block
)
{
    // If input is invalid:
    if (!codec || !blockData || dataBytes < 1) {
        return Wirehair_InvalidInput;
    }

    wirehair::Codec* decoder = reinterpret_cast<wirehair::Codec*>(codec);

    return WirehairCudaDispatchDecode(decoder, blockId, blockData, dataBytes);
}

WIREHAIR_EXPORT WirehairResult wirehair_recover(
    WirehairCodec    codec, ///< Codec object
    void*       messageOut, ///< Buffer where reconstructed message will be written
    uint64_t  messageBytes  ///< Bytes in the message
)
{
    // If input is invalid:
    if (!codec || !messageOut) {
        return Wirehair_InvalidInput;
    }

    wirehair::Codec* decoder = reinterpret_cast<wirehair::Codec*>(codec);

    return decoder->ReconstructOutput(messageOut, messageBytes);
}

WIREHAIR_EXPORT WirehairResult wirehair_recover_block(
    WirehairCodec codec, ///< Codec object
    unsigned    blockId, ///< ID of the block to reconstruct between 0..N-1
    void*  blockDataOut, ///< Pointer to block data
    uint32_t*  bytesOut  ///< Set to the number of data bytes in the block
)
{
    // If input is invalid:
    if (!codec || !blockDataOut || !bytesOut) {
        return Wirehair_InvalidInput;
    }

    wirehair::Codec* decoder = reinterpret_cast<wirehair::Codec*>(codec);

    return decoder->ReconstructBlock((uint16_t)blockId, blockDataOut, bytesOut);
}

WIREHAIR_EXPORT WirehairResult wirehair_decoder_becomes_encoder(
    WirehairCodec codec ///< Codec to change
)
{
    // If input is invalid:
    if (!codec) {
        return Wirehair_InvalidInput;
    }

    wirehair::Codec* encoder = reinterpret_cast<wirehair::Codec*>(codec);

    return encoder->InitializeEncoderFromDecoder();
}

WIREHAIR_EXPORT void wirehair_free(
    WirehairCodec codec ///< Codec object to free
)
{
    wirehair::Codec* object = reinterpret_cast<wirehair::Codec*>(codec);

    delete object;
}

WIREHAIR_EXPORT WirehairPipeline wirehair_pipeline_create(
    const WirehairPipelineConfig* config
)
{
    if (!m_init.load(std::memory_order_acquire) || !config) {
        return nullptr;
    }
    PipelineRuntime* runtime = new (std::nothrow) PipelineRuntime(*config);
    return reinterpret_cast<WirehairPipeline>(runtime);
}

WIREHAIR_EXPORT void wirehair_pipeline_free(
    WirehairPipeline pipeline
)
{
    PipelineRuntime* runtime = reinterpret_cast<PipelineRuntime*>(pipeline);
    delete runtime;
}

WIREHAIR_EXPORT WirehairResult wirehair_encode_batch_async(
    WirehairPipeline pipeline,
    const WirehairEncodeBatchRequest* request
)
{
    if (!pipeline || !request) {
        return Wirehair_InvalidInput;
    }
    PipelineRuntime* runtime = reinterpret_cast<PipelineRuntime*>(pipeline);
    return runtime->SubmitEncode(*request);
}

WIREHAIR_EXPORT WirehairResult wirehair_decode_batch_async(
    WirehairPipeline pipeline,
    const WirehairDecodeBatchRequest* request
)
{
    if (!pipeline || !request) {
        return Wirehair_InvalidInput;
    }
    PipelineRuntime* runtime = reinterpret_cast<PipelineRuntime*>(pipeline);
    return runtime->SubmitDecode(*request);
}

WIREHAIR_EXPORT WirehairResult wirehair_pipeline_submit_generation(
    WirehairPipeline pipeline,
    const WirehairPipelineGenerationRequest* request
)
{
    if (!pipeline || !request || !request->request) {
        return Wirehair_InvalidInput;
    }
    switch (request->job_type)
    {
    case WirehairPipelineJob_EncodeBatch:
        return wirehair_encode_batch_async(
            pipeline,
            reinterpret_cast<const WirehairEncodeBatchRequest*>(request->request));
    case WirehairPipelineJob_DecodeBatch:
        return wirehair_decode_batch_async(
            pipeline,
            reinterpret_cast<const WirehairDecodeBatchRequest*>(request->request));
    default:
        return Wirehair_InvalidInput;
    }
}

WIREHAIR_EXPORT WirehairResult wirehair_pipeline_poll(
    WirehairPipeline pipeline,
    uint32_t timeout_msec,
    WirehairPipelineEvent* event_out
)
{
    if (!pipeline) {
        return Wirehair_InvalidInput;
    }
    PipelineRuntime* runtime = reinterpret_cast<PipelineRuntime*>(pipeline);
    return runtime->Poll(timeout_msec, event_out);
}


} // extern "C"
