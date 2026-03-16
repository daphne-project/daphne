/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MTWrapper.h"
#include <runtime/local/vectorized/Tasks.h>

#ifdef USE_CUDA
#include <runtime/local/vectorized/TasksCUDA.h>
#endif

template <typename VT>
[[maybe_unused]] void MTWrapper<DenseMatrix<VT>>::executeSingleQueue(
    std::vector<std::function<typename MTWrapper<DenseMatrix<VT>>::PipelineFunc>> funcs, DenseMatrix<VT> ***res,
    const bool *isScalar, Structure **inputs, size_t numInputs, size_t numOutputs, int64_t *outRows, int64_t *outCols,
    VectorSplit *splits, VectorCombine *combines, DCTX(ctx), const bool verbose) {
    auto inputProps = this->getInputProperties(inputs, numInputs, splits);
    auto len = inputProps.first;
    auto mem_required = inputProps.second;
    mem_required += this->allocateOutput(res, numOutputs, outRows, outCols, combines);
    auto row_mem = mem_required / len;

    // create task queue (w/o size-based blocking)
    std::unique_ptr<TaskQueue> q = std::make_unique<BlockingTaskQueue>(len);

    std::vector<TaskQueue *> tmp_q{q.get()};
    size_t logBatchSizeCPU = std::max(1ul, ctx->getUserConfig().phyBatchSize / row_mem);
    this->initCPPWorkers(tmp_q, logBatchSizeCPU, verbose, 1, QueueTypeOption::CENTRALIZED, false);

#ifdef USE_CUDA
    size_t logBatchSize8M = std::max(100ul, static_cast<size_t>(std::ceil(8 * 1024 * 1024 / row_mem)));
    if (this->_numCUDAThreads) {
        this->initCUDAWorkers(q.get(), logBatchSize8M * 4, verbose);
        this->cudaPrefetchInputs(inputs, numInputs, mem_required, splits);
        ctx->logger->info("MTWrapper_dense: \nRequired memory (ins/outs): {} "
                          "Required mem/row: {}\n batchsizeCPU={} "
                          "batchsizeGPU={}",
                          mem_required, row_mem, logBatchSizeCPU, logBatchSize8M * 4);
    }
#endif

    // lock for aggregation combine
    // TODO: multiple locks per output
    std::mutex resLock;

    // create tasks and close input
    uint64_t startChunk = 0;
    uint64_t endChunk = 0;
    SelfSchedulingScheme schedulingScheme = ctx->config.taskPartitioningScheme;
    int chunkParam = ctx->config.minimumTaskSize;
    if (chunkParam <= 0)
        chunkParam = 1;
    bool autoChunk = false;
    if (schedulingScheme == SelfSchedulingScheme::AUTO)
        autoChunk = true;
    LoadPartitioning lp(schedulingScheme, len, chunkParam, this->_numThreads, autoChunk);
    while (lp.hasNextChunk()) {
        endChunk += lp.getNextChunk();
        q->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(
            CompiledPipelineTaskData<DenseMatrix<VT>>{funcs, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                                                      splits, combines, startChunk, endChunk, outRows, outCols, 0, ctx},
            resLock, res));
        startChunk = endChunk;
    }
    q->closeInput();

    this->joinAll();
}

template <typename VT>
[[maybe_unused]] void MTWrapper<DenseMatrix<VT>>::executeCpuQueues(
    std::vector<std::function<typename MTWrapper<DenseMatrix<VT>>::PipelineFunc>> funcs, DenseMatrix<VT> ***res,
    const bool *isScalar, Structure **inputs, size_t numInputs, size_t numOutputs, int64_t *outRows, int64_t *outCols,
    VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
    auto inputProps = this->getInputProperties(inputs, numInputs, splits);
    auto len = inputProps.first;
    auto mem_required = inputProps.second;
    mem_required += this->allocateOutput(res, numOutputs, outRows, outCols, combines);
    auto row_mem = mem_required / len;

    std::vector<std::unique_ptr<TaskQueue>> q;
    std::vector<TaskQueue *> qvector;
    if (ctx->getUserConfig().pinWorkers) {
        for (int i = 0; i < this->_numQueues; i++) {
#ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#endif
            std::unique_ptr<TaskQueue> tmp = std::make_unique<BlockingTaskQueue>(len);
            q.push_back(std::move(tmp));
            qvector.push_back(q[i].get());
        }
    } else {
        for (int i = 0; i < this->_numQueues; i++) {
            std::unique_ptr<TaskQueue> tmp = std::make_unique<BlockingTaskQueue>(len);
            q.push_back(std::move(tmp));
            qvector.push_back(q[i].get());
        }
    }

    size_t logBatchSizeCPU = std::max(1ul, ctx->getUserConfig().phyBatchSize / row_mem);
    this->initCPPWorkers(qvector, logBatchSizeCPU, verbose, this->_numQueues, this->_queueMode,
                         ctx->getUserConfig().pinWorkers);

    // lock for aggregation combine
    // TODO: multiple locks per output
    std::mutex resLock;

    // create tasks and close input
    uint64_t startChunk = 0;
    uint64_t endChunk = 0;
    uint64_t currentItr = 0;
    uint64_t target = 0;
    SelfSchedulingScheme schedulingScheme = ctx->config.taskPartitioningScheme;
    int chunkParam = ctx->config.minimumTaskSize;
    if (chunkParam <= 0)
        chunkParam = 1;
    if (ctx->getUserConfig().prePartitionRows) {
        uint64_t oneChunk = len / this->_numQueues;
        int remainder = len - (oneChunk * this->_numQueues);
        std::vector<LoadPartitioning> lps;
        lps.emplace_back(schedulingScheme, oneChunk + remainder, chunkParam, this->_numThreads, false);
        for (int i = 1; i < this->_numQueues; i++) {
            lps.emplace_back(schedulingScheme, oneChunk, chunkParam, this->_numThreads, false);
        }
        if (ctx->getUserConfig().pinWorkers) {
            for (int i = 0; i < this->_numQueues; i++) {
                while (lps[i].hasNextChunk()) {
                    endChunk += lps[i].getNextChunk();
                    qvector[i]->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(
                                                CompiledPipelineTaskData<DenseMatrix<VT>>{
                                                    funcs, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                                                    splits, combines, startChunk, endChunk, outRows, outCols, 0, ctx},
                                                resLock, res),
                                            this->_topology.responsibleThreads[i]);
                    startChunk = endChunk;
                }
            }
        } else {
            for (int i = 0; i < this->_numQueues; i++) {
                while (lps[i].hasNextChunk()) {
                    endChunk += lps[i].getNextChunk();
                    qvector[i]->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(
                        CompiledPipelineTaskData<DenseMatrix<VT>>{funcs, isScalar, inputs, numInputs, numOutputs,
                                                                  outRows, outCols, splits, combines, startChunk,
                                                                  endChunk, outRows, outCols, 0, ctx},
                        resLock, res));
                    startChunk = endChunk;
                }
            }
        }
    } else {
        bool autoChunk = false;
        if (schedulingScheme == SelfSchedulingScheme::AUTO)
            autoChunk = true;
        LoadPartitioning lp(schedulingScheme, len, chunkParam, this->_numThreads, autoChunk);
        if (ctx->getUserConfig().pinWorkers) {
            while (lp.hasNextChunk()) {
                endChunk += lp.getNextChunk();
                target = currentItr % this->_numQueues;
                qvector[target]->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(
                                                 CompiledPipelineTaskData<DenseMatrix<VT>>{
                                                     funcs, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                                                     splits, combines, startChunk, endChunk, outRows, outCols, 0, ctx},
                                                 resLock, res),
                                             this->_topology.uniqueThreads[target]);
                startChunk = endChunk;
                currentItr++;
            }
        } else {
            while (lp.hasNextChunk()) {
                endChunk += lp.getNextChunk();
                target = currentItr % this->_numQueues;
                qvector[target]->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(
                    CompiledPipelineTaskData<DenseMatrix<VT>>{funcs, isScalar, inputs, numInputs, numOutputs, outRows,
                                                              outCols, splits, combines, startChunk, endChunk, outRows,
                                                              outCols, 0, ctx},
                    resLock, res));
                startChunk = endChunk;
                currentItr++;
            }
        }
    }
    for (int i = 0; i < this->_numQueues; i++) {
        qvector[i]->closeInput();
    }

    this->joinAll();
}

template <typename VT>
[[maybe_unused]] void MTWrapper<DenseMatrix<VT>>::executeQueuePerDeviceType(
    std::vector<std::function<typename MTWrapper<DenseMatrix<VT>>::PipelineFunc>> funcs, DenseMatrix<VT> ***res,
    const bool *isScalar, Structure **inputs, size_t numInputs, size_t numOutputs, int64_t *outRows, int64_t *outCols,
    VectorSplit *splits, VectorCombine *combines, DCTX(ctx), bool verbose) {
    size_t device_task_len = 0ul;
    auto inputProps = this->getInputProperties(inputs, numInputs, splits);
    auto len = inputProps.first;
    auto mem_required = inputProps.second;
    mem_required += this->allocateOutput(res, numOutputs, outRows, outCols, combines);
    auto row_mem = mem_required / len;
    size_t logBatchSizeCPU = std::max(1ul, ctx->getUserConfig().phyBatchSize / row_mem);
    // lock for aggregation combine
    // TODO: multiple locks per output
    std::mutex resLock;

#ifdef USE_CUDA
    // ToDo: multi-device support :-P
    float taskRatioCUDA = 0.25f;
    auto gpu_task_len = static_cast<size_t>(std::ceil(static_cast<float>(len) * taskRatioCUDA));
    device_task_len += gpu_task_len;
    std::unique_ptr<TaskQueue> q_cuda = std::make_unique<BlockingTaskQueue>(gpu_task_len);
    size_t logBatchSize8M = std::max(100ul, static_cast<size_t>(std::ceil(8 * 1024 * 1024 / row_mem)));
    this->initCUDAWorkers(q_cuda.get(), logBatchSize8M * 4, verbose);

    auto ***res_cuda = new DenseMatrix<VT> **[numOutputs];
    auto blksize = gpu_task_len / ctx->cuda_contexts.size();
    ctx->logger->debug("gpu_task_len: {}\ntaskRatioCUDA: {}\nBlock size: {}", gpu_task_len, taskRatioCUDA, blksize);
    for (size_t i = 0; i < numOutputs; ++i) {
        res_cuda[i] = new DenseMatrix<VT> *;
        if (combines[i] == mlir::daphne::VectorCombine::ROWS) {
            auto rc2 = static_cast<DenseMatrix<VT> *>((*res[i]))->sliceRow(0, gpu_task_len);
            (*res_cuda[i]) = rc2;
        } else if (combines[i] == mlir::daphne::VectorCombine::COLS) {
            (*res_cuda[i]) = static_cast<DenseMatrix<VT> *>((*res[i]))->slice(0, outRows[i], 0, gpu_task_len);
        } else {
            (*res_cuda[i]) = (*res[i]);
        }
    }

    for (uint32_t k = 0; k < gpu_task_len; k += blksize) {
        q_cuda->enqueueTask(new CompiledPipelineTaskCUDA<DenseMatrix<VT>>(
            CompiledPipelineTaskData<DenseMatrix<VT>>{funcs, isScalar, inputs, numInputs, numOutputs, outRows, outCols,
                                                      splits, combines, k, std::min(k + blksize, len), outRows, outCols,
                                                      0, ctx},
            resLock, res_cuda));
    }
    q_cuda->closeInput();
#endif

    auto cpu_task_len = len - device_task_len;
    DenseMatrix<VT> ***res_cpp{};
    std::unique_ptr<TaskQueue> q_cpp;

    std::vector<std::unique_ptr<TaskQueue>> q;
    std::vector<TaskQueue *> qvector;
    if (cpu_task_len > 0) {
        // Multiple Queues addition
        if (ctx->getUserConfig().pinWorkers) {
            for (int i = 0; i < this->_numQueues; i++) {
#ifdef __linux__
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(i, &cpuset);
                sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#endif
                std::unique_ptr<TaskQueue> tmp = std::make_unique<BlockingTaskQueue>(cpu_task_len);
                q.push_back(std::move(tmp));
                qvector.push_back(q[i].get());
            }
        } else {
            for (int i = 0; i < this->_numQueues; i++) {
                std::unique_ptr<TaskQueue> tmp = std::make_unique<BlockingTaskQueue>(cpu_task_len);
                q.push_back(std::move(tmp));
                qvector.push_back(q[i].get());
            }
        }
        this->initCPPWorkers(qvector, logBatchSizeCPU, verbose, this->_numQueues, this->_queueMode,
                             ctx->getUserConfig().pinWorkers);
        // End Multiple Queues

        res_cpp = new DenseMatrix<VT> **[numOutputs];
        auto offset = device_task_len;

        for (size_t i = 0; i < numOutputs; ++i) {
            res_cpp[i] = new DenseMatrix<VT> *;
            if (combines[i] == mlir::daphne::VectorCombine::ROWS) {
                (*res_cpp[i]) = static_cast<DenseMatrix<VT> *>((*res[i]))->sliceRow(device_task_len, len);
            } else if (combines[i] == mlir::daphne::VectorCombine::COLS) {
                (*res_cpp[i]) = static_cast<DenseMatrix<VT> *>((*res[i]))->sliceCol(device_task_len, len);
            } else {
                (*res_cpp[i]) = (*res[i]);
            }
        }

        uint64_t startChunk = device_task_len;
        uint64_t endChunk = device_task_len;
        uint64_t currentItr = 0;
        uint64_t target = 0;
        SelfSchedulingScheme method = ctx->config.taskPartitioningScheme;
        int chunkParam = ctx->config.minimumTaskSize;
        if (chunkParam <= 0)
            chunkParam = 1;
        bool autoChunk = false;
        if (method == SelfSchedulingScheme::AUTO)
            autoChunk = true;

        LoadPartitioning lp(method, cpu_task_len, chunkParam, this->_numCPPThreads, autoChunk);
        while (lp.hasNextChunk()) {
            endChunk += lp.getNextChunk();
            target = currentItr % this->_numQueues;
            qvector[target]->enqueueTask(new CompiledPipelineTask<DenseMatrix<VT>>(
                CompiledPipelineTaskData<DenseMatrix<VT>>{funcs, isScalar, inputs, numInputs, numOutputs, outRows,
                                                          outCols, splits, combines, startChunk, endChunk, outRows,
                                                          outCols, offset, ctx},
                resLock, res_cpp));
            startChunk = endChunk;
            currentItr++;
        }
        for (int i = 0; i < this->_numQueues; i++) {
            qvector[i]->closeInput();
        }
    }
    this->joinAll();

#ifdef USE_CUDA
    this->combineOutputs(res, res_cuda, numOutputs, combines, ctx);
#endif

    if (cpu_task_len > 0) {
        for (size_t i = 0; i < numOutputs; ++i) {
            if (combines[i] == mlir::daphne::VectorCombine::ROWS || combines[i] == mlir::daphne::VectorCombine::COLS)
                DataObjectFactory::destroy((*res_cpp[i]));
        }
    }
}

#ifdef USE_CUDA
template <typename VT>
void MTWrapper<DenseMatrix<VT>>::combineOutputs(DenseMatrix<VT> ***&res_, DenseMatrix<VT> ***&res_cuda_,
                                                size_t numOutputs, mlir::daphne::VectorCombine *combines, DCTX(ctx)) {
    const size_t deviceID = 0; // ToDo: multi device support
    AllocationDescriptorCUDA alloc_desc(ctx, deviceID);
    for (size_t i = 0; i < numOutputs; ++i) {
        auto *res = static_cast<DenseMatrix<VT> *>((*res_[i]));
        auto *res_cuda = static_cast<DenseMatrix<VT> *>((*res_cuda_[i]));
        if (combines[i] == mlir::daphne::VectorCombine::ROWS) {
            const auto &const_res_cuda = *res_cuda;
            auto data_dest = res->getValues();
            CHECK_CUDART(cudaMemcpy(data_dest, const_res_cuda.getValues(&alloc_desc), const_res_cuda.getBufferSize(),
                                    cudaMemcpyDeviceToHost));
            //            debugPrintCUDABuffer("MTWrapperDense: combine
            //            outputs", const_res_cuda.getValues(&alloc_desc),
            //            const_res_cuda.getNumItems());
            DataObjectFactory::destroy(res_cuda);
        } else if (combines[i] == mlir::daphne::VectorCombine::COLS) {
            const auto &const_res_cuda = *res_cuda;
            auto dst_base_ptr = res->getValues();
            auto src_base_ptr = const_res_cuda.getValues(&alloc_desc);
            for (auto j = 0u; j < res_cuda->getNumRows(); ++j) {
                // ToDo: rowSkip would be correct here if res_cuda wasn't a
                // shallow copy
                //                auto data_src = src_base_ptr +
                //                res_cuda->getRowSkip() * j;
                auto data_src = src_base_ptr + res_cuda->getNumCols() * j;
                auto data_dst = dst_base_ptr + res->getRowSkip() * j;
                CHECK_CUDART(
                    cudaMemcpy(data_dst, data_src, res_cuda->getNumCols() * sizeof(VT), cudaMemcpyDeviceToHost));
            }
            DataObjectFactory::destroy(res_cuda);
        }
    }
}
#else
template <typename VT>
void MTWrapper<DenseMatrix<VT>>::combineOutputs(DenseMatrix<VT> ***&res_, DenseMatrix<VT> ***&res_cuda_,
                                                size_t numOutputs, mlir::daphne::VectorCombine *combines, DCTX(ctx)) {}
#endif

template class MTWrapper<DenseMatrix<double>>;
template class MTWrapper<DenseMatrix<float>>;
template class MTWrapper<DenseMatrix<int64_t>>;
