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

#pragma once

#include <ir/daphneir/Daphne.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/vectorized/VectorizedDataSink.h>

#include <functional>
#include <mutex>
#include <vector>

using mlir::daphne::VectorCombine;
using mlir::daphne::VectorSplit;

class Task {
  public:
    virtual ~Task() = default;

    virtual void execute(uint32_t fid, uint32_t batchSize) = 0;
    virtual uint64_t getTaskSize() = 0;
};

// task for signaling closed input queue (no more tasks)
class EOFTask : public Task {
  public:
    EOFTask() = default;
    ~EOFTask() override = default;
    void execute(uint32_t fid, uint32_t batchSize) override {}
    uint64_t getTaskSize() override { return 0; }
};

template <class DT> struct CompiledPipelineTaskData {
    std::vector<std::function<void(DT ***, Structure **, DCTX(ctx))>> _funcs;
    const bool *_isScalar;
    Structure **_inputs;
    const size_t _numInputs;
    const size_t _numOutputs;
    const int64_t *_outRows;
    const int64_t *_outCols;
    const VectorSplit *_splits;
    const VectorCombine *_combines;
    const uint64_t _rl;              // row lower index
    const uint64_t _ru;              // row upper index
    const int64_t *_wholeResultRows; // number of rows of the complete result
    const int64_t *_wholeResultCols; // number of cols of the complete result
    const uint64_t _offset;
    DCTX(_ctx);

    [[maybe_unused]] CompiledPipelineTaskData<DT> withDifferentRange(uint64_t newRl, uint64_t newRu) {
        return CompiledPipelineTaskData<DT>{_funcs, _isScalar, _inputs, _numInputs, _numOutputs, _outRows, _outCols,
                                            _splits, _combines, newRl, newRu, _wholeResultRows, _wholeResultCols,
                                            _offset, _ctx};
    }
};

template <class DT> class CompiledPipelineTaskBase : public Task {
  protected:
    CompiledPipelineTaskData<DT> _data;

  public:
    explicit CompiledPipelineTaskBase(CompiledPipelineTaskData<DT> data) : _data(data) {}
    void execute(uint32_t fid, uint32_t batchSize) override = 0;
    uint64_t getTaskSize() override = 0;

  protected:
    bool isBroadcast(mlir::daphne::VectorSplit splitMethod, Structure *input) {
        return splitMethod == VectorSplit::NONE || (splitMethod == VectorSplit::ROWS && input->getNumRows() == 1);
    }

    std::vector<Structure *> createFuncInputs(uint64_t rowStart, uint64_t rowEnd) {
        std::vector<Structure *> linputs;
        for (auto i = 0u; i < _data._numInputs; i++) {
            if (isBroadcast(_data._splits[i], _data._inputs[i])) {
                linputs.push_back(_data._inputs[i]);
                // We need to increase the reference counter, since the
                // pipeline manages the reference counter itself.
                // This might be a scalar disguised as a Structure*.
                if (!_data._isScalar[i])
                    // Note that increaseRefCounter() synchronizes the access
                    // via a std::mutex. If that turns out to slow down things,
                    // creating a shallow copy of the input would be an
                    // alternative.
                    _data._inputs[i]->increaseRefCounter();
            } else if (VectorSplit::ROWS == _data._splits[i]) {
                linputs.push_back(_data._inputs[i]->sliceRow(rowStart, rowEnd));
            } else {
                llvm_unreachable("Not all vector splits handled");
            }
        }
        return linputs;
    }
};

template <class DT> class CompiledPipelineTask : public CompiledPipelineTaskBase<DT> {};

template <typename VT> class CompiledPipelineTask<DenseMatrix<VT>> : public CompiledPipelineTaskBase<DenseMatrix<VT>> {
    std::mutex &_resLock;
    DenseMatrix<VT> ***_res;
    using CompiledPipelineTaskBase<DenseMatrix<VT>>::_data;

  public:
    CompiledPipelineTask(CompiledPipelineTaskData<DenseMatrix<VT>> data, std::mutex &resLock, DenseMatrix<VT> ***res)
        : CompiledPipelineTaskBase<DenseMatrix<VT>>(data), _resLock(resLock), _res(res) {}

    void execute(uint32_t fid, uint32_t batchSize) override;
    uint64_t getTaskSize() override;

  private:
    void accumulateOutputs(std::vector<DenseMatrix<VT> *> &localResults, std::vector<DenseMatrix<VT> *> &localAggRes,
                           uint64_t rowStart, uint64_t rowEnd);
};

template <typename VT> class CompiledPipelineTask<CSRMatrix<VT>> : public CompiledPipelineTaskBase<CSRMatrix<VT>> {
    std::vector<VectorizedDataSink<CSRMatrix<VT>> *> &_resultSinks;
    using CompiledPipelineTaskBase<CSRMatrix<VT>>::_data;

  public:
    CompiledPipelineTask(CompiledPipelineTaskData<CSRMatrix<VT>> data,
                         std::vector<VectorizedDataSink<CSRMatrix<VT>> *> &resultSinks)
        : CompiledPipelineTaskBase<CSRMatrix<VT>>(data), _resultSinks(resultSinks) {}

    void execute(uint32_t fid, uint32_t batchSize) override;
    uint64_t getTaskSize() override;
};
