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

#include "Worker.h"
#include <runtime/local/vectorized/TaskQueues.h>
#include <spdlog/spdlog.h>
#include <utility>

class WorkerCPU : public Worker {
    std::vector<TaskQueue *> _q;
    std::vector<int> _physical_ids;
    std::vector<int> _unique_threads;
    std::array<bool, 256> eofWorkers{};
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numQueues;
    QueueTypeOption _queueMode;
    VictimSelectionLogic _victimSelection;
    bool _pinWorkers;

  public:
    // ToDo: remove compile-time verbose parameter and use logger
    WorkerCPU(std::vector<TaskQueue *> deques, std::vector<int> physical_ids, std::vector<int> unique_threads,
              DCTX(dctx), bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0,
              QueueTypeOption queueMode = QueueTypeOption::CENTRALIZED,
              VictimSelectionLogic victimSelection = VictimSelectionLogic::SEQ, bool pinWorkers = false)
        : Worker(dctx), _q(std::move(deques)), _physical_ids(std::move(physical_ids)),
          _unique_threads(std::move(unique_threads)), _verbose(verbose), _fid(fid), _batchSize(batchSize),
          _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode), _victimSelection(victimSelection),
          _pinWorkers(pinWorkers) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPU::run, this);
    }

    ~WorkerCPU() override = default;

    void run() override {
        if (_pinWorkers) {
#ifdef __linux__
            // pin worker to CPU core
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(_threadID, &cpuset);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#endif
        }

        int currentDomain = _physical_ids[_threadID];
        ctx->logger->debug("Thread{}, _physical_ids.size()={}, capacity={}, currentDomain={}", _threadID,
                           _physical_ids.size(), _physical_ids.capacity(), currentDomain);
        int targetQueue = _threadID;
        if (_queueMode == QueueTypeOption::CENTRALIZED) {
            targetQueue = 0;
        } else if (_queueMode == QueueTypeOption::PERGROUP) {
            targetQueue = currentDomain;
        } else if (_queueMode == QueueTypeOption::PERCPU) {
            targetQueue = _threadID;
        } else {
            ctx->logger->error("WorkerCPU: queue not found");
        }
        int startingQueue = targetQueue;

        Task *t = _q[targetQueue]->dequeueTask();

        while (!isEOF(t)) {
            // execute self-contained task
            if (_verbose)
                ctx->logger->trace("WorkerCPU: executing task.");
            t->execute(_fid, _batchSize);
            delete t;
            // get next tasks (blocking)
            t = _q[targetQueue]->dequeueTask();
        }

        // All tasks from own queue have completed. Now stealing from other
        // queues.

        if (_numQueues > 1) {
            if (_victimSelection == VictimSelectionLogic::SEQ) {
                // Stealing in sequential order

                targetQueue = (targetQueue + 1) % _numQueues;

                while (targetQueue != startingQueue) {
                    t = _q[targetQueue]->dequeueTask();
                    if (isEOF(t)) {
                        targetQueue = (targetQueue + 1) % _numQueues;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                }
            } else if (_victimSelection == VictimSelectionLogic::SEQPRI) {
                // Stealing in sequential order from same domain first
                if (_queueMode == QueueTypeOption::PERCPU) {
                    targetQueue = (targetQueue + 1) % _numQueues;

                    while (targetQueue != startingQueue) {
                        if (_physical_ids[targetQueue] == currentDomain) {
                            t = _q[targetQueue]->dequeueTask();
                            if (isEOF(t)) {
                                targetQueue = (targetQueue + 1) % _numQueues;
                            } else {
                                t->execute(_fid, _batchSize);
                                delete t;
                            }
                        } else {
                            targetQueue = (targetQueue + 1) % _numQueues;
                        }
                    }
                }

                // No more tasks on this domain, now switching to other domain

                targetQueue = (targetQueue + 1) % _numQueues;

                while (targetQueue != startingQueue) {
                    t = _q[targetQueue]->dequeueTask();
                    if (isEOF(t)) {
                        targetQueue = (targetQueue + 1) % _numQueues;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                }
            } else if (_victimSelection == VictimSelectionLogic::RANDOM) {
                // stealing from random workers until all workers EOF

                eofWorkers.fill(false);
                while (std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues) {
                    targetQueue = rand() % _numQueues;
                    if (eofWorkers[targetQueue] == false) {
                        t = _q[targetQueue]->dequeueTask();
                        // std::cout << "Execute task stolen from: " <<
                        // targetQueue << std::endl;
                        if (isEOF(t)) {
                            eofWorkers[targetQueue] = true;
                        } else {
                            t->execute(_fid, _batchSize);
                            delete t;
                        }
                    }
                }

            } else if (_victimSelection == VictimSelectionLogic::RANDOMPRI) {
                // stealing from random workers from same socket first
                int queuesThisDomain = 0;
                eofWorkers.fill(false);

                for (int i = 0; i < _numQueues; i++) {
                    if (_physical_ids[i] == currentDomain) {
                        queuesThisDomain++;
                    }
                }
                if (_queueMode == QueueTypeOption::PERCPU) {
                    while (std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < queuesThisDomain) {
                        targetQueue = rand() % _numQueues;
                        if (_physical_ids[targetQueue] == currentDomain) {
                            if (eofWorkers[targetQueue] == false) {
                                t = _q[targetQueue]->dequeueTask();
                                if (isEOF(t)) {
                                    eofWorkers[targetQueue] = true;
                                } else {
                                    t->execute(_fid, _batchSize);
                                    delete t;
                                }
                            }
                        }
                    }
                }

                // all workers on same domain are EOF, now also allowing
                // stealing from other domain This could also be done by keeping
                // a list of EOF workers on the other domain

                while (std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues) {
                    targetQueue = rand() % _numQueues;
                    // no need to check if they are on the other domain, because
                    // otherwise they would be EOF anyway
                    if (eofWorkers[targetQueue] == false) {
                        t = _q[targetQueue]->dequeueTask();
                        if (isEOF(t)) {
                            eofWorkers[targetQueue] = true;
                        } else {
                            t->execute(_fid, _batchSize);
                            delete t;
                        }
                    }
                }
            }
        }

        // No more tasks available anywhere
        if (_verbose)
            ctx->logger->debug("WorkerCPU: received EOF, finalized.");
    }
};
