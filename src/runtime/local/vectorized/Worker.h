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

#include <runtime/local/vectorized/Tasks.h>

#include <numeric>
#ifdef __linux__
#include <sched.h>
#endif
#include <thread>

class Worker {
  protected:
    std::unique_ptr<std::thread> t;

    DCTX(ctx);

    // Worker only used as derived class, which starts the thread after the
    // class has been constructed (order matters).
    explicit Worker(DCTX(dctx)) : ctx(dctx) {}

  public:
    // Worker is move only due to std::thread. Therefore, we delete the copy
    // constructor and the assignment operator.
    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;

    // move constructor
    Worker(Worker &&obj) noexcept : t(std::move(obj.t)), ctx(obj.ctx) {}

    // move assignment operator
    Worker &operator=(Worker &&obj) noexcept {
        if (t->joinable())
            t->join();
        t = std::move(obj.t);
        ctx = obj.ctx;
        return *this;
    }

    virtual ~Worker() {
        if (t->joinable())
            t->join();
    };

    void join() { t->join(); }
    virtual void run() = 0;
    static bool isEOF(Task *t) { return dynamic_cast<EOFTask *>(t); }
};
