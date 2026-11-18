#!/bin/bash

# Copyright 2026 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Baseline test runner for issue #512 (Simplification Rewrites for Linear
# Algebra) work.
#
# This wrapper invokes ./test.sh with two Catch2 tag exclusions that filter
# out pre-existing aarch64-only failures unrelated to #512:
#
#   ~[distributed]  issue #1011: "Simple distributed worker functionality
#                      test" aborts the run_tests binary via SIGABRT on
#                      aarch64, preventing the remaining ~2,111 downstream
#                      test cases from executing at all.
#
#   ~[sql]          issue #1012: --columnar mode crashes on aarch64 with
#                      an llvm::cast<TypedValue<IndexType>> assertion when
#                      running columnar_4..7 and group_6 scripts.
#
# Both failures are aarch64-only; upstream x86_64 CI is unaffected.
# Drop the corresponding flag once #1011 / #1012 are fixed.
#
# All arguments to this script are forwarded to ./test.sh (which in turn
# forwards them to the Catch2 main function).
#
# Invoke from the repository root:
#   ./scripts/testing/test-issue512.sh --no-build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# Catch2 tag exclusions. Each is documented against its upstream issue above.
EXCLUDE_DISTRIBUTED="~[distributed]"  # issue #1011
EXCLUDE_SQL="~[sql]"                  # issue #1012

exec ./test.sh "$@" "$EXCLUDE_DISTRIBUTED" "$EXCLUDE_SQL"
