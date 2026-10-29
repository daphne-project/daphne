#!/usr/bin/env python3

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

"""
Fails fast at CMake configure time if any kernel that ships its template
definition out-of-line (via a hand-written .cpp with explicit template
instantiations) has drifted from what is listed in kernels.json.

Currently guards a single kernel:

    DistributedPipeline.cpp   <->   kernels.json entry for `distributedPipeline`

Shipping distributedPipeline's definition out-of-line in
DistributedPipeline.cpp removes a large transitive header closure from the
generated kernels_*.cpp that call it, but the price is that every type
listed under `distributedPipeline.api[].instantiations` in kernels.json
must ALSO appear as an explicit `template void distributedPipeline<...>`
line in DistributedPipeline.cpp. Otherwise the generated wrapper linked
by KernelObjLib will fail to find the definition, a linker error that
surfaces late in the build.

Add further entries to CHECKS below if this pattern is applied to more
kernels in the future.
"""

import json
import re
import sys
from pathlib import Path

# List of (kernel opName, .cpp path relative to this script's directory) pairs
# to verify. Each .cpp is expected to carry one `template void <opName><...>`
# line per instantiation listed under that opName in kernels.json.
CHECKS = [
    ("distributedPipeline", "DistributedPipeline.cpp"),
]


def flatten_type(spec):
    """Turn kernels.json's nested list representation of a C++ type into a
    canonical single-line spelling. Whitespace is normalized so that the
    comparison against DistributedPipeline.cpp is not defeated by formatting.

    Examples:
        ["DenseMatrix", "double"]     ->  "DenseMatrix<double>"
        [["DenseMatrix", "double"]]   ->  "DenseMatrix<double>"   (unary tuple)
    """
    if isinstance(spec, str):
        return spec
    if isinstance(spec, list):
        if len(spec) == 1 and isinstance(spec[0], list):
            # kernels.json wraps single-template-arg instantiations in an
            # extra list layer. Peel it.
            return flatten_type(spec[0])
        if len(spec) >= 1 and isinstance(spec[0], str):
            head, *args = spec
            if not args:
                return head
            return f"{head}<{', '.join(flatten_type(a) for a in args)}>"
    raise ValueError(f"cannot flatten {spec!r}")


def instantiations_from_json(json_path, op_name):
    """Return the set of canonical type spellings listed for op_name under
    the CPP api in kernels.json."""
    with open(json_path) as f:
        data = json.load(f)
    for entry in data:
        kt = entry.get("kernelTemplate", {})
        if kt.get("opName") != op_name:
            continue
        for api in entry.get("api", []):
            names = api.get("name", [])
            if "CPP" not in names:
                continue
            result = set()
            for inst in api.get("instantiations", []):
                # Each `inst` is a list of template-arg specs (one per
                # templateParams entry). For distributedPipeline that's a
                # single DTRes -> so a one-element list.
                types = [flatten_type(t) for t in inst]
                result.add(", ".join(types))
            return result
    return None  # opName not found


# Match `template void <opName><...>(`, capturing the template argument list.
def instantiations_from_cpp(cpp_path, op_name):
    """Return the set of canonical type spellings appearing in explicit
    `template void <op_name><...>` instantiations in the given .cpp."""
    text = Path(cpp_path).read_text()
    # Strip C++ line comments and block comments before matching so that a
    # comment that mentions the pattern (e.g. a doc line describing what
    # explicit instantiations look like) is not misread as a real one.
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    pattern = re.compile(
        r"template\s+void\s+" + re.escape(op_name) + r"\s*<(.+?)>\s*\(",
        re.DOTALL,
    )
    result = set()
    for m in pattern.finditer(text):
        # Normalize whitespace: collapse runs to single spaces.
        canonical = re.sub(r"\s+", " ", m.group(1).strip())
        result.add(canonical)
    return result


def main():
    if len(sys.argv) != 2:
        print(
            f"usage: {sys.argv[0]} <path-to-kernels.json>",
            file=sys.stderr,
        )
        return 2

    json_path = Path(sys.argv[1])
    here = Path(__file__).resolve().parent

    all_ok = True
    for op_name, cpp_rel in CHECKS:
        cpp_path = here / cpp_rel
        expected = instantiations_from_json(json_path, op_name)
        if expected is None:
            print(
                f"error: kernels.json has no CPP api entry for opName "
                f"'{op_name}'; either add one or remove this check.",
                file=sys.stderr,
            )
            all_ok = False
            continue
        found = instantiations_from_cpp(cpp_path, op_name)

        missing = expected - found
        extra = found - expected
        if missing or extra:
            all_ok = False
            print(
                f"error: explicit-instantiation drift for '{op_name}' "
                f"between kernels.json and {cpp_rel}:",
                file=sys.stderr,
            )
            for t in sorted(missing):
                print(
                    f"  kernels.json lists '{op_name}<{t}>' but "
                    f"{cpp_rel} does NOT explicitly instantiate it.",
                    file=sys.stderr,
                )
                print(
                    f"    fix: add `template void {op_name}<{t}>(...);` to "
                    f"{cpp_rel}, matching the signature already there.",
                    file=sys.stderr,
                )
            for t in sorted(extra):
                print(
                    f"  {cpp_rel} explicitly instantiates '{op_name}<{t}>' "
                    f"but kernels.json does not list it.",
                    file=sys.stderr,
                )
                print(
                    f"    fix: either add the type to kernels.json under "
                    f"'{op_name}', or drop the explicit instantiation.",
                    file=sys.stderr,
                )

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
