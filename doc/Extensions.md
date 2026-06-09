<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Custom Extensions to DAPHNE

DAPHNE will be extensible in various respects.
Users will be able to add their own kernels, data types, value types, compiler passes, runtime schedulers, etc. without changing the DAPHNE source code itself.

So far, DAPHNE has initial support for adding custom kernels.

## Custom Kernel Extensions

Users can add their own custom kernels (physical operators) to DAPHNE following a three-step approach:

1. The extension is implemented as a stand-alone code base.
2. The extension is compiled as a shared library.
3. The extension is used in a DaphneDSL script or via DAPHNE's Python API.

Since this feature is still in an early stage, we only mention the most important points here rather than providing a full reference of what's supported.

Furthermore, we include a running example of adding two custom kernels for the summation of a dense matrix of single-precision floating-point values. 
We are interested in two variants: one sequential implementation for the CPU and one implementation that uses SIMD instructions from Intel's AVX (256-bit vector registers) on the CPU.
All files shown below can be found in `/scripts/examples/extensions/myKernels/`.

### Step 1: Implementing a Kernel Extension

A kernel extension consists at least of the following:

- A *C++ source file*, which includes some essential DAPHNE headers and defines one or multiple kernel functions.
    The kernel functions have to follow a certain interface **(*)** and have `extern "C"` linkage.
    Within the kernel functions, extension developers have a lot of freedom.
    Nevertheless, we also plan to provide some best practices and helpers to make extension development more productive.
- A *kernel catalog JSON file*, which provides some essential information on the kernels provided in the extension, such that DAPHNE knows how to use them.
    This information includes: the mnemonic of the DaphneIR operation **(*)**, the name of the kernel function, the list of result/argument types, the backend (e.g. CPU or a specific hardware accelerator), and the path to the shared library of the extension (relative to this JSON file).
- To build the extension, it is recommendable (but not required) to include a Makefile or similar as well.

**(*)** *We will add a concrete list of DaphneIR operations for which custom kernels can be added later.
This list will be understandable by DAPHNE users, and will contain the operations' mnemonics, arguments, results, as well as expected C++ kernel function interfaces.
In the meantime, developers familiar with DAPHNE internals can already find references of the DaphneIR operations in `src/ir/daphneir/DaphneOps.td` and a reference of the kernel interfaces in `build/runtime/local/kernels/kernels.cpp` (generated during the DAPHNE build).*

*Running example:*

C++ source file `myKernels.cpp`:
```c++
#include <runtime/local/datastructures/DenseMatrix.h>

#include <immintrin.h> // for the SIMD-enabled kernel
#include <iostream>
#include <stdexcept>

class DaphneContext;

extern "C" {
// Custom sequential sum-kernel.
void mySumSeq(float *res, const DenseMatrix<float> *arg, int kernelId, DaphneContext *ctx) {
    std::cerr << "hello from mySumSeq()" << std::endl;

    const float *valuesArg = arg->getValues();
    *res = 0;
    for (size_t r = 0; r < arg->getNumRows(); r++) {
        for (size_t c = 0; c < arg->getNumCols(); c++)
            *res += valuesArg[c];
        valuesArg += arg->getRowSkip();
    }
}

// Custom SIMD-enabled sum-kernel.
void mySumSIMD(float *res, const DenseMatrix<float> *arg, int kernelId, DaphneContext *ctx) {
    std::cerr << "hello from mySumSIMD()" << std::endl;

    // Validation.
    const size_t numCells = arg->getNumRows() * arg->getNumCols();
    if (numCells % 8)
        throw std::runtime_error("for simplicity, the number of cells must be "
                                 "a multiple of 8");
    if (arg->getNumCols() != arg->getRowSkip())
        throw std::runtime_error("for simplicity, the argument must not be "
                                 "a column segment of another matrix");

    // SIMD accumulation (8x f32).
    const float *valuesArg = arg->getValues();
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = 0; i < numCells / 8; i++) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(valuesArg));
        valuesArg += 8;
    }

    // Summation of accumulator elements.
    *res = (reinterpret_cast<float *>(&acc))[0] + (reinterpret_cast<float *>(&acc))[1] +
           (reinterpret_cast<float *>(&acc))[2] + (reinterpret_cast<float *>(&acc))[3] +
           (reinterpret_cast<float *>(&acc))[4] + (reinterpret_cast<float *>(&acc))[5] +
           (reinterpret_cast<float *>(&acc))[6] + (reinterpret_cast<float *>(&acc))[7];
}
}
```

Kernel catalog file `myKernels.json`:
```json
[
  {
    "opMnemonic": "sumAll",
    "kernelFuncName": "mySumSeq",
    "resTypes": ["float"],
    "argTypes": ["DenseMatrix<float>"],
    "backend": "CPP",
    "libPath": "libMyKernels.so"
  },
  {
    "opMnemonic": "sumAll",
    "kernelFuncName": "mySumSIMD",
    "resTypes": ["float"],
    "argTypes": ["DenseMatrix<float>"],
    "backend": "CPP",
    "libPath": "libMyKernels.so"
  }
]
```

`Makefile`:
```make
libMyKernels.so: myKernels.o
	g++ -shared myKernels.o -o libMyKernels.so

myKernels.o: myKernels.cpp
	g++ -c -fPIC myKernels.cpp -I../../../../src/ -std=c++17 -O3 -mavx2 -o myKernels.o
    
clean:
	rm -rf myKernels.o libMyKernels.so
```

### Step 2: Building a Kernel Extension

The kernel extension must be built as a shared library.
Additional details will follow.

*Running example:*

Given the Makefile above, the extension is built by simply running `make` in the extension's directory, which produces the shared library `libMyKernels.so`:

```bash
make
```

### Step 3: Using a Kernel Extension

The kernels in a kernel extension can be used either automatically by DAPHNE or manually by the user.
The manual use has precedence over the automatic use.

#### Manual Use of Custom Kernels

The manual employment of custom kernels is very useful for experimentation, e.g., to see the impact of a particular kernel at a certain point of a larger integrated data analysis pipeline.
To this end, DaphneDSL [compiler hints](/doc/DaphneDSL/LanguageRef.md#compiler-hints) tell DAPHNE to use a specific kernel in a specific place, even though DAPHNE's optimizing compiler may not choose the kernel, otherwise.

*Running example:*

A minimal example using a summation on a matrix of single-precision floating-point values could look as follows:

`demo.daphne`:
```R
# Create a matrix of random f32 values in [0, 1] (400 MiB).
X = rand(10^4, 10^4, as.f32(0), as.f32(1), 1, 12345);
# Calculate the sum over the matrix.
s = sum(X);
# Print the sum.
print(s);
```

We execute this script from the DAPHNE root directory by:
```bash
bin/daphne scripts/examples/extensions/myKernels/demo.daphne
```

In order to manually use our custom sequential `sum`-kernel, we add the DAPHNE compiler hint `::mySumSeq` to the script:

`demoSeq.daphne`:
```R
X = rand(10^4, 10^4, as.f32(0), as.f32(1), 1, 12345);
s = sum::mySumSeq(X);
print(s);
```

We execute this script with the following command, whereby the argument `--kernel-ext` specifies the kernel catalog JSON file of the extension to use:
```bash
bin/daphne --kernel-ext scripts/examples/extensions/myKernels/myKernels.json scripts/examples/extensions/myKernels/demoSeq.daphne
```

Alternatively, we can try our custom SIMD-enabled `sum`-kernel by adapting the compiler hint accordingly:

`demoSIMD.daphne`:
```R
X = rand(10^4, 10^4, as.f32(0), as.f32(1), 1, 12345);
s = sum::mySumSIMD(X);
print(s);
```

We execute this script by:
```bash
bin/daphne --kernel-ext scripts/examples/extensions/myKernels/myKernels.json scripts/examples/extensions/myKernels/demoSIMD.daphne
```

#### Automatic Use of Custom Kernels

The automatic use of custom kernels is currently restricted to the selection of a kernel based on its result/argument data/value types and its priority level.
In the future we plan to support custom cost models as well.

*Running example:*

Continuing the running example from above, we can make DAPHNE use the custom kernels `mySumSeq()` or `mySumSIMD()` even without a manual kernel hint by specifying a suitable *priority* when registering the `myKernels` extension with DAPHNE.

Priority levels can optionally be specified with the `--kernel-ext` command line argument by appending a colon (`:`) followed by the priority as an integer.
The default priority of `0` is used for all built-in kernels and for extension kernels in case no priority is specified.
When registering a kernel extension, the given priority is assigned to *all* kernels provided by the extension.
When multiple kernels are applicable for an operation based on the combination of argument/result data/value types as well as the backend, DAPHNE chooses the kernel with the highest priority.
If there are multiple kernels with the highest priority, it is not specified which of them is used.

By registering a kernel extension with a priority greater than zero, one can enforce that the kernels provided by the extension are always preferred over the built-in ones whenever they are applicable.
For instance, the following command registers the `myKernels` extension with a priority of `1`.
As the `myKernels` extension provides two kernels for the same operation, argument/result types, and backend, we cannot tell, based on priorities, which of these kernels will be used, but we can be sure that the built-in kernel will not be employed.

```bash
bin/daphne --kernel-ext scripts/examples/extensions/myKernels/myKernels.json:1 scripts/examples/extensions/myKernels/demo.daphne
```