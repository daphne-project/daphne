<!--
Copyright 2025 The DAPHNE Consortium

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

<!-- TODO make paths to source files links -->
<!-- TODO maybe add an overview picture (might be ASCII art) -->
<!-- TODO elaborate on testing related to data properties -->
<!-- TODO blueprints for the files to create would be helpful -->
<!-- TODO maybe generally add more little examples throughout the text -->
<!-- TODO maybe split this doc into multiple and create a folder for them in the nav (might be just too long otherwise) -->

# Data Properties: Representation, Inference, and Exploitation

DAPHNE data objects (matrices/frames) can be described by various *data properties* (synonym: *data characteristics*), such as the data type and value type, the shape, the sparsity, the number of distinct values, the value distribution, etc.

Information about the data properties can be exploited for optimizing runtime and memory consumption in several ways, e.g., through:

- *Simplification rewrites* (e.g., eliminate transpose on a symmetric matrix, eliminate row-wise sum on a single-column matrix, etc.)
- *Physical data/value type selection* (e.g., sparse representations like CSR for sparse data, compressed representations for small integers, etc.)
- *Physical operator/algorithm selection* (e.g., special algorithms on sparse data, special algorithms on sorted data, etc.)

This document explains how DAPHNE handles interesting data properties and justifies why it is done this way. Furthermore, it explains how to extend DAPHNE with additional data properties.

## Currently Supported Data Properties

DAPHNE supports an individual set of data properties for each data type as described in the list below.
We view the data and value type as a special kind of data property.
The fact that DAPHNE supports a certain data property does not mean that the data property will be known for every SSA value; some data properties may not be inferred because (a) they simply cannot be known in a certain situation, or (b) there are gaps due to the current state of the implementation.

<!-- TODO for reference: add possible value ranges, unknown value, etc. -->
- **Matrix**
    - *value type:* the common type of all matrix elements
    - *shape (number of rows, number of columns):* the logical size of the matrix
    - *sparsity:* the number of non-zero elements divided by the total number of elements
    - *symmetry:* whether the *(n x n)* matrix *X* is symmetric, i.e., for any *0 ≤ i, j ≤ n: X[i, j] = X[j, i]*
    - *numDistinct:* the number of distinct matrix values (-1 denotes unknown)
    - *value bounds:* known lower and upper bounds on matrix values, represented as optional doubles
    - *physical representation:* the way the matrix is stored in memory (e.g., dense or sparse)
- **Frame**
    - *value types:* the array of types for each column
    - *shape (number of rows, number of columns):* the logical size of the frame
    - *column labels:* the array of textual labels for each column (labels must be unique for each frame)
- **Column**
    - *value type:* the common type of all column elements
    - *shape (number of rows):* the logical size of the column
- **Scalar**
    - *(none so far)*

## Representation of Data Properties

**Data properties are represented in the DaphneIR and in the DAPHNE runtime.**
Thus, the information on data properties is available at compile-time and at run-time.
The rationale for a compile-time representation (as opposed to a pure run-time representation) is twofold:

- Some ways of inferring data properties require a global view on the program.
    This global view is only available at compile-time.
    For instance, the result of `t(X) @ X` is a symmetric matrix for any matrix `X`, but that cannot easily be derived from looking at the individual ops.
- Some ways of exploiting data properties for optimizations need to take place at compile-time, because they apply global optimizations to the IR (e.g., simplification rewrites, operator ordering, code generation, and operator fusion).

**In DaphneIR, data properties are stored as [parameters](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#parameters) of custom MLIR types** (see `src/ir/daphneir/DaphneTypes.td`) (e.g., `Matrix`/`mlir::daphne::MatrixType` and `Frame`/`mlir::daphne::FrameType`).

There would have been some alternative approaches to using MLIR type parameters.
Those alternatives would not necessarily be impossible to use for the purpose of representing data properties.
However, we decided against them.
We considered the following alternatives:

- *MLIR attributes on MLIR operations*.
    We decided against this approach, because attributes attached to ops typically denote the data properties of the op's result(s).
    We want to be able to represent data properties of any SSA value (`mlir::Value`), but not all SSA values are produced by an MLIR operation (some are block arguments).
    An operation can have multiple results, would lead to duplication of attributes.
- *MLIR [analysis passes](https://mlir.llvm.org/docs/PassManagement/#analysis-management)*.
    **TODO** Explain why not.

Nevertheless, the chosen approach of representing data properties as MLIR type parameters also has some downsides:

- Data properties technically become a part of the MLIR type.
    Hence, we cannot use MLIR's type inference interface unless we make type inference always infer all data properties (which might be costly).
    However, this point is not too bad, since we have our own type inference interface.
- As a consequence, type comparisons can have, at first glance, unexpected results (e.g., a matrix of value type `f64` with sparsity `0.5` is not the same MLIR type as matrix of value type `f64` with sparsity `0.1`).
    As a workaround, we introduced the `withSameValueType()` utility function on our custom MLIR types.
    This functions sets all data properties to *unknown* except the value type(s).
    After invoking this utility function, type comparisons will only consider the data/value type, so the comparison result should be as expected.

**In the DAPHNE runtime, data properties are stored as member variables of DAPHNE data objects.**
More precisely, the data properties are stored at the level of the data type superclasses `Matrix` and `Frame` (see `src/runtime/local/datastructures/Matrix.h` and `src/runtime/local/datastructures/Frame.h`).
The rationale for not storing data properties at the level of individual data type subclasses (such as `DenseMatrix` and `CSRMatrix`) is that data properties describe the logical data.
While certain data properties may make certain physical representations more suitable, the data properties and the physical representation of a data object can be viewed independently of each other.

**Each data property can have an *unknown value*.**
After DaphneDSL parsing, most (almost all) data properties are unknown in the initial DaphneIR representation.
Through data property inference, some/many data properties can become known at compile-time, but not necessarily all.
At any point in time during compilation, a data property may not be known (yet).
By the end of the DAPHNE compiler (more precisely, by `RewriteToCallKernelOpPass`) the data/value type of all SSA values must be known (otherwise DAPHNE cannot select the right runtime kernel for the operation), except for the column value types of frames, which may remain unknown at compile-time (because the column value types of frames are interpreted dynamically at run-time).

For each data property, there is some special value that represents *unknown*:

- Data/value type: DAPHNE's custom `Unknown` type (`mlir::daphne::UnknownType`)
- Non-negative numeric data properties (e.g., the number of rows, the number of columns, number of distinct values, sparsity, etc.): typically `-1`
<!-- TODO maybe we will use sth like Optional<T> in the future -->
- Value bounds: each endpoint is a `std::optional<double>`. `std::nullopt` denotes unknown
- Complex object-valued data properties (e.g., the array of column labels of a frame): typically `nullptr`

**The compile-time information on the data properties is also available to the runtime.**

- On the one hand, some *fundamental data properties* are anyway available at run-time, because they are part of the fundamental meta data of the runtime data structures:
    <!-- TODO code snippets for accessing these -->
    - **Matrix**
        - value type (usually as a template parameter of a kernel)
        - shape (number of rows, number of columns)
    - **Frame**
        - column value types
        - column labels

    Even in case these data properties are not known at compile-time for a particular intermediate result, they are *always* known at run-time *at no extra cost*.

- On the other hand, there are some *additional data properties*, whose analysis would incur non-negligible extra effort at run-time:
    - **Matrix**
        - sparsity
        - symmetry
        - number of distinct values (`numDistinct`)
        - value bounds

    So far, these properties are not analyzed at run-time due to the implied extra cost.
    Instead, the compile-time information on these properties is transferred to the runtime data objects.
    To this end, there is a dedicated compiler pass, `TransferDataPropertiesPass` (see `src/compiler/inference/TransferDataPropertiesPass.cpp`), which inserts a `TransferPropertiesOp` (see `src/ir/daphneir/DaphneOps.td`) for every matrix-valued op result.
    The op gets a matrix as well as the scalar values of the data properties mentioned above as its arguments.
    The compiler pass extracts the values of the data properties (which may be unknown) from the MLIR types and wraps them in `ConstantOp`s, which become the arguments of the `TransferPropertiesOp`.
    Like most DaphneIR operations, `TransferPropertiesOp` gets lowered to a kernel call; the corresponding `transferProperties`-kernel can be found in `src/runtime/local/kernels/TransferProperties.h`.

## Data Property Inference

At compile-time, DAPHNE tries to find out the data properties of intermediate results in the IR.
This process is called *property inference* (synonyms: *property propagation*, *property estimation*).
The basic idea is to infer the data properties of an op's results from the data properties of the op's arguments (and, perhaps, from compile-time constant argument values).
Currently, DAPHNE does this inference only at the level of each individual op.
In the future, it could also infer data properties for patterns/DAGs of ops (e.g., the result of `t(X) @ X` is a symmetric matrix for any matrix `X`, which can hardly be inferred based on the individual ops `t` and `@`).

**The compiler pass responsible for data property inference is the `InferencePass`** (`src/compiler/inference/InferencePass.cpp`).
Besides that, `SelectMatrixRepresentationsPass` (`src/compiler/inference/SelectMatrixRepresentationsPass.cpp`) also plays a role as it selects the physical matrix representation (dense/sparse) based on the estimated sparsity, but maybe we will factor it into `InferencePass` in the future.

**Procedure for data property inference.**
The `InferencePass` walks the IR op by op and tries to infer all data properties for all results of the current op before moving on to the next op.

The rationale for this *"for each op: for each prop: infer"*-procedure rather than a *"for each prop: for each op: infer"*-procedure is that a data property *X* of an op's result does not always depend on the data property *X* of the op's argument(s), but sometimes on another data property *Y* of the argument(s).
For instance, the number of rows of the result of `FilterRowOp` depends on the sparsity of the input bit vector.
Thus, before inferring any data property of the result of an op, we should have as much information on all data properties of the argument(s) of the op as possible.

Given a certain op, the `InferencePass` considers the supported data properties one by one.
If the value of a data property is unknown for any result of the op, the inference logic for this data property on this op is invoked.
The rationale for not re-inferring already known data properties is efficiency.

**The op-specific inference of data properties is handled by custom MLIR [interfaces](https://mlir.llvm.org/docs/Interfaces/) and [traits](https://mlir.llvm.org/docs/Traits/).**
By default, there is one custom MLIR interface per supported data property.
The functions defined by the interface become part of the op the interface is attached to.
The interface returns the inferred value of the data property (usually some kind of list, e.g., a `std::vector`, because an op can have multiple results).

The rationale for not making the interface set the data property on the results of the op are as follows:
(1) The logic for setting the data property value would need to be redundantly implemented by every single interface implementation.
Even though this logic is usually just a one-liner, there is a potential for bugs.
Moreover, one would hard-code the way data properties are represented (e.g., type parameters vs. attributes) in every single interface implementation.
(2) We want a single point of insertion into the IR, because data properties could also be inferred by traits (see below).

The **source files involved in the inference interface of a data property** `${prop}` are the following (look at the files of some existing data property for the details):

- `src/ir/daphneir/DaphneInfer${prop}OpInterface.td`
    - A TableGen-based declaration of the custom MLIR interface.
    - Typically the interface defines a single function `infer${prop}()`.
    - In justified cases, there can be multiple interfaces.
        For instance, the interface for shape inference comes with separate interfaces for the inference of the number of rows and columns (because sometimes, just one of the two dimensions can be expressed by a trait, see below).
- `src/ir/daphneir/DaphneInfer${prop}OpInterface.h`
    - Defines the custom MLIR traits for this data property (C++ part).
    - Include the TableGen'erated header file `ir/daphneir/DaphneInfer${prop}OpInterface.h.inc`
    - Declares the function `tryInfer${prop}(mlir::Operation *op)`.
        (Type inference also has a function `setInferredTypes()`, because we want to set the data/value type also in the parser and compiler passes.)
- `src/ir/daphneir/DaphneInfer${prop}OpInterface.cpp`
    - Op-specific implementations of the inference interface, i.e., functions `${op}::infer${prop}()` (for an op `${op}`).
    - Trait implementations.
    - Definition of `tryInfer${prop}(mlir::Operation *op)`.
        This function first checks if op has the respective inference interface; if so, it calls `infer${prop}()` on the op.
        Otherwise: it checks if the op has exactly one result and a related inference trait; if so, it applies the trait.
        Otherwise: it sets the data property to unknown for all results.
        <!-- TODO explain why not leave it as it is -->

In fact, many ops have a quite similar implementation of an inference interface.
Thus, we use **custom [MLIR traits](https://mlir.llvm.org/docs/Traits/)** to express common inference behavior.
Traits can refer to the entire data property or an aspect of it.
For instance, type inference has some traits that express how to infer the complete data and value type, and some traits for either the data or the value type (for better reuse of traits).

The **source files involved in the inference traits of a data property** `${prop}` are the following (again, see concrete examples in the code for more details):

- `src/ir/daphneir/DaphneInfer{$prop}Traits.td`
    - TableGen-based definition of custom (parametric) MLIR traits; essentially just references to the C++ part in `src/ir/daphneir/DaphneInfer${prop}OpInterface.h`.

**Property inference and control flow constructs.**

**TODO**

**Relation to constant folding.**
In the DAPHNE compilation chain, data property inference is executed multiple times and interleaved with the MLIR `CanonicalizationPass`.

The rationale for this repetition and interleaving is that [canonicalization](https://mlir.llvm.org/docs/Canonicalization/) includes [constant folding](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-the-fold-method), i.e., calculating the result of an op at compile-time if all inputs are compile-time constants.
On the one hand, the result *value* of some ops depends on a *data property* of an argument (e.g., for `NumRowsOp`, the result value is the argument's number of rows).
On the other hand, a result *data property* of some ops depends on an argument *value* (e.g., for `FillOp`, the result shape is given by argument values).
Thus, data property inference and constant folding depend on each other.

**Viewing the data properties inferred by the DAPHNE compiler** is possible through DAPHNE's explain functionality.
When invoking DAPHNE with the argument `--explain property_inference`, the IR after data property inference is printed.
The data properties of an SSA value are displayed as a part of its type, whereby the concrete syntax is determined by `mlir::daphne::DaphneDialect::printType()` (see `src/ir/daphneir/DaphneDialect.cpp`).

**Command-line arguments related to data property inference.**

- `--select-matrix-repr`
    Makes DAPHNE use the estimated sparsities of intermediate results for selecting a suitable physical representation of a matrix (especially `DenseMatrix` vs. `CSRMatrix`) and, hence, the respective kernels, e.g., those for `CSRMatrix`.
    Currently, *DAPHNE will only use sparse representations and kernels if this argument is provided*, because sparsity support is still experimental in some respects.
    In the future, sparsity support will be on by default.
- `--explain property_inference`
    Makes DAPHNE print the IR after data property inference (see above).

**Relation to file meta data.**

**TODO**

## Data Property Exploitation

### Declaring Relevant Input Properties

The `DataPropertyRelevance` operation interface declares which input data properties runtime kernels (implementation variants of operations) may exploit to choose specialized execution paths. Actual relevance depends on the concrete kernel implementation.

The method `getRelevantProperties(operandIndex, properties)` appends the relevant property kinds for a given operand, allowing a future compiler pass to determine which properties to analyze at run-time.

To add a declaration, attach `DeclareOpInterfaceMethods<DataPropertyRelevanceOpInterface>` in `DaphneOps.td` and implement `getRelevantProperties` in `DaphneDataPropertyRelevanceOpInterface.cpp`.

## Extending DAPHNE's Support for Data Properties

DAPHNE's support for data properties can be extended by modifying the source code straightforwardly.
This section provides step-by-step instructions for typical tasks, summarizing what needs to be done in which files.
We focus on giving high-level explanations and pointers; for the details, the existing code (which contains several examples of data properties) can be used as a blueprint.
Furthermore, the [MLIR documentation](https://mlir.llvm.org/docs/) provides more detailed information on how to do various things in MLIR.

### Adding a New Data Property

New data properties can be added to DAPHNE's abstract data types (matrix, frame, etc.).
Before adding a new data property called `${prop}`, think about (a) which DAPHNE data type you want to add the data property to (e.g., matrix or frame), and (b) of what C++ type the new data property shall be (e.g., `ssize_t`, `double`, `mlir::Type`, some custom C++ enum class, or some custom C++ class).
If the new data property shall be of a custom C++ type, add this new type to `src/ir/DataPropertyTypes.h` (or use one of the existing types there). In particular, if the new property shall have a small set of discrete values, you can use an enum class as the C++ type of the new data property.
In the following, we assume the new data property is of C++ type `${pt}` and we want to add it to the DAPHNE data type `${dt}`.
The following steps are required (commit da35f1eb197c5e993167d6c2862d98a513b40852 is an example for adding a new data property; however, note that some details of adding a new data property have evolved since this commit):

*Support for `${prop}` in the DAPHNE compiler:*

1. Add a new [parameter](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#parameters) `${prop}` to the custom MLIR type `${dt}` in `src/ir/daphneir/DaphneTypes.td`.
1. Adapt `mlir::daphne::detail::${dt}TypeStorage` (see `src/ir/daphneir/DaphneDialect.cpp`) by adding the new data property.
    In particular, update the constructor, the member functions `operator==()`, `hashKey()`, and `construct()`, and add a member variable for the new data property.
1. Add the default value (usually the *unknown* value) of the new data property to the custom builder in the `builders` section of the custom MLIR type `${dt}`.
1. Update the custom methods in the [`extraClassDeclaration`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#extra-declarations) of the custom MLIR type `${dt}`:
    - Add a method `with${prop}(${pt} ${prop})` that gets a value for the new data property and returns an instance of the MLIR type that sets the value of the new data property to the given value and retains the current value for all other data properties of `${dt}`.
    - Add a call to `get${prop}()` to all other `with*(...)` methods such that they retain the current value of the new data property `${prop}`.
    - Take the new data property into account in `isSpecializationOf(...)`.
1. Update the method `mlir::daphne::${dt}Type::verify(...)` (see `src/ir/daphneir/DaphneDialect.cpp`) by adding a validity check for the value of the new data property `${prop}`.
    This step is necessary only if the C++ type `${pt}` contains any values that are not valid for the data property `${prop}` or if any values might conflict with the values of other data properties.
1. Make sure the new data property gets represented in the textual IR, such that it is displayed, e.g., when running `bin/daphne --explain property_inference`, (optional step, but highly recommended).
    To that end, take the data property into account in `mlir::daphne::DaphneDialect::printType()` and `mlir::daphne::DaphneDialect::parseType()` (see `src/ir/daphneir/DaphneDialect.cpp`).
1. Create the inference interface for `${prop}`.
    - Add the files `src/ir/daphneir/DaphneInfer${prop}OpInterface.td`, `src/ir/daphneir/DaphneInfer${prop}OpInterface.h`, and `src/ir/daphneir/DaphneInfer${prop}OpInterface.cpp`.
        See the description [above](#data-property-inference) for more details on the meaning of these files.
        The existing files of other data properties can be used as blueprints (up to this point, you don't need any interface implementations for specific ops and you don't need any inference traits).
    - Update `src/ir/daphneir/CMakeLists.txt`: Add a line `add_mlir_interface(DaphneInfer${prop}OpInterface)` and add the `.cpp` file.
    - Add the include `#include <ir/daphneir/DaphneInfer${prop}OpInterface.h>` to `src/ir/daphneir/Daphne.h`.
    - Add the line `include "ir/daphneir/DaphneInfer${prop}OpInterface.td"` to the top of `src/ir/daphneir/DaphneOps.td` for future use.
1. Make the `InferencePass` consider the new data property (see `src/compiler/inference/InferencePass.cpp`):
    - Update `daphne::InferenceConfig::InferenceConfig`.
        Update `struct InferenceConfig` as well as the line `std::unique_ptr<Pass> createInferencePass(...)` in `src/ir/daphneir/Passes.h` accordingly.
        Update the lines `pm.addPass(daphne::createInferencePass(...));` in `src/compiler/lowering/SpecializeGenericFunctionsPass.cpp` accordingly.
    - Update the function `getTypeWithCommonInfo()`.
    - Update the lambda `walkOp`: In the then-branch of `if (!isScfOp)` (if the current op is not a control flow operation), there is a sequence of if-statements, one for each supported data properties.
        Add an if-statement for the new data property `${prop}` akin to the existing ones here.
        In that context, you will most likely want to add a new helper function `returnsUnknown${prop}()` for the new data property akin to the existing ones.

*Support for `${prop}` in the DAPHNE runtime:*

1. Add a member variable for `${prop}` in the runtime class corresponding to `${dt}` (e.g., in `src/runtime/local/datastructures/Matrix.h` or `src/runtime/local/datastructures/Frame.h`).
    This member variable should be of the C++ type `${pt}`.
    Ensure that the constructor of the class initializes the new member variable to the unknown value of `${prop}`.
1. Add a new argument for `${prop}` to the definition of the DaphneIR op `TransferPropertiesOp` in `src/ir/daphneir/DaphneOps.td`.
    Update the corresponding kernel `transferProperties` (`src/runtime/local/kernels/TransferProperties.h`) accordingly by adding a new argument for `${prop}` to the kernel; the kernel should set the new member variable on the argument to the provided value.
    Reflect the additional kernel argument in `src/runtime/local/kernels/kernels.json`.
1. Update `TransferDataPropertiesPass` (`src/compiler/inference/TransferDataPropertiesPass.cpp`) by extracting the value of `${prop}` from the MLIR type and passing it as an argument to the `TransferPropertiesOp` created by this pass.

*Documentation:*

1. Update the documentation on data properties (this document, `doc/development/PropertyInference.md`) by mentioning `${prop}` in the list of supported data properties at the top of the page.

### Making a New Op Support Inference for an Existing Data Property

To enable the op `${op}` to infer the data property `${prop}` for its result(s), we can either add the respective inference interface or an existing related inference trait to the TableGen record of the op `${op}` in `src/ir/daphneir/DaphneOps.td`.

#### Adding an Existing Inference Interface to the Op

1. Add the item `DeclareOpInterfaceMethods<Infer${prop}OpInterface>` to the definition of `${op}` in `src/ir/daphneir/DaphneOps.td` (see other ops for examples).
1. Make sure that `src/ir/daphneir/DaphneOps.td` has an include `include "ir/daphneir/DaphneInfer${prop}OpInterface.td"`.
1. Add the C++ implementation of the inference interface in `src/ir/daphneir/DaphneInfer${prop}OpInterface.cpp`.
    - Remember that any data property of any argument could have an unknown value.
    - Take inspiration from the existing inference interface implementations (of this or another data property).

Commit 4b120162ae26a2af1906f51fed48661926c87a48 is an example.

#### Adding an Existing Inference Trait to the Op

1. To add the inference trait `${trait}`, add the item `${trait}` to the definition of `${op}` in `src/ir/daphneir/DaphneOps.td` (see other ops for examples).
1. Make sure that `src/ir/daphneir/DaphneOps.td` has an include `include "ir/daphneir/DaphneInfer${prop}Traits.td"`.

### Adding a New Inference Trait for an Existing Data Property

It is not uncommon that multiple ops share a quite similar way of inferring the value of a data property `${prop}` of their result(s).
Rather than writing essentially the same inference interface implementation multiple times for these ops, we can optionally add a custom MLIR [trait](https://mlir.llvm.org/docs/Traits/) that expresses how the inference works.
That way, we can make the source code simpler and easier to maintain.
The following steps are required:

1. Create a new trait.
    - Add the C++ part of the trait in `src/ir/daphneir/DaphneInfer${prop}OpInterface.h`.
        That is, something like:

        ```c++
        // Example: Always infer ${prop} as 123 (non-parametric trait).
        template <class ConcreteOp> class ${prop}123 : public TraitBase<ConcreteOp, ${prop}123> {};

        // Example: Retain the property value from the i-th argument (parametric trait).
        namespace mlir::OpTrait {
        template <size_t i> struct ${prop}FromIthArg {
            template <class ConcreteOp> class Impl : public TraitBase<ConcreteOp, Impl> {};
        };
        } // namespace mlir::OpTrait
        ```
    - Create the file `src/ir/daphneir/DaphneInfer${prop}Traits.td`, if it doesn't exist yet.
        Furthermore, add an include of this `.td` file in `src/ir/daphneir/DaphneOps.td` such that the inference traits for data property `${prop}` can be attached to ops.
    - Write the TableGen part of the trait in `src/ir/daphneir/DaphneInfer${prop}Traits.td`.
        That is, something like:
        
        ```text
        // For non-parametric traits.
        def ${prop}123 : NativeOpTrait<"${prop}123">;

        // For parametric traits.
        class ${prop}FromIthArg<int i> : ParamNativeOpTrait<"${prop}FromIthArg", !cast<string>(i)>;
        def ${prop}FromArg : ${prop}FromIthArg<0>; // short form for a concrete value of i
        ```
1. Implement the effect the trait should have during data property inference.
    We need to make the DAPHNE compiler know how to infer data property `${prop}` for ops that have the newly added trait.
    To that end, we need to:

    - Update the function `tryInfer${prop}(mlir::Operation *op)` (see `src/ir/daphneir/DaphneInfer${prop}OpInterface.cpp`) such that it checks for the presence of the new trait and reacts accordingly.
        You can check for a non-parametric trait by, e.g., `op->hasTrait<${prop}123>()`.
        For parametric traits, you can only check for the trait including a concrete value of the parameter, e.g., `op->hasTrait<${prop}FromIthArg<0>>`.
        To check for a range of values of a trait parameter, we frequently use a custom utility `tryParamTraitUntil`; see the code for details.