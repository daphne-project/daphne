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

# API Reference

This document is a hand-crafted reference of the DaphneLib API.
A general introduction to [DaphneLib (DAPHNE's Python API)](/doc/DaphneLib/Overview.md) can be found in a separate document.
DaphneLib offers numerous methods for *obtaining DAPHNE matrices and frames* as well as for *building complex computations* based on them, including complex control flow with if-then-else, loops, and user-defined functions.
Ultimately, DaphneLib will support all [DaphneDSL built-in functions](/doc/DaphneDSL/Builtins.md) on matrices and frames.
Furthermore, **we also plan to create a library of higher-level primitives** allowing users to productively implement integrated data analysis pipelines at a much higher level of abstraction.

At the moment, the documentation is still rather incomplete.
However, as the methods largely map to DaphneDSL built-in functions, you can find some more information in the [List of DaphneDSL built-in functions](/doc/DaphneDSL/Builtins.md), for the time being.

## Obtaining DAPHNE Matrices and Frames

### `DaphneContext` API Reference

**Importing data from Python lists:**

- **`from_python`**`(mat: [], shared_memory=True, verbose=False, return_shape=False) -> Matrix`

**Importing data from other Python libraries:**

- **`from_numpy`**`(mat: np.ndarray, shared_memory=True, verbose=False, return_shape=False) -> Matrix`
- **`from_pandas`**`(df: pd.DataFrame, shared_memory=True, verbose=False, keepIndex=False) -> Frame`
- **`from_tensorflow`**`(tensor: tf.Tensor, shared_memory=True, verbose=False, return_shape=False) -> Matrix`
- **`from_pytorch`**`(tensor: torch.Tensor, shared_memory=True, verbose=False, return_shape=False) -> Matrix`

  
**Generating data in DAPHNE:**

- **`fill`**`(arg, rows:int, cols:int) -> Matrix`
- **`seq`**`(start, end, inc = 1) -> Matrix`
- **`rand`**`(rows: int, cols: int, min: Union[float, int] = None, max: Union[float, int] = None, sparsity: Union[float, int] = 0, seed: Union[float, int] = 0) -> Matrix`
- **`createFrame`**`(columns: List[Matrix], labels: List[str] = None) -> 'Frame'`
- **`diagMatrix`**`(self, arg: Matrix) -> 'Matrix'`
- **`sample`**`(range, size, withReplacement: bool, seed = -1) -> 'Matrix'`

**Reading files using DAPHNE's readers:**

- **`readMatrix`**`(file:str) -> Matrix`
- **`readFrame`**`(file:str) -> Frame`

**Extended relational algebra:**

- **`sql`**`(query) -> Frame`

## Building Complex Computations

Complex computations can be built using Python operators (see [DaphneLib](/doc/DaphneLib/Overview.md)) and using DAPHNE matrix/frame/scalar methods.
In the following, we describe only the latter.

### `Matrix` API Reference

**Matrix meta data:**

- **`ncol`**`()`
- **`nrow`**`()`
- **`ncell`**`()`

**Elementwise unary:**

- **`abs`**`()`
- **`sign`**`()`
- **`exp`**`()`
- **`ln`**`()`
- **`sqrt`**`()`
- **`round`**`()`
- **`floor`**`()`
- **`ceil`**`()`
- **`sin`**`()`
- **`cos`**`()`
- **`tan`**`()`
- **`asin`**`()`
- **`acos`**`()`
- **`atan`**`()`
- **`sinh`**`()`
- **`cosh`**`()`
- **`tanh`**`()`
- **`isNan`**`()`

**Elementwise binary:**

- **`pow`**`(other: 'Matrix')`
- **`log`**`(other: 'Matrix')`
- **`mod`**`(other: 'Matrix')`
- **`max`**`(other: 'Matrix')`
- **`min`**`(other: 'Matrix')`

**Outer binary:**

- **`outerAdd`**`(other: 'Matrix')`
- **`outerSub`**`(other: 'Matrix')`
- **`outerMul`**`(other: 'Matrix')`
- **`outerDiv`**`(other: 'Matrix')`
- **`outerPow`**`(other: 'Matrix')`
- **`outerLog`**`(other: 'Matrix')`
- **`outerMod`**`(other: 'Matrix')`
- **`outerMin`**`(other: 'Matrix')`
- **`outerMax`**`(other: 'Matrix')`
- **`outerAnd`**`(other: 'Matrix')`
- **`outerOr`**`(other: 'Matrix')`
- **`outerXor`**`(other: 'Matrix')` *(not supported yet)*
- **`outerConcat`**`(other: 'Matrix')` *(not supported yet)*
- **`outerEq`**`(other: 'Matrix')`
- **`outerNeq`**`(other: 'Matrix')`
- **`outerLt`**`(other: 'Matrix')`
- **`outerLe`**`(other: 'Matrix')`
- **`outerGt`**`(other: 'Matrix')`
- **`outerGe`**`(other: 'Matrix')`

**Aggregation:**

- **`sum`**`(axis: int = None)`
- **`aggMin`**`(axis: int = None)`
- **`aggMax`**`(axis: int = None)`
- **`mean`**`(axis: int = None)`
- **`var`**`(axis: int = None)`
- **`stddev`**`(axis: int = None)`
- **`idxMin`**`(axis: int)`
- **`idxMax`**`(axis: int)`

**Cumulative aggregation**

- **`cumSum`**`()`
- **`cumProd`**`()`
- **`cumMin`**`()`
- **`cumMax`**`()`

**Reorganization:**

- **`t`**`()`
- **`reshape`**`(numRows: int, numCols: int)`
- **`cbind`**`(other: Matrix)`
- **`rbind`**`(other: Matrix)`
- **`reverse`**`()`
- **`lowerTri`**`(diag: bool, values: bool)`
- **`upperTri`**`(diag: bool, values: bool)`
- **`replace`**`(pattern, replacement)`
- **`order`**`(colIdxs: List[int], ascs: List[bool], returnIndexes: bool)`

**Data preprocessing:**

- **`oneHot`**`(info:matrix)`
- **`bin`**`(numBins:int, Min = None, Max = None)`

**Other matrix operations:**

- **`diagVector`**`()`
- **`solve`**`(other: 'Matrix')`

**Input/output:**

- **`print`**`()`
- **`write`**`(file: str)`

**Conversions and casts:**

- **`asType`**`(dtype=None, vtype=None) -> Matrix`

**Conditional:**

- **`ifElse`**`(thenVal: Union['Matrix', 'Scalar'], elseVal: Union['Matrix', 'Scalar'])`

### `Frame` API Reference

**Frame meta data:**

- **`nrow`**`()`
- **`ncol`**`()`
- **`ncell`**`()`

**Frame label manipulation:**

- **`setColLabels`**`(labels)`
- **`setColLabelsPrefix`**`(prefix)`

**Reorganization:**

- **`cbind`**`(other)`
- **`rbind`**`(other)`
- **`order`**`(colIdxs: List[int], ascs: List[bool], returnIndexes: bool)`

**Extended relational algebra:**

- **`registerView`**`(table_name: str)`
- **`cartesian`**`(other)`
- **`innerJoin`**`(right_frame, left_on, right_on)`

**Input/output:**

- **`print`**`()`
- **`write`**`(file: str)`

**Conversions and casts:**

- **`toMatrix`**`(value_type="f64") -> Matrix`

### `Scalar` API Reference

**Elementwise unary:**

- **`abs`**`()`
- **`sign`**`()`
- **`exp`**`()`
- **`ln`**`()`
- **`sqrt`**`()`
- **`round`**`()`
- **`floor`**`()`
- **`ceil`**`()`
- **`sin`**`()`
- **`cos`**`()`
- **`tan`**`()`
- **`asin`**`()`
- **`acos`**`()`
- **`atan`**`()`
- **`sinh`**`()`
- **`cosh`**`()`
- **`tanh`**`()`
- **`isNan`**`()`

**Elementwise binary:**

- **`pow`**`(other)`
- **`log`**`(other)`
- **`mod`**`(other)`
- **`min`**`(other)`
- **`max`**`(other)`

**Input/output:**

- **`print`**`()`

### `DaphneContext` API Reference

**Logical operators**

Logical *and* (`&&`) and *or* (`||`) operators can be used for the conditions for while-loops and do-while-loops as well as for the predicates for if-then-else statements.
*Note that these logical operators may be provided in another way than via the `DaphneContext` in the future.*

- **`logical_and`**`(left_operand: Scalar, right_operand: Scalar) -> Scalar`
- **`logical_or`**`(left_operand: Scalar, right_operand: Scalar) -> Scalar`

## Building Complex Control Structures

Complex control structures like if-then-else, for-loops, while-loops and do-while-loops can be built using methods of the `DaphneContext`.
These control structures can be used to manipulate matrices, frames, and scalars, and are lazily evaluated. Furthermore, user-defined functions can be created to build reusable code which can then be again lazily evaluated.
User-defined functions can manipulate matrices, frames, and scalars, too.

### `DaphneContext` API Reference

**If-then-else**

- **`cond`**`(input_nodes, pred, then_fn, else_fn)`
    * input_nodes: Iterable[VALID_COMPUTED_TYPES]
    * pred: Callable *(0 arguments, 1 return value)*
    * then_fn: Callable *(n arguments, n return values, n=[1, ...])*
    * else_fn: Callable *(n arguments, n return values, n=[1, ...])*
    * returns: Tuple[VALID_COMPUTED_TYPES] *(length n)*

**Loops**

- **`for_loop`**`(input_nodes, callback, start, end, step)`
    * input_nodes: Iterable[VALID_COMPUTED_TYPES]
    * callback: Callable  *(n+1 arguments, n return values, n=[1, ...]; the last argument is the iteration variable)*
    * start: int
    * end: int
    * step: Union[int, None]
    * returns: Tuple[VALID_COMPUTED_TYPES]  *(length n)*
- **`while_loop`**`(input_nodes, cond, callback)`
    * input_nodes: Iterable[VALID_COMPUTED_TYPES]
    * cond: Callable  *(n arguments, 1 return value, n=[1, ...])*
    * callback: Callable  *(n arguments, n return values)*
    * returns: Tuple[VALID_COMPUTED_TYPES]  *(length n)*
- **`do_while_loop`**`(input_nodes, cond, callback)`
    * input_nodes: Iterable[VALID_COMPUTED_TYPES]
    * cond: Callable  *(n arguments, 1 return value, n=[1, ...])*
    * callback: Callable  *(n arguments, n return values)*
    * returns: Tuple[VALID_COMPUTED_TYPES]  *(length n)*

**User-defined functions**

- **`@function`**, **`function`**`(callback)`
    * callback: Callable
        - This function requires adding typing hints in case the arguments are supposed to be handled as `Scalar` or `Frame`, all arguments without hints are handled as `Matrix` objects.
            Hinting `Matrix` is optional.
            Wrong or missing typing hints can trigger errors before and during computing (lazy evaluation).
    * returns: Tuple[VALID_COMPUTED_TYPES]  *(length equals the return values of callback)*
    * if the decorator `@function` is used the *callback* is defined right below it like regular Python method
