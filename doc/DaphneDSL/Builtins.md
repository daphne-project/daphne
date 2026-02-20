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

# Built-in Functions

DaphneDSL offers numerous built-in functions, which can be used in every DaphneDSL script without requiring any imports.
The general syntax for calling a built-in function is `func(param1, param2, ...)` (see the [DaphneDSL Language Reference](/doc/DaphneDSL/LanguageRef.md)).

This document provides an overview of the DaphneDSL built-in functions.
Note that we are still extending this set of built-in functions.
Furthermore, **we also plan to create a library of higher-level ML primitives** allowing users to productively implement integrated data analysis pipelines at a much higher level of abstraction.
Those library functions will internally be implemented using the built-in functions described in this document.

We use the following notation (deviating from the DaphneDSL function syntax):

- square brackets `[]` mean that a parameter is optional
- `...` stands for an arbitrary repetition of the previous parameter (including zero).
- `/` means alternative options, e.g., `matrix/frame` means the parameter could be a matrix or a frame

## List of Categories

DaphneDSL's built-in functions can be categorized as follows:

- Data generation
- Matrix/frame meta data
- Elementwise unary
- Elementwise binary
- Outer binary (generalized outer product)
- Aggregation and statistical
- Reorganization
- Matrix decomposition & co
- Deep neural network
- Other matrix operations
- Extended relational algebra
- Conversions and casts
- Input/output
- Data preprocessing
- Measurements
- List operations

## Data Generation

- **`fill`**`(value:scalar, numRows:size, numCols:size)`

    Creates a *(`numRows` x `numCols`)* matrix and sets all elements to `value`.
  
- **`createFrame`**`(column:matrix, ...[, labels:str, ...])`

    Creates a frame from an arbitrary number of column matrices.
    Optionally, a label can be specified for each column (the number of provided columns and labels must be equal).
  
- **`diagMatrix`**`(arg:matrix)`

    Creates an *(n x n)* diagonal matrix by placing the elements of the given *(n x 1)* column-matrix `arg` on the diagonal of an otherwise empty (zero) square matrix.
  
- **`rand`**`(numRows:size, numCols:size, min:scalar, max:scalar, sparsity:double, seed:si64)`

    Generates a *(`numRows` x `numCols`)* matrix of random values.
    The values are drawn uniformly from the range *[`min`, `max`]* (both inclusive). A value of zero in the `min` parameter
    will be ignored, as the insertion of zeros in the output is controlled by the `sparsity` parameter.
    The `sparsity` can be chosen between `0.0` (all zeros) and `1.0` (all non-zeros).
    The `seed` can be set to `-1` (randomly chooses a seed), or be provided explicitly to enable reproducible random values.
  
- **`sample`**`(range:scalar, size:size, withReplacement:bool, seed:si64)`

    Generates a *(`size` x 1)* column-matrix of values drawn from the range *[0, `range - 1`]*.
    The parameter `withReplacement` determines if a value can be drawn multiple times (`true`) or not (`false`).
    The `seed` can be set to `-1` (randomly chooses a seed), or be provided explicitly to enable reproducible random values.
  
- **`seq`**`(from:scalar, to:scalar[, inc:scalar])`

    Generates a column matrix containing an arithmetic sequence of values starting at `from`, going through `to`, in increments of `inc`.
    Note that `from` may be greater than `to`, and `inc` may be negative.
    The scalar `inc` is an optional argument and defaults to `1`.

## Matrix/Frame Meta Data

The following built-in functions allow to find out meta data of matrices and frames.

- **`typeOf`**`(arg:scalar/matrix/frame)`

    Returns a string with the data type of `arg`, which can be a scalar, matrix, or a frame containing mixed value types.
    For matrices and frames, this includes their dimensions and value type or value type per column in the case of frames.
    Value types are named after their C++ representation.

- **`nrow`**`(arg:matrix/frame)`

    Returns the number of rows in `arg`.
  
- **`ncol`**`(arg:matrix/frame)`

    Returns the number of columns in `arg`.
  
- **`ncell`**`(arg:matrix/frame)`

    Returns the number of cells in `arg`.
    This is the product of the number of rows and the number of columns.
  
- **`sparsity`**`(arg:matrix)`

    Returns the DAPHNE compiler's *estimate* of the argument's sparsity.
    Note that this value may deviate from the *actual* sparsity of the data at run-time.
  
- **`isSymmetric`**`(arg:matrix)`

    Returns `true` if and only if the given *(n x n)* matrix is symmetric, i.e., for any *i in {0, 1, ..., n-1}* and *j in {0, 1, ..., n-1}*, the value at position *(i, j)* is the same as the value at position *(j, i)*.
    The given matrix must be a square matrix.

## Elementwise Unary

The following built-in functions all follow the same scheme:

- ***`unaryFunc`***`(arg:scalar/matrix)`
  
    Applies the respective unary function (see table below) to the given scalar `arg` or to each element of the given matrix `arg`.

### Arithmetic/General Math

| function | meaning |
| ----- | ----- |
| **`abs`** | absolute value |
| **`sign`** | signum (`1` for positive, `0` for zero, `-1` for negative) |
| **`exp`** | exponentiation (*e* to the power of `arg`) |
| **`ln`** | natural logarithm (logarithm of `arg` to the base of *e*) |
| **`sqrt`** | square root |

### Trigonometric/Hyperbolic

`arg` unit must be radians (conversion: $x^\circ * \frac{\pi}{180^\circ} = y$ radians)

| function | meaning |
| ----- | ----- |
| **`sin`** | sine |
| **`cos`** | cosine |
| **`tan`** | tangent |
| **`asin`** | arc sine  (inverse of sine) |
| **`acos`** | arc cosine (inverse of cosine) |
| **`atan`** | arc tangent (inverse of tangent) |
| **`sinh`** | hyperbolic sine $\left( \frac{\exp(\text{arg}) \, - \, \exp(\text{ - arg})}{2} \right)$ |
| **`cosh`** | hyperbolic cosine $\left( \frac{\exp(\text{arg}) \, + \, \exp(\text{ - arg})}{2} \right)$ |
| **`tanh`** | hyperbolic tangent $\left( \frac{\text{sinh arg}}{\text{cosh arg}} \right)$ |

### Rounding

| function | meaning |
| ----- | ----- |
| **`round`** | round to nearest |
| **`floor`** | round down |
| **`ceil`** | round up |

### Comparison

| function | meaning |
| ----- | ----- |
| **`isNan`** | `1` if argument is NaN, `0` otherwise |

## Elementwise Binary

DaphneDSL supports various elementwise binary operations.
Some of those can be used through *operators in infix notation*, e.g., `+`; and some through *built-in functions*, e.g., `log()`.
Some operations even support both, e.g., `pow(a, b)` and `a^b` have the same semantics.

The built-in functions all follow the same scheme:

- ***`binaryFunc`***`(lhs:scalar/matrix, rhs:scalar/matrix)`
  
    Applies the respective binary function (see table below) to the corresponding pairs of a value in the left-hand-side argument `lhs` and the right-hand-side argument `rhs`.
    Regarding the combinations of scalars and matrices, the same broadcasting semantics apply as for binary operations like `+`, `*`, etc. (see the [DaphneDSL Language Reference](/doc/DaphneDSL/LanguageRef.md)).

### Arithmetic

| function | operator | meaning |
| ----- | ----- | ----- |
| | **`+`** | addition |
| | **`-`** | subtraction |
| | **`*`** | multiplication |
| | **`/`** | division |
| **`pow`** | **`^`** | exponentiation (`lhs` to the power of `rhs`) |
| **`log`** | | logarithm (logarithm of `lhs` to the base of `rhs`) |
| **`mod`** | **`%`** | modulo |

### Min/Max

| function | operator | meaning |
| ----- | ----- | ----- |
| **`min`** | | minimum |
| **`max`** | | maximum |

### Logical

| function | operator | meaning |
| ----- | ----- | ----- |
| | **`&&`** | logical conjunction |
| | **`||`** | logical disjunction |

### Strings

| function | operator | meaning |
| ----- | ----- | ----- |
| **`concat`** | **`+`** | string concatenation |

### Comparison

| function | operator | meaning |
| ----- | ----- | ----- |
| | **`==`** | equal |
| | **`!=`** | not equal |
| | **`<`** | less than |
| | **`<=`** | less or equal |
| | **`>`** | greater than |
| | **`>=`** | greater or equal |

## Outer Binary (Generalized Outer Product)

The following built-in functions all follow the same scheme:

- ***`outerBinaryFunc`***`(lhs:matrix, rhs:matrix)`
  
    The argument `lhs` is expected to be a column *(m x 1)* matrix, and the argument `rhs` is expected to be a row *(1 x n)* matrix.
    The result is a *(m x n)* matrix, whereby the element at position *(i, j)* is calculated by applying the respective binary function (see the table below) to the *i*-th element in `lhs` and the *j*-th element in `rhs`.
    Schematically, this looks as follows (where `∘` is some binary operation):

    ```text
           |    b0    b1 ...    bn rhs
        ---+----------------------
    lhs a0 | a0∘b0 a0∘b1 ... a0∘bn res
        a1 | a1∘b0 a1∘b1 ... a1∘bn
        .. | ..... .....     .....
        am | am∘b0 am∘b1 ... am∘bn
    ```
  
### Arithmetic

| function | meaning |
| ----- | ----- |
| **`outerAdd`** | addition |
| **`outerSub`** | subtraction |
| **`outerMul`** | multiplication (the well-known *outer product*) |
| **`outerDiv`** | division |
| **`outerPow`** | exponentiation (`lhs` to the power of `rhs`) |
| **`outerLog`** | logarithm (logarithm of `lhs` to the base of `rhs`) |
| **`outerMod`** | modulo |

### Min/Max

| function | meaning |
| ----- | ----- |
| **`outerMin`** | minimum |
| **`outerMax`** | maximum |

### Logical

| function | meaning |
| ----- | ----- |
| **`outerAnd`** | logical conjunction |
| **`outerOr`** | logical disjunction |
| **`outerXor`** | logical exclusive disjunction *(not supported yet)* |

### Strings

| function | meaning |
| ----- | ----- |
| **`outerConcat`** | string concatenation *(not supported yet)* |

### Comparison

| function | meaning |
| ----- | ----- |
| **`outerEq`** | equal |
| **`outerNeq`** | not equal |
| **`outerLt`** | less than |
| **`outerLe`** | less or equal |
| **`outerGt`** | greater than |
| **`outerGe`** | greater or equal |

## Aggregation and Statistical

### Full/Row/Column Aggregation

The following built-in functions all follow the same scheme:

- **`agg`**`(arg:matrix)`

    Full aggregation over all elements of the matrix `arg` using aggregation function `agg` (see table below).
    Returns a scalar.

- **`agg`**`(arg:matrix, axis:si64)`

    Row or column aggregation over a *(n x m)* matrix `arg` using aggregation function `agg` (see table below).
    
    - `axis` == 0: calculate one aggregate per row; the result is a *(n x 1)* (column) matrix
    - `axis` == 1: calculate one aggregate per column; the result is a *(1 x m)* (row) matrix

| function | meaning |
| ----- | ----- |
| `sum` | summation |
| `aggMin` | minimum |
| `aggMax` | maximum |
| `mean` | arithmetic mean |
| `var` | variance |
| `stddev` | standard deviation |
| `idxMin` | argmin (the index of the minimum value, only for row/column-wise aggregation) |
| `idxMax` | argmax (the index of the maximum value, only for row/column-wise aggregation) |

### Cumulative Aggregation

The following built-in functions all follow the same scheme:

- **`cumAgg`**`(arg:matrix)`

    Cumulative aggregation over each column of the matrix `arg`.
    Returns a matrix of the same shape as `arg`.

| function | meaning |
| ----- | ------ |
| `cumSum` | cumulative sum |
| `cumProd` | cumulative product |
| `cumMin` | cumulative minimum |
| `cumMax` | cumulative maximum |

## Reorganization

- **`reshape`**`(arg:matrix, numRows:size, numCols:size)`

    Changes the shape of `arg` to *(`numRows` x `numCols`)*.
    Note that the number of cells must be retained, i.e., the product of `numRows` and `numCols` must be equal to the product of the number of rows in `arg` and the number of columns in `arg`.
  
- **`transpose/t`**`(arg:matrix)`

    Transposes the given matrix `arg`.

- **`cbind`**`(lhs:matrix/frame, rhs:matrix/frame)`

    Concatenates two matrices or two frames horizontally.
    The two inputs must have the same number of rows.

- **`rbind`**`(lhs:matrix/frame, rhs:matrix/frame)`

    Concatenates two matrices or two frames vertically.
    The two inputs must have the same number of columns.

- **`reverse`**`(arg:matrix)`

    Reverses the rows in the given matrix `arg`.

- **`order`**`(arg:matrix/frame, colIdxs:size, ..., ascs:bool, ..., returnIndexes:bool)`

    Sorts the given matrix or frame by an arbitrary number of columns.
    The columns are specified in terms of their indexes (counting starts at zero).
    Each column can be sorted either in ascending (`true`) or descending (`false`) order (as determined by parameter `ascs`).
    The provided number of columns and sort orders must match.
    The parameter `returnIndexes` determines whether to return the sorted data (`false`) or a column-matrix of positions representing the permutation applied by the sorting (`true`).

## Matrix Decomposition & Co

- **`eigen`**`(arg:matrix)`

    Calculates the eigenvalues and eigenvectors of the given matrix.
    This built-in function has two results: (1) the eigenvalues as a column matrix, and (2) the eigenvectors as a matrix, where each column is an eigenvector.

We plan to support additional matrix decompositions like **`lu`**, **`qr`**, and **`svd`** in the future.

## Deep Neural Network

Note that most of these operations only have a CUDNN-based kernel for GPU execution at the moment.

- **`affine`**`(inputData:matrix, weightData:matrix, biasData:matrix)`

- **`avg_pool2d`**`(inputData:matrix, numImages:size, numChannels:size, imgHeight:size, imgWidth:size, poolHeight:size, poolWidth:size, strideHeight:size, strideWidth:size, paddingHeight:size, paddingWidth:size)`

    Performs average pooling operation.

- **`max_pool2d`**`(inputData:matrix, numImages:size, numChannels:size, imgHeight:size, imgWidth:size, poolHeight:size, poolWidth:size, strideHeight:size, strideWidth:size, paddingHeight:size, paddingWidth:size)`

    Performs max pooling operation.

- **`batch_norm2d`**`(inputData:matrix, gamma, beta, emaMean, emaVar, eps)`

    Performs batch normalization operation.

- **`biasAdd`**`(input:matrix, bias:matrix)`

    Adds the *(1 x `numChannels`)* row-matrix `bias` to the `input` with the given number of channels.

- **`conv2d`**`(input:matrix, filter:matrix, numImages:size, numChannels:size, imgHeight:size, imgWidth:size, filterHeight:size, filterWidth:size, strideHeight:size, strideWidth:size, paddingHeight:size, paddingWidth:size)`

    2D convolution.
  
- **`relu`**`(inputData:matrix)`

- **`softmax`**`(inputData:matrix)`

## Other Matrix Operations

- **`diagVector`**`(arg:matrix)`

    Extracts the diagonal of the given *(n x n)* matrix `arg` as a *(n x 1)* column-matrix.

- **`lowerTri`**`(arg:matrix, diag:bool, values:bool)`

    Extracts the lower triangle of the given square matrix `arg` by setting all elements in the upper triangle to zero.
    If `diag` is `true`, the elements on the diagonal are retained; otherwise, they are set to zero, too.
    If `values` is `true`, the non-zero elements in the lower triangle are retained; otherwise, they are set to one.

- **`upperTri`**`(arg:matrix, diag:bool, values:bool)`

    Extracts the upper triangle of the given square matrix `arg` by setting all elements in the lower triangle to zero.
    If `diag` is `true`, the elements on the diagonal are retained; otherwise, they are set to zero, too.
    If `values` is `true`, the non-zero elements in the upper triangle are retained; otherwise, they are set to one.

- **`solve`**`(A:matrix, b:matrix)`

    Solves the system of linear equations given by the *(n x n)* matrix `A` and the *(n x 1)* column-matrix `b` and returns the result as a *(n x 1)* column-matrix.

- **`replace`**`(arg:matrix, pattern:scalar, replacement:scalar)`

    Replaces all occurrences of the element `pattern` in the matrix `arg` by the element `replacement`.

- **`ctable`**`(ys:matrix, xs:matrix[, weight:scalar][, numRows:int, numCols:int])`

    Returns the contingency table of two *(n x 1)* column-matrices `ys` and `xs`.
    The resulting matrix `res` consists of `max(ys) + 1` rows and `max(xs) + 1` columns.
    More precisely,
    $\text{res}[x, y] = \left| \{ k \bigm| \text{ys}[k, 0] = y \wedge \text{xs}[k, 0] = x, \; 0 \leq k \leq n-1 \} \right| * \text{weight} \quad \forall x \in \text{xs}, y \in \text{ys}$.
  
    In other words, starting with an all-zero result matrix, all pairs of values $\{ (\text{xs}[k, 0],\text{ys}[k, 0]) \mid 0 \leq k \leq n-1 \}$
    are used to index the result matrix and increase the corresponding value by `weight`.  
    Note that `ys` and `xs` must not contain negative numbers.
  
    The scalar weight is an optional argument and defaults to `1.0`.
    The weight also determines the value type of the result.

    Moreover, optionally, the result shape in terms of the number of rows and columns can be specified.
    If omitted, it defaults to the smallest numbers required to accommodate all given `y`/`x`-coordinates, as expressed above.
    If specified, the result can be either cropped or padded with zeros to the desired shape.
    If a value less than zero is provided as the number of rows/columns, the respective dimension will also be determined from the input data.

    This built-in function can be called with 2, 3, 4, or 5 arguments, depending on which optional arguments are given.

- **`syrk`**`(A:martix)`

    Calculates `t(A) @ A` by symmetric rank-k update operations.

- **`gemv`**`(A:matrix, x:matrix)`

    Calcuates `t(A) @ x` for the given *(n x m)* matrix `A` and *(n x 1)* column-matrix `x`.

## Extended Relational Algebra

DaphneDSL supports relational algebra on frames in two ways:
On the one hand, entire SQL queries can be executed over previously registered *views*.
This aspect is described in detail in a [separate tutorial](/doc/tutorial/sqlTutorial.md).
On the other hand, built-in functions for individual operations of extended relational algebra can be used on frames in DaphneDSL.

### Entire SQL Query

- **`registerView`**`(viewName:str, arg:frame)`

    Registers the frame `arg` to be accessible to SQL queries by the name `viewName`.

- **`sql`**`(query:str)`

    Executes the SQL query `query` on the frames previously registered with `registerView()` and returns the result as a frame.
  
### Set Operations

We will support set operations such as **`intersect`**, **`merge`**, and **`except`**.
<!--
- **`intersect`**`(lhs:frame, rhs:frame)`

  Returns the intersection of the two given frames (in set or bag semantics, tbd).

- **`merge`**`(lhs:frame, rhs:frame)`

  Returns the union of the two given frames (in set or bag semantics, tbd).

- **`except`**`(lhs:frame, rhs:frame)`

  Returns the tuples in `lhs`, which are not in `rhs` (in set or bag semantics, tbd).
-->

### Cartesian Product and Joins

- **`cartesian`**`(lhs:frame, rhs:frame)`

    Calculates the cartesian product of the two input frames.

- **`innerJoin`**`(lhs:frame, rhs:frame, lhsOn:str, rhsOn:str[, numRowRes:si64])`

    Performs an inner join of the two input frames on `lhs`.`lhsOn` == `rhs`.`rhsOn`.

    The parameter `numRowRes` is an optional hint for an upper bound of the number or result rows.
    If specified, it determines the number of rows that will be allocated for the result, whereby `-1` stands for an automatically chosen size.
    Otherwise, it defaults to `-1`.

- **`semiJoin`**`(lhs:frame, rhs:frame, lhsOn:str, rhsOn:str[, numRowRes:si64])`

    Performs a semi join of the two input frames on `lhs`.`lhsOn` == `rhs`.`rhsOn`.
    Returns only the columns belonging to `lhs`.
    
    The parameter `numRowRes` is an optional hint for an upper bound of the number or result rows.
    If specified, it determines the number of rows that will be allocated for the result, whereby `-1` stands for an automatically chosen size.
    Otherwise, it defaults to `-1`.

- **`groupJoin`**`(lhs:frame, rhs:frame, lhsOn:str, rhsOn:str, rhsAgg:str)`

    Group-join of `lhs` and `rhs` on `lhs.lhsOn == rhs.rhsOn` with summation of `rhs.rhsAgg`.
  
We will support more variants of joins, including (left/right) outer joins, theta joins, anti-joins, etc.

### Grouping and Aggregation

- **`groupSum`**`(arg:frame, grpColNames:str[, grpColNames, ...], sumColName:str)`

    Groups the rows in the given frame `arg` by the specified columns `grpColNames` (at least one column) and calculates the per-group sum of the column denoted by `sumColName`.

    *This built-in function is currently limited in terms of functionality (aggregation only on a single column, sum as the only aggregation function). It will be extended in the future. Meanwhile, consider using DAPHNE's `sql()` built-in function for more comprehensive grouping and aggregation support.*

### Frame Label Manipulation

- **`setColLabels`**`(arg:frame, labels:str, ...)`

    Sets the column labels of the given frame `arg` to the given `labels`.
    There must be as many `labels` as columns in `arg`.

- **`setColLabelsPrefix`**`(arg:frame, predfix:str, ...)`

    Prepends the given `prefix` to the labels of all columns in `arg`.

## Conversions and Casts

Note that DaphneDSL offers casts in form of the `as.()`-expression.
See the [DaphneDSL Language Reference](/doc/DaphneDSL/LanguageRef.md) for details.

- **`quantize`**`(arg:matrix<f32>, min:f32, max:f32)`

    Performs a `min`/`max` quantization of the values in `arg`.
    The result matrix is of value type `ui8`.

## Input/Output

DAPHNE supports local file I/O for various file formats.
The format is determined by the specified file name extension.
Currently, the following formats are supported:

- ".csv": comma-separated values
- ".mtx": matrix market
- ".parquet": Apache Parquet format
- ".dbdf": [DAPHNE's binary data format](/doc/BinaryFormat.md)

For both reading and writing, file names can be specified as absolute or relative paths.

For most formats, DAPHNE requires additional information on the data and value types as well as dimensions, *when reading files*.
These must be provided in a separate [`.meta`-file](/doc/FileMetaDataFormat.md).

- **`print`**`(arg:scalar/matrix/frame[, newline:bool[, toStderr:bool]])`

    Prints the given scalar, matrix, or frame `arg` to `stdout`.
    The parameter `newline` is optional; `true` (the default) means a new line is started after `arg`, `false` means no new line is started.
    The parameter `toStderr` is optional; `true` means the text is printed to `stderr`, `false` (the default) means it is printed to `stdout`.

- **`readFrame`**`(filename:str)`

    Reads the file `filename` into a frame.
    Assumes that a `.meta`-file is present for the specified `filename`.

- **`readMatrix`**`(filename:str)`

    Reads the file `filename` into a matrix.
    Assumes that a `.meta`-file is present for the specified `filename`.

- **`write/writeFrame/writeMatrix`**`(arg:matrix/frame, filename:str)`

    Writes the given matrix or frame `arg` into the specified file `filename`.
    Note that the type of `arg` determines how to store the data; thus, it suffices to call `write()` (but `writeFrame()` and `writeMatrix()` can be used synonymously for consistency with reading).
    At the same time, this creates a `.meta`-file for the written file, so that it can be read again using `readMatrix()`/`readFrame()`.

- **`stop`**`([message:str])`

    Terminates the DaphneDSL script execution with the given optional message.

## Data Preprocessing

- **`oneHot`**`(arg:matrix, info:matrix<si64>)`

    Applies one-hot-encoding to the given *(n x m)* matrix `arg`.
    The *(1 x m)* row-matrix `info` specifies the details (in the following, *d[j]* is short for `info[0, j]`):

    - If *d[j]* == -1, then the *j*-th column of `arg` will remain as it is.
    - If *d[j]* == 0, then the *j*-th column of `arg` will be omitted in the output.
    - If *d[j]* > 0, then the *j*-th column of `arg` will be encoded to a vector of length *d[j]*.

        More precisely, if *d[j]* > 0 the *j*-th column of `arg` must contain only integral values in the range *[0, d[j] - 1]*, and will be replaced by *d[j]* columns containing only zeros and ones.
        For each row *i* in `arg`, the value in the `as.scalar(arg[i, j])`-th of those columns is set to 1, while all others are set to 0.

- **`recode`**`(arg:matrix, orderPreserving:bool)`

    Applies dictionary encoding to the given *(n x 1)* matrix `arg`, i.e., assigns an integral code to each distinct value in `arg`.
    The codes start at *0*.
    Iff the parameter `orderPreserving` is `true`, then the distinct values are sorted in ascending order before assigning codes.
    That way, range predicates are possible on the encoded output.
    Otherwise, new codes are assigned to distinct values are they are encountered.
    That way, only point predicates are possible on the encoded output.

    There are two results:

    - The first result is the encoded data, a *(n x 1)* matrix of the codes for each element in the input `arg`.
    - The second result is the decoding dictionary, a *(#distinct(`arg`) x 1)* matrix of the distinct values in `arg`.
        The value at position *i* is the value that is mapped to the code *i*.

    The encoded data can be decoded by right indexing on the dictionary using the codes as positions.

    Example:

    ```r
    data = [10, 5, 20, 5, 20];
    codes, dict = recode(data, false);
    print(codes);
    print(dict);
    decoded = dict[codes, ];
    print(decoded);
    ```

    ```text
    DenseMatrix(5x1, int64_t)
    0
    1
    2
    1
    2
    DenseMatrix(3x1, int64_t)
    10
    5
    20
    DenseMatrix(5x1, int64_t)
    10
    5
    20
    5
    20
    ```

- **`bin`**`(arg:matrix, numBins:size[, min:scalar, max:scalar])`

    Applies binning to the given matrix `arg`, i.e., each value is mapped to the number of its bin (counting starts at zero).
    The bin boundaries are determined by splitting the interval *[`min`, `max`]* into `numBins` equi-width bins.
    The number of bins `numBins` is required.
    Specifying `min` and `max` is optional; if they are omitted, they are automatically calculated, whereby NaNs in `arg` are not taken into account.

    Example:

    ```r
    print(bin(t([10, 20, 30, 40, 50, 60, 70]), 3));
    print(bin(t([10, 20, 30, 40, 50, 60, 70]), 3, 10, 70));
    print(bin(t([5.0, 20.0, nan, 40.0, inf, 60.0, 100.0]), 3, 10.0, 70.0));
    ```

    ```text
    DenseMatrix(1x7, int64_t)
    0 0 0 1 1 2 2
    DenseMatrix(1x7, int64_t)
    0 0 0 1 1 2 2
    DenseMatrix(1x7, double)
    0 0 nan 1 2 2 2
    ```

## Measurements

- **`now`**`()`

    Returns the current time since the epoch in nanoseconds.

## List Operations

- **`createList`**`(elm:matrix, ...)`

    Creates and returns a new list from the given elements `elm`.
    At least one element must be specified.

- **`length`**`(lst:list)`

    Returns the number of elements in the given list `lst`.

- **`append`**`(lst:list, elm:matrix)`

    Appends the given matrix `elm` to the given list `lst`.
    Returns the result as a new list (the argument list stays unchanged).

- **`remove`**`(lst:list, idx:size)`

    Removes the element at position `idx` (counting starts at zero) from the given list `lst`.
    Returns (1) the result as a new list (the argument list stays unchanged), and (2) the removed element.

- **`replaceElementInList`**`(lst:list, idx:size, elm:matrix)`

    Replaces the element at position `idx` (counting starts at zero) from the given list `lst` with the given element `elm`.
    Returns (1) the result as a new list (the argument list stays unchanged), and (2) the old element.
- **`getElementInList`**`(lst:list, idx:size)`

    Returns the element at position `idx` (counting starts at zero) from the given list `lst`.
