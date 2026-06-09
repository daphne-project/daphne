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

# Language Reference

DaphneDSL is DAPHNE's domain-specific language (DSL).
DaphneDSL is written in plain text files, typically ending with `.daphne` or `.daph`.
It is a case-sensitive language inspired by ML systems as well as
languages and libraries for numerical computation like Julia, Python NumPy,
R, and SystemDS DML.
Its syntax is inspired by C/Java-like languages.

## Hello World

A simple hello-world script can look as follows:

```csharp
print("hello world");
```

Assuming this script is stored in the file `hello.daphne`, it can be executed by the following command:

```shell
bin/daphne hello.daphne
```

The remainder of this document discusses the language features of DaphneDSL in detail *as they are right now*, but *note that DaphneDSL is still evolving*.

## Variables

Variables are used to refer to values.

**Valid identifiers** start with a letter (`a-z`, `A-Z`) or an underscore (`_`) that can be followed by any number of letters (`a-z`, `A-Z`), underscores (`_`), and decimal digits (`0-9`).

The following reserved keywords must not be used as identifiers: `if`, `else`, `while`, `do`, `for`, `in`, `true`, `false`, `as`, `def`, `return`, `import`, `matrix`, `frame`, `scalar`, `list`, `f64`, `f32`, `si64`, `si8`, `ui64`, `ui32`, `ui8`, `str`, `nan`, and `inf`.

*Examples:*

```text
X
y
_hello123
a_long_Variable123_456NAME
```

Variables do not need to be (and cannot be) declared.
Instead, simply assign a value to a variable and its type will be inferred.
Variables must have been assigned to before they are used in an expression.

## Types

DaphneDSL differentiates *data types* and *value types*.

Currently, DaphneDSL supports the following *abstract* **data types**:

- `matrix`: homogeneous value type for all cells
- `frame`: a table with columns of potentially different value types
- `scalar`: a single value
- `list`: an ordered sequence of elements of homogeneous data/value type; currently, only matrices can be elements of lists

**Value types** specify the representation of individual values. We currently support:

- floating-point numbers of various widths: `f64`, `f32`
- signed and unsigned integers of various widths: `si64`, `si32`, `si8`, `ui64`, `ui32`, `ui8`
- strings `str` *(currently only for scalars, support for matrix elements is still experimental)*
- booleans `bool` *(currently only for scalars and dense matrices)*

Data types and value types can be combined, e.g.:

- `matrix<f64>` is a matrix of double-precision floating point values

## Comments

DaphneDSL supports single-line comments (starting with `#` or `//`) and multi-line comments (everything enclosed in `/*` and `*/`).

*Examples:*

```csharp
# this is a comment
print("Hello World!"); // this is also a comment
/* comments can
span multiple
lines */
```

## Expressions

### Simple Expressions

Simple expressions constitute the basis of all expressions, and DaphneDSL offers three kinds:

#### Literals

Literals represent hard-coded values and can be of various data and value types:

##### Scalar Literals

**Integer literals** are specified in decimal digits.
By default, they have the type `si64`, but if the optional suffix `u` is appended, the type is `ui64`.
They support the digit separators `'` and `_`.

*Examples*: `0`, `123`, `-456`, `1_000`, `18446744073709551615u`

**Floating-point literals** are specified in decimal or scientific notation.
The following literals stand for special floating-point values: `nan`, `inf`, `-inf`.
By default, the type of a floating-point literal is `f64`, but if an `f` is appended to the literal, it is of type `f32` instead.
Floating-point literals support the digit separators `'` and `_`.

*Examples*: `0.0`, `123.0`, `-456.78f`, `inf`, `nan`, `1.2e5f`, `1E-14`

**Boolean literals** can be `false` and `true`.

**String literals** are enclosed in quotation marks `"`.
Special characters must be escaped using a backslash:

- `\n`: new line
- `\t`: tab
- `\"`: quotation mark
- `\\`: backslash
- `\b`: backspace
- `\f`: line feed
- `\r`: carriage return

*Examples*:

```csharp
"Hello World!"
"line 1\nline 2\nline 3"
"This is \"hello.daphne\"."
```

##### Matrix Literals

A matrix literal consists of a comma-separated list of elements enclosed in square brackets,
optionally followed by parentheses with a comma separated pair of dimensions (number of rows and/or columns).
The elements and the dimensions can be complex expressions.
The matrix's value type is the most general value type among its elements.
Matrix literals are by default column matrices if no dimensions are specified.
If only one dimension is given, the other one will be inferred automatically.
Note that the [built-in function](/doc/DaphneDSL/Builtins.md) `reshape` can also be used to modify the shape of matrices.

*Examples:*

```r
[1, 0, -4.0]                    # matrix<f64> with shape (3 x 1)
[1, 2, sqrt(3), 4](1,)          # matrix<f64> with shape (1 x 4)
[1, 2, 3, sum([4, 5])](2, 2)    # matrix<si64> with shape (2 x 2)
```

##### Frame Literals

Frame literals can be defined using either columns or rows and are enclosed in curly brackets.

**Frame literals of columns** consist of comma-separated pairs of a `str`-typed label
and a (n x 1) matrix, in the format `label: column`.
Both labels and columns can be complex expressions.

*Examples:*

```r
# frame with two columns: "col 1" of type f64 and "col 2" of type si64
{"col 1": [1, 2, sqrt(3)], "col 2": seq(1, 3, 1)}
# frame with two columns: "col 1" of type si64 and "col 2" of type f64
{"col 1": [1, 2, 3], "col 2": [1.1, -2.2, 3.3]}
```

**Frame literals of rows** consist of comma-separated rows, whereby each row is enclosed in square brackets.
The first row contains the `str`-typed column labels.
All remaining rows represent the data of the frame.
Both labels and elements of a row can be complex expressions.
However, it is currently not possible to specify an entire row by an expression.

*Examples:*

```r
# equivalent to the 2nd example above, specified by rows
{["col 1", "col 2"], [1, 1.1], [2, -2.2], [3, 3.3]}
```

#### Variable Expressions

Variables are referenced by their name.

*Examples:*

```text
x
```

#### Script Arguments

Script arguments are named *literals* that can be passed to a DaphneDSL script.
They are referenced by a dollar sign `$` followed by the argument's name.

*Examples:*

`bin/daphne my_script.daphne a=12 b=-2 c=false d=true e=2.0 f=-1.1 g=\"string\" h="\"white spaces\""`

```r
print($g);
myVar = $a + $b;
```

Note that matrix literals are not supported as script arguments yet. Check out [Running DAPHNE Locally](/doc/RunningDaphneLocally.md) for more information.

### Complex Expressions

DaphneDSL offers several ways to build more complex expressions.

#### Operators

DaphneDSL currently supports the following binary operators:

| Operator | Meaning |
| --- | --- |
| `-`, `+` | additive inverse (unary operators) |
| `@` | matrix multiplication (highest precedence) |
| `^` | exponentiation |
| `%` | modulo |
| `*`, `/` | multiplication, division |
| `+`, `-` | addition/string concatenation, subtraction |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | comparison |
| `&&` | logical AND |
| `\|\|` | logical OR (lowest precedence) |

*We plan to add more unary and binary operators in the future.*

*Matrix multiplication (`@`):*
The inputs must be matrices of compatible shapes, and the output is always a matrix.

*All other binary operators:*
The following table shows which combinations of inputs are allowed and which result they yield:

| Left input | Right input | Result | Details |
| --- | --- | --- | --- |
| scalar | scalar | scalar | binary operation of two scalars |
| matrix (n x m) | scalar | matrix (n x m) | element-wise operation of each value with scalar |
| scalar | matrix (n x m) | matrix (n x m) | element-wise operation of scalar with each value **(*)** |
| matrix (n x m) | matrix (n x m) | matrix (n x m) | element-wise operation on corresponding values |
| matrix (n x m) | matrix (1 x m) | matrix (n x m) | broadcasting of row-vector |
| matrix (n x m) | matrix (n x 1) | matrix (n x m) | broadcasting of column-vector |

**(\*)** *Scalar-`op`-matrix* operations are so far only supported for `+`, `-`, `*`, `/`; for `/` only if the matrix is of a floating-point value type.

In the future, we will fully support *scalar-`op`-matrix* operations as well as row/column-matrices as the left-hand-side operands.

*Examples:*

```r
1.5 * X @ y + 0.001
x == 1 && y < 3.5
```

#### Parentheses

Parentheses can be used to manually control operator precedence.

*Examples:*

```r
1 * (2 + 3)
```

#### Indexing

(Right) indexing enables the extraction of a part of the rows and/or columns of a data object (matrix/frame) into a new data object.
The result is always a data object of the same data type as the input (even *1 x 1* results need to be casted to scalars explicitly, if needed).

The rows and columns to extract can be specified independently in any of the following ways:

##### Omit Indexing

Omitting the specification of rows/columns means extracting all rows/columns.

*Examples:*

```r
X[, ] # same as X (all rows and columns)
```

##### Indexing by Position

This is supported for addressing rows and columns in matrices and frames.

- *Single row/column position:*
    Extracts only the specified row/column.

    *Examples:*

    ```r
    X[2, 3] # extracts the cell in row 2, column 3 as a 1 x 1 matrix
    ```

- *Row/column range:*
    Extracts all rows/columns between a lower bound (inclusive) and an upper bound (exclusive).
    The lower and upper bounds can be omitted independently of each other.
    In that case, they are replaced by zero and the number of rows/columns, respectively.

    *Examples:*

    ```r
    X[2:5, 3] # extracts rows 2, 3, 4 of column 3
    X[2, 3:]  # extracts row 2 of all columns from column 3 onward
    X[:5, 3]  # extracts rows 0, 1, 2, 3, 4 of column 3
    X[:, 3]   # extracts all rows of column 3, same as X[, 3]
    ```

- *Arbitrary sequence of row/column positions:*
    Expects a sequence of row/column positions *as a column (n x 1) matrix*.
    There are no restrictions on these positions, except that they must be in bounds.
    In particular, they do *not* need to be contiguous, sorted, or unique.

    *Examples:*

    ```r
    X[ [5, 1, 3], ] # extracts rows 5, 1, and 3
    X[, [2, 2, 2] ] # extracts column 2 three times
    ```

    Note that, when using matrix literals to specify the positions, a space must be left between the opening/closing bracket `[`/`]` of the indexing and that of the matrix literal, in order to avoid confusion with the indexing by bit vector.

A few remarks on positions:

- Counting starts at zero.
    For instance, a 5 x 3 matrix has row positions 0, 1, 2, 3, and 4, and column positions 0, 1, and 2.
- Using negative positions starts counting from the end.
    For instance, the rows of a 5 x 3 matrix can alternatively be addressed by -5, -4, -3, -2, and -1, while the columns can alternatively be addressed by -3, -2, -1.
- They can be provided as integers or floating-point numbers (the latter are rounded down to integers).
- They can be given as literals or as any expression evaluating to a suitable value.

*Examples:*

```r
X[-1, ]               # the last row of X
X[1.2, ]              # same as X[1, ]
X[1.9, ]              # same as X[1, ]
X[i, (j + 2*sum(Y)):] # expressions
```

##### Indexing by Label

So far, this is only supported for addressing columns of frames.

- *Single column label:*
    Extracts only the column with the given label.

    *Examples:*

    ```r
    X[, "revenue"]        # extracts the column labeled "revenue"
    X[100:200, "revenue"] # extracts rows 100 through 199 of the column labeled "revenue"
    ```

##### Indexing by Bit Vector

This is not supported for addressing columns of frames yet.

For each row/column, a single zero/one entry ("bit") must be provided.
More precisely, a (*n x 1*) matrix is required on data objects with *n* rows, and a (*m x 1*) matrix is required on data objects with *m* columns.
Only the rows/columns with a corresponding 1-value in the bit vector are present in the result.

Note that double square brackets (`[[...]]`) must be used to distinguish indexing by bit vector from indexing by an arbitrary sequence of positions.

*Examples:*

```r
# Assume X is a 4x3 matrix.
X[[[0, 1, 1, 0], ]]           # extracts rows 1 and 2
                              # same as X[[1, 2], ]
X[[, [1, 0, 1] ]]             # extracts columns 0 and 2
                              # same as X[, [0, 2]]
X[[[0, 1, 1, 0], [1, 0, 1] ]] # extracts columns 0 and 2 of rows 1 and 2
                              # same as X[[1, 2], [0, 2]]
```
  
Note that, when using a matrix literal to provide the column bit vector, there must be a space between the closing bracket `]` of the matrix literal and the closing double bracket `]]` of the indexing expression, e.g., `X[[, [0] ]]` instead of `X[[, [0]]]`.

#### Casts

Values can be casted to a particular type explicitly.
Currently, it is possible to cast:

- between scalars of different types
- between matrices of different value types
- between matrix and frame
- between scalar and *1x1* matrix/frame

Casts can either fully specify the target data *and* value type, or specify only the target data type *or* the target value type.
In the latter case, the unspecified part of the type will be retained from the argument.

*Examples:*

```r
as.scalar<f64>(x)  # casts x to f64 scalar
as.matrix<ui32>(x) # casts x to a matrix of ui32

as.scalar(x) # casts x to a scalar of the same value type as x
as.matrix(x) # casts x to a matrix of the same value type as x
as.frame(x)  # casts x to a frame whose column types are the value type of x

as.f32(x) # casts x to the same data type as x, but with value type f32
as.ui8(x) # casts x to the same data type as x, but with value type ui8
```

Note that casting to frames does not support changing the value/column type yet, i.e., expressions like `as.frame<f64, si32, f32>(x)` and `as.f64(x)` (on a frame `x`) do not work yet.

#### Function Calls

Function calls can address [*built-in* functions](/doc/DaphneDSL/Builtins.md) as well as [*user-defined* functions](#user-defined-functions-udfs).
Arguments can be provided positionally or as named keyword arguments (similar to Python), but positional arguments must precede keyword arguments and every parameter can only be specified once.
Keyword names must match the parameter names of the called function (see the built-ins reference or the UDF definition).

*Examples:*

```r
print("hello");
t(myMatrix);
seq(0, 10, 2);
fill(value=2, numRows=3, numCols=4);
seq(from=0, to=6, inc=2);
```

#### Conditional Expression

DaphneDSL supports the conditional expression with the general syntax:

```csharp
condition ? then-value : else-value
```

The condition can be either a scalar or a matrix.

- *Condition is a scalar:*
    If the condition is `true` (when casted to boolean), then the result is the `then-value`.
    Otherwise, the result is the `else-value`.
    The `then-value` and the `else-value` must have the same type.
- *Condition is a matrix (elementwise application):*
    In this case, the condition matrix can be of any value type, but must only contain 0 or 1 values of that type (for all other values, the behavior is unspecified).
    The `then-value` and `else-value` must be matrices of the same shape as the condition and must have the same value type as each other.
    The `?:`-operator is applied in an elementwise fashion, i.e., individually for each triple of corresponding elements in condition/`then-value`/`else-value`.
    The `then-value` and `else-value` may also be scalars, in which case they are treated like matrices with a constant value.
    The result is a matrix of the same shape as the condition and the same value type as the `then-value`/`else-value`.

*Examples:*

```r
(i > 5) ? 42.0 : -42.0                      # 42.0 if i > 5, -42.0 otherwise
[1, 0, 0, 1] ? [1.0, 2.0, 3.0, 4.0] : 99.9  # [1.0, 99.9, 99.9, 4.0]
```

## Statements

At the highest level, a DaphneDSL script is a sequence of statements.
Statements comprise assignments, various forms of control flow, and declarations of user-defined functions.

### Expression Statement

Every expression followed by a semicolon `;` can be used as a statement.
This is useful for expressions (especially function calls) which do not return a value.
Nevertheless, it can also be used for expressions with one or more return values, in which case these values are ignored.

*Examples:*

```r
print("hello"); # built-in function without return value
1 + 2;          # value is ignored, useless but possible
doSomething();  # possible return values are ignored, but the execution 
                # of the user-defined function could have side effects
```

### Assignment Statement

The return value(s) of an expression can be assigned to one (or more) variable(s).

**Single-assignments** are used for expressions with exactly one return value.

*Examples:*

```r
x = 1 + 2;
```

**Multi-assignments** are used for expressions with more than one return value.

*Examples:*

```r
evals, evecs = eigen(A); # eigen() returns two values, the (n x 1)-matrix of
                         # eigen-values and the (n x n)-matrix of eigen-vectors
                         # of the input matrix A.
```

#### Indexing

The value of an expression can also be assigned to a *partition* of an *existing data object*.
This is done by (left) indexing, whose syntax is similar to (right) indexing in expressions.

Currently, left indexing is supported only for matrices.
Furthermore, the rows/columns cannot be addressed by arbitrary positions lists or bit vectors (yet).

*Examples:*

```r
X[5, -2]      = [123];            # insert (1 x 1)-matrix
X[10:20, 2:5] = fill(123, 10, 3); # insert (10 x 3)-matrix
```

The following conditions must be fulfilled:

- The left-hand-side variable must have been initialized.
- The left-hand-side variable must be of data type matrix.
- The right-hand-side expression must return a matrix.
- The shapes of the partition addressed on the left-hand side and the return value of the right-hand-side expression must match.
- The value type of the left-hand-side and right-hand-side matrices must match.<!--TODO this should be relaxed-->

Left indexing can be used with both single and multi-assignments.
With the latter, it can be used with each variable on the left-hand side individually and independently.

*Examples:*

```r
x, Y[3, :], Z = calculateSomething();
```

**Copy-on-write semantics**

Left indexing enables the modification of existing data objects, whereby the semantics is *copy-on-write*.
That is, if two different variables represent the same runtime data object, then left indexing on one of these variables does not have any effects on the other one.
This is achieved by transparently copying the data as necessary.

*Examples:*

```r
A = ...;           # some matrix
B = A;             # copy-by-reference
B[..., ...] = ...; # copy-on-write: changes B, but no effect on A
A[..., ...] = ...; # copy-on-write: changes A, but no effect on B
```

### Control Flow Statements

DaphneDSL supports block statements, conditional branching, and various kinds of loops.
These control flow constructs can be nested arbitrarily.
Besides that, the [`stop()` built-in function](/doc/DaphneDSL/Builtins.md) can be used to terminate the DaphneDSL script execution.

#### Block Statement

A block statement allows to view an enclosed sequence of statements like a single statement.
This is very useful in combination with the control flow statements described below.
Besides that, a block statement starts a new scope in terms of visibility of variables.
Within a block, all variables from outside the block can be read and written.
However, variables created inside a block are not visible anymore after the block.

The syntax of a block statement is:

```r
{
    statement1
    statement2
    ...
}
```

*Examples:*

```r
x = 1;
{
    print(x); # read access
    x = 2;    # write access
    y = 1;    # variable created inside the block
}
print(x);     # prints 2
print(y);     # error
```

#### If-Then-Else

The syntax of an if-then-else statement is as follows:

```csharp
if (condition)
    then-statement
else
    else-statement
```

*condition* is an expression returning a single value.
If this value is `true` (when casted to value type `bool`, if necessary), the *then-statement* is executed.
Otherwise, the *else-statement* is executed, *if it is present*.
Note that the *else*-branch (keyword and statement) may be omitted.
Furthermore, *then-statement* and *else-statement* can be block statements, to allow any number of statements in the then and else-branches.

*Examples:*

```r
if (sum(X) == 0)
    X = X + 1;
```

```r
if (2 * x > y) {
    z = z / 2;
    a = true;
}
else
    z = z * 2;
```

```r
if (a)
    print("a");
else if (b)
    print("not a, but b");
else
    print("neither a nor b");
```

#### Loops

DaphneDSL supports for-loops, while-loops, and do-while-loops.
In the future we plan to support also parfor-loops as well as `break` and `continue` statements.

##### For-Loops

For-loops are used to iterate over the elements of a sequence of integers.
The syntax of a for-loop is as follows:

```r
for (var in start:end[:step])
    body-statement
```

*var* must be a valid identifier and is assigned the values from *start* to *end* in increments of *step*.
*start*, *end*, and *step* are expressions evaluating to a single number.
*step* is optional and defaults to 1 if *end* is greater than *start*, or -1 otherwise.
In that sense, for-loops can also be used to count backwards by setting *start* greater than *end*.
The *body-statement* is executed for each value in the sequence, and within the *body-statement*, this value is accessible via the read-only variable `var`.
Note that the *body-statement* may be a block statement enclosing an arbitrary number of statements.

*Examples:*

```csharp
for(i in 1:3)
    print(i); # 1 2 3
```

```r
x = 0; y = 0;
for(i in 10:1:-3) {
    x = x + i;
    y = y + 1;
}
print(x); # 22
print(y); #  4
```

##### While-Loops

While loops are used to execute a (block of) statement(s) as long as an arbitrary condition holds true.
The syntax of a while-loop is as follows:

```csharp
while (condition)
    body-statement
```

*condition* is an expression returning a single value, and is evaluated before each iteration.
If this value is `true` (when casted to value type `bool`, if necessary), the *body-statement* is executed, and the loop starts anew.
Otherwise, the program continues after the loop.
Note that the *body-statement* may be a block statement enclosing an arbitrary number of statements.

*Examples:*

```r
i = 0;
while(i < 10 && !converged) {
    A = A @ B;
    converged = sum(A);
    i = i + 1;
}
```

##### Do-While-Loops

Do-while-loops are a variant of while-loops, which checks the condition after each iteration.
Consequently, a do-while-loop always executes at least one iteration.
The syntax of a do-while-loop is as follows:

```csharp
do
    body-statement
while (condition);
```

The semicolon at the end is optional.
Note that the *body-statement* may be a block statement enclosing an arbitrary number of statements.

*Examples:*

```csharp
i = 5;
do {
    A = sqrt(A);
    i = i - 1;
} while (mean(A) > 100 && i > 0);
```

## User-defined Functions (UDFs)

DaphneDSL allows users to define their own functions.
The syntax of a function definition looks as follows:

```csharp
def funcName(paramName1[:paramType1], paramName2[:paramType2], ...) [-> returnType1, returnType2, ...] {
    statement1
    statement2
    ...
}
```

The function name must be a valid and unique identifier.
A function can have zero, one, or more parameters, and their names must be valid and unique identifiers.
Furthermore, a function may return zero, one, or more values.
The types of parameters are optional and can be provided or omitted for each parameter individually.
The types of the return values are optional, but if omitted, exactly one return value is implicitly assumed.
Functions with multiple return values must specify the types of all return values.
See also *typed and untyped functions* below.
The body of a function definition is always a block statement, i.e., it must be enclosed in curly braces `{}` even if it is just a single statement.
<!--TODO Overloading is allowed-->

The parameter names given in the signature can be used at call sites as keyword arguments.
When using keyword arguments, the same rules as for built-ins apply: positional arguments must precede keyword arguments and a parameter may only be specified once.

Functions must be defined in the top-level scope of a DaphneDSL script, i.e., a function definition must not be nested within a control-flow statement or within another function definition.

### Returning Values

User-defined functions can return zero, one, or more (comma-separated) values using the `return`-statement.
The number of returned values must match the function signature.

*Examples:*
```csharp
return;       # don't return any values
return x;     # return exactly one value
return x, y;  # return two values
```

Currently, the return statement must be the last statement of a function.
Alternatively, it can be nested into if-then-else (early return), as long as it is ensured that there is exactly one return statement at the end of each path through the function (experimental).

*Examples:*

```csharp
def fib(n: si64) -> si64 {
    if (n <= 0)
        return 0;
    if (n <= 1)
        return 1;
    return fib(n - 1) + fib(n - 2);
}
```
```csharp
def nextTwo(a: si64) -> si64, si64 {
    return a + 1, a + 2;
}
```

### Calling User-defined Functions

A user-defined function can be called like any other (built-in) function (see *function-call expressions* above).

*Examples:*

```r
x = 2 * fib(5) + 123;
y, z = nextTwo(123);
```

### Typed and Untyped Functions (Experimental)

DaphneDSL supports both typed and untyped functions.

The definition of a *typed function* specifies the data and value types of all parameters and return values.
Hence, a typed function can only be called with inputs of the specified types (if a provided input has an unexpected type, it is automatically casted to the expected type, if possible), and always returns outputs of the specified types.
A typed function is compiled exactly once and specialized to the specified parameter and return types.

In contrast to that, the definition of an *untyped function* leaves the data and value type, or just the value type, of one or more parameters and/or return values unspecified.
At call sites, a value of any type, or any value type, can be passed to an untyped parameter.
As a consequence, an untyped function is compiled and specialized on demand according to the types at a call site.
Consistently, the types of untyped return values are inferred from the parameter types and operations.

## Compiler Hints

One of DAPHNE's strengths is its (WIP) ability to make various decisions on its own, e.g., regarding physical data representation (such as dense/sparse), physical operators (kernels), and data/operator placement (such as local/distributed, CPU/GPU/FPGA, computational storage).
However, expert users may optionally provide hints to influence compiler decisions.
This feature is useful for experimentation and in the context of DAPHNE's extensibility.
For instance, a user could force the use of a certain custom kernel at a certain point in a larger DaphneDSL script to measure the impact of that custom kernel, even if the DAPHNE compiler would normally not choose that kernel in that situation.

*The support for compiler hints is still experimental and it is currently not guaranteed that the DAPHNE compiler respects these hints.*

### Kernel Hints

Users can provide hints on the physical kernel that should be used for a specific occurrence of a DaphneDSL operation.
So far, kernel hints are only supported for DaphneDSL built-in functions and *some* DaphneDSL operators (`+`, `-`, `*`, `/`).
Here, the name of the pre-compiled kernel function can optionally be attached to the name of the built-in function or the operator symbol, separated by `::`.

*Examples:*

```r
res = sum::my_custom_sum_kernel(X);
res = X +::my_custom_add_kernel Y;
```

## Example Scripts

A few example DaphneDSL scripts can be found in:

- [scripts/algorithms/](/scripts/algorithms/)
- [scripts/examples/](/scripts/examples/)
- [test/api/cli/algorithms/](/test/api/cli/algorithms/)
