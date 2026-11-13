<!--
Copyright 2026 The DAPHNE Consortium

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

# Reading Apache ORC Files

DAPHNE reads Apache ORC files into `DenseMatrix` or `Frame` values.
The reader uses the Apache ORC C++ library directly and supports
column projection and single-clause predicate push-down.

## Status

The reader is fully implemented. It exposes two option-map keys,
`columns` and `predicate`, that control which columns are decoded
and which rows are returned.

Passing options from DaphneDSL requires the file-I/O extension
infrastructure introduced in [PR #993](https://github.com/daphne-eu/daphne/pull/993)
(An Infrastructure for Extendable Input/Output File Format Support).
Until #993 is merged:

- `readMatrix("data.orc")` and `readFrame("data.orc")` work
  end-to-end for whole-file reads.
- Options (`columns`, `predicate`) can be exercised from the C++
  API and are covered by the reader's Catch2 tests.
- Once #993 merges, the two-argument DSL form
  `readFrame("data.orc", options)` becomes available and the
  examples below start working from a `.daphne` script.

## Prerequisites

The reader links against Apache ORC's C++ library directly (not
Arrow's ORC adapter). Apache ORC must be built locally at
`.deps/orc/` before running `build.sh`. CMake looks for
`.deps/orc/lib/liborc.a` and fails with a clear error if it is
missing. `.deps/orc/` is git-ignored.

The build steps inside the DAPHNE dev container are:

```bash
git clone --branch v2.0.0 https://github.com/apache/orc.git .deps/orc-src
cd .deps/orc-src && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/../../orc \
      -DBUILD_JAVA=OFF -DBUILD_TOOLS=OFF -DBUILD_CPP_TESTS=OFF \
      -DBUILD_POSITION_INDEPENDENT_LIB=ON \
      -DPROTOBUF_HOME=/usr/local ..
make -j$(nproc) install
```

Once `.deps/orc/lib/liborc.a` exists, `./build.sh` finds it and
links it automatically. No extra `build.sh` flag is required.

## Supported Target Types and Value Types

| Target | Value types |
|---|---|
| `DenseMatrix<double>` | ORC `DOUBLE` columns |
| `DenseMatrix<int64_t>` | ORC `LONG` columns |
| `Frame` | Any mix of `DOUBLE`, `LONG`, `STRING` columns |

The reader rejects ORC `VARCHAR`, `CHAR`, `BINARY`, `TIMESTAMP`,
`DECIMAL`, and other types not listed above with a clear error
message. Null values are rejected across all types.
`DenseMatrix<std::string>` is not supported (matches the posture
of the CSV and Parquet readers).

## The `.meta` Sidecar for ORC Files

Like every other reader in DAPHNE, ORC requires a matching
`filename.orc.meta` sidecar describing the target shape. See
[FileMetaDataFormat.md](FileMetaDataFormat.md) for the general
format.

Example sidecar for a Frame with four columns:

```json
{
    "numRows": 3,
    "numCols": 4,
    "schema": [
        {"label": "name",   "valueType": "str"},
        {"label": "age",    "valueType": "si64"},
        {"label": "dept",   "valueType": "str"},
        {"label": "salary", "valueType": "f64"}
    ]
}
```

Example sidecar for a homogeneous `DenseMatrix<double>`:

```json
{
    "numRows": 1000,
    "numCols": 8,
    "valueType": "f64"
}
```

## Whole-File Read

The simplest case reads every row and every column into the target
type declared by the sidecar:

```text
X = readFrame("data.orc");
print(X);
```

Available today. No changes needed after #993 merges.

## Column Projection

Pass `options["columns"]` to read only a subset of the file's
columns. Columns are named by their ORC file column labels, are
returned in the user-specified order, and duplicate names are
rejected.

Example (usable in DaphneDSL after #993 merges):

```text
Y = readFrame("data.orc", {"columns": "age,salary"});
print(Y);
```

Rules:

- Names are comma-separated. Whitespace around each name is trimmed.
- Duplicate names are rejected.
- Missing column names are rejected with a message that lists the
  file's actual column names.
- User-specified order becomes output order (SQL `SELECT` semantics).
- Output type must match the projected columns' types. For example,
  projecting one string column into a `DenseMatrix<double>` fails
  with a type-mismatch error.

## Predicate Push-Down

Pass `options["predicate"]` to filter rows during the read. Rows
that do not satisfy the predicate never appear in the output.

Grammar (one clause):

```text
<column> <operator> <literal>
```

- `<column>` must be one of the output columns.
- `<operator>` is one of `=`, `!=`, `<`, `<=`, `>`, `>=`.
- `<literal>` is an integer (`60`), a float (`72.25`), or a
  single-quoted string (`'eng'`).
- Compound predicates with `AND` / `OR` / `IS NULL` are not
  supported yet.

Example (usable after #993):

```text
Y = readFrame("data.orc", {"predicate": "age > 60"});
```

Example combining projection and predicate:

```text
Y = readFrame("data.orc", {"columns": "age,salary", "predicate": "age > 60"});
```

Under the hood, the reader builds an `orc::SearchArgument` and asks
liborc to skip whole stripes whose per-column statistics prove the
predicate cannot match. On rows that survive stripe-level skipping,
the reader applies the predicate row by row so the output is exact.

Because the number of matching rows is not known before the scan
completes, when a predicate is set the reader allocates the final
`DenseMatrix` or `Frame` at the true matching-row count. Callers
that construct the reader directly from C++ must therefore pass
`*res = nullptr` when supplying a predicate.

## Error Cases

The reader throws `std::runtime_error` with a message prefixed
`"ORC reader: "` for:

- File open failures (path missing, permission denied).
- Sidecar mismatch on row count or column count (when projection
  is not active).
- File column type does not match the declared output type.
- Null values in any column.
- ORC types outside the supported set (`VARCHAR`, `CHAR`,
  `TIMESTAMP`, etc.).
- Projection: empty options, empty entry between commas, duplicate
  column, column not present in the file.
- Predicate: empty option, missing operator or literal, unknown
  operator, string literal without a closing quote, literal type
  that does not match the column's DAPHNE value type, or a
  predicate column that is not in the output.

## Future Work

- DaphneDSL surface for options once PR #993 lands.
- Compound predicates (`AND` / `OR`, `IS NULL`).
- Optional null-replacement value passed via `options`.
- Making the `.meta` sidecar optional for ORC files (schema and row
  count can be recovered from the ORC file footer).
