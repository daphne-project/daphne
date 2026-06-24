<!--
Copyright 2024 The DAPHNE Consortium

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

# `dml2daph`: Translator from DML to DaphneDSL

[Apache SystemDS](https://github.com/apache/systemds) offers a variety of data science primitives as built-in functions written in DML (SystemDS's R-inspired domain-specific language). `dml2daphnedsl` is a translator tool that aims to convert a given DML script to an equivalent DaphneDSL script.

## Running the Translator

The translator can be found in `tools/dml2daph.py`. In order to run the script the path to the `.dml` file has to be provided as an argument. For instance, converting `shortestPath.dml` to DaphneDSL would look like this:

```bash
python3 dml2daph.py path/to/systemds/scripts/builtin/shortestPath.dml
```

The translated script can then be found in `tools/translated_files/shortestPath.daph`.

## Regenerating the Translator

`dml2daph` is based on ANTLR. The ANTLR-generated lexer, parser, and visitor are already provided in `tools/`. In order to newly create them, DML's grammar file `Dml.g4` is needed. The files can then be created using the following command:

```bash
antlr4 -Dlanguage=Python3 -visitor Dml.g4
```

## Tests

The tests for `dml2daph` can be found in `tools/tests/`. Each test folder contains a Python script as well as `.dml` and `.daphne` files needed for calling the scripts.

Running the test for shortest path in `tools/tests/test_shortestPath` would look like this:

```bash
python3 test_shortestPath.py
```

After running the tests the folders `tools/data` and `tools/output` are created. The `data` folder contains the input matrices. The `output` folder contains the output matrices for both the DML script and the translated DaphneDSL script.

### Implemented Tests

* `sigmoid`
* `shortestPath`
* `dbscan`
* `lm`

## Known Limitations

The translator works for a variety of DML scripts, but it is still missing certain functionalities.

* `visitImportStatement()`: Could be implemented similar to `handleImplicitImport()`
* dynamically retrieving parameter types and return types of imported functions
* extend mappings from native DML functions to equivalent functions in DaphneDSL
* extend type conversion
* ...