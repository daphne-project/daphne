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

# A Few Early Example Algorithms in DaphneDSL

## Example data

To get some toy example data, you may execute the following commands from the repository's root directory:

```bash
mkdir data
curl https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv -o data/wine.csv
sed -i '1d' data/wine.csv
sed -i 's/;/,/g' data/wine.csv
echo '{"numRows": 4898, "numCols": 12, "valueType": "f64", "numNonZeros": 58776}' > data/wine.csv.meta
```

This downloads a dataset in CSV format, slightly transforms it to be readable with DAPHNE's current CSV reader, and creates a [metadata file](/doc/FileMetaDataFormat.md) for the CSV file.

## Some example invocations

Run these example invocations from the repository's root directory.
Note that all of them are toy examples on small datasets (either the one downloaded above or random data).

### Connected components

```bash
bin/daphne scripts/algorithms/components.daph n=100 e=500 C=\"outC.csv\"
```
*Does not work with vectorized execution (`--vec`) at the moment.*
<!-- error with --vec
loc(fused["scripts/algorithms/components.daph":40:17, "scripts/algorithms/components.daph":40:8, "scripts/algorithms/components.daph":40:22, "scripts/algorithms/components.daph":37:7]): error: operand #2 does not dominate this use
-->

### Gaussian Nonnegative Matrix Factorization (GNMF)

```bash
bin/daphne scripts/algorithms/gnmf.daph rank=2 n=100 e=500 W=\"outW.csv\" H=\"outH.csv\"
```
*Does not work with vectorized execution (`--vec`) at the moment (executes, but with different results).*

### Linear Regression using the Direct Solve method

```bash
bin/daphne scripts/algorithms/lmDS.daph XY=\"data/wine.csv\" icpt=0 reg=0.0000001 verbose=true
```
<!-- successful with --vec -->

### Linear Regression using the Conjugate Gradient method

```bash
bin/daphne scripts/algorithms/lmDS.daph XY=\"data/wine.csv\" icpt=0 reg=0.0000001 tol=0.0000001 maxi=0 verbose=true
```
<!-- successful with --vec -->

### Multinomial Logistic Regression using Trust Region method

```bash
bin/daphne scripts/algorithms/multiLogReg.daph XY=\"data/wine.csv\" B=\"output.csv\"
```
*Does not work with vectorized execution (`--vec`) at the moment*
<!-- error with --vec:
daphne: /daphne/src/runtime/local/datastructures/DenseMatrix.cpp:62: DenseMatrix<ValueType>::DenseMatrix(const DenseMatrix<ValueType>*, size_t, size_t, size_t, size_t) [with ValueType = double; size_t = long unsigned int]: Assertion `((rowLowerIncl < src->numRows) || rowLowerIncl == 0) && "rowLowerIncl is out of bounds"' failed.
daphne: /daphne/src/runtime/local/datastructures/DenseMatrix.cpp:63: DenseMatrix<ValueType>::DenseMatrix(const DenseMatrix<ValueType>*, size_t, size_t, size_t, size_t) [with ValueType = double; size_t = long unsigned int]: Assertion `(rowUpperExcl <= src->numRows) && "rowUpperExcl is out of bounds"' failed.
double free or corruption (out)
Segmentation fault (core dumped)
-->

### Principal Component Analysis (PCA)

```bash
bin/daphne scripts/algorithms/pca.daph X=\"data/wine.csv\" K=2 center=true scale=false Xout=\"outX.csv\" Mout=\"outM.csv\"
```
<!-- successful with --vec -->

<!--
bin/daphne test/api/cli/algorithms/kmeans.daphne r=1000 f=10 c=5 i=3
bin/daphne test/api/cli/algorithms/lm.daphne r=1000 c=100
-->


### Breadth First Search (bfs) 

Under development. See issue: https://github.com/daphne-eu/daphne/issues/781

### K-Core (k-core)

Under development. See issue: https://github.com/daphne-eu/daphne/issues/782

### Simple N-Body Simulation (nbody.daph)
The only parameter (aside from the DAPHNE specific ones) that nbody.daph requires is "--args nb_particules=" where the user should provide the number of particles for the simple simulation.

### Triangle Count (TC) (tc.daph)

Needs the sparse by sparse matrix multiplication which is not yet available in DAPHNE. See issue: https://github.com/daphne-eu/daphne/issues/783 related to TC development.



