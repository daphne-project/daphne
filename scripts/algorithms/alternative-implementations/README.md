# Alternative Implementations
This folder contains alternative implementations of various algorithms wirtten in C++, Python, and Julia. These implementations can serve as benchmarks to compare the performance and efficiency of DAPHNE against well-established programming languages.

## Software requierements

The files `flake.nix` and `flake.lock` capture the software environment for the non-DAPHNE implementation.
You can learn more about Nix at [https://nixos.org](https://nixos.org) and download it at [https://nixos.org/download](https://nixos.org/download).
You will need to activate the "Flake" feature, see: [https://nixos.wiki/wiki/Flakes#Other_Distros.2C_without_Home-Manager](https://nixos.wiki/wiki/Flakes#Other_Distros.2C_without_Home-Manager)

To enter the software environment execute: `nix develop`

If you do not use Nix, here are the packages that you will need to install via your package manager:

- gcc
- pkg-config
- eigen
- julia
- python
  - numpy package
  - scipy package

## Architecture

Each folder correspond to an application/alogrithm/benchmark.
In each of folder, there is one folder per implementation.
As of now, the applications are written in Python, Julia, and C++. 
Some applications also include their multithread and/or multiprocess respective implementations.

## Setting up

Only C++ and Julia need some setting up.

### Julia

For Julia, you will need to install the packages.

```console
cd connected_components/julia/
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

or

```console
julia --project=connected_components/julia/ -e "using Pkg; Pkg.instantiate()"
```

This will install the Julia packages required for the Connected Components.

You will need to do this for every application what you want to execute with Julia.

### C++

For C++, you will need to compile the source code.
In each `cpp` folder there is `build.sh` script which basically does:

```console
# Sequential version
g++ connected_components.cpp -o connected_components_seq `pkg-config --cflags eigen3` -O3 -DEIGEN_DONT_PARALLELIZE

# Parallel version
g++ connected_components.cpp -o connected_components_par `pkg-config --cflags eigen3` -O3 -fopenmp
```

## Executing

All the languages have (almost) the same interface.
The first argument is the path to the matrix in the MatrixMarket format (`.mtx`), and the output (on `stdout`) is the execution time of the application (without the loading of the matrix).

In the case of C++, we also need to give the size of the matrix as the second argument (i.e., the number of rows, or columns -> *we assume a square matrix*)

In some languages, the function to import the MatrixMarket matrix differ a bit in behavior.
The `fix_matrices.py` script will explictily add "ones" to each coordinates of the matrix.

```console
python3 fix_matrice.py --input path/to/matrix.mtx --output path/to/matrix_ones.mtx
```

### Python

```console
python3 connected_components/py/connected_components.py path/to/matrix_ones.mtx
```

### Julia

```console
julia --project=connected_components/julia/ connected_components/julia/connected_components.jl path/to/matrix_ones.mtx
```

### C++

```console
./connected_components/cpp/connected_components_seq path/to/matrix_ones.mtx nb_cols
```

## Available Applications

### Breadth First Search (`bfs` folder)

### Connected Components (`connected_components` folder)

### K-Core (`k-core` folder)

### N-Body Simulation (`nbody` folder)

Needs to be checked double checked for all the implementations.

### Pagerank (`pagerank` folder)

### Triangle Count (TC) (`tc` folder)
