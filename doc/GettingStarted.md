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

# Getting Started

This document provides quick-start instructions for DAPHNE users and DAPHNE developers, which should work in *most* cases. Furthermore, it contains background information to help with custom setups.

- [Quickstart for Users](#quickstart-for-users)
- [Quickstart for Developers](#quickstart-for-developers)
- [Additional Details for Custom Setups](#additional-details-for-custom-setups)

## Quickstart for Users

Follow these instructions if you want to *use* DAPHNE to define and run your own integrated data analysis pipelines.
These simple steps should suffice to get started for *most* users.
If required, you can find more details for custom setups later in this document.

### 1. Get DAPHNE

**Option 1: Use a binary release**

Download and extract the file `daphne-<flavor>-<version>-bin.tgz` from the [release page](https://github.com/daphne-eu/daphne/releases).
It is recommended to use the latest version and to choose the flavor based on your platform and needs (e.g., `X86-64`, `cuda-X86-64`, or `AArch64`).
Using DAPHNE with CUDA requires Nvidia Pascal hardware or newer.

**Option 2: Build from sources**

To build from the *latest sources* with *the most up-to-date state of DAPHNE*, please follow the [Quickstart for Developers](#quickstart-for-developers) below.
If you want to use a *source release*, also follow those instructions, but instead of cloning the DAPHNE repository, download the source code of the latest release (file `<version>.zip`) from the [release page](https://github.com/daphne-eu/daphne/releases).
After the build, you can simply use the system without making changes to the source code.

### 2. Run DAPHNE

**Ubuntu.**
The DAPHNE release binaries should run natively on recent Ubuntu systems (20.04 or later) and perhaps on other GNU/Linux distributions.
However, if there are any issues running DAPHNE natively, please try the pre-built DAPHNE container image.

**Windows.**
The container image can also be used in Windows and WSL.
Installing WSL and Docker should be straightforward using the documentation provided by [Microsoft](https://learn.microsoft.com/en-us/windows/wsl/).
On an installed WSL container, launching DAPHNE via Docker should work the same way as in a native installation.

The following commands should be executed in a bash (or a compatible shell) from the extracted binary release directory (e.g., `daphne-X86-64-v<version>-bin`).

#### Option 1: Run DAPHNE in the Container

**Download the container image**
```bash
docker pull daphneeu/daphne:latest_X86-64_BASE
```
***Hint:** In case of Docker permission errors, consider prepending `sudo` to the command.*

***Hint:** You may want to choose another image tag based on your platform and needs, e.g., `latest_X86-64_CUDA` (for GPU support) or `latest_ARMV8_BASE` (for ARM support).*

**Run a DaphneDSL hello-world script in the container**, which should print the following (besides some numbers):
```bash
containers/quickstart.sh scripts/examples/hello-world.daph
```
```text
Hello World!
```

#### Option 2: Run DAPHNE Natively

**Run a DaphneDSL hello-world script natively**, which should print the following (besides some numbers):

```bash
./run-daphne.sh scripts/examples/hello-world.daph
```
```text
Hello World!
```

***Hint:** Run with `--cuda` to activate CUDA ops, see also `./run-daphne.sh --help`.*

**Run a DaphneLib example natively**, which should print something like the following (the concrete numbers are random):

```bash
./run-python.sh scripts/examples/daphnelib/shift-and-scale.py
```
```text
[[ 0.60676254 -0.88097088 -0.33785961]
 [-0.48030447 -1.04911355  0.97716037]
 [ 1.44938037  1.31980374 -0.2036432 ]
 [-0.060317   -0.46788873 -1.59526103]
 [-1.51552144  1.07816942  1.15960347]]
```

***Hint:** DaphneLib requires numpy and pandas, consider installing them in a virtual environment.*

### Next Steps

- Browse the [user documentation](/doc/), especially the parts on [DaphneDSL](/doc/DaphneDSL/LanguageRef.md) and [DaphneLib](/doc/DaphneLib/Overview.md).
- Have a look at some [examples](/scripts/examples/).
- Start using DAPHNE for your own integrated data analysis pipelines.

## Quickstart for Developers

Follow these instructions if you want to *make modifications* to DAPHNE in terms of the source code, tests, examples, tooling, or documentation.
These simple steps should suffice to get started for *most* developers.
If required, you can find more details for custom setups later in this document.

### 1. Clone the DAPHNE Source Code Repository

**Clone the source code** of the main DAPHNE repository (command below) or your own fork (adapt the command below as necessary).

```bash
git clone https://github.com/daphne-eu/daphne.git
```

### 2. Download the DAPHNE Development Container Image

The development container image already contains all necessary (¹) dependencies of a DAPHNE development environment as well as a useful initialization of environment variables etc., such that you don't need to worry about these things and can have a productive start.

**Get the container image**

```bash
docker pull daphneeu/daphne-dev:latest_X86-64_BASE
```

or

```bash
docker pull daphneeu/daphne-dev:latest_AArch64
```

***Hint:** In case of Docker permission errors, try prepending `sudo` to the command.*

***Hint:** You may want to choose another image tag based on your platform and needs, e.g., `latest_X86-64_CUDA` (for GPU support) or `latest_AArch64` (for ARM support).*

***Hint:** (¹) TensorFlow and PyTorch are *optional* for DaphneLib and *not included* in the `daphne-dev`container due to their footprint of several gigabytes. Please follow the [instructions](/doc/development/InstallPythonLibsInContainer.md) on installing Python libraries in the `daphne-dev` container if you need these libraries.*

**Enter the container**, which should finally print something like the following (where `xyz` is your user name on your system, and the password and IP address may vary):

```bash
cd daphne
./containers/run-docker-example.sh
```
```text
Use xyz with password Docker!0147 for SSH login
Docker Container IP address(es):
172.17.0.2
xyz@daphne-container:/daphne$
```

***Hint:** In case of Docker permission errors, consider setting `USE_SUDO=sudo` in `run-docker-example.sh` near line 26.*

***Hint:** If you pulled a container image of another tag above (e.g., `latest_X86-64_CUDA`), set `DOCKER_TAG=` accordingly in `run-docker-example.sh` near line 22.*

***Hint:** You can leave the container at any time by executing `exit` or by typing `[Ctrl]`+`[D]`.*

### 3. Build DAPHNE and Run the Test Suite

*Everything related to building and running DAPHNE should be done inside the DAPHNE development container*, as described above.

**Build DAPHNE**, which should finally print something like the following:
```bash
./build.sh --no-deps --target all
```
```text
[DAPHNE]..Successfully built Daphne://all
```

**Run the test suite**, which should finally print something like the following (the concrete numbers of assertions and test cases may have changed since the time of this writing):
```bash
./test.sh -nb -d yes
```
```text
===============================================================================
All tests passed (307532 assertions in 1186 test cases)
```

**Run a hello-world script**, which should print the following:
```bash
bin/daphne scripts/examples/hello-world.daph
```
```text
Hello World!
```

### Next Steps

- Browse the [user and developer documentation](/doc/).
- Get familiar with the [contribution guidelines](/CONTRIBUTING.md).
- Start working in DAPHNE (for inspiration, see the list of [open issues](https://github.com/daphne-eu/daphne/issues), including ["good first issues"](https://github.com/daphne-eu/daphne/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22))

## Additional Details for Custom Setups

The [Quickstart for Users](#quickstart-for-users) and [Quickstart for Developers](#quickstart-for-developers) above should be enough for most users and developers to get started.
Nevertheless, there are cases when more details are required, e.g., when DAPHNE shall be run natively, without one of the pre-built containers.

### System Requirements

Please ensure that your development system meets the following requirements before trying to build DAPHNE.
Note that the DAPHNE container images already ship with all these dependencies installed.
There is a [convenience script](/install-ubuntu-packages.sh) to install all necessary dependencies to compile DAPHNE on Ubuntu distributions.
However, in order to use optional features, such as CUDA or FPGA support, the respective requirements listed below still need to be installed separately.

**(*)**
You can view the version numbers as an orientation rather than a strict requirement.
Newer versions should work as well, older versions might work as well.

#### Operating system

| OS         | distribution/version known to work (*) | Comment                                                                             |
|------------|----------------------------------------|-------------------------------------------------------------------------------------|
| GNU/Linux  | Manjaro                                | Last checked in April 2024                                                          |
| GNU/Linux  | Ubuntu 20.04 - 24.04                   | All versions in that range work. 20.04 needs CMake installed from Snap.             |
| GNU/Linux  | Ubuntu 18.04                           | Used with Intel PAC D5005 FPGA, [custom toolchain](/doc/BuildEnvironment.md) needed |
| GNU/Linux | RHEL 8 | [Custom toolchain](/doc/BuildEnvironment.md) needed                                 |
| MS Windows | 10 Build 19041, 11                     | Works on Ubuntu WSL2, using the provided Docker images is recommended               |

##### Windows

Installing WSL and Docker should be straightforward using the documentation provided by [Microsoft](https://learn.microsoft.com/en-us/windows/wsl/). On an installed WSL container,
launching DAPHNE via Docker (see below) should work the same way as in a native installation.

#### Software

| tool/lib                             | version known to work (*)    | comment                                                                                                                                 |
|--------------------------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| clang                                | 10.0.0                       |                                                                                                                                         |
| cmake                                | 3.20                         | On Ubuntu 20.04, install by `sudo snap install cmake --classic` to fulfill the version requirement; `apt` provides only version 3.16.3. |
| GCC/G++                              | 9.3.0                        | Last checked version: 13.2 (compiles with warnings)                                                                                     |
| gfortran                             | 9.3.0                        |                                                                                                                                         |
| git                                  | 2.25.1                       |                                                                                                                                         |
| java (e.g. openjdk)                  | 11 (1.7 should be fine)      |                                                                                                                                         |
| jq                                   |                              | JSON command-line processor used in docker image generation scripts.                                                                     |
| libpfm4-dev                          | 4.10                         | This dependency is needed for profiling support [DAPHNE-#479]                                                                           |
| gRPC                                 | 1.38.0                       |                                                                                                                                         |
| libssl-dev                           | 1.1.1                        | Dependency introduced while optimizing grpc build (which used to build ssl unnecessarily)                                               |
| lld                                  | 10.0.0                       |                                                                                                                                         |
| llvm-10-tools                        | 10, 15, 18                   | `apt` provides up to `llvm-10-tools` for Ubuntu 20.04 whereas 22.04 / 24.04 require a newer version such as `llvm-15-tools`.            |
| ninja                                | 1.10.0                       |                                                                                                                                         |
| numpy                                | 1.19.5                       |                                                                                                                                         |
| pandas                               | 0.25.3                       |                                                                                                                                         |
| pkg-config                           | 0.29.1                       |                                                                                                                                         |
| python3                              | 3.8.5                        |                                                                                                                                         |
| unzip                                | 6.0                          |                                                                                                                                         |
| uuid-dev                             |                              |                                                                                                                                         |
| wget                                 |                              | Used to fetch additional dependencies and other artefacts.                                                                              |
| zlib1g-dev                           |                              | Needed to satisfy cmake dependencies.                                                                                                   |
| ***                                  | ***                          | ***                                                                                                                                     |
| CUDA SDK                             | 11.7.1                       | Optional for CUDA ops                                                                                                                   |
| Intel FPGA SDK or OneAPI FPGA Add-On | 2022.x                       | Optional for FPGAOPENCL ops                                                                                                             |
| OneAPI SDK                           | 2022.x                       | Optional for OneAPI ops                                                                                                                 |
| tensorflow                           | 2.13.1                       | Optional for data exchange between DaphneLib and TensorFlow                                                                             |
| torch                                | 2.3.0+cu121                  | Optional for data exchange between DaphneLib and PyTorch                                                                                |

#### Hardware

- about 7.5 GB of free disk space to build from source (mostly due to dependencies)
- on memory-constrained hosts (WSL, small VMs, Docker Desktop, containers with a `--memory` limit), `build.sh` automatically caps build parallelism to prevent OOM, with no manual configuration needed; see [Memory Requirements](development/BuildingDaphne.md#memory-requirements) for details and manual override options
- Optional:
    - NVidia GPU for CUDA ops (tested on Pascal and newer architectures); 8GB for CUDA SDK
    - Intel GPU for OneAPI ops (tested on Coffeelake graphics); 23 GB for OneAPI
    - Intel FPGA for FPGAOPENCL ops (tested on PAC D5005 accelerator); 23 GB for OneAPI

### Obtaining the Source Code

DAPHNE is based on MLIR, which is a part of the LLVM monorepo.
The LLVM monorepo is included in this repository as a submodule.
Thus, clone this repository as follows to also clone the submodule:

```bash
git clone --recursive https://github.com/daphne-eu/daphne.git
```

Upstream changes to this repository might contain changes to the submodule (we might have upgraded to a newer version of MLIR/LLVM).
Thus, please pull as follows:

```bash
# in git >= 2.14
git pull --recurse-submodules

# in git < 2.14
git pull && git submodule update --init --recursive

# or use this little convenience script
./pull.sh
```

### Building DAPHNE

Simply build the system using the build-script without any arguments:

```bash
./build.sh
```

When you do this the first time, or when there were updates to the LLVM submodule, this will also download and build the third-party material, which might increase the build time significantly.
Subsequent builds, e.g., when you changed something in this repository, will be much faster.

If the build fails in between (e.g., due to missing packages), multiple build directories (e.g., daphne, antlr, llvm)
require cleanup. To only remove build output use the following two commands:

```bash
./build.sh --clean
./build.sh --cleanDeps
```

If you want to remove downloaded and extracted artifacts, use this:

```bash
./build.sh --cleanCache
```

For convenience, you can call the following to remove them all.

```bash
./build.sh --cleanAll
```

See [this page](/doc/development/BuildingDaphne.md) for more information.

### Setting up the Environment

As DAPHNE uses shared libraries, these need to be found by the operating system's loader to link them at runtime.
Since most DAPHNE setups will not end up in one of the standard directories (e.g., `/usr/local/lib`), environment variables
are a convenient way to set everything up without interfering with system installations (where you might not even have
administrative privileges to do so).

```bash
# from your cloned DAPHNE repo or your otherwise extracted sources/binaries: 
export DAPHNE_ROOT=$PWD 
export LD_LIBRARY_PATH=$DAPHNE_ROOT/lib:$DAPHNE_ROOT/thirdparty/installed/lib:$LD_LIBRARY_PATH
# optionally, you can add the location of the DAPHNE executable to your PATH:
export PATH=$DAPHNE_ROOT/bin:$PATH
```

If you're running/compiling DAPHNE from a container you'll most probably **_*_**not**_*_** need to set these environment
variables (unless you have reason to customize your setup - then it is assumed that you know what you are doing).

### Running the Tests

```bash
./test.sh
```

We use [catch2](https://github.com/catchorg/Catch2) as the unit test framework. You can use all [command line arguments](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md#top) of catch2 with `test.sh`.

### Running DAPHNE

Write a little DaphneDSL script or use [`scripts/examples/hello-world.daph`](/scripts/examples/hello-world.daph)...

```csharp
print("Hello World!");
```

... and execute it as follows: `bin/daphne scripts/examples/hello-world.daph` (This command works if `daphne` is run
after building from source. Omit `bin/` in the path to the DAPHNE binary if executed from the binary distribution).

Optionally flags like `--cuda` can be added after the `daphne` command and before the script file to activate support
for accelerated ops (see [software requirements](#software) above and [build instructions](development/BuildingDaphne.md)).
For further flags that can be set at runtime to activate additional functionality, run `daphne --help`.

### Building and Running with Containers

To avoid installing dependencies and to circumvent conflicts with existing installed libraries, one may use containers.

- You need to install Docker or Singularity: Docker version 20.10.2 or higher | Singularity version 3.7.0-1.el7 or
higher are sufficient
- You can use the provided docker files and scripts to create and run DAPHNE.

**A full description on containers is available in the [containers](/containers) subdirectory.**

The following recreates all images provided by [daphneeu](https://hub.docker.com/u/daphneeu)

```bash
cd containers
./build-containers.sh
```

Running in an interactive container can be done with this run script, which takes care of mounting your
current directory and handling permissions:

```bash
# please customize this script first
./containers/run-docker-example.sh
```

For more about building and running with containers, refer (once again) to the directory `containers/` and its
[README.md](/containers/README.md).
For documentation about using containers in conjunction with our cluster deployment scripts, refer to [Deploy.md](/doc/Deploy.md).

### Exploring the Source Code

As an **entry point for exploring the source code**, you might want to have a look at the code behind the `daphne` executable, which can be found in `src/api/cli/daphne.cpp`.

On the top-level, there are the following directories:

- `bin`: after compilation, generated binaries will be placed here (e.g., daphne)
- `build`: temporary build output
- [`containers`:](/containers) scripts and configuration files to get/build/run with Docker or Singularity containers
- [`deploy`:](/deploy) shell scripts to ease deployment in Slurm clusters
- [`doc`:](/doc) documentation written in markdown (e.g., what you are reading at the moment)
- `lib`: after compilation, generated library files will be placed here (e.g., libAllKernels.so, libCUDAKernels.so, ...)
- [`scripts`:](/scripts) a collection of algorithms and examples written in DAPHNE's own domain specific language ([DaphneDSL](/doc/DaphneDSL/LanguageRef.md))
- [`src`:](/src) the actual source code, subdivided into the individual components of the system
- [`test`:](/test) test cases
- `thirdparty`: required external software
