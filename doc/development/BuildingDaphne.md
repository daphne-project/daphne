<!--
Copyright 2023 The DAPHNE Consortium

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

# Building DAPHNE

The DAPHNE project provides a full-fledged build script. After cloning, it does everything from dependency setup to
generation of the executable.

## What Does the Build Script Do? (Simplified)

- Download & build all code dependencies
- Build DAPHNE
- Clean Project

## How Long Does a Build Take?

The first run will take a while, due to long compilation times of the dependencies (~1 hour on a 16 vcore desktop, ~10 minutes on a 128 vcore cluster node). But they only have to be compiled once (except updates).
Following builds only take a few seconds/minutes.

Contents:

- [Usage of the build script](#usage-of-the-build-script)
- [Extension of the build script](#extension)

---

## Usage of the Build Script

This section shows the possibilities of the build script.

### Build

The default command to build the default target **`daphne`**.

```bash
./build.sh
```

Print the CLI build help page. This also shows all the following options.

```bash
./build.sh --help
```

Build a specific **target**.

```bash
./build.sh --target "target"
```

For example the following builds the main test target.

```bash
./build.sh --target "run_tests"
```

### Clean

Clean all build directories, i.e., the DAPHNE build dir `<project_root>/build` and the build output in
`<project_root>/bin` and `<project_root>/lib`:

```bash
./build.sh --clean
```

Clean all downloads and extracted archive directories, i.e., `<thirdparty_dir>/download-cache`, `<thirdparty_dir>/sources`
and `<thirdparty_dir>/*.download.success` files:

```bash
./build.sh --cleanCache
```

Clean third-party build output, i.e., `<thirdparty_dir>/installed`, `<thirdparty_dir>/build` and
`<thirdparty_dir>/*.install.success` files:

```bash
./build.sh --cleanDeps
```

Clean everything (DAPHNE build output and third-party directory):

```bash
./build.sh --cleanAll
```

### Minimize Compile Times of Dependencies

The most time-consuming part of getting DAPHNE compiled is building the third-party dependencies.
To avoid this, one can either use a prebuilt container image (in combination with some parameters to the build script
see below) or at least build the dependencies once and subsequently point to the directory where the third-party
dependencies get installed. The build script must be invoked with the following two parameters to achieve this:

`./build.sh --no-deps --installPrefix path/to/installed/deps`

If you have built DAPHNE and **change the installPrefix directory**, it is required to clean up and build again:
`./build.sh --clean`.

### Options

All possible options for the build script:

| Option                  | Effect                                                                                     |
|-------------------------|--------------------------------------------------------------------------------------------|
| -h, --help              | Print the help page                                                                        |
| --installPrefix <path\> | Install third-party dependencies in `<path>` (default: `<project_root>/thirdparty/installed`) |
| --clean                 | Clean DAPHNE build output (`<project_root>/{bin,build,lib}`)                                 |
| --cleanCache            | Clean downloaded and extracted third-party artifacts                                       |
| --cleanDeps             | Clean third-party dependency build output and installed files                              |
| --cleanAll              | Clean DAPHNE build output and reset the third-party directory to the state in the git repo |
| --target <target\>      | Build specific target                                                                      |
| -nf, --no-fancy         | Disable colorized output                                                                   |
| --no-deps               | Avoid building third-party dependencies                                                    |
| -y, --yes               | Accept prompt (e.g., when executing the clean command)                                     |
| --debug                 | Compile the `daphne` binary with debug symbols                                               |
| --cuda                  | Compile with support for GPU operations using the CUDA SDK                                 |
| --oneapi                | Compile with support for accelerated operations using the OneAPI SDK                       |
| --fpgaopencl            | Compile with support for FPGA operations using the Intel FPGA SDK or OneAPI+FPGA Add-On    |
| --no-papi               | Compile without support for PAPI-based profiling                                       |

---

## Building on macOS (Experimental)

DAPHNE can be built natively on macOS with Apple Silicon (ARM64). This support is experimental.

### Prerequisites

Install [Homebrew](https://brew.sh/), then install the required packages:

```bash
brew install gcc@15 cmake ninja wget zstd libomp
```

GCC is required because DAPHNE and all its dependencies must be built with the
same C++ standard library (libstdc++). Apple Clang uses libc++, which is
ABI-incompatible and causes link failures.

### Building

```bash
./build.sh --no-papi
```

The build script automatically detects macOS and configures GCC-15 as the compiler. PAPI is not supported on macOS, so `--no-papi` is required.

### Known Limitations

- CPU pinning (thread affinity) is disabled on macOS
- PAPI-based profiling is not available
- CUDA and FPGA options are not supported on macOS
- Only Apple Silicon (ARM64) has been tested

## Building on WSL

When using Windows Subsystems for Linux (WSL), the default memory limit for WSL is 50% of the total memory of the underlying Windows host. This can lead to build fails due to SIGKILL for DAPHNE builds. [Advanced settings configuration in WSL](https://learn.microsoft.com/en-us/windows/wsl/wsl-config) describes how the memory limit can be configured.

## Extension

### Overview of the Build Script

The build script is divided into sections, visualized by

```bash
#******************************************************************************
# #1 Section name
#******************************************************************************
```

Each section should only contain functionality related to the section name.

The following list contains a rough overview of the sections and the concrete functions or functionality done here.

1. Help message
    1. **printHelp()** // prints help message
2. Build message helper
    1. **daphne_msg(** <message\> **)** // prints a status message in DAPHNE style
    2. **printableTimestamp(** <timestamp\> **)** // converts a unix epoch timestamp into a human readable string (e.g., 5min 20s 100ms)
    3. **printLogo()** // prints a DAPHNE logo to the console
3. Clean build directories
    1. **clean(** <array ref dirs\> <array ref files\> **)** // removes all given directories (1. parameter) and all given files (2. parameter) from disk
    2. **cleanBuildDirs()** // cleans build dirs (DAPHNE and dependency build dirs)
    3. **cleanAll()** // cleans DAPHNE build dir and wipes all dependencies from disk (resetting the third-party directory)
    4. **cleanDeps()** // removes third-party build output
    5. **cleanCache()** // removes downloaded third-party artifacts (but leaving git submodules (only LLVM/MLIR at the time of writing)
4. Create / Check Indicator-files
    1. **dependency_install_success(** <dep\> **)** // used after successful build of a dependency; creates related indicator file
    2. dependency_download_success(<dep\>) // used after successful download of a dependency; creates related indicator file
    3. **is_dependency_installed(** <dep\> **)** // checks if dependency is already installed/built successfully
    4. **is_dependency_downloaded(** <dep\> **)** // checks if dependency is already downloaded successfully
5. Versions of third-party dependencies
    1. Versions of the software dependencies are configured here
6. Set some prefixes, paths and dirs
    1. Definition of project related paths
    2. Configuration of path prefixes. For example all build directories are prefixed with `buildPrefix`. If fast storage
        is available on the system, build directories could be redirected with this central configuration.
7. Parse arguments
    1. Parsing
    2. Updating git submodules
8. Download and install third-party dependencies if requested (default is yes, omit with --no-deps))
    1. Antlr
    2. catch2
    3. OpenBLAS
    4. nlohmannjson
    5. abseil-cpp - Required by gRPC. Compiled separately to apply a patch.
    6. MPI (Default is MPI library is OpenMPI but cut can be any)
    7. gRPC
    8. Arrow / Parquet
    9. MLIR
9. Build DAPHNE target
    1. Compilation of the DAPHNE-target (`daphne` is the default)

### Adding a Dependency

1. If the dependency is fixed to a specific version, add it to the dependency versions section (section 5).
1. Create a new segment in section 8 for the new dependency.
1. Define needed dependency variables:
    1. Directory name (which is used by the script to locate the dependency in different stages)
    1. Create an internal version variable in form of an array with two entries. Those are used for internal versioning and updating of the dependency without rebuilding each time.
        1. First: Name and version of the dependency as a string of the form `<dep_name>_v${dep_version}` (This one is updated, if a new version of the dependency is chosen.)
        1. Second: Third-party Version of the dependency as a string of the form `v1` (This one is incremented each time by hand, if something changes on the path system of the dependency or DAPHNE itself. This way already existing projects are updated automatically, if something changes.)
    1. Optional: Dep-specific paths, Dep-specific files, etc.
1. Download the dependency, encased by:

    ```bash
    # in segment 5
    <dep>_version="<dep_version>"

    # in segment 8
    # ----
    # 8.x Your dependency
    # ----
    <dep>_dirname="<dep_name>" # 3.1
    <dep>_version_internal=("<dep_name>_v${<dep>_version}" "v1") # 3.2
    <dep>... # 3.3
    if ! is_dependency_downloaded "${<dep>_version_internal[@]}"; then
    
        # do your stuff here
    
        dependency_download_success "${<dep>_version_internal[@]}"
    fi
    ```

    **Hint:** It is recommended to use the paths defined in section 6 for dependency downloads and installations. There are predefined paths like `cacheDir`, `sourcePrefix`, `buildPrefix` and `installPrefix`. Take a look at other dependencies to see how to use them.
1. Install the dependency (if necessary), encased by:

    ```bash
    if ! is_dependency_installed "${<dep>_version_internal[@]}"; then
    
        # do your stuff here
    
        dependency_install_success "${<dep>_version_internal[@]}"
    fi
    ```

1. Define a flag for the build script if your dependency is optional or poses unnecessary
    overhead for users (e.g., CUDA is optional as the CUDA SDK is a considerably-sized package that only owners of Nvidia hardware would want to install).

    See section 7 about argument parsing. Quick guide: define a variable and its default value and add an item to the argument handling loop.
