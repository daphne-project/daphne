#!/bin/bash

# Copyright 2021 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This is the main build script of Daphne. It downloads and builds all third-party dependencies and then builds Daphne.
# It is required if you freshly cloned the repository or if you want to update the third-party dependencies.
# It is not required if you only want to build Daphne from your IDE for example after you already built it once.
# In this case you need to adapt the cmake settings according to the cmake call in section 9.
#
# For more information about the build script and contribution guidelines please refer to the doc/development/BuildingDaphne.md


# Stop immediately if any command fails.
set -e

build_ts_begin=$(date +%s%N)

#******************************************************************************
# #1 Help message
#******************************************************************************

function printHelp {
    printLogo
    echo "Build the DAPHNE prototype."
    echo ""
    echo "Usage: $0 [-h|--help] [--target <target>]"
    echo ""
    echo "This includes downloading and building all required third-party "
    echo "material, if necessary. Simply invoke this script without any "
    echo "arguments to build the prototype. You should only invoke it from "
    echo "the prototype's root directory (where this script resides)."
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help        Print this help message and exit."
    echo "  --installPrefix <path>"
    echo "                    Set the prefix for the install path of third party dependencies (default: <projectRoot>/thirdparty)"
    echo "  --target <target> Build the cmake target TARGET (defaults to '$target')"
    echo "  --clean           Remove DAPHNE build output (DAPHNE_ROOT/{bin,build,lib}"
    echo "  --cleanCache      Remove downloaded and extracted third party artifacts"
    echo "  --cleanDeps       Remove build output of third party dependencies (<thirdpartyPath>/{build,installed})"
    echo "  --cleanAll        Remove all build output and reset the directory of the third party dependencies"
    echo "  -nf, --no-fancy   Suppress all colored and animated output"
    echo "  -nd, --no-deps    Avoid building third party dependencies at all"
    echo "  -y, --yes         Accept all prompts"
    echo "  --cuda            Compile with support for CUDA ops"
    echo "  --debug           Compile with support for debug mode"
    echo "  --fpgaopencl      Compile with support for Intel PAC D5005 FPGA"
    echo "  --mpi             Compile with support for MPI"
    echo "  --hdfs            Compile with support for HDFS"
    echo "  --io_uring        Compile with support for io_uring"
    echo "  --no-papi         Compile without support for PAPI"
}

#******************************************************************************
# #2 Build message helper
#******************************************************************************
# Daphne Colors
daphne_red_fg='\e[38;2;247;1;70m'
#daphne_red_bg='\e[48;2;247;1;70m'
daphne_blue_fg='\e[38;2;120;137;251m'
#daphne_blue_bg='\e[48;2;36;42;76m'
#daphne_black_fg='\e[38;2;0;0;0m'
#daphne_black_bg='\e[48;2;0;0;0m'
reset='\e[00m'
fancy="1"

# Prints info message with style ... Daphne style 8-)
function daphne_msg() {
    local message date dotSize dots textSize columnWidth

    # check if output is directed to file, if true, set width to 120
    if ! [ -t 1 ]; then
        columnWidth=120
        fancy="0"
    else
        # get width of terminal
        columnWidth="$(tput cols)"
    fi
    prefix="[DAPHNE]"
    message="..${*}"
    date="[$(date +"%d.%m.%y %H:%M:%S")]"
    textSize=$((columnWidth - ${#date} - ${#prefix}))
    dotSize=$((textSize - ${#message}))

    dots=""
    for ((i = 0; i < dotSize; i++)); do
        dots="${dots}."
    done

    message="${message}${dots}"

    # no fancy output (if disabled or not standard output, e.g. piped into file)
    if [ "$fancy" -eq 0 ] || ! [ -t 1 ]; then
        printf "%s%s%s\n" "${prefix}" "${message}" "${date}"
        return 0
    fi

    # colored output
    printf "${daphne_red_fg}%s${daphne_blue_fg}%s${daphne_red_fg}%s${reset}\n" "${prefix}" "${message}" "${date}"
    return 0
}

function printableTimestamp() {
    local t="0"
    local result=""
    local tmp
    if [ -n "$1" ]; then
        t="$1"
    fi

    if [ "$t" -eq 0 ]; then
        printf "0ns"
        return 0
    fi

    tmp=$((t % 1000))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}ns"
    fi
    # t < 1000 ns
    if [ "$t" -lt 1000 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 1000))" #us
    tmp=$((t % 1000))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}us${result}"
    fi
    # t < 1000 us
    if [ "$t" -lt 1000 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 1000))" #ms
    tmp=$((t % 1000))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}ms${result}"
    fi
    # t < 1000 ms
    if [ "$t" -lt 1000 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 1000))" #s
    tmp=$((t % 60))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}s${result}"
    fi
    # t < 60 s
    if [ "$t" -lt 60 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 60))" #min
    tmp=$((t % 60))
    if [ "$tmp" -gt 0 ]; then
        result="${tmp}min${result}"
    fi
    # t < 60 min
    if [ "$t" -lt 60 ]; then
        printf "%s\n" "${result}"
        return 0
    fi
    # add space
    if [ "$tmp" -gt 0 ]; then result=" $result"; fi

    t="$((t / 60))" #h
    result="${t}h${result}"
    printf "%s\n" "${result}"
}

function printLogo() {
    daphne_msg ""
    daphne_msg ".Welcome to"
    daphne_msg ""
    daphne_msg ".._______.......................__"
    daphne_msg ".|       \\.....................|  \\"
    daphne_msg ".| €€€€€€€\\  ______    ______  | €€____   _______    ______"
    daphne_msg ".| €€  | €€ |      \\  /      \\ | €€    \\ |       \\  /      \\"
    daphne_msg ".| €€  | €€  \\€€€€€€\\|  €€€€€€\\| €€€€€€€\\| €€€€€€€\\|  €€€€€€\\"
    daphne_msg ".| €€  | €€ /      €€| €€  | €€| €€  | €€| €€  | €€| €€    €€"
    daphne_msg ".| €€__/ €€|  €€€€€€€| €€__/ €€| €€  | €€| €€  | €€| €€€€€€€€"
    daphne_msg ".| €€    €€ \\€€    €€| €€    €€| €€  | €€| €€  | €€ \€€     \\"
    daphne_msg ". \\€€€€€€€   \\€€€€€€€| €€€€€€€  \\€€   \\€€ \\€€   \\€€  \\€€€€€€€"
    daphne_msg ".....................| €€"
    daphne_msg ".....................| €€"
    daphne_msg "......................\\€€..................EU-H2020.//.957407"
    daphne_msg ""
    printf "\n\n"
}

#******************************************************************************
# #3 Clean build directories
#******************************************************************************

function clean() {
    # Throw error, if clean is executed, but output is piped to a file. In this case the user has to accept the
    # cleaning via parameter --yes
    if ! [ -t 1 ] && ! [ "$par_acceptAll" -eq 1 ]; then
        printf >&2 "${daphne_red_fg}"
        printf "Error: To clean Daphne while piping the output into a file set the --yes option, to accept cleaning.\n" | tee /dev/stderr
        printf >&2 "${reset}"
        exit 1
    fi

    cd "$projectRoot"
    local -n __dirs
    local -n __files

    if [ "$#" -gt 0 ]; then
        if [ -n "$1" ]; then
            __dirs=$1
        fi
        shift
    fi

    if [ "$#" -gt 0 ]; then
        if [ -n "$1" ]; then
            __files=$1
        fi
        shift
    fi
    if [ "$fancy" -eq 0 ] || ! [ -t 1 ]; then
        printf "WARNING. This will delete the following..."
    else
        printf "${daphne_red_fg}WARNING.${reset} This will delete following..."
    fi
    echo "Directories:"
    for dir in "${__dirs[@]}"; do
        echo " > $dir"
    done
    echo "Files:"
    for file in "${__files[@]}"; do
        echo " > $file"
    done

    echo

    # prompt confirmation, if not set by --yes
    if [ "$par_acceptAll" == "0" ]; then
        read -r -p "Are you sure? (y/n) " answer

        if [[ "$answer" != [yY] ]]; then
            echo "Abort."
            exit 0
        fi
    fi

    # Delete entire directories.
    for dir in "${__dirs[@]}"; do
        if [ -d "${dir}" ]; then
            echo "---- cleanup ${dir}"
            rm -rf "${dir}"
        else
            echo "---- cleanup ${dir} - non-existing"
        fi
    done

    # Delete individual files.
    for file in "${__files[@]}"; do
        if [ -f "$file" ]; then
            echo "---- cleanup $file"
            rm -f "$file"
        else
            echo "---- cleanup $file - non-existing"
        fi
    done
}

function cleanBuildDirs {
    echo "-- Cleanup of DAPHNE build directories in ${projectRoot} ..."

    local dirs=("${daphneBuildDir}" "${projectRoot}/bin" "${projectRoot}/lib")

    clean dirs
}

function cleanDeps {
    echo "-- Cleanup of third party build directories in ${thirdpartyPath} ..."

    local dirs=("${buildPrefix}" "${installPrefix}")
    local files=(
        "${thirdpartyFlagsDir}/"*".install."*".success"
        "${llvmCommitFilePath}")

    clean dirs files
}

function cleanCache {
    echo "-- Cleanup of downloaded/extracted artifacts of the third party dependencies"

    local dirs=("${sourcePrefix}" "${cacheDir}")

    local files=(
        "${thirdpartyFlagsDir}/"*".download."*".success"
    )

    clean dirs files
}

function cleanAll {
    cd "$projectRoot"
    message="This will delete the DAPHNE build output and everyting in $thirdpartyPath and reset this directory to its"
    message+=" last state in git.\n"
    if [ "$fancy" -eq 0 ] || ! [ -t 1 ]; then
        printf "WARNING! ${message}"
    else
        printf "${daphne_red_fg}WARNING!${reset} ${message}"
    fi

    # prompt confirmation, if not set by --yes
    if [ "$par_acceptAll" == "0" ]; then
        read -r -p "Are you sure? (y/n) " answer

        if [[ "$answer" != [yY] ]]; then
            echo "Abort."
            exit 0
        fi
    fi

    rm -rf "$thirdpartyPath"
    git checkout "$thirdpartyPath"

    local par_acceptAll_old=par_acceptAll
    par_acceptAll="1"
    cleanBuildDirs
    par_acceptAll=$par_acceptAll_old
    cd - >/dev/null
}

#******************************************************************************
# #4 Create / Check Indicator-files
#******************************************************************************

#// creates indicator file which indicates successful dependency installation in <projectRoot>/thirdparty/
#// param 1 dependency name
function dependency_install_success() {
    dep_version="${2}"
    if [ -z "${dep_version}" ]; then dep_version="v1"; fi
    daphne_msg "Successfully installed ${1}."
    touch "${thirdpartyFlagsDir}/${1}.install.${dep_version}.success"
}
function dependency_download_success() {
    dep_version="${2}"
    if [ -z "${dep_version}" ]; then dep_version="v1"; fi
    daphne_msg "Successfully downloaded ${1}."
    touch "${thirdpartyFlagsDir}/${1}.download.${dep_version}.success"
}

#// checks if dependency is installed successfully
#// param 1 dependency name
function is_dependency_installed() {
    dep_version="${2}"
    if [ -z "${dep_version}" ]; then dep_version="v1"; fi
    [ -e "${thirdpartyFlagsDir}/${1}.install.${dep_version}.success" ]
}
function is_dependency_downloaded() {
    dep_version="${2}"
    if [ -z "${dep_version}" ]; then dep_version="v1"; fi
    [ -e "${thirdpartyFlagsDir}/${1}.download.${dep_version}.success" ]
}

function clean_param_check() {
    if [ "$par_clean" -gt 0 ]; then
        echo "Only *one* clean parameter (clean/cleanAll/cleanDeps/cleanCache) is allowed!"
        exit 1
    fi
}

#******************************************************************************
# #5 Versions of third party dependencies
#******************************************************************************
source software-package-versions.txt

#******************************************************************************
# #6 Set some prefixes, paths and dirs
#******************************************************************************

projectRoot="$(pwd)"
thirdpartyPath="${projectRoot}/thirdparty"

# a convenience indirection to set build/cache/install/source dirs at once (e.g., to /tmp/daphne)
myPrefix=$thirdpartyPath
#myPrefix=/tmp/daphne

daphneBuildDir="$projectRoot/build"
llvmCommitFilePath="${thirdpartyPath}/llvm-last-built-commit.txt"
patchDir="$thirdpartyPath/patches"
installPrefix="${myPrefix}/installed"
buildPrefix="${myPrefix}/build"
sourcePrefix="${myPrefix}/sources"
cacheDir="${myPrefix}/download-cache"

mkdir -p "$cacheDir"
mkdir -p "$sourcePrefix"

thirdpartyFlagsDir="${thirdpartyPath}/flags"

# Create the flags directory and migrate any flags that might be there to avoid unnecessary recompilation.
if [ ! -d "$thirdpartyFlagsDir" ]; then
    dep_version="v1."
    p=-7
    mkdir -p "${thirdpartyPath}/flags"
    for file in $thirdpartyPath/*.success; do
        if [ -f $file ]; then
            old_filename=$(basename "$file")
            new_filename="${old_filename:0:p}$dep_version${old_filename:p}"
            echo "Moving $old_filename to flags/$new_filename"
            mv "$file" "$thirdpartyFlagsDir/$new_filename"
        fi
    done
fi

#******************************************************************************
# #7 Parse arguments
#******************************************************************************

# Defaults.
target="daphne"
par_printHelp="0"
par_clean="0"
par_acceptAll="0"
unknown_options=""
BUILD_CUDA="-DUSE_CUDA=OFF"
BUILD_FPGAOPENCL="-DUSE_FPGAOPENCL=OFF"
BUILD_DEBUG="-DCMAKE_BUILD_TYPE=Release"
BUILD_MPI="-DUSE_MPI=OFF"
BUILD_HDFS="-DUSE_HDFS=OFF"
BUILD_IO_URING="-DUSE_IO_URING=OFF"
BUILD_PAPI="-DUSE_PAPI=ON"
WITH_DEPS=1
WITH_SUBMODULE_UPDATE=1

while [[ $# -gt 0 ]]; do
    key=$1
    shift
    case $key in
    -h | --help)
        par_printHelp="1"
        ;;
    --clean)
        clean_param_check
        par_clean="1"
        ;;
    --cleanAll)
        clean_param_check
        par_clean="2"
        ;;
    --cleanDeps)
        clean_param_check
        par_clean="3"
        ;;
    --cleanCache)
        clean_param_check
        par_clean="4"
        ;;
    --target)
        target=$1
        shift
        ;;
    -nf | --no-fancy)
        fancy="0"
        ;;
    -y | --yes)
        par_acceptAll="1"
        ;;
    --cuda)
        echo using CUDA
        export BUILD_CUDA="-DUSE_CUDA=ON"
        ;;
    --fpgaopencl)
        echo using FPGAOPENCL
        export BUILD_FPGAOPENCL="-DUSE_FPGAOPENCL=ON"
        ;;
    --mpi)
        echo using MPI
        export BUILD_MPI="-DUSE_MPI=ON"
        ;;
    --hdfs)
        echo using HDFS
        export BUILD_HDFS="-DUSE_HDFS=ON"
        ;;
    --io-uring)
        echo using io_uring
        export BUILD_IO_URING="-DUSE_IO_URING=ON"
        ;;
    --no-papi)
        echo not using PAPI
        export BUILD_PAPI="-DUSE_PAPI=OFF"
        ;;
    --debug)
        echo building DEBUG version
        export BUILD_DEBUG="-DCMAKE_BUILD_TYPE=Debug"
        ;;
    --installPrefix)
        installPrefix=$1
        shift
        ;;
    -nd | --no-deps)
        WITH_DEPS=0
        ;;
    -ns | --no-submodule-update)
        WITH_SUBMODULE_UPDATE=0
        ;;
      *)
        unknown_options="${unknown_options} ${key}"
        ;;
    esac
done

if [ -n "$unknown_options" ]; then
    printf "Unknown option(s): '%s'\n\n" "$unknown_options"
    printHelp
    exit 1
fi

if [ "$par_printHelp" -eq 1 ]; then
    printHelp
    exit 0
fi

if [ "$par_clean" -gt 0 ]; then
    case $par_clean in
    1)
        cleanBuildDirs
        ;;
    2)
        cleanAll
        ;;
    3)
        cleanDeps
        ;;
    4)
        cleanCache
        ;;
    esac
    exit 0
fi

# Print Daphne-Logo when first time executing
if [ "$fancy" -eq 1 ] && [ ! -d "${projectRoot}/build" ]; then
    printLogo
    sleep 1
fi

#******************************************************************************
# #8 Download and install third-party dependencies if requested (default is yes, omit with --no-deps))
#******************************************************************************
if [ $WITH_DEPS -gt 0 ]; then
    LLVM_ARCH=X86
    # optimizes for multiple x86_64 architectures
    PAPI_OBLAS_ARCH=NEHALEM
    # Determine CPU architecture to compile for
    if [ $(arch) == 'armv*'  ] || [ $(arch) == 'aarch64' ]; then
      echo "Building for ARMv8 architecture"
      LLVM_ARCH=AArch64
      PAPI_OBLAS_ARCH=ARMV8
    fi

    # Directory name of the LLVM dependency
    llvmName="llvm-project"

    # Make sure that the submodule(s) have been updated since the last clone/pull.
    # But only if this is a git repo.
    # Extra care needs to be taken if the submodule was loaded from gh action cache
    if [ -d .git ]; then
        if [ -e "${thirdpartyPath}/${llvmName}/.git" ]; then # Note: .git in the submodule is not a directory.
            submodule_path=$(cut -d ' ' -f 2 < "${thirdpartyPath}/${llvmName}/.git")

            # if the third party directory was loaded from gh action cache, this path will not exist
            if [ -d "$submodule_path" ] && [ $WITH_SUBMODULE_UPDATE -ne 0 ]; then
                git submodule update --init --recursive
            fi
        # do a submodule update only if llvm path is empty (e.g., initial repo checkout)
        elif [ ! "$(ls -A ${thirdpartyPath}/${llvmName})" ] && [ $WITH_SUBMODULE_UPDATE -ne 0 ]; then
            git submodule update --init --recursive
        fi
    fi


    #------------------------------------------------------------------------------
    # PAPI (Performance Application Programming Interface)
    #------------------------------------------------------------------------------
    if [ $BUILD_PAPI == "-DUSE_PAPI=ON" ]; then
        papiDirName="papi-$papiVersion"
        papiTarName="${papiDirName}.tar.gz"
        papiInstDirName=$installPrefix
        if ! is_dependency_downloaded "papi_v${papiVersion}"; then
            daphne_msg "Get PAPI version ${papiVersion}"
            wget "https://icl.utk.edu/projects/papi/downloads/${papiTarName}" \
                -qO "${cacheDir}/${papiTarName}"
            tar -xf "$cacheDir/$papiTarName" -C "$sourcePrefix"
            dependency_download_success "papi_v${papiVersion}"
        fi
        if ! is_dependency_installed "papi_v${papiVersion}"; then
            cd "$sourcePrefix/$papiDirName/src"
            # FIXME: Add accelerator components (cuda, nvml, rocm, intel_gpu)
            CFLAGS="-fPIC" ./configure --prefix="$papiInstDirName" \
                --with-components="coretemp infiniband io lustre net powercap rapl sde stealtime" \


            CFLAGS="-fPIC -DPIC" make -j"$(nproc)" DYNAMIC_ARCH=1 TARGET="$PAPI_OBLAS_ARCH"
            make install
            cd - > /dev/null
            dependency_install_success "papi_v${papiVersion}"
        else
            daphne_msg "No need to build PAPI again."
        fi
    fi

    #------------------------------------------------------------------------------
    # hwloc
    #------------------------------------------------------------------------------
    hwlocDirName="hwloc-$hwlocVersion"
    hwlocTarName="${hwlocDirName}.tar.gz"
    hwlocInstDirName=$installPrefix
    if ! is_dependency_downloaded "hwloc_v${hwlocVersion}"; then
        daphne_msg "Get hwloc version ${hwlocVersion}"
        wget "https://download.open-mpi.org/release/hwloc/v2.9/${hwlocTarName}" \
            -qO "${cacheDir}/${hwlocTarName}"
        tar -xf "$cacheDir/$hwlocTarName" -C "$sourcePrefix"
        dependency_download_success "hwloc_v${hwlocVersion}"
    fi
    if ! is_dependency_installed "hwloc_v${hwlocVersion}"; then
        cd "$sourcePrefix/$hwlocDirName/"
        ./configure --prefix="$hwlocInstDirName"
        make -j"$(nproc)" DYNAMIC_ARCH=1 TARGET="$PAPI_OBLAS_ARCH"
        make install
        cd - > /dev/null
        dependency_install_success "hwloc_v${hwlocVersion}"
    else
        daphne_msg "No need to build hwloc again."
    fi

    #------------------------------------------------------------------------------
    # Antlr4 (parser)
    #------------------------------------------------------------------------------
    antlrJarName="antlr-${antlrVersion}-complete.jar"
    antlrCppRuntimeDirName="antlr4-cpp-runtime-${antlrVersion}-source"
    antlrCppRuntimeZipName="${antlrCppRuntimeDirName}.zip"
    dep_antlr=("antlr_v${antlrVersion}" "v1")

    # Download antlr4 C++ run-time if it does not exist yet.
    if ! is_dependency_downloaded "${dep_antlr[@]}"; then
        daphne_msg "Get Antlr version ${antlrVersion}"
        # Download antlr4 jar if it does not exist yet.
        daphne_msg "Download Antlr v${antlrVersion} java archive"
        wget "https://www.antlr.org/download/${antlrJarName}" -qO "${cacheDir}/${antlrJarName}"
        daphne_msg "Download Antlr v${antlrVersion} Runtime"
        wget https://www.antlr.org/download/${antlrCppRuntimeZipName} -qO "${cacheDir}/${antlrCppRuntimeZipName}"
        rm -rf "${sourcePrefix:?}/$antlrCppRuntimeDirName"
        mkdir -p "$sourcePrefix/$antlrCppRuntimeDirName"
        unzip -q "$cacheDir/$antlrCppRuntimeZipName" -d "$sourcePrefix/$antlrCppRuntimeDirName"
        dependency_download_success "${dep_antlr[@]}"
    fi
    # build antlr4 C++ run-time
    if ! is_dependency_installed "${dep_antlr[@]}"; then
        mkdir -p "$installPrefix"/share/antlr4/
        cp "$cacheDir/$antlrJarName" "$installPrefix/share/antlr4/$antlrJarName"

        daphne_msg "Applying 0000-antlr-silence-compiler-warnings.patch"
        # disable fail on error as first build might fail and patches might be rejected
        set +e
        patch -Np0 -i "$patchDir/0000-antlr-silence-compiler-warnings.patch" \
            -d "$sourcePrefix/$antlrCppRuntimeDirName"
        # Github disabled the unauthenticated git:// protocol, patch antlr4 to use https://
        # until we upgrade to antlr4-4.9.3+
        sed -i 's#git://github.com#https://github.com#' "$sourcePrefix/$antlrCppRuntimeDirName/runtime/CMakeLists.txt"

        daphne_msg "Build Antlr v${antlrVersion}"
        cmake -S "$sourcePrefix/$antlrCppRuntimeDirName" -B "${buildPrefix}/${antlrCppRuntimeDirName}" \
            -G Ninja -DANTLR4_INSTALL=ON -DCMAKE_INSTALL_PREFIX="$installPrefix" -DCMAKE_BUILD_TYPE=Release
        cmake --build "${buildPrefix}/${antlrCppRuntimeDirName}"
        daphne_msg "Applying 0001-antlr-gtest-silence-warnings.patch"
        patch -Np1 -i "$patchDir/0001-antlr-gtest-silence-warnings.patch" \
            -d "$buildPrefix/$antlrCppRuntimeDirName/runtime/thirdparty/utfcpp/extern/gtest/"

        # enable fail on error again
        set -e
        cmake --build "${buildPrefix}/${antlrCppRuntimeDirName}" --target install/strip

        dependency_install_success "${dep_antlr[@]}"
    else
        daphne_msg "No need to build Antlr4 again."
    fi

    #------------------------------------------------------------------------------
    # catch2 (unit test framework)
    #------------------------------------------------------------------------------
    # Download catch2 release zip (if necessary), and unpack the single header file
    # (if necessary).
    catch2Name="catch2"
    catch2ZipName="v$catch2Version.zip"
    catch2SingleHeaderInstalledPath=$installPrefix/include/catch.hpp
    dep_catch2=("catch2_v${catch2Version}" "v1")
    
    if ! is_dependency_installed "${dep_catch2[@]}"; then
        daphne_msg "Get catch2 version ${catch2Version}"
        mkdir -p "${thirdpartyPath}/${catch2Name}"
        cd "${thirdpartyPath}/${catch2Name}"
        if [ ! -f "$catch2ZipName" ] || [ ! -f "$catch2SingleHeaderInstalledPath" ]; then
            daphne_msg "Download catch2 version ${catch2Version}"
            wget "https://github.com/catchorg/Catch2/archive/refs/tags/${catch2ZipName}" -qO "${cacheDir}/catch2-${catch2ZipName}"
            unzip -q -p "$cacheDir/$catch2Name-$catch2ZipName" "Catch2-$catch2Version/single_include/catch2/catch.hpp" \
                >"$catch2SingleHeaderInstalledPath"
        fi
        dependency_install_success "${dep_catch2[@]}"
    else
        daphne_msg "No need to download Catch2 again."
    fi

    #------------------------------------------------------------------------------
    # OpenBLAS (basic linear algebra subprograms)
    #------------------------------------------------------------------------------
    openBlasDirName="OpenBLAS-$openBlasVersion"
    openBlasZipName="${openBlasDirName}.zip"
    openBlasInstDirName=$installPrefix
    dep_openBlas=("openBlas_v${openBlasVersion}" "v1")

    if ! is_dependency_downloaded "${dep_openBlas[@]}"; then
        daphne_msg "Get OpenBlas version ${openBlasVersion}"
        wget "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${openBlasVersion}/${openBlasZipName}" \
            -qO "${cacheDir}/${openBlasZipName}"
        unzip -q "$cacheDir/$openBlasZipName" -d "$sourcePrefix"
        dependency_download_success "${dep_openBlas[@]}"
    fi
    if ! is_dependency_installed "${dep_openBlas[@]}"; then
        cd "$sourcePrefix/$openBlasDirName"
        make clean
        make -j"$(nproc)" DYNAMIC_ARCH=1 TARGET="$PAPI_OBLAS_ARCH"
        make PREFIX="$openBlasInstDirName" install
        cd - >/dev/null
        dependency_install_success "${dep_openBlas[@]}"
    else
        daphne_msg "No need to build OpenBlas again."
    fi

    #------------------------------------------------------------------------------
    # nlohmann/json (library for JSON parsing)
    #------------------------------------------------------------------------------
    nlohmannjsonDirName=nlohmannjson
    nlohmannjsonSingleHeaderName=json.hpp
    dep_nlohmannjson=("nlohmannjson_v${nlohmannjsonVersion}" "v1")

    if ! is_dependency_installed "${dep_nlohmannjson[@]}"; then
        daphne_msg "Get nlohmannjson version ${nlohmannjsonVersion}"
        mkdir -p "${installPrefix}/include/${nlohmannjsonDirName}"
        wget "https://github.com/nlohmann/json/releases/download/v$nlohmannjsonVersion/$nlohmannjsonSingleHeaderName" \
            -qO "${installPrefix}/include/${nlohmannjsonDirName}/${nlohmannjsonSingleHeaderName}"
        dependency_install_success "${dep_nlohmannjson[@]}"
    else
        daphne_msg "No need to download nlohmannjson again."
    fi

    #------------------------------------------------------------------------------
    # abseil (compiled separately to apply a patch)
    #------------------------------------------------------------------------------
    abslPath=$sourcePrefix/abseil-cpp
    if [ $(arch) == 'armv64'  ] || [ $(arch) == 'aarch64' ]; then
        abslVersion=20211102.0
    fi
    dep_absl=("absl_v${abslVersion}" "v1")

    if ! is_dependency_downloaded "${dep_absl[@]}"; then
        daphne_msg "Get abseil version ${abslVersion}"
        rm -rf "$abslPath"
        git clone --depth 1 --branch "$abslVersion" https://github.com/abseil/abseil-cpp.git "$abslPath"
        if [ $(arch) == 'armv*'  ] || [ $(arch) == 'aarch64' ]; then
           daphne_msg "Applying 0002-absl-stdmax-params.patch"
           patch -Np1 -i "${patchDir}/0002-absl-stdmax-params.patch" -d "$abslPath"
        fi
        dependency_download_success "${dep_absl[@]}"
    fi
    if ! is_dependency_installed "${dep_absl[@]}"; then
        cmake -S "$abslPath" -B "$buildPrefix/absl" -G Ninja -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE \
            -DCMAKE_INSTALL_PREFIX="$installPrefix" -DCMAKE_CXX_STANDARD=17 -DABSL_PROPAGATE_CXX_STD=ON
        cmake --build "$buildPrefix/absl" --target install/strip
        dependency_install_success "${dep_absl[@]}"
    else
        daphne_msg "No need to build Abseil again."
    fi

    #------------------------------------------------------------------------------
    # MPI (Default is MPI library is OpenMPI but cut can be any)
    #------------------------------------------------------------------------------
    MPIZipName=openmpi-$openMPIVersion.tar.gz
    MPIInstDirName=$installPrefix
    dep_mpi=("openmpi_v${openMPIVersion}" "v1")

    if ! is_dependency_downloaded "${dep_mpi[@]}"; then
        daphne_msg "Get openmpi version ${openMPIVersion}"
        wget "https://download.open-mpi.org/release/open-mpi/v4.1/$MPIZipName" -qO "${cacheDir}/${MPIZipName}"
        tar -xf "$cacheDir/$MPIZipName" --directory "$sourcePrefix"
        dependency_download_success "${dep_mpi[@]}"
        mkdir -p "$MPIInstDirName"
    fi
    if ! is_dependency_installed "${dep_mpi[@]}"; then
        cd "$sourcePrefix/openmpi-$openMPIVersion"
        ./configure --prefix="$MPIInstDirName"
        make -j"$(nproc)" all
        make install
        cd -
        dependency_install_success "${dep_mpi[@]}"
    else
        daphne_msg "No need to build OpenMPI again"
    fi
    #------------------------------------------------------------------------------
    # gRPC
    #------------------------------------------------------------------------------
    grpcDirName="grpc"
    grpcInstDir=$installPrefix
    dep_grpc=("grpc_v${grpcVersion}" "v1")

    if ! is_dependency_downloaded "${dep_grpc[@]}"; then
        daphne_msg "Get grpc version ${grpcVersion}"
        # Download gRPC source code.
        if [ -d "${sourcePrefix}/${grpcDirName}" ]; then
            rm -rf "${sourcePrefix}/${grpcDirName:?}"
        fi
        git clone -b v$grpcVersion --depth 1 https://github.com/grpc/grpc "$sourcePrefix/$grpcDirName"
        pushd "$sourcePrefix/$grpcDirName"
        git submodule update --init --depth 1 third_party/boringssl-with-bazel
        git submodule update --init --depth 1 third_party/cares/cares
        git submodule update --init --depth 1 third_party/protobuf
        git submodule update --init --depth 1 third_party/re2
        daphne_msg "Applying 0003-protobuf-override.patch"
        patch -Np1 -i "${patchDir}/0003-protobuf-override.patch" -d "$sourcePrefix/$grpcDirName/third_party/protobuf"
        popd
        dependency_download_success "${dep_grpc[@]}"
    fi
    if ! is_dependency_installed "${dep_grpc[@]}"; then
        cmake -G Ninja -S "$sourcePrefix/$grpcDirName" -B "$buildPrefix/$grpcDirName" \
            -DCMAKE_INSTALL_PREFIX="$grpcInstDir" \
            -DCMAKE_BUILD_TYPE=Release \
            -DgRPC_INSTALL=ON \
            -DgRPC_BUILD_TESTS=OFF \
            -DCMAKE_CXX_STANDARD=17 \
            -DCMAKE_INCLUDE_PATH="$installPrefix/include" \
            -DgRPC_ABSL_PROVIDER=package \
            -DgRPC_ZLIB_PROVIDER=package
        cmake --build "$buildPrefix/$grpcDirName" --target install/strip
        dependency_install_success "${dep_grpc[@]}"
    else
        daphne_msg "No need to build GRPC again."
    fi
    #------------------------------------------------------------------------------
    # Arrow / Parquet
    #------------------------------------------------------------------------------
    arrowDirName="apache-arrow-$arrowVersion"
    arrowArtifactFileName=$arrowDirName.tar.gz
    dep_arrow=("arrow_v${arrowVersion}" "v1")

    if ! is_dependency_downloaded "${dep_arrow[@]}"; then
        rm -rf "${sourcePrefix:?}/${arrowDirName}"
        wget "https://archive.apache.org/dist/arrow/arrow-$arrowVersion/$arrowArtifactFileName" -qP "$cacheDir"
        tar xzf "$cacheDir/$arrowArtifactFileName" --directory="$sourcePrefix"
        daphne_msg "Applying 0004-arrow-git-log.patch"
        patch -Np0 -i "$patchDir/0004-arrow-git-log.patch" -d "$sourcePrefix/$arrowDirName"
        dependency_download_success "${dep_arrow[@]}"
    fi

    # this works around a build error that occurs on Ubuntu with Boost installed
    if [ $(lsb_release -is) == "Ubuntu" ]; then
        if [ $(dpkg -l | grep libboost | wc -l) == "" ]; then
            daphne_msg "Setting BOOST_ROOT=/usr on Ubuntu Linux with libboost installed"
            sleep 5
            export BOOST_ROOT=/usr
        fi
    fi

    if ! is_dependency_installed "${dep_arrow[@]}"; then
        cmake -G Ninja -S "${sourcePrefix}/${arrowDirName}/cpp" -B "${buildPrefix}/${arrowDirName}" \
            -DCMAKE_INSTALL_PREFIX="${installPrefix}" -DARROW_CSV=ON -DARROW_FILESYSTEM=ON -DARROW_PARQUET=ON \
            -DARROW_WITH_BROTLI=ON -DARROW_WITH_BZ2=ON -DARROW_WITH_LZ4=ON -DARROW_WITH_SNAPPY=ON -DARROW_WITH_ZLIB=ON \
            -DARROW_WITH_ZSTD=ON

        cmake --build "${buildPrefix}/${arrowDirName}" --target install/strip
        dependency_install_success "${dep_arrow[@]}"
    else
        daphne_msg "No need to build Arrow again."
    fi
    #------------------------------------------------------------------------------
    # fmt
    #------------------------------------------------------------------------------
    fmtDirName="fmt-$fmtVersion"
    fmtArtifactFileName=$fmtDirName.zip
    if ! is_dependency_downloaded "fmt_v${fmtVersion}"; then
        rm -rf "${sourcePrefix:?}/${fmtDirName}"
        wget "https://github.com/fmtlib/fmt/releases/download/${fmtVersion}/$fmtArtifactFileName" -qO  "$cacheDir/$fmtArtifactFileName"
        unzip -q "$cacheDir/$fmtArtifactFileName" -d "$sourcePrefix"
        dependency_download_success "fmt_v${fmtVersion}"
    fi
    if ! is_dependency_installed "fmt_v${fmtVersion}"; then
        cmake -G Ninja -S "${sourcePrefix}/${fmtDirName}" -B "${buildPrefix}/${fmtDirName}" \
            -DCMAKE_INSTALL_PREFIX="${installPrefix}" -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DFMT_MASTER_PROJECT=OFF
        cmake --build "${buildPrefix}/${fmtDirName}" --target install/strip
        dependency_install_success "fmt_v${fmtVersion}"
    else
        daphne_msg "No need to build fmt again."
    fi
    #------------------------------------------------------------------------------
    # spdlog
    #------------------------------------------------------------------------------
    spdlogDirName="spdlog-$spdlogVersion"
    spdlogArtifactFileName=$spdlogDirName.tar.gz
    if ! is_dependency_downloaded "spdlog_v${spdlogVersion}"; then
        rm -rf "${sourcePrefix:?}/${spdlogDirName}"
        # changed URL scheme due to  temporarily use tip of main branch (2024-10-03)
#        wget "https://github.com/gabime/spdlog/archive/refs/tags/v$spdlogVersion.tar.gz" -qO \
        wget https://github.com/gabime/spdlog/archive/$spdlogVersion.tar.gz -qO \
            "$cacheDir/$spdlogArtifactFileName"
        tar xzf "$cacheDir/$spdlogArtifactFileName" --directory="$sourcePrefix"
        dependency_download_success "spdlog_v${spdlogVersion}"
    fi
    if ! is_dependency_installed "spdlog_v${spdlogVersion}"; then
        cmake -G Ninja -S "${sourcePrefix}/${spdlogDirName}" -B "${buildPrefix}/${spdlogDirName}" \
            -DSPDLOG_FMT_EXTERNAL=ON -DCMAKE_INSTALL_PREFIX="${installPrefix}" -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        cmake --build "${buildPrefix}/${spdlogDirName}" --target install/strip
        dependency_install_success "spdlog_v${spdlogVersion}"
    else
        daphne_msg "No need to build spdlog again."
    fi
    #------------------------------------------------------------------------------
    # Eigen
    #------------------------------------------------------------------------------
    eigenDirName="eigen-${eigenVersion}"
    if ! is_dependency_downloaded "eigen_v${eigenVersion}"; then
      wget https://gitlab.com/libeigen/eigen/-/archive/${eigenVersion}/eigen-${eigenVersion}.tar.bz2 -qP "${cacheDir}"
      rm -rf ${sourcePrefix}/${eigenDirName}
      tar xf "$cacheDir/eigen-${eigenVersion}.tar.bz2" --directory "$sourcePrefix"
      cd ${sourcePrefix}/${eigenDirName}
      dependency_download_success "eigen_v${eigenVersion}"
    fi
    if ! is_dependency_installed "eigen_v${eigenVersion}"; then
      cmake -G Ninja -S "${sourcePrefix}/${eigenDirName}" -B "${buildPrefix}/${eigenDirName}" \
          -DCMAKE_INSTALL_PREFIX=${installPrefix}
      cmake --build "${buildPrefix}/${eigenDirName}" --target install/strip
      dependency_install_success "eigen_v${eigenVersion}"
    else
      daphne_msg "No need to build eigen again."
    fi
    #------------------------------------------------------------------------------
    # HAWQ (libhdfs3)
    #------------------------------------------------------------------------------
    hawqDirName="hawq-rel-v$hawqVersion"
    hawqDlTarName="v${hawqVersion}.tar.gz"
    hawqTarName="${hawqDirName}.tar.gz"
    hawqInstDirName=$installPrefix

    if [ $BUILD_HDFS == "-DUSE_HDFS=ON" ]; then
        if ! is_dependency_downloaded "hawq_v${hawqVersion}"; then
            daphne_msg "Get HAWQ (libhdfs3) version ${hawqVersion}"
            wget "https://github.com/apache/hawq/archive/refs/tags/rel/${hawqDlTarName}" \
                -qO "${cacheDir}/${hawqTarName}"
            tar -xf "$cacheDir/$hawqTarName" -C "$sourcePrefix"
            daphne_msg "Applying 0005-libhdfs3-remove-gtest-dep.patch"
            patch -Np1 -i "${patchDir}/0005-libhdfs3-remove-gtest-dep.patch" -d "$sourcePrefix/$hawqDirName"
            daphne_msg "Applying 0006-libhdfs3-add-cstdint-include.patch"
            patch -Np1 -i "${patchDir}/0006-libhdfs3-add-cstdint-include.patch" -d "$sourcePrefix/$hawqDirName"
            dependency_download_success "hawq_v${hawqVersion}"
        fi
        if ! is_dependency_installed "hawq_v${hawqVersion}"; then
            cmake -G Ninja -S "$sourcePrefix/$hawqDirName/depends/libhdfs3" -B "${buildPrefix}/${hawqDirName}" \
                -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$installPrefix"
            cmake --build "${buildPrefix}/${hawqDirName}" --target install/strip
            dependency_install_success "hawq_v${hawqVersion}"
        else
              daphne_msg "No need to build HAWQ (libhdfs3) again."
        fi
    fi

    #------------------------------------------------------------------------------
    # Build MLIR
    #------------------------------------------------------------------------------
    # We rarely need to build MLIR/LLVM, only during the first build of the
    # prototype and after upgrades of the LLVM sub-module. To avoid unnecessary
    # builds (which take several seconds even if there is nothing to do), we store
    # the LLVM commit hash we built into a file, and only rebuild MLIR/LLVM if this
    # file does not exist (first build of the prototype) or does not contain the
    # expected hash (upgrade of the LLVM sub-module).

    llvmCommit="llvmCommit-local-none"
    cd "${thirdpartyPath}/${llvmName}"
    if [ -e .git ]; then # Note: .git in the submodule is not a directory.
        submodule_path=$(cut -d ' ' -f 2 < .git)

        # if the third party directory was loaded from gh action cache, this path will not exist
        if [ -d "$submodule_path" ]; then
            llvmCommit="$(git log -1 --format=%H)"
        else
            llvmCommit="3623fe661ae35c6c80ac221f14d85be76aa870f1"
        fi
    else
        # download and set up LLVM code if compilation is run without the local working copy being checked out from git
        # e.g., compiling from released source artifact
        llvmCommit="3623fe661ae35c6c80ac221f14d85be76aa870f1"
        llvmSnapshotArtifact="llvm_${llvmCommit}.tar.gz"
        llvmSnapshotPath="${cacheDir}/${llvmSnapshotArtifact}"
        if ! is_dependency_downloaded "llvm_v${llvmCommit}"; then
            daphne_msg "Geting LLVM/MLIR snapshot ${llvmCommit}"
            wget https://github.com/llvm/llvm-project/archive/${llvmCommit}.tar.gz -qO "${llvmSnapshotPath}"
            tar xzf "${llvmSnapshotPath}" --strip-components=1
            dependency_download_success "llvm_v${llvmCommit}"
        fi
    fi

    cd - >/dev/null # return back from llvm 3rd party subdir

    if ! is_dependency_installed "llvm_v${llvmCommit}" || [ "$(cat "${llvmCommitFilePath}")" != "$llvmCommit" ]; then
        daphne_msg "Building LLVM/MLIR from ${llvmCommit}"
        cd "${thirdpartyPath}/${llvmName}"
        echo "Need to build MLIR/LLVM."
        cmake -G Ninja -S llvm -B "$buildPrefix/$llvmName" \
            -DLLVM_ENABLE_PROJECTS=mlir \
            -DLLVM_BUILD_EXAMPLES=OFF \
            -DLLVM_TARGETS_TO_BUILD="$LLVM_ARCH" \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_ENABLE_ASSERTIONS=ON \
            -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
            -DLLVM_ENABLE_RTTI=ON \
            -DCMAKE_INSTALL_PREFIX="$installPrefix"
        cmake --build "$buildPrefix/$llvmName" --target check-mlir
        cmake --build "$buildPrefix/$llvmName" --target install/strip
        echo "$llvmCommit" >"$llvmCommitFilePath"
        cd - >/dev/null
        dependency_install_success "llvm_v${llvmCommit}"
    else
        daphne_msg "No need to build MLIR/LLVM again."
    fi
    #------------------------------------------------------------------------------
    # Liburing
    #------------------------------------------------------------------------------
    liburingDirName="liburing-$liburingVersion"
    liburingTarName="${liburingDirName}.tar.gz"
    liburingInstDirName=$installPrefix
    liburing_cc=$([ "$CC" = "" ] && echo "gcc" || echo "$CC")
    liburing_cxx=$([ "$CXX" = "" ] && echo "g++" || echo "$CXX")

    if [ $BUILD_IO_URING == "-DUSE_IO_URING=ON" ]; then
        if ! is_dependency_downloaded "liburing_v${liburingVersion}"; then
            daphne_msg "Get liburing version ${liburingVersion}"
            wget "https://github.com/axboe/liburing/archive/refs/tags/${liburingTarName}" \
                -qO "${cacheDir}/${liburingTarName}"
            mkdir "$sourcePrefix/$liburingDirName"
            tar -xf "$cacheDir/$liburingTarName" -C "$sourcePrefix/$liburingDirName" --strip-components=1
            dependency_download_success "liburing_v${liburingVersion}"
        fi
        if ! is_dependency_installed "liburing_v${liburingVersion}"; then
            cd "$sourcePrefix/$liburingDirName"
            ./configure --cc="$liburing_cc" --cxx="$liburing_cxx" --prefix="$liburingInstDirName"
            make -j"$(nproc)"
            cp ./src/liburing.a "$installPrefix/lib/"
            cp -r ./src/include/* "$installPrefix/include"
            cd - > /dev/null
            dependency_install_success "liburing_v${liburingVersion}"
        else
            daphne_msg "No need to build liburing again."
        fi
    fi
    #------------------------------------------------------------------------------
    # Fetch bitstreams
    #------------------------------------------------------------------------------
    if [[ $BUILD_FPGAOPENCL = *"ON"* ]]; then
        FPGAOPENCL_BISTREAM_DIR="$projectRoot/src/runtime/local/kernels/FPGAOPENCL/bitstreams"
        FPGAOPENCL_BISTREAM_URL="https://github.com/daphne-eu/supplemental-binaries/raw/main/fpga_bitstreams/"
        if [ ! -d "$FPGAOPENCL_BISTREAM_DIR" ]; then
            echo fetching FPGA bitstreams
            mkdir -p "$FPGAOPENCL_BISTREAM_DIR"
            cd "$FPGAOPENCL_BISTREAM_DIR"
            wget $FPGAOPENCL_BISTREAM_URL/sgemm.aocx
            wget $FPGAOPENCL_BISTREAM_URL/sgemv.aocx
            wget $FPGAOPENCL_BISTREAM_URL/ssyrk.aocx
            cd - >/dev/null
        fi
    fi
fi

#******************************************************************************
# #9 Build DAPHNE target.
#******************************************************************************

daphne_msg "Build Daphne"

cmake -S "$projectRoot" -B "$daphneBuildDir" -G Ninja -DANTLR_VERSION="$antlrVersion" \
    -DCMAKE_PREFIX_PATH="$installPrefix" \
    $BUILD_CUDA $BUILD_FPGAOPENCL $BUILD_DEBUG $BUILD_MPI $BUILD_HDFS $BUILD_PAPI

cmake --build "$daphneBuildDir" --target "$target" #-- -j2

build_ts_end=$(date +%s%N)
daphne_msg "Successfully built Daphne://${target} (took $(printableTimestamp $((build_ts_end - build_ts_begin))))"

set +e
