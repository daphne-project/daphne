#!/bin/bash
#
# Shell-level unit tests for the memory-detection and JOBS-picking helpers
# defined in build.sh: get_memory_gb() and calc_daphne_jobs().
#
# Why this file exists: those two functions have several code paths that
# cannot be exercised by a real docker build (cgroup v1 sentinel, memory.max
# = "max", missing /proc/meminfo, well-resourced host suppression). This
# script mocks the filesystem inputs, runs each branch, and asserts the
# expected output.
#
# Invoked via test/build/BuildHelpersTest.cpp as part of the run_tests
# Catch2 suite ("./test.sh [build]"), and can also be run standalone:
#
#   bash test/build/test_build_helpers.sh
#
# Exits 0 on success, non-zero if any assertion fails.

set -u

# ---------------------------------------------------------------------------
# Test framework (kept minimal; no external deps)
# ---------------------------------------------------------------------------
FAIL=0
PASS=0

assert_eq() {
    # $1 = expected, $2 = actual, $3 = test name
    if [ "$1" = "$2" ]; then
        printf '  \033[32mPASS\033[0m  %s\n' "$3"
        PASS=$((PASS + 1))
    else
        printf '  \033[31mFAIL\033[0m  %s\n        expected: %s\n        actual:   %s\n' \
            "$3" "$1" "$2"
        FAIL=$((FAIL + 1))
    fi
}

# ---------------------------------------------------------------------------
# get_memory_gb (mockable version)
#
# The version in build.sh reads absolute paths (/proc/self/cgroup,
# /sys/fs/cgroup/..., /proc/meminfo) plus macOS sysctl. To exercise each
# path deterministically we take a MOCK_ROOT prefix and read from
# $MOCK_ROOT/proc/... and $MOCK_ROOT/sys/fs/cgroup/... instead. The
# arithmetic and control flow are byte-for-byte the same as build.sh.
# ---------------------------------------------------------------------------
get_memory_gb_mocked() {
    local mock="$1"
    local mem_bytes=0
    local os="${2:-Linux}"   # optional 2nd arg: "Darwin" to force macOS branch

    if [ "$os" = "Darwin" ]; then
        # For macOS we can't mock sysctl portably; only exercised on real
        # Darwin hosts. This branch is a placeholder to document intent.
        # The interesting branches for coverage are the Linux ones.
        local raw="${MOCK_DARWIN_MEMSIZE:-0}"
        if [[ "$raw" =~ ^[0-9]+$ ]] && [ "$raw" -gt 0 ]; then
            echo $(( raw / 1024 / 1024 / 1024 ))
        else
            echo 4
        fi
        return
    fi

    # cgroup v2 walk
    local cg_path
    cg_path=$(awk -F: '/^0:/{print $3; exit}' "$mock/proc/self/cgroup" 2>/dev/null)
    if [ -n "$cg_path" ]; then
        local path="$cg_path" val min_limit=0 mem_max_file
        while true; do
            mem_max_file="$mock/sys/fs/cgroup${path}/memory.max"
            if [ -r "$mem_max_file" ]; then
                val=$(cat "$mem_max_file" 2>/dev/null)
                if [ "$val" != "max" ] && [[ "$val" =~ ^[0-9]+$ ]]; then
                    if [ "$min_limit" -eq 0 ] || [ "$val" -lt "$min_limit" ]; then
                        min_limit="$val"
                    fi
                fi
            fi
            [ "$path" = "/" ] || [ -z "$path" ] && break
            path=$(dirname "$path")
            [ "$path" = "." ] && break
        done
        [ "$min_limit" -gt 0 ] && mem_bytes="$min_limit"
    fi

    # cgroup v1 fallback
    if [ "$mem_bytes" -eq 0 ]; then
        local v1="$mock/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if [ -r "$v1" ]; then
            local v1val
            v1val=$(cat "$v1" 2>/dev/null)
            if [[ "$v1val" =~ ^[0-9]+$ ]] && [ "$v1val" -gt 0 ] && \
               [ "$v1val" -lt 1125899906842624 ]; then
                mem_bytes="$v1val"
            fi
        fi
    fi

    # /proc/meminfo fallback
    if [ "$mem_bytes" -eq 0 ] && [ -r "$mock/proc/meminfo" ]; then
        # Multiply outside awk (see build.sh): keeps large byte counts out of
        # mawk's scientific-notation formatting so the guard below accepts them.
        local mem_kb
        mem_kb=$(awk '/^MemTotal:/{print $2; exit}' "$mock/proc/meminfo")
        mem_kb="${mem_kb:-0}"
        [[ "$mem_kb" =~ ^[0-9]+$ ]] && mem_bytes=$(( mem_kb * 1024 ))
    fi

    if ! [[ "$mem_bytes" =~ ^[0-9]+$ ]] || [ "$mem_bytes" -eq 0 ]; then
        echo 4
        return
    fi
    echo $(( mem_bytes / 1024 / 1024 / 1024 ))
}

# calc_daphne_jobs is pulled verbatim from build.sh (no filesystem reads,
# just arithmetic). Parametrized by nproc.
calc_daphne_jobs_mocked() {
    local avail_gb=$1
    local ncpu=$2
    local usable=$(( avail_gb > 1 ? avail_gb - 1 : 1 ))
    local cj=$(( usable / 2 ))
    [ "$cj" -lt 1 ] && cj=1
    [ "$cj" -gt "$ncpu" ] && cj=$ncpu
    local kj=$cj
    echo "$cj $kj"
}

# ---------------------------------------------------------------------------
# Helper: build a mock filesystem tree from a fixture spec
# ---------------------------------------------------------------------------
make_mock_root() {
    local root
    root=$(mktemp -d)
    mkdir -p "$root/proc/self" "$root/sys/fs/cgroup"
    echo "$root"
}

# ---------------------------------------------------------------------------
# get_memory_gb tests
# ---------------------------------------------------------------------------
printf '\n== get_memory_gb ==\n'

# --- v2, single limit at container path ---
r=$(make_mock_root)
mkdir -p "$r/sys/fs/cgroup/docker/abcd"
printf '0::/docker/abcd\n'                 > "$r/proc/self/cgroup"
printf '8589934592'                         > "$r/sys/fs/cgroup/docker/abcd/memory.max"   # 8 GiB
printf 'max'                                > "$r/sys/fs/cgroup/memory.max" 2>/dev/null   # root
assert_eq "8" "$(get_memory_gb_mocked "$r")" "v2: container 8g limit"
rm -rf "$r"

# --- v2, memory.max == "max" everywhere; falls to meminfo ---
r=$(make_mock_root)
mkdir -p "$r/sys/fs/cgroup/docker/xyz"
printf '0::/docker/xyz\n'                   > "$r/proc/self/cgroup"
printf 'max'                                > "$r/sys/fs/cgroup/docker/xyz/memory.max"
printf 'max'                                > "$r/sys/fs/cgroup/memory.max"
# meminfo says 16 GiB
printf 'MemTotal:       16777216 kB\n'      > "$r/proc/meminfo"
assert_eq "16" "$(get_memory_gb_mocked "$r")" "v2: all max, fall through to meminfo 16g"
rm -rf "$r"

# --- v2, tighter limit in PARENT than in leaf: min-of-chain wins ---
r=$(make_mock_root)
mkdir -p "$r/sys/fs/cgroup/parent-slice/leaf"
printf '0::/parent-slice/leaf\n'            > "$r/proc/self/cgroup"
printf '8589934592'                         > "$r/sys/fs/cgroup/parent-slice/leaf/memory.max"   # 8 GiB
printf '4294967296'                         > "$r/sys/fs/cgroup/parent-slice/memory.max"        # 4 GiB (tighter)
printf 'max'                                > "$r/sys/fs/cgroup/memory.max"
assert_eq "4" "$(get_memory_gb_mocked "$r")" "v2: walks up, picks tighter parent 4g"
rm -rf "$r"

# --- v1: sentinel value ~unlimited (9223372036854771712 bytes ≈ 8 EiB) → rejected, fall to meminfo ---
r=$(make_mock_root)
printf ''                                   > "$r/proc/self/cgroup"           # v2 absent
mkdir -p "$r/sys/fs/cgroup/memory"
printf '9223372036854771712'                > "$r/sys/fs/cgroup/memory/memory.limit_in_bytes"   # v1 unlimited sentinel
printf 'MemTotal:       8388608 kB\n'       > "$r/proc/meminfo"               # 8 GiB fallback
assert_eq "8" "$(get_memory_gb_mocked "$r")" "v1: sentinel rejected, meminfo 8g wins"
rm -rf "$r"

# --- v1: real limit 4 GiB ---
r=$(make_mock_root)
printf ''                                   > "$r/proc/self/cgroup"
mkdir -p "$r/sys/fs/cgroup/memory"
printf '4294967296'                         > "$r/sys/fs/cgroup/memory/memory.limit_in_bytes"   # 4 GiB
printf 'MemTotal:       16777216 kB\n'      > "$r/proc/meminfo"               # 16g (should be ignored)
assert_eq "4" "$(get_memory_gb_mocked "$r")" "v1: real 4g limit beats meminfo 16g"
rm -rf "$r"

# --- meminfo only ---
r=$(make_mock_root)
printf ''                                   > "$r/proc/self/cgroup"
# no v1 file
printf 'MemTotal:       33554432 kB\n'      > "$r/proc/meminfo"   # 32 GiB
assert_eq "32" "$(get_memory_gb_mocked "$r")" "meminfo 32g bare host"
rm -rf "$r"

# --- meminfo on a big host: bytes exceed 2^31, must not be lost to awk's ---
# --- scientific notation (would report 4g if multiplied inside awk on mawk) ---
r=$(make_mock_root)
printf ''                                   > "$r/proc/self/cgroup"
printf 'MemTotal:       134217728 kB\n'     > "$r/proc/meminfo"   # 128 GiB
assert_eq "128" "$(get_memory_gb_mocked "$r")" "meminfo 128g host, no sci-notation loss"
rm -rf "$r"

# --- nothing readable → fallback 4 ---
r=$(make_mock_root)
# leave everything empty / missing
assert_eq "4" "$(get_memory_gb_mocked "$r")" "no source available, fallback 4"
rm -rf "$r"

# --- malformed memory.max content is ignored (should not crash, should fall through) ---
r=$(make_mock_root)
mkdir -p "$r/sys/fs/cgroup/docker/abc"
printf '0::/docker/abc\n'                   > "$r/proc/self/cgroup"
printf 'not-a-number\n'                     > "$r/sys/fs/cgroup/docker/abc/memory.max"
printf 'MemTotal:       8388608 kB\n'       > "$r/proc/meminfo"
assert_eq "8" "$(get_memory_gb_mocked "$r")" "malformed memory.max ignored, meminfo 8g"
rm -rf "$r"

# --- v2 present at path but file empty ---
r=$(make_mock_root)
mkdir -p "$r/sys/fs/cgroup/docker/def"
printf '0::/docker/def\n'                   > "$r/proc/self/cgroup"
printf ''                                   > "$r/sys/fs/cgroup/docker/def/memory.max"   # empty
printf 'MemTotal:       4194304 kB\n'       > "$r/proc/meminfo"
assert_eq "4" "$(get_memory_gb_mocked "$r")" "empty memory.max falls through, meminfo 4g"
rm -rf "$r"

# ---------------------------------------------------------------------------
# calc_daphne_jobs tests
# ---------------------------------------------------------------------------
printf '\n== calc_daphne_jobs (nproc=12) ==\n'

# Standard table with 12-core host (nproc doesn't clip anything under 26 GB)
declare -a cases_12=(
    "1 1 1"
    "2 1 1"
    "3 1 1"
    "4 1 1"
    "5 2 2"
    "6 2 2"
    "7 3 3"
    "8 3 3"
    "12 5 5"
    "16 7 7"
    "26 12 12"    # 12 == nproc, no clamp yet
    "32 12 12"    # clamp at nproc
    "64 12 12"    # deep clamp at nproc
)
for row in "${cases_12[@]}"; do
    read -r gb ecj ekj <<<"$row"
    read -r cj kj <<<"$(calc_daphne_jobs_mocked "$gb" 12)"
    assert_eq "${ecj} ${ekj}" "${cj} ${kj}" "gb=${gb} nproc=12 → ${ecj}/${ekj}"
done

printf '\n== calc_daphne_jobs (nproc=1): tiny host clamp ==\n'
# On a 1-CPU host we can never exceed nproc=1 regardless of RAM.
for row in "1 1 1" "4 1 1" "8 1 1" "32 1 1"; do
    read -r gb ecj ekj <<<"$row"
    read -r cj kj <<<"$(calc_daphne_jobs_mocked "$gb" 1)"
    assert_eq "${ecj} ${ekj}" "${cj} ${kj}" "gb=${gb} nproc=1 → clamp to 1"
done

printf '\n== calc_daphne_jobs (nproc=4): modest host clamp ==\n'
# 4-CPU host clamps at 4 for anything ≥ 10 GB.
declare -a cases_4=(
    "1 1 1"
    "4 1 1"
    "6 2 2"
    "8 3 3"
    "10 4 4"      # clamp begins
    "16 4 4"      # deep clamp
    "64 4 4"
)
for row in "${cases_4[@]}"; do
    read -r gb ecj ekj <<<"$row"
    read -r cj kj <<<"$(calc_daphne_jobs_mocked "$gb" 4)"
    assert_eq "${ecj} ${ekj}" "${cj} ${kj}" "gb=${gb} nproc=4 → ${ecj}/${ekj}"
done

# ---------------------------------------------------------------------------
# Cross-check: the well-resourced host suppression condition
# ---------------------------------------------------------------------------
# In build.sh the auto-detect banner prints ONLY when auto_cj < nproc.
# Verify that for hosts where 2×nproc GB of RAM is present, auto_cj == nproc
# (suppression fires) so we don't spam the message on beefy hosts.
printf '\n== well-resourced-host suppression ==\n'
for nproc in 1 2 4 8 12 16; do
    # 2*nproc + 1 GB should give cj = nproc exactly
    gb=$(( 2 * nproc + 1 ))
    read -r cj _kj <<<"$(calc_daphne_jobs_mocked "$gb" "$nproc")"
    assert_eq "$nproc" "$cj" "gb=${gb} nproc=${nproc}: cj hits nproc (no banner)"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
printf '\n== summary ==\n'
printf '  pass: %d\n' "$PASS"
printf '  fail: %d\n' "$FAIL"
exit "$FAIL"
