#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Where we are (script at repo root: /daphne)
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # /daphne
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"  # ensure CWD is repo root so relative catalogs resolve

# ------------------------------------------------------------
# Arrow/Daphne runtime linking (adjust if paths differ)
# ------------------------------------------------------------
DAPHNE_ROOT="${DAPHNE_ROOT:-$REPO_ROOT}"
TP_LIB_DIR="${TP_LIB_DIR:-$DAPHNE_ROOT/thirdparty/installed/lib}"
DS_LIB_DIR="${DS_LIB_DIR:-$DAPHNE_ROOT/build/src/runtime/local/datastructures}"
export LD_LIBRARY_PATH="$TP_LIB_DIR:$DS_LIB_DIR:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$TP_LIB_DIR:${LIBRARY_PATH:-}"
export CPATH="${DAPHNE_ROOT}/thirdparty/installed/include:${CPATH:-}"

# ------------------------------------------------------------
# Find Daphne binary
# ------------------------------------------------------------
if [[ -z "${DAPHNE_BIN:-}" ]]; then
  for c in "$REPO_ROOT/bin/daphne" "$REPO_ROOT/build/bin/daphne"; do
    [[ -x "$c" ]] && { DAPHNE_BIN="$c"; break; }
  done
  [[ -n "${DAPHNE_BIN:-}" ]] || { echo "[error] daphne not found in $REPO_ROOT/bin or $REPO_ROOT/build/bin. Set \$DAPHNE_BIN."; exit 1; }
fi

# ------------------------------------------------------------
# Daphne DSL + extension JSONs
# ------------------------------------------------------------
SCRIPT_ROOT="${SCRIPT_ROOT:-$REPO_ROOT/scripts/examples/extensions/experiment}"
DSL_DIR_MATRIX="$SCRIPT_ROOT/matrix"
[[ -d "$DSL_DIR_MATRIX" ]] || { echo "[error] DSL folder not found at $DSL_DIR_MATRIX"; exit 1; }

CSV_EXT_JSON="${CSV_EXT_JSON:-$REPO_ROOT/scripts/examples/extensions/csv/myIO.json}"
PARQUET_EXT_JSON="${PARQUET_EXT_JSON:-$REPO_ROOT/scripts/examples/extensions/parquetReader/parquet.json}"

# CSV
CSV_SCRIPTS_NORMAL=(
  "$DSL_DIR_MATRIX/csv_eval.daphne"
  "$DSL_DIR_MATRIX/csv_small_eval.daphne"
)
CSV_SCRIPTS_PLUGIN=(
  "$DSL_DIR_MATRIX/csv_plugin_eval.daphne"
  "$DSL_DIR_MATRIX/csv_plugin_small_eval.daphne"
)

# Parquet
PARQUET_SCRIPTS_NORMAL=(
  "$DSL_DIR_MATRIX/parquet_small_eval.daphne"
)
PARQUET_SCRIPTS_PLUGIN=(
  "$DSL_DIR_MATRIX/parquet_plugin_small_eval.daphne"
  "$DSL_DIR_MATRIX/parquet_single_thread_eval.daphne"
  "$DSL_DIR_MATRIX/parquet_multi_thread_eval.daphne"
)


# ------------------------------------------------------------
# Generator + Meta binaries (live in /daphne/experiment)
# ------------------------------------------------------------
GEN_DATA_BIN="${GEN_DATA_BIN:-$REPO_ROOT/experiment/gen_data}"
GEN_META_BIN="${GEN_META_BIN:-$REPO_ROOT/experiment/gen_meta}"
[[ -x "$GEN_DATA_BIN" ]] || { echo "[error] generator not found or not executable at: $GEN_DATA_BIN"; exit 1; }
[[ -x "$GEN_META_BIN" ]] || { echo "[error] gen_meta not found or not executable at: $GEN_META_BIN"; exit 1; }

# ------------------------------------------------------------
# Optional: show downloader
# ------------------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  echo "[net] using curl"
elif command -v wget >/dev/null 2>&1; then
  echo "[net] using wget"
else
  echo "[net] using python urllib (no curl/wget found)"
fi

# ------------------------------------------------------------
# CLI / defaults
# ------------------------------------------------------------
TARGET_MB=1500
FORMATS=("csv" "parquet")
ONLY=""
KEEP=0
SKIP_KNOWN=0
SKIP_GENERATED=0
OUTDIR=""

usage() {
  cat <<EOF
Usage: $0 [options]
  --target-size-mb N   Per-file target size in MB (default: $TARGET_MB)
  --only FMT           Only run one: csv | parquet
  --formats ...        Space-separated list of formats to run
  --outdir PATH        Output dir (default: experiment/eval)
  --keep               Keep files (skip cleanup)
  --skip-known         Skip known/downloaded datasets
  --skip-generated     Skip generated datasets
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-size-mb) TARGET_MB="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --formats) shift; FORMATS=(); while [[ $# -gt 0 && "$1" != --* ]]; do FORMATS+=("$1"); shift; done;;
    --outdir) OUTDIR="$2"; shift 2;;
    --keep) KEEP=1; shift;;
    --skip-known) SKIP_KNOWN=1; shift;;
    --skip-generated) SKIP_GENERATED=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done
[[ -n "$ONLY" ]] && FORMATS=("$ONLY")

# ------------------------------------------------------------
# Workspace (fixed name) + persistent data cache
# ------------------------------------------------------------
DEFAULT_OUTDIR="$REPO_ROOT/experiment/eval"   # fixed name (no timestamp)
WORKDIR="${OUTDIR:-$DEFAULT_OUTDIR}"
DATA_DIR="$REPO_ROOT/experiment/data"         # persistent cache for known datasets
LOGDIR="$WORKDIR/logs"
mkdir -p "$WORKDIR" "$DATA_DIR" "$LOGDIR"
echo "[workdir] $WORKDIR"
echo "[cache ] $DATA_DIR"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
have_cmd() { command -v "$1" >/dev/null 2>&1; }
bytes() { stat --printf="%s" "$1" 2>/dev/null || wc -c <"$1"; }
mb() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1024/1024}'; }

# High-precision monotonic-ish timestamp in nanoseconds
now_ns() {
  if ts="$(date +%s%N 2>/dev/null)"; then
    printf '%s\n' "$ts"
  else
    python3 - <<'PY'
import time
print(int(time.time()*1_000_000_000))
PY
  fi
}

# ----- Configurable sampling interval (seconds, float ok) -----
MEM_SAMPLE_INTERVAL="${MEM_SAMPLE_INTERVAL:-0.05}"

# Return current RSS (kB) for a PID (parent only; threads included)
rss_kb_of_pid() {
  local pid="$1"
  ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo ""
}

# Robust downloader: curl → wget → Python stdlib
download() { # url outpath
  local url="$1" out="$2"
  # If file already exists and is non-empty, skip
  if [[ -s "$out" ]]; then
    echo "[cache] found $(basename "$out") ($(mb "$(bytes "$out")") MB) – skip download"
    return 0
  fi
  echo "[download] $(basename "$out") from: $url"
  if have_cmd curl; then
    # try resume if partial exists
    curl -L --fail --retry 3 -C - -o "$out" "$url"
  elif have_cmd wget; then
    wget -c -q -O "$out" "$url"
  else
    python3 - "$url" "$out" <<'PY'
import sys, urllib.request
url, out = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url, timeout=1800) as r, open(out, 'wb') as f:
    while True:
        chunk = r.read(1024*1024)
        if not chunk: break
        f.write(chunk)
PY
  fi
}

# ------------------------------------------------------------
# Portable time(1) detection + Linux RSS sampler fallback
# ------------------------------------------------------------
TIME_BIN=""; TIME_MODE=""  # {GNU_V,GNU_F,BSD_L,PS_SAMPLER}

# GNU time -v
if [[ -x /usr/bin/time ]]; then
  if /usr/bin/time -v true >/dev/null 2>"$WORKDIR/.time_test" || grep -qi "Command exited" "$WORKDIR/.time_test"; then
    TIME_BIN="/usr/bin/time -v"; TIME_MODE="GNU_V"
  fi
fi
# Homebrew gtime -v
if [[ -z "$TIME_MODE" ]] && command -v gtime >/dev/null 2>&1; then
  TIME_BIN="gtime -v"; TIME_MODE="GNU_V"
fi
# GNU time -f '%M' (kB)
if [[ -z "$TIME_MODE" ]] && command -v /usr/bin/time >/dev/null 2>&1; then
  if /usr/bin/time -f "%M" true >/dev/null 2>"$WORKDIR/.time_test2"; then
    TIME_BIN="/usr/bin/time -f %M"; TIME_MODE="GNU_F"
  fi
fi
# BSD/macOS time -l
if [[ -z "$TIME_MODE" ]] && command -v /usr/bin/time >/dev/null 2>&1; then
  if /usr/bin/time -l true >/dev/null 2>"$WORKDIR/.time_test3"; then
    TIME_BIN="/usr/bin/time -l"; TIME_MODE="BSD_L"
  fi
fi
# Linux fallback: sample RSS via ps every MEM_SAMPLE_INTERVAL
if [[ -z "$TIME_MODE" && "$(uname -s)" == "Linux" ]]; then
  TIME_MODE="PS_SAMPLER"
fi

rm -f "$WORKDIR/.time_test" "$WORKDIR/.time_test2" "$WORKDIR/.time_test3"
echo "[measure] using: ${TIME_BIN:-ps sampler} (${TIME_MODE})"

# ------------------------------------------------------------
# OFF → CSV (no empties) with dynamic column selection (cached)
# ------------------------------------------------------------
fetch_off_as_csv_noempties() { # url outpath min_mb max_mb
  local url="$1" out="$2" min_mb="$3" max_mb="$4"

  # If final output exists and non-empty, skip entire step
  if [[ -s "$out" ]]; then
    echo "[cache] found $(basename "$out") ($(mb "$(bytes "$out")") MB) – skip OFF conversion"
    return 0
  fi

  local tmp_tsv="$DATA_DIR/off.tmp.tsv"
  echo "[download] OFF raw -> $(basename "$tmp_tsv")"
  download "$url" "$tmp_tsv"

  echo "[convert] OFF (TSV -> CSV; quote-free; sanitize commas) -> $(basename "$out") [${min_mb}-${max_mb} MB]"
  python3 - "$tmp_tsv" "$out" "$min_mb" "$max_mb" <<'PY'
import sys, csv, io, os
tsv_path, out_csv, min_mb, max_mb = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
MIN = min_mb * 1024 * 1024
MAX = max_mb * 1024 * 1024
SAMPLE_ROWS = 200000
K_COLS       = 12
CHUNK_TARGET = 64 * 1024 * 1024
def sanitize(cell: str) -> str:
    return cell.replace('\r',' ').replace('\n',' ').replace(',', ';').strip()
def pass1_select_columns():
    with open(tsv_path, newline='', encoding='utf-8', errors='ignore') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r, None)
        if not header:
            return [], []
        n = len(header)
        nonempty = [0]*n
        seen = [0]*n
        for i, row in enumerate(r):
            if i >= SAMPLE_ROWS: break
            if len(row) != n: continue
            for j, cell in enumerate(row):
                seen[j] += 1
                if cell.strip() != "": nonempty[j] += 1
        ratios = [(nonempty[j]/seen[j] if seen[j] else 0.0, j) for j in range(n)]
        ratios.sort(key=lambda x: (-x[0], x[1]))
        for thr in (0.999, 0.995, 0.990, 0.975, 0.950, 0.900, 0.800, 0.700, 0.0):
            cand = [j for (ratio,j) in ratios if ratio >= thr]
            if len(cand) >= K_COLS:
                sel = cand[:K_COLS]
                break
        else:
            sel = [j for (_,j) in ratios[:K_COLS]]
        sel.sort()
        return header, sel
header, sel = pass1_select_columns()
if not header or not sel:
    open(out_csv, 'w', encoding='utf-8').close()
    raise SystemExit(0)
written = 0
chunk_buf = io.StringIO()
chunk_written = 0
with open(tsv_path, newline='', encoding='utf-8', errors='ignore') as f, \
     open(out_csv, 'w', newline='', encoding='utf-8') as g:
    r = csv.reader(f, delimiter='\t')
    _hdr = next(r, None)
    out_header = [sanitize(header[j]) for j in sel]
    line = ','.join(out_header) + '\n'
    g.write(line); written += len(line.encode('utf-8'))
    for row in r:
        if len(row) < len(header):
            continue
        out_row = [sanitize(row[j]) for j in sel]
        if any(c == "" for c in out_row):
            continue
        s = ','.join(out_row) + '\n'
        b = s.encode('utf-8')
        if written + len(b) > MAX:
            break
        g.write(s); written += len(b)
        if chunk_written < CHUNK_TARGET:
            chunk_buf.write(s); chunk_written += len(b)
if written < MIN and chunk_written > 0:
    with open(out_csv, 'a', newline='', encoding='utf-8') as g:
        chunk = chunk_buf.getvalue(); cb = chunk.encode('utf-8')
        while written + len(cb) <= MAX and written < MIN:
            g.write(chunk); written += len(cb)
        if written < MIN:
            for line in chunk.splitlines(True):
                lb = line.encode('utf-8')
                if written + len(lb) > MAX: break
                g.write(line); written += len(lb)
with open(out_csv, 'ab+') as g:
    g.seek(0,2); sz=g.tell()
    if sz>0:
        g.seek(max(sz-1,0)); last=g.read(1)
        if last!=b'\n': g.write(b'\n')
print(f"[csv-known] wrote {(written/1024/1024):.1f} MB to {out_csv}")
PY
}

# ------------------------------------------------------------
# Make a safe --args with quoted filePath and optional extras
# ------------------------------------------------------------
make_args() { # filepath extras
  local p="$1"; local extras="${2:-}"
  if [[ -n "$extras" ]]; then
    echo "--args 'filePath=\"${p}\"; ${extras}'"
  else
    echo "--args 'filePath=\"${p}\"'"
  fi
}

# ------------------------------------------------------------
# Execute a Daphne command; record timing + memory + status
# Capture full program logs into $LOGDIR and print them if the run FAILs
# ------------------------------------------------------------
run_eval() { # format which filepath label full_command
  local fmt="$1" which="$2" path="$3" label="$4" fullcmd="$5"
  local start_ns end_ns elapsed_ns elapsed_s rc mem_mb="-"

  # log files
  local safe_label="$(echo "${label}_${fmt}_${which}" | tr -c 'A-Za-z0-9._-' '_')"
  local lfile="$LOGDIR/${safe_label}.log"        # program stdout+stderr
  local tfile="$LOGDIR/${safe_label}.time"       # time(1) stderr / sampler notes

  start_ns="$(now_ns)"
  set +e
  pushd "$REPO_ROOT" >/dev/null

  case "$TIME_MODE" in
    GNU_V|GNU_F|BSD_L)
      # Redirect program output inside the shell so time's own stderr is clean in $tfile
      # We double-nest bash -lc to preserve user's $fullcmd context.
      $TIME_BIN bash -lc "exec bash -lc \"$fullcmd\" >\"$lfile\" 2>&1" 2> "$tfile"
      rc=$?
      ;;
    PS_SAMPLER)
      # Run command with logs captured; sample RSS while it runs
      bash -lc "$fullcmd" >"$lfile" 2>&1 &
      local pid=$!
      local max_kb=0 cur
      while kill -0 "$pid" 2>/dev/null; do
        cur="$(rss_kb_of_pid "$pid")"
        if [[ -n "$cur" ]]; then
          (( cur > max_kb )) && max_kb="$cur"
        fi
        sleep "$MEM_SAMPLE_INTERVAL"
      done
      wait "$pid"; rc=$?
      cur="$(rss_kb_of_pid "$pid")"
      if [[ -n "$cur" && "$cur" -gt "$max_kb" ]]; then max_kb="$cur"; fi
      if [[ "$max_kb" -gt 0 ]]; then
        mem_mb="$(awk -v k="$max_kb" 'BEGIN{printf("%.1f", k/1024)}')"
      fi
      # write a small sampler note
      printf 'ps_sampler_max_kb=%s\n' "${max_kb:-0}" > "$tfile"
      ;;
    *)
      bash -lc "$fullcmd" >"$lfile" 2>&1
      rc=$?
      ;;
  esac

  popd >/dev/null
  set -e
  end_ns="$(now_ns)"

  # elapsed seconds with 5 decimals
  elapsed_ns=$(( end_ns - start_ns ))
  elapsed_s="$(awk -v ns="$elapsed_ns" 'BEGIN{printf("%.5f", ns/1e9)}')"

  # Parse memory for time(1) modes
  if [[ "$TIME_MODE" == "GNU_V" ]]; then
    mem_mb="$(awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/,"",$2); printf("%.1f", $2/1024)}' "$tfile" || true)"
  elif [[ "$TIME_MODE" == "GNU_F" ]]; then
    kb="$(tr -dc '0-9\n' < "$tfile" | tail -n1)"; [[ -n "$kb" ]] && mem_mb="$(awk -v k="$kb" 'BEGIN{printf("%.1f", k/1024)}')"
  elif [[ "$TIME_MODE" == "BSD_L" ]]; then
    b="$(awk -F: 'BEGIN{IGNORECASE=1} /maximum resident set size/ {gsub(/[^0-9]/,"",$2); print $2}' "$tfile")"
    if [[ -n "$b" ]]; then
      if (( b > 1024*1024 )); then mem_mb="$(awk -v x="$b" 'BEGIN{printf("%.1f", x/1024/1024)}')"
      else mem_mb="$(awk -v k="$b" 'BEGIN{printf("%.1f", k/1024)}')"
      fi
    fi
  fi
  [[ -z "${mem_mb:-}" ]] && mem_mb="-"

  local status="FAIL"; [[ $rc -eq 0 ]] && status="OK"

  # If FAILED, print the logs to the terminal (last 200 lines to keep noise down)
  if [[ "$status" == "FAIL" ]]; then
    echo
    echo "----- RUN FAILED: ${label} (${fmt}/${which}) -----"
    echo "Command   : $fullcmd"
    echo "Log file  : $lfile"
    echo "Time info : $tfile"
    echo "----------- BEGIN LAST 200 LINES OF LOG -----------"
    if [[ -s "$lfile" ]]; then
      tail -n 200 "$lfile"
    else
      echo "(no program output captured)"
    fi
    echo "---------------------- END LOG ---------------------"
    echo
  fi

  echo -e "$fmt\t$which\t$label\t$elapsed_s\t$mem_mb\t$status\t$rc\t$path" >> "$WORKDIR/summary.tsv"
}

# ------------------------------------------------------------
# Known dataset sources (cached)
# ------------------------------------------------------------
OFF_URL="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv"

NYC_TLC_MONTHS=(
  "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"
  # add more months/urls if you want larger known parquet total
)

fetch_tlc_parquet_bundle() { # outdir target_mb
  local outdir="$1" target="$2"; mkdir -p "$outdir"
  local accum=0
  for url in "${NYC_TLC_MONTHS[@]}"; do
    local fn="$outdir/$(basename "$url")"
    if [[ -s "$fn" ]]; then
      local sz; sz=$(bytes "$fn")
      echo "[cache] found $(basename "$fn") ($(mb "$sz") MB) – skip download"
    else
      echo "[download] $url -> $(basename "$fn")"
      download "$url" "$fn"
    fi
    local sz; sz=$(bytes "$fn")
    echo "           size: $(mb "$sz") MB (accum $(mb "$accum") MB)"
    "$GEN_META_BIN" "$fn"
    accum=$((accum + sz))
    if (( accum >= target*1024*1024 )); then break; fi
  done
}

# ------------------------------------------------------------
# Generate (C++ generator) + Known downloads
# ------------------------------------------------------------
SUMMARY="$WORKDIR/summary.tsv"
printf "format\twhich\tlabel\ttime_s\tmax_rss_mb\tstatus\texit_code\tfile\n" > "$SUMMARY"

declare -A GEN_PATH
declare -A KNOWN_PATH

if (( SKIP_GENERATED == 0 )); then
  echo "[gen] calling gen_data (C++/Arrow) for CSV/Parquet..."
  "$GEN_DATA_BIN" --outdir "$WORKDIR" --target-size-mb "$TARGET_MB" --cols 8
  GEN_PATH[csv]="$WORKDIR/gen.csv"
  GEN_PATH[parquet]="$WORKDIR/gen.parquet"
fi

if (( SKIP_KNOWN == 0 )); then
  for f in "${FORMATS[@]}"; do
    case "$f" in
      csv)
        out="$DATA_DIR/known_off_clean.csv"
        fetch_off_as_csv_noempties "$OFF_URL" "$out" 1000 2000
        KNOWN_PATH[csv]="$out"
        "$GEN_META_BIN" "$out"
        ;;
      parquet)
        outdir="$DATA_DIR/known_tlc_parquet"
        fetch_tlc_parquet_bundle "$outdir" "$TARGET_MB"
        ;;
    esac
  done
fi

# ------------------------------------------------------------
# Evaluate (CSV×2, Parquet×2)
# ------------------------------------------------------------
for f in "${FORMATS[@]}"; do
  case "$f" in
    csv)
      CSV_EXTRAS_GEN=""
      CSV_EXTRAS_KNOWN=""

      if (( SKIP_GENERATED == 0 )) && [[ ${GEN_PATH[csv]+x} ]]; then
        fp="${GEN_PATH[csv]-}"
        args_normal=$(make_args "$fp" "$CSV_EXTRAS_GEN")
        for scr in "${CSV_SCRIPTS_NORMAL[@]}"; do
          cmd="$DAPHNE_BIN $args_normal \"$scr\""
          run_eval "csv" "generated" "$fp" "$(basename "$scr")" "$cmd"
        done
        for scr in "${CSV_SCRIPTS_PLUGIN[@]}"; do
          cmd="$DAPHNE_BIN $args_normal --FileIO-ext \"$CSV_EXT_JSON\" \"$scr\""
          run_eval "csv" "generated" "$fp" "$(basename "$scr")" "$cmd"
        done
      fi

      if (( SKIP_KNOWN == 0 )) && [[ ${KNOWN_PATH[csv]+x} ]]; then
        fp="${KNOWN_PATH[csv]-}"
        args_known=$(make_args "$fp" "$CSV_EXTRAS_KNOWN")
        for scr in "${CSV_SCRIPTS_NORMAL[@]}"; do
          cmd="$DAPHNE_BIN $args_known \"$scr\""
          run_eval "csv" "known" "$fp" "$(basename "$scr")" "$cmd"
        done
        for scr in "${CSV_SCRIPTS_PLUGIN[@]}"; do
          cmd="$DAPHNE_BIN $args_known --FileIO-ext \"$CSV_EXT_JSON\" \"$scr\""
          run_eval "csv" "known" "$fp" "$(basename "$scr")" "$cmd"
        done
      fi
      ;;
    parquet)
      PARQUET_EXTRAS_GEN=""
      PARQUET_EXTRAS_KNOWN=""

      if (( SKIP_GENERATED == 0 )) && [[ ${GEN_PATH[parquet]+x} ]] && [[ -f "${GEN_PATH[parquet]-}" ]]; then
        fp="${GEN_PATH[parquet]-}"
        args_normal=$(make_args "$fp" "$PARQUET_EXTRAS_GEN")
        for scr in "${PARQUET_SCRIPTS_NORMAL[@]}"; do
          cmd="$DAPHNE_BIN $args_normal --FileIO-ext \"$PARQUET_EXT_JSON\" \"$scr\""
          run_eval "parquet" "generated" "$fp" "$(basename "$scr")" "$cmd"
        done
        for scr in "${PARQUET_SCRIPTS_PLUGIN[@]}"; do
          cmd="$DAPHNE_BIN $args_normal --FileIO-ext \"$PARQUET_EXT_JSON\" \"$scr\""
          run_eval "parquet" "generated" "$fp" "$(basename "$scr")" "$cmd"
        done
      fi

      if (( SKIP_KNOWN == 0 )) && [[ -d "$DATA_DIR/known_tlc_parquet" ]]; then
        shopt -s nullglob
        for fp in "$DATA_DIR"/known_tlc_parquet/*.parquet; do
          args_known=$(make_args "$fp" "$PARQUET_EXTRAS_KNOWN")
          for scr in "${PARQUET_SCRIPTS_NORMAL[@]}"; do
            cmd="$DAPHNE_BIN $args_known --FileIO-ext \"$PARQUET_EXT_JSON\" \"$scr\""
            run_eval "parquet" "known" "$fp" "$(basename "$scr")" "$cmd"
          done
          for scr in "${PARQUET_SCRIPTS_PLUGIN[@]}"; do
            cmd="$DAPHNE_BIN $args_known --FileIO-ext \"$PARQUET_EXT_JSON\" \"$scr\""
            run_eval "parquet" "known" "$fp" "$(basename "$scr")" "$cmd"
          done
        done
        shopt -u nullglob
      fi
      ;;
  esac
done

# ------------------------------------------------------------
# FINAL RESULTS
# ------------------------------------------------------------
echo
echo "=== RESULTS (time & memory) ==="
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$WORKDIR/summary.tsv"
else
  cat "$WORKDIR/summary.tsv"
fi

echo
echo "TSV saved at: $WORKDIR/summary.tsv"
echo "Logs in     : $LOGDIR"

# ------------------------------------------------------------
# Cleanup or keep
# ------------------------------------------------------------
if (( KEEP )); then
  echo "[keep] Files preserved at $WORKDIR"
else
  # Keep logs even on cleanup
  find "$WORKDIR" -mindepth 1 -maxdepth 1 ! -name logs -exec rm -rf {} +
  echo "[cleanup] Removed all work files except logs at $LOGDIR"
  echo "[note ] Cached data kept at $DATA_DIR"
fi
