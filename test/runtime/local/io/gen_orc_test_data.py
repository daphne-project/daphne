#!/usr/bin/env python3
"""
gen_orc_test_data.py — generate the ORC fixtures consumed by ReadOrcTest.cpp.

Run once whenever fixtures need to be (re)generated:

    source daphne-venv/bin/activate
    pip install pyarrow
    cd test/runtime/local/io
    python3 gen_orc_test_data.py

The script is deterministic. All produced .orc files and .meta sidecars are
committed alongside the script so reviewers don't need pyarrow to run the
tests.

ORC fixtures here intentionally use TINY tables (2-3 rows) for fast tests.
Real ORC files in production are megabytes-to-gigabytes. The reader is
agnostic to size — Apache ORC's stripe iteration handles both transparently.
"""

import json
import os

import pyarrow as pa
import pyarrow.orc as orc

HERE = os.path.dirname(os.path.abspath(__file__))


def write_orc(table, basename):
    path = os.path.join(HERE, basename)
    orc.write_table(table, path)
    print(f"  wrote {basename}")


def write_meta(basename, payload):
    path = os.path.join(HERE, basename + ".meta")
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
    print(f"  wrote {basename}.meta")


# ---------------------------------------------------------------------------
# 1. DenseMatrix<double> 2x4
# ---------------------------------------------------------------------------
write_orc(
    pa.table(
        {
            "c0": pa.array([-0.1, 3.14], type=pa.float64()),
            "c1": pa.array([-0.2, 5.41], type=pa.float64()),
            "c2": pa.array([0.1, 6.22216], type=pa.float64()),
            "c3": pa.array([0.2, 5.0], type=pa.float64()),
        }
    ),
    "ReadOrc_DenseDouble.orc",
)
write_meta(
    "ReadOrc_DenseDouble.orc",
    {"numRows": 2, "numCols": 4, "valueType": "f64"},
)


# ---------------------------------------------------------------------------
# 2. DenseMatrix<int64_t> 2x4
# ---------------------------------------------------------------------------
write_orc(
    pa.table(
        {
            "c0": pa.array([1, 5], type=pa.int64()),
            "c1": pa.array([2, 6], type=pa.int64()),
            "c2": pa.array([3, 7], type=pa.int64()),
            "c3": pa.array([4, 8], type=pa.int64()),
        }
    ),
    "ReadOrc_DenseInt64.orc",
)
write_meta(
    "ReadOrc_DenseInt64.orc",
    {"numRows": 2, "numCols": 4, "valueType": "si64"},
)


# ---------------------------------------------------------------------------
# 3. Frame mixed types 3x3
# ---------------------------------------------------------------------------
write_orc(
    pa.table(
        {
            "a": pa.array([1.1, 2.2, 3.3], type=pa.float64()),
            "b": pa.array([10, 20, 30], type=pa.int64()),
            "c": pa.array([0.5, 0.6, 0.7], type=pa.float64()),
        }
    ),
    "ReadOrc_Frame.orc",
)
write_meta(
    "ReadOrc_Frame.orc",
    {
        "numRows": 3,
        "numCols": 3,
        "schema": [
            {"label": "a", "valueType": "f64"},
            {"label": "b", "valueType": "si64"},
            {"label": "c", "valueType": "f64"},
        ],
    },
)


# ---------------------------------------------------------------------------
# 4. Negative fixture: string column (the meta declares a string type
#    so that the string-rejection check fires in readOrc()).
# ---------------------------------------------------------------------------
write_orc(
    pa.table({"s": pa.array(["alpha", "beta"], type=pa.string())}),
    "ReadOrc_StringCol.orc",
)
write_meta(
    "ReadOrc_StringCol.orc",
    {"numRows": 2, "numCols": 1, "valueType": "str"},
)


# ---------------------------------------------------------------------------
# 5. Negative fixture: null values (one cell is null).
# ---------------------------------------------------------------------------
write_orc(
    pa.table({"x": pa.array([1.0, None], type=pa.float64())}),
    "ReadOrc_HasNulls.orc",
)
write_meta(
    "ReadOrc_HasNulls.orc",
    {"numRows": 2, "numCols": 1, "valueType": "f64"},
)

print("\nAll fixtures generated.")
