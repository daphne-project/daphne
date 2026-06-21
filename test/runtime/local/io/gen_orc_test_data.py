#!/usr/bin/env python3
"""
Generate the ORC test fixtures used by ReadOrcTest.cpp.

You only need to run this if you're changing or adding a fixture. The .orc
files and their .meta sidecars are committed to the repo, so running the
tests does not require pyarrow.

To regenerate:

    source daphne-venv/bin/activate
    pip install pyarrow
    cd test/runtime/local/io
    python3 gen_orc_test_data.py

The output is deterministic, so the same inputs always produce the same
bytes on disk.

Each fixture has only 2 or 3 rows. That is enough to exercise the reader
and keeps the test suite fast. The reader does not care about file size:
ORC's stripe iteration scales the same way for tiny and large files.
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


# DenseMatrix<double>, 2x4
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


# DenseMatrix<int64_t>, 2x4
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


# Frame with mixed types, 3x3
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


# String column. The meta declares str so the reader's string-rejection
# path is exercised.
write_orc(
    pa.table({"s": pa.array(["alpha", "beta"], type=pa.string())}),
    "ReadOrc_StringCol.orc",
)
write_meta(
    "ReadOrc_StringCol.orc",
    {"numRows": 2, "numCols": 1, "valueType": "str"},
)


# A column with one null cell, for the null-rejection test.
write_orc(
    pa.table({"x": pa.array([1.0, None], type=pa.float64())}),
    "ReadOrc_HasNulls.orc",
)
write_meta(
    "ReadOrc_HasNulls.orc",
    {"numRows": 2, "numCols": 1, "valueType": "f64"},
)

print("\nDone.")
