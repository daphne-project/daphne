# Edge cases for numpy string transfer to DAPHNE via FixedStr16 shared memory.

import numpy as np

from daphne.context.daphne_context import DaphneContext
from daphne.operator.operation_node import OperationNode
from daphne.script_building.dag import OutputType


def assert_raises(expected_exception, func):
    try:
        func()
    except expected_exception:
        return
    raise AssertionError(f"expected {expected_exception.__name__}")


def print_type(dctx, matrix):
    OperationNode(
        dctx,
        "print",
        [OperationNode(dctx, "typeOf", [matrix], output_type=OutputType.SCALAR)],
        output_type=OutputType.NONE,
    ).compute()


dctx = DaphneContext()

# Exactly 15 bytes is the maximum payload for FixedStr16.
print_type(dctx, dctx.from_numpy(np.array([b"123456789012345"], dtype="S15"), shared_memory=True))
print_type(dctx, dctx.from_numpy(np.array([b"123456789012345"], dtype="S16"), shared_memory=True))

# Unicode input is accepted if its UTF-8 representation fits in 15 bytes.
print_type(dctx, dctx.from_numpy(np.array(["é" * 7], dtype=str), shared_memory=True))

# Object arrays cover the explicit Python-side conversion path.
object_values = np.array(
    [["plain", np.str_("npstr"), b"bytes"], [bytearray(b"ba"), memoryview(b"mv"), None]],
    dtype=object,
)
print_type(dctx, dctx.from_numpy(object_values, shared_memory=True))

# 16 bytes must be rejected because FixedStr16 needs one byte for null termination.
assert_raises(
    RuntimeError,
    lambda: dctx.from_numpy(np.array([b"1234567890123456"], dtype="S16"), shared_memory=True),
)
assert_raises(
    RuntimeError,
    lambda: dctx.from_numpy(np.array(["é" * 8], dtype=str), shared_memory=True),
)
assert_raises(
    RuntimeError,
    lambda: dctx.from_numpy(np.array([b"1234567890123456"], dtype=object), shared_memory=True),
)
assert_raises(
    TypeError,
    lambda: dctx.from_numpy(np.array([object()], dtype=object), shared_memory=True),
)

print("ok")
