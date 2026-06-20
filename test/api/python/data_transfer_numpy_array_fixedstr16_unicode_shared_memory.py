# Data transfer from numpy unicode strings to DAPHNE via shared memory.

import numpy as np

from daphne.context.daphne_context import DaphneContext
from daphne.operator.operation_node import OperationNode
from daphne.script_building.dag import OutputType

m1 = np.array(["red", "green", "blue"], dtype=str)

dctx = DaphneContext()
m2 = dctx.from_numpy(m1, shared_memory=True)

OperationNode(
    dctx,
    "print",
    [OperationNode(dctx, "typeOf", [m2], output_type=OutputType.SCALAR)],
    output_type=OutputType.NONE,
).compute()
