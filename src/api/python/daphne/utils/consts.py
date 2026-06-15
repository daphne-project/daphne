# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Modifications Copyright 2022 The DAPHNE Consortium
#
# -------------------------------------------------------------
from __future__ import annotations
import os
from pathlib import Path
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from daphne.script_building.dag import DAGNode
    from daphne.operator.nodes.matrix import Matrix
    from daphne.operator.nodes.frame import Frame
    from daphne.operator.nodes.scalar import Scalar

VALID_INPUT_TYPES = Union['DAGNode', str, int, float, bool, object]
# These are the operator symbols used in DaphneDSL (not in Python).
BINARY_OPERATIONS = ['+', '-', '/', '*', '^', '%', '<', '<=', '>', '>=', '==', '!=', '@', '&&', '||']
VALID_ARITHMETIC_TYPES = Union['DAGNode', int, float]
VALID_COMPUTED_TYPES = Union['Matrix', 'Frame', 'Scalar']

TMP_PATH = str(Path.home() / ".daphnelib")
os.makedirs(TMP_PATH, exist_ok=True)

_PROTOTYPE_PATH_ENV_VAR_NAME = "DAPHNELIB_DIR_PATH"
PROTOTYPE_PATH = os.environ.get(_PROTOTYPE_PATH_ENV_VAR_NAME)
if not PROTOTYPE_PATH:
    raise ValueError(f"Environment variable '{_PROTOTYPE_PATH_ENV_VAR_NAME}' NOT SET")
DAPHNELIB_FILENAME = "libdaphnelib.so"

# DAPHNE value type codes.
# The values need to be updated according to the value type codes in ValueTypeCode.h as this is a 1:1 copy.
SI8 = 0
SI32 = 1
SI64 = 2
UI8 = 3
UI32 = 4
UI64 = 5
F32 = 6
F64 = 7
STR = 8
FIXEDSTR16 = 9
INVALID = 10
