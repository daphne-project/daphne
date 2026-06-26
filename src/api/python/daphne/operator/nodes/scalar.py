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

from daphne.operator.operation_node import OperationNode
from daphne.script_building.dag import OutputType
from daphne.utils.consts import VALID_INPUT_TYPES, VALID_ARITHMETIC_TYPES

import numpy as np

from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from daphne.context.daphne_context import DaphneContext

class Scalar(OperationNode):
    __assign: bool
    __copy: bool

    def __init__(self, daphne_context: 'DaphneContext', operation: str,
                 unnamed_input_nodes: Iterable[VALID_INPUT_TYPES] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
                 output_type: OutputType = OutputType.SCALAR,
                 assign: bool = False, copy: bool = False) -> 'Scalar':
        self.__assign = assign
        self.__copy = copy
        super().__init__(daphne_context, operation, unnamed_input_nodes=unnamed_input_nodes,
                         named_input_nodes=named_input_nodes, output_type=output_type)

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        if self.__assign:
            return f'{var_name}={self.operation};'
        if self.__copy:
            return f'{var_name}={unnamed_input_vars[0]};'
        else:
            return super().code_line(var_name, unnamed_input_vars, named_input_vars)

    def compute(self) -> Union[np.ndarray]:
        return super().compute()
    
    def __add__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '+', [self, other])

    # Left hand side
    def __radd__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '+', [other, self])

    def __sub__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '-', [self, other])

    # Left hand side
    def __rsub__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '-', [other, self])

    def __mul__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar( self.daphne_context,'*', [self, other])

    def __rmul__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '*', [other, self])

    def __truediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '/', [self, other])

    def __rtruediv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '/', [other, self])

    def __floordiv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '//', [self, other])

    def __rfloordiv__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '//', [other, self])

    def __pow__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        # "**" in Python, "^" in DaphneDSL.
        return Scalar(self.daphne_context, '^', [other, self])

    def __mod__(self, other: VALID_ARITHMETIC_TYPES) -> 'Scalar':
        return Scalar(self.daphne_context, '%', [other, self])

    def __lt__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '<', [self, other])

    def __rlt__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '<', [other, self])

    def __le__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '<=', [self, other])

    def __rle__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '<=', [other, self])

    def __gt__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '>', [self, other])

    def __rgt__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '>', [other, self])

    def __ge__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '>=', [self, other])

    def __rge__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '>=', [other, self])

    def __eq__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '==', [self, other])

    def __req__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '==', [other, self])

    def __ne__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '!=', [self, other])

    def __rne__(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, '!=', [other, self])

    def __neg__(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'minus', [self])
    
    def abs(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'abs', [self])
    
    def sign(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'sign', [self])
    
    def exp(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'exp', [self])
    
    def ln(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'ln', [self])
    
    def sqrt(self) -> 'Scalar':
        return Scalar(self.daphne_context,'sqrt',[self])
    
    def round(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'round', [self])
    
    def floor(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'floor', [self])
    
    def ceil(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'ceil', [self])
    
    def sin(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'sin', [self])
    
    def cos(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'cos', [self])
    
    def tan(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'tan', [self])
    
    def sinh(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'sinh', [self])
    
    def cosh(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'cosh', [self])
    
    def tanh(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'tanh', [self])
    
    def asin(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'asin', [self])
    
    def acos(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'acos', [self])
    
    def atan(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'atan', [self])
    
    def isNan(self) -> 'Scalar':
        return Scalar(self.daphne_context, 'isNan', [self])
    
    def pow(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, 'pow', [self, other])
    
    def log(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, 'log', [self, other])
    
    def mod(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, 'mod', [self, other])
    
    def min(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, 'min', [self, other])
    
    def max(self, other) -> 'Scalar':
        return Scalar(self.daphne_context, 'max', [self, other])
    
    def print(self):
        return OperationNode(self.daphne_context,'print',[self], output_type=OutputType.NONE)