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

from daphne.script_building.dag import DAGNode, OutputType
from daphne.script_building.script import DaphneDSLScript
from daphne.utils.consts import BINARY_OPERATIONS, TMP_PATH, VALID_INPUT_TYPES, F64, F32, SI64, SI32, SI8, UI64, UI32, UI8
from daphne.utils.daphnelib import DaphneLib, DaphneLibResult
from daphne.utils.helpers import create_params_string

import numpy as np
import pandas as pd
try:
    import torch as torch
except ImportError as e:
    torch = e
try:
    import tensorflow as tf
except ImportError as e:
    tf = e

import ctypes
import json
import os
import time
from typing import Dict, Iterable, Optional, Sequence, Union, TYPE_CHECKING, List

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from daphne.context.daphne_context import DaphneContext
    
class OperationNode(DAGNode):  
    _result_var:Optional[Union[float,np.ndarray]]
    _script:Optional[DaphneDSLScript]
    _output_types: Optional[Iterable[VALID_INPUT_TYPES]]
    _source_node: Optional["DAGNode"]
    _brackets: bool
    data: Optional[Union[pd.DataFrame, np.ndarray]]

    def __init__(self, daphne_context,operation:str, 
                unnamed_input_nodes: Union[str, Iterable[VALID_INPUT_TYPES]]=None,
                named_input_nodes: Dict[str, VALID_INPUT_TYPES]=None, 
                output_type:OutputType = OutputType.MATRIX, is_python_local_data: bool = False,
                brackets: bool = False, left_brackets: bool = False,
                consumer_list: List['OperationNode'] = None):
        if unnamed_input_nodes is None:
            unnamed_input_nodes = []
        if named_input_nodes is None:
            named_input_nodes = []
        if consumer_list is None:
            self.consumer_list = []
        self.daphne_context = daphne_context
        self.operation = operation
        self._unnamed_input_nodes = unnamed_input_nodes
        self._named_input_nodes = named_input_nodes
        self._result_var = None
        self._script = None
        self._source_node = None
        self._already_added = False
        self.daphnedsl_name = ""
        self._is_python_local_data = is_python_local_data
        self._brackets = brackets
        self._left_brackets = left_brackets
        self._output_type = output_type

        # add this node to the consumer lists of all the nodes it uses.
        for i in range(len(unnamed_input_nodes)):
            if isinstance(unnamed_input_nodes[i], OperationNode):
                self._unnamed_input_nodes[i].consumer_list.append(self)
              
    def update_node_in_input_list(self, new_node, current_node):
        current_index = self._unnamed_input_nodes.index(current_node)
        self._unnamed_input_nodes[current_index] = new_node

    def compute(self, type="shared memory", verbose=False, asTensorFlow=False, asPyTorch=False, shape=None, useIndexColumn=False) -> Union[np.ndarray, pd.DataFrame, 'tf.Tensor', 'torch.Tensor', float]:
        """
        Compute function for processing the Daphne Object or operation node and returning the results.
        The function builds a DaphneDSL script from the node and its context, executes it, and processes the results
        to produce a pandas DataFrame, numpy array, or TensorFlow/PyTorch tensors.

        :param type: Execution type, either "shared memory" for in-memory data transfer or "files" for file-based data transfer.
        :param verbose: If True, outputs verbose logs, including timing information for each step.
        :param asTensorFlow: If True and the result is a matrix, the output will be converted to a TensorFlow tensor.
        :param asPyTorch: If True and the result is a matrix, the output will be converted to a PyTorch tensor.
        :param shape: If provided and the result is a matrix, it defines the shape to reshape the resulting tensor (either TensorFlow or PyTorch).
        :param useIndexColumn: If True and the result is a DataFrame, uses the column named "index" as the DataFrame's index.

        :return: Depending on the parameters and the operation's output type, this function can return:
            - A pandas DataFrame for frame outputs.
            - A numpy array for matrix outputs.
            - A scalar value for scalar outputs.
            - TensorFlow or PyTorch tensors if `asTensorFlow` or `asPyTorch` is set to True respectively.
        """
        
        if self._result_var is None:
            if verbose:
                start_time = time.time()

            self._script = DaphneDSLScript(self.daphne_context)
            for definition in self.daphne_context._functions.values():
                self._script.daphnedsl_script += definition
            result = self._script.build_code(self, type)

            if verbose:
                exec_start_time = time.time()

            self._script.execute()
            self._script.clear(self)

            if verbose:
                print(f"compute(): Python-side execution time of the execute() function: {(time.time() - exec_start_time):.10f} seconds")
            
            if self._output_type == OutputType.FRAME and type=="shared memory":
                if verbose:
                    dt_start_time = time.time()

                daphneLibResult = DaphneLib.getResult()
                
                # Read the frame's address into a numpy array.
                if daphneLibResult.columns is not None:
                    # Read the column labels and dtypes from the Frame's labels and dtypes directly.
                    labels = [ctypes.cast(daphneLibResult.labels[i], ctypes.c_char_p).value.decode() for i in range(daphneLibResult.cols)]
                    
                    # Create a new type representing an array of value type codes.
                    VTArray = ctypes.c_int64 * daphneLibResult.cols
                    # Cast the pointer to this type and access its contents.
                    vtcs_array = ctypes.cast(daphneLibResult.vtcs, ctypes.POINTER(VTArray)).contents
                    # Convert the value types into numpy dtypes.
                    dtypes = [self.getNumpyType(vtc) for vtc in vtcs_array]

                    data = {label: None for label in labels}

                    # Using ctypes cast and numpy array view to create dictionary directly.
                    for idx in range(daphneLibResult.cols):
                        c_data_type = self.getType(daphneLibResult.vtcs[idx])
                        array_view = np.ctypeslib.as_array(
                            ctypes.cast(daphneLibResult.columns[idx], ctypes.POINTER(c_data_type)),
                            shape=[daphneLibResult.rows]
                        )
                        label = labels[idx]
                        data[label] = array_view

                    # Create DataFrame from dictionary.
                    df = pd.DataFrame(data, copy=False)

                    # If useIndexColumn is True, set "index" column as the DataFrame's index
                    # TODO What if there is no column named "index"?
                    if useIndexColumn and "index" in df.columns:
                        df.set_index("index", inplace=True, drop=True)
                else:
                    # TODO Raise an error.
                    # TODO When does this happen?
                    print("Error: NULL pointer access")
                    labels = []
                    dtypes = []
                    df = pd.DataFrame()
                
                result = df
                self.clear_tmp()

                if verbose:
                    print(f"compute(): time for Python side data transfer (Frame, shared memory): {(time.time() - dt_start_time):.10f} seconds")
            elif self._output_type == OutputType.FRAME and type=="files":
                df = pd.read_csv(result)
                with open(result + ".meta", "r") as f:
                    fmd = json.load(f)
                    df.columns = [x["label"] for x in fmd["schema"]]
                result = df
                self.clear_tmp()
            elif self._output_type == OutputType.MATRIX and type=="shared memory":
                daphneLibResult = DaphneLib.getResult()
                result = np.ctypeslib.as_array(
                    ctypes.cast(daphneLibResult.address, ctypes.POINTER(self.getType(daphneLibResult.vtc))),
                    shape=[daphneLibResult.rows, daphneLibResult.cols]
                )
                self.clear_tmp()
            elif self._output_type == OutputType.MATRIX and type=="files":
                # Ensure string data is handled correctly
                arr = np.genfromtxt(result, delimiter=',', dtype=None, encoding='utf-8')
                meta_file_name = result + ".meta"
                if os.path.exists(meta_file_name):
                    with open(meta_file_name, "r") as meta_file:
                        meta_data = json.load(meta_file)
                        if meta_data.get("valueType") == "str":
                            arr = arr.astype(str)
                else:
                    print(f"metadata file not found: {meta_file_name}")
                self.clear_tmp()
                return arr
            elif self._output_type == OutputType.SCALAR:
                # We transfer scalars back to Python by wrapping them into a 1x1 matrix.
                daphneLibResult = DaphneLib.getResult()
                result = np.ctypeslib.as_array(
                    ctypes.cast(daphneLibResult.address, ctypes.POINTER(self.getType(daphneLibResult.vtc))),
                    shape=[daphneLibResult.rows, daphneLibResult.cols]
                )[0, 0]
                self.clear_tmp()
            
            # TODO asTensorFlow and asPyTorch should be mutually exclusive.
            if asTensorFlow and self._output_type == OutputType.MATRIX:
                # This feature is only available if TensorFlow is available.
                if isinstance(tf, ImportError):
                    raise tf
                
                if verbose:
                    tc_start_time = time.time()

                # Convert the Matrix to a TensorFlow Tensor.
                result = tf.convert_to_tensor(result)

                # If a shape is provided, reshape the TensorFlow Tensor.
                if shape is not None:
                    result = tf.reshape(result, shape)

                if verbose:
                    print(f"compute(): time to convert to TensorFlow Tensor: {(time.time() - tc_start_time):.10f} seconds")
            elif asPyTorch and self._output_type == OutputType.MATRIX:
                # This feature is only available if PyTorch is available.
                if isinstance(torch, ImportError):
                    raise torch
                
                if verbose:
                    tc_start_time = time.time()

                # Convert the Matrix to a PyTorch Tensor.
                result = torch.from_numpy(result)

                # If a shape is provided, reshape the PyTorch Tensor.
                if shape is not None:
                    result = torch.reshape(result, shape)

                if verbose:
                    print(f"compute(): time to convert to PyTorch Tensor: {(time.time() - tc_start_time):.10f} seconds")
            
            if verbose:
                print(f"compute(): total Python-side execution time: {(time.time() - start_time):.10f} seconds")    

            if result is None:
                return
            return result

    def clear_tmp(self):
       for f in os.listdir(TMP_PATH):
          os.remove(os.path.join(TMP_PATH, f))

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str], named_input_vars: Dict[str, str])->str:
        if self._left_brackets:
            line_1 = f'{unnamed_input_vars[0]}[{",".join(unnamed_input_vars[2:])}] = {unnamed_input_vars[1]};'
            line_2 = f'{var_name} = {unnamed_input_vars[0]};'
            return line_1 + "\n" + line_2
        if self._brackets:
            return f'{var_name}={unnamed_input_vars[0]}[{",".join(unnamed_input_vars[1:])}];'
        if self.operation in BINARY_OPERATIONS:
            assert len(
                named_input_vars) == 0, 'named parameters can not be used with binary operations'
            assert len(unnamed_input_vars)==2, 'Binary operations need exactly two input variables'
            return f'{var_name}={unnamed_input_vars[0]} {self.operation} {unnamed_input_vars[1]};'
        if self.operation == "ifElse":
            assert len(
                named_input_vars) == 0, 'named parameters can not be used with ifElse operation'
            assert len(unnamed_input_vars)==3, 'ifElse operation needs exactly three input variables'
            return f'{var_name}={unnamed_input_vars[0]} ? {unnamed_input_vars[1]} : {unnamed_input_vars[2]};'
        inputs_comma_sep = create_params_string(unnamed_input_vars, named_input_vars).format(file_name=var_name)
        if self.output_type == OutputType.NONE:
            return f'{self.operation}({inputs_comma_sep});'
        else:
            return f'{var_name}={self.operation}({inputs_comma_sep});'

    def getType(self, vtc):
        if vtc == F64:
            return ctypes.c_double
        elif vtc == F32:
            return ctypes.c_float
        elif vtc == SI64:
            return ctypes.c_int64
        elif vtc == SI32:
            return ctypes.c_int32
        elif vtc == SI8:
            return ctypes.c_int8
        elif vtc == UI64:
            return ctypes.c_uint64
        elif vtc == UI32:
            return ctypes.c_uint32
        elif vtc == UI8:
            return ctypes.c_uint8
        else:
            raise RuntimeError(f"unknown value type code: {vtc}")
    
    def getNumpyType(self, vtc):
        """Convert DAPHNE value type to numpy dtype."""
        if vtc == F64:
            return np.float64
        elif vtc == F32:
            return np.float32
        elif vtc == SI64:
            return np.int64
        elif vtc == SI32:
            return np.int32
        elif vtc == SI8:
            return np.int8
        elif vtc == UI64:
            return np.uint64
        elif vtc == UI32:
            return np.uint32
        elif vtc == UI8:
            return np.uint8
        else:
            raise RuntimeError(f"unknown value type code: {vtc}")