# Copyright 2024 The DAPHNE Consortium
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

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

# Can be installed by `pip install antlr4-python3-runtime`.
from antlr4 import *

# ANTLR-generated files.
from DmlLexer import DmlLexer
from DmlParser import DmlParser
from DmlVisitor import DmlVisitor

import argparse
import os

# -------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------

INDENT = "  "

# TODO Get parameter types of imported functions dynamically.
# Global variable storing the data types of the function parameters (only needed for imported functions).
param_types = {
    "lm": [["X", "matrix<f64>", None], ["y", "matrix<f64>", None], ["icpt", "si64", "0"], ["reg", "f64", "0.0"], ["tol", "f64", "0.0"], ["maxi", "si64", "0"], ["verbose", "bool", "true"]],
    "lmCG": [["X", "matrix<f64>", None], ["y", "matrix<f64>", None], ["icpt", "si64", "0"], ["reg", "f64", "0.0"], ["tol", "f64", "0.0"], ["maxi", "si64", "0"], ["verbose", "bool", "true"]],
    "lmDS": [["X", "matrix<f64>", None], ["y", "matrix<f64>", None], ["icpt", "si64", "0"], ["maxi", "si64", "0"], ["verbose", "bool", "true"]],
    "dist": [["X", "matrix<f64>", None]],
    "components": [["G", "matrix<f64>", None], ["maxi", "si64", "0"], ["verbose", "bool", "true"]]
}
class Context:
    def __init__(self):
        self.then_assigns = set()     # Variables assigned inside then-block
        self.else_assigns = set()     # Variables assigned inside else-block
        self.already_assigned = set() # Variables that have already been assigned
        self.in_thenBody = False      # Flag needed for special handling of variables that are initialized inside if-then-else statements
        self.in_elseBody = False      # Flag needed for special handling of variables that are initialized inside if-then-else statements
        # TODO Is data_types valid only for the current function? If so, is it reset before each function?
        self.data_types = dict()      # Each variables current data type
        self.current_function = ""    # The current function (a script can contain multiple functions)
        self.function_names = []      # Names of all the functions defined inside this script
        self.imports = ""             # String containing the import statements for built-in DML scripts that are not explicitly imported
        self.call_graph = {}          # Graph of function calls: needed to correctly reorder functions inside the file
        
        # TODO Get return types of imported functions dynamically.
        # Stores the return types of each imported function (internally defined functions are added during the translation).
        self.return_types = {
            "lm": ["matrix<f64>"],
            "lmCG": ["matrix<f64>"],
            "lmDS": ["matrix<f64>"],
            "dist": ["matrix<f64>"],
            "components": ["matrix<f64>"]
        }

# -------------------------------------------------------------------------
# Visitor class
# -------------------------------------------------------------------------

class Translator(DmlVisitor):
    def __init__(self):
        self.context = Context()

        # Mapping functions from DML to their respective alternatives in DaphneDSL.
        self.FUNCTIONS_MAP = {
            "matrix": self.matrix_function,
            "ncol": self.ncol_function,
            "nrow": self.nrow_function,
            "t": self.t_function,
            "sum": self.sum_function,
            "ifelse": self.ifelse_function,
            "print": self.print_function,
            "stop": self.stop_function,
            "colMaxs": self.colMaxs_function,
            "rowMaxs": self.rowMaxs_function,
            "colMins": self.colMins_function,
            "rowMins": self.rowMins_function,
            "colSums": self.colSums_function,
            "rowSums": self.rowSums_function,
            "toString": self.toString_function,
            "replace" : self.replace_function,
            "min": self.min_function,
            "max": self.max_function,
            "seq": self.seq_function,
            "diag": self.diag_function,
            "outer": self.outer_function,
            "sqrt": self.sqrt_function,
            "removeEmpty": self.removeEmpty_function,
            "mean": self.mean_function,
            "log": self.log_function,
            "as.integer": self.asInteger_function,
            "as.scalar": self.asScalar_function,
            "as.float": self.asScalar_function,
            "as.matrix": self.asMatrix_function,
            "cbind": self.cbind_function,
            "rbind": self.rbind_function,
            "solve": self.solve_function,
            "abs": self.abs_function,
            "exp": self.exp_function,
            "colIndexMin": self.colIndexMin_function,
            "rowIndexMin": self.rowIndexMin_function,
            "table": self.table_function,
            "time": self.time_function
        }

    # -------------------------------------------------------------------------
    # Mappings of native DML functions to DaphneDSL built-in functions
    # -------------------------------------------------------------------------
 
    def matrix_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None, "rows", "cols"])
        
        # TODO Why?
        if args[1] == "0":
            args[1] = "1"
        if args[2] == "0":
            args[2] = "1"

        dtype = self.inferType(args[0], exprs[0])
        if dtype in {"si64", "f64", "bool", "str"}:
            dtype = "f64" if dtype == "si64" else dtype
            function_call = f"fill(as.{dtype}({args[0]}), {args[1]}, {args[2]})"
            dtype = f"matrix<{dtype}>"
        else:
            function_call = f"reshape({args[0]}, {args[1]}, {args[2]})"
            
        return function_call, dtype
        
    def ncol_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"as.si64(ncol({args[0]}))"
        dtype = "si64"
        
        return function_call, dtype

    def nrow_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"as.si64(nrow({args[0]}))"
        dtype = "si64"
        
        return function_call, dtype

    def t_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"t({args[0]})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype
        
    def sum_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"sum({args[0]})"
        dtype = self.inferType(args[0], exprs[0])
        dtype = self.getValueType(dtype)
        
        return function_call, dtype

    def ifelse_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None, None, None])
        function_call = f"{args[0]} ? {args[1]} : {args[2]}"
        dtype = self.inferType(args[1], exprs[1])
        
        return function_call, dtype

    # Handles toString() as well by splitting combined print() into seperate print() statements.
    def print_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"{args[0]}"
        
        # TODO Don't create individual prints, or if so, make them not end with newline.
        split_str = function_call.split("+")
        transformed_strs = []
        for part in split_str:
            stripped_part = part.strip()
            if stripped_part.startswith("toString(") and stripped_part.endswith(")"):
                var_name = stripped_part[len("toString("):-1]
                transformed_strs.append(f"print({var_name});")
            else:
                transformed_strs.append(f"print({stripped_part});")
        all_prints = " ".join(transformed_strs)
        
        return all_prints.rstrip(";"), None

    # Prints message and then calls "return" with default values.
    def stop_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        return_string = ", ".join(self.default_value(arg) for arg in self.context.return_types[self.context.current_function])
        function_call = f"print({args[0]});\nreturn {return_string}"
        
        return function_call, None

    def colMaxs_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"aggMax({args[0]}, 1)"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def rowMaxs_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"aggMax({args[0]}, 0)"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def colMins_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"aggMin({args[0]}, 1)"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def rowMins_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"aggMin({args[0]}, 0)"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def colSums_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"sum({args[0]}, 1)"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def rowSums_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"sum({args[0]}, 0)"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype
                
    # TODO DaphneDSL has no toString() function, don't create it here.
    # Don't delay treatment to print(), can cause problems.
    # toString() is handled in print_function.
    def toString_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"toString({args[0]})"
        dtype = "str"
        
        return function_call, dtype

    def replace_function(self, arguments):
        args, exprs = self.reorder_args(arguments, ["target", "pattern", "replacement"])
        args[1] = "nan" if args[1] == "0 / 0" else args[1]
        function_call = f"replace({args[0]}, {args[1]}, {args[2]})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype
       
    def min_function(self, arguments):
        function_call = ""
        dtype = ""
        if len(arguments) == 1:
            args, exprs = self.reorder_args(arguments, [None])
            function_call = f"aggMin({args[0]})"
            
            dtype = self.inferType(args[0], exprs[0])
            dtype = self.getValueType(dtype)
        elif len(arguments) == 2:
            args, exprs = self.reorder_args(arguments, [None, None])
            function_call = f"min({args[0]}, {args[1]})"
            
            dtype_arg1 = self.inferType(args[0], exprs[0])
            dtype_arg2 = self.inferType(args[1], exprs[1])
            
            if self.isMatrix(dtype_arg1):
                dtype = self.getValueType(dtype_arg1)
            elif self.isMatrix(dtype_arg2):
                dtype = self.getValueType(dtype_arg2)
            else:
                dtype = dtype_arg1
                
        return function_call, dtype

    def max_function(self, arguments):
        function_call = ""
        dtype = ""
        if len(arguments) == 1:
            args, exprs = self.reorder_args(arguments, [None])
            function_call = f"aggMax({args[0]})"
            
            dtype = self.inferType(args[0], exprs[0])
            dtype = self.getValueType(dtype)
        elif len(arguments) == 2:
            args, exprs = self.reorder_args(arguments, [None, None])
            function_call = f"max({args[0]}, {args[1]})"
            
            dtype_arg1 = self.inferType(args[0], exprs[0])
            dtype_arg2 = self.inferType(args[1], exprs[1])
            
            if self.isMatrix(dtype_arg1):
                dtype = self.getValueType(dtype_arg1)
            elif self.isMatrix(dtype_arg2):
                dtype = self.getValueType(dtype_arg2)
            else:
                dtype = dtype_arg1
                
        return function_call, dtype
        
    def seq_function(self, arguments):
        function_call = ""
        dtype = ""
        if len(arguments) == 2:
            args, exprs = self.reorder_args(arguments, [None, None])
            function_call = f"seq(as.f64({args[0]}), {args[1]}, {args[0]} <= {args[1]} ? 1 : -1)"
            dtype = "matrix<f64>"
        elif len(arguments) == 3:
            args, exprs = self.reorder_args(arguments, [None, None, None])
            function_call = f"seq(as.f64({args[0]}), {args[1]}, {args[2]})"
            dtype = "matrix<f64>"
            
        return function_call, dtype

    # TODO Implement diagVector.
    def diag_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None, None])
        function_call = f"diagMatrix({args[0]})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def outer_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None, None, None])
        
        dml_to_daph = {
            "*": "outerMul",
            "-": "outerSub",
            "+": "outerAdd",
            "/": "outerDiv",
            "^": "outerPow",
            "%": "outerMod",
            "min": "outerMin",
            "max": "outerMax"
        }
        op = args[2].replace('"', '')
        function_name = dml_to_daph.get(op)
        
        if function_name is None:
            raise ValueError(f"unknown operation {op} for outer()")
        
        # outerAdd(diagMatrix(X), t(diagMatrix(X))) would fail due to wrong shapes, therefore reshape and transformation needed.
        function_call = ""
        if args[0].startswith("diagMatrix(") or args[0].startswith("t(diagMatrix("):
            arg1 = f"reshape({args[0]}, ncol({args[0]})*nrow({args[0]}), 1)"
            arg1 += f"[[sum({arg1} != 0, 0) > 0, ]]"
            arg2 = f"reshape({args[1]}, 1, ncol({args[1]})*nrow({args[1]}))"
            arg2 += f"[[, t(sum({arg2} != 0, 1)) > 0]]"
            function_call = f"{function_name}({arg1}, {arg2})"
        else:
            function_call = f"{function_name}({args[0]}, {args[1]})"

        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def sqrt_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"sqrt({args[0]})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def removeEmpty_function(self, arguments):
        args, exprs = self.reorder_args(arguments, ["target", "margin", "select"])
        
        function_call = ""
        args[1] = args[1].replace('"', '')
        if args[1] == "cols":
            function_call = f"{args[0]}[[, {args[2]}]]"
        elif args[1] == "rows":
            function_call = f"{args[0]}[[{args[2]}, ]]"
            
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    # TODO Handle the case of two arguments.
    def mean_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"mean({args[0]})"
        dtype = "f64"
        
        return function_call, dtype

    def log_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"log({args[0]}, 2)"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def asInteger_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"as.si64({args[0]})"
        dtype = "si64"
        
        return function_call, dtype

    def asScalar_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"as.scalar({args[0]})"
        
        dtype = self.inferType(args[0], exprs[0])
        if self.isMatrix(dtype):
            dtype = self.getValueType(dtype)
            
        if dtype == "si64":
            function_call = f"as.scalar(as.si64({args[0]}))"
        elif dtype == "f64":
            function_call = f"as.scalar(as.f64({args[0]}))"
        elif dtype == "bool":
            function_call = f"as.scalar(as.bool({args[0]}))"
        elif dtype == "str":
            function_call = f"as.scalar(as.str({args[0]}))"
        
        return function_call, dtype
        
    def asFloat_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"as.f64({args[0]})"
        dtype = "f64"
        
        return function_call, dtype

    def asMatrix_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"as.matrix({args[0]})"
        dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
        
        return function_call, dtype

    def cbind_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None, None])
        function_call = f"cbind({args[0]}, {args[1]})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def rbind_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None, None])
        function_call = f"rbind({args[0]}, {args[1]})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def solve_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None, None])
        function_call = f"solve({args[0]}, {args[1]})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def abs_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"abs({args[0]})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def exp_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"exp({args[0]})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def colIndexMin_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"idxMin({args[0]}, 1)"
        dtype = "si64"
        
        return function_call, dtype

    def rowIndexMin_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"idxMin({args[0]}, 0)"
        dtype = "si64"
        
        return function_call, dtype

    # TODO Handle multiple named arguments.
    def table_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None]*len(arguments))
        args_str = ", ".join(args)
        function_call = f"ctable({args_str})"
        dtype = self.inferType(args[0], exprs[0])
        
        return function_call, dtype

    def time_function(self, arguments_order, arguments_expr, arguments):
        function_call = f"now()"
        dtype = "f64"
        
        return function_call, dtype

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------
    
    # Function for indenting the code.
    def indent(self, code, level):
        # TODO Repeated splitting and joining could be inefficient.
        indentation = INDENT * level
        lines = code.split("\n")
        indented_lines = [indentation + line if i > 0 else line for i, line in enumerate(lines)]
        
        return "\n".join(indented_lines)
      
    # Checking if a data type is a matrix.
    def isMatrix(self, dtype):
        if "<" in dtype and ">" in dtype:
            return True
        else:
            return False
        
    # Return values for initialization of variables based on their data/value type.
    def default_value(self, dtype):
        if dtype == "si64":
            return "0"
        elif dtype == "f64":
            return "0.0"
        elif dtype == "bool":
            return "false"
        elif dtype == "str":
            return '""'
        elif self.isMatrix(dtype):
            value_type = self.getValueType(dtype)
            if value_type == "si64":
                return "[0]"
            elif value_type == "f64":
                return "[0.0]"
            elif value_type == "bool":
                return "[false]"
            elif value_type == "str":
                return '[""]'
        else:
            raise ValueError(f"unknown data type: {dtype}")
        
    # Retrieves data type of expression with two operands (e.g., data type of a + b).
    def getDataType(self, ctx):
        dtype_left = self.inferType(self.visitExpression(ctx.expression(0)), ctx.expression(0))
        dtype_right = self.inferType(self.visitExpression(ctx.expression(1)), ctx.expression(1))

        if self.isMatrix(dtype_left) and self.isMatrix(dtype_right):
            vtype_left = self.getValueType(dtype_left)
            vtype_right = self.getValueType(dtype_right)
            
            # TODO What about other value types (maybe ok when translating from DML)?
            if vtype_left == "f64" or vtype_right == "f64":
                return "matrix<f64>"
            else:
                return vtype_left
        elif not self.isMatrix(dtype_left) and not self.isMatrix(dtype_right):
            # TODO What about other value types (maybe ok when translating from DML)?
            if dtype_left == "f64" or dtype_right == "f64":
                return "f64"
            else:
                return dtype_left
        elif self.isMatrix(dtype_left):
            return dtype_left
        else:
            return dtype_right
        
    # Determines the data/value type of a given variable.
    def inferType(self, source, ctx):
        source_type = self.context.data_types.get(source)
        if source_type is not None:
            return source_type
        else:
            if isinstance(ctx, DmlParser.ConstIntIdExpressionContext):
                return "si64"
            elif isinstance(ctx, DmlParser.ConstDoubleIdExpressionContext):
                return "f64"
            elif isinstance(ctx, DmlParser.ConstStringIdExpressionContext):
                return "str"
            elif isinstance(ctx, DmlParser.ConstTrueExpressionContext) or isinstance(ctx, DmlParser.ConstFalseExpressionContext):
                return "bool"
            elif isinstance(ctx, DmlParser.BuiltinFunctionExpressionContext):
                return self.visitBuiltinFunctionExpression(ctx, True)
            elif isinstance(ctx, DmlParser.AddSubExpressionContext):
                return self.getDataType(ctx)
            elif isinstance(ctx, DmlParser.MultDivExpressionContext):
                return self.getDataType(ctx)
            elif isinstance(ctx, DmlParser.AtomicExpressionContext):
                return self.inferType(self.visitExpression(ctx.expression()), ctx.expression())
            elif isinstance(ctx, DmlParser.MatrixMulExpressionContext):
                return self.inferType(self.visitExpression(ctx.expression(0)), ctx.expression(0))
            elif isinstance(ctx, DmlParser.RelationalExpressionContext):
                return self.visitRelationalExpression(ctx, True)
            elif isinstance(ctx, DmlParser.UnaryExpressionContext):
                return self.inferType(self.visitExpression(ctx.expression()), ctx.expression())
            elif isinstance(ctx, DmlParser.BooleanAndExpressionContext):
                return "bool"
            elif isinstance(ctx, DmlParser.PowerExpressionContext):
                return self.getDataType(ctx)
            elif isinstance(ctx, DmlParser.BooleanNotExpressionContext):
                return "bool"
            elif isinstance(ctx, DmlParser.ModIntDivExpressionContext):
                return "si64"
            elif isinstance(ctx, DmlParser.BooleanOrExpressionContext):
                return "bool"
            elif "[" in source and "]" in source:
                return self.visitDataIdentifier(ctx, True)
            elif source == "nan" or source == "inf":
                return "f64"

            raise ValueError(f"data type of {source} unknown")

    # Extracts value type from data type (e.g., f64 from matrix<f64>).
    def getValueType(self, dtype):
        value_type = dtype.split("<", 1)
        
        if len(value_type) == 1:
            raise ValueError(f"expected matrix but got {dtype}")

        value_type = value_type[1].rsplit(">", 1)[0]
        if "<" in value_type:
            return self.getValueType(value_type)
        else:
            return value_type
        
        return value_type
    
    # Order function arguments correctly and extract values/expressions.
    def reorder_args(self, args, correct_order):
        named_args = {k: [v, expr] for k, v, expr in args if k is not None}
        unnamed_args = [[v, expr] for k, v, expr in args if k is None]
        
        ordered_values = []
        ordered_exprs = []
        
        for order in correct_order:
            if order in named_args:
                v, expr = named_args[order]
                ordered_values.append(v)
                ordered_exprs.append(expr)
            elif unnamed_args:
                v, expr = unnamed_args.pop(0)
                ordered_values.append(v)
                ordered_exprs.append(expr)
                
        return ordered_values, ordered_exprs
    
    def retrieve_function_parameters(self, ctx):
        target = None
        if ctx:
            try:
                target = self.visitDataIdentifier(ctx.dataIdentifier())
            except:
                pass
                
        function_name = ctx.ID().getText()
        args = []
        for param_expr in ctx.parameterizedExpression():
            param_name, param_val = self.visitParameterizedExpression(param_expr)
            args.append([param_name, param_val, param_expr.expression()])
            
        return target, function_name, args

    # Handle implicit import of built-in DML scripts.
    def handleImplicitImport(self, script_name):
        # Get path to built-in DML script.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # TODO Don't hardcode the path to SystemDS.
        file_path_dml = os.path.join(script_dir, "../thirdparty/systemds/scripts/builtin/", script_name + ".dml")
        file_path_daph = os.path.join(script_dir, "translated_files/", script_name + ".daph")
        
        # Translate the script.
        try:
            with open(file_path_dml, "r") as f:
                dml_code = f.read()
                
            # Create parse tree.
            lexer = DmlLexer(InputStream(dml_code))
            tokens = CommonTokenStream(lexer)
            parser = DmlParser(tokens)
            tree = parser.programroot()
            
            # Traverse parse tree and translate it to DaphneDSL.
            translator = Translator()
            daph_code = translator.visitProgramroot(tree)

            # Write the translated code to a new file.
            with open(file_path_daph, "w") as f:
                f.write(daph_code)

        except FileNotFoundError:
            raise ValueError(f"unknown function: {script_name}")
            
        # Translate the import statement.
        translated_import = f'import "{script_name}.daph";\n'
        
        return translated_import, self.context.return_types[script_name]

    def create_function_call(self, function_name, arguments):
        function_call = ""
        dtype = ""
    
        arg_list = [arg[1] for arg in arguments]
        arg_str = ", ".join(arg_list)
    
        # Cases: built-in function, function inside this file, DML script that was not explicitly imported (explicit imports are handled in visitImportStatement).
        if function_name in self.FUNCTIONS_MAP:
            function_call, dtype = self.FUNCTIONS_MAP[function_name](arguments)
        elif function_name in self.context.function_names:
            i = 0
            converted_arg_list = []
            for arg in arg_list:
                param_type = param_types[function_name][i]
                if param_type in ["si64", "f64", "bool", "str", "matrix<si64>", "matrix<f64>", "matrix<bool>", "matrix<str>"]:
                    converted_arg_list.append(f"as.{param_type}({arg})")
                else:
                    converted_arg_list.append(arg)
                i += 1
            
            arg_str = ", ".join(converted_arg_list)
            function_call = f"{function_name}({arg_str})"
            dtype = self.context.return_types[function_name]
            self.context.call_graph[self.context.current_function].add(function_name)
        else:
            # Add default values for missing parameters in function call (and reorder them correctly).
            if len(arguments) != len(param_types[function_name]):
                arg_dict = {arg[0]: arg for arg in arguments}
                new_arguments = []
                for param in param_types[function_name]:
                    param_name, _, default_value = param
                    if param_name in arg_dict:
                        new_arguments.append(arg_dict[param_name])
                    else:
                        new_arguments.append([param_name, default_value, None])
                arguments = new_arguments

            arg_list = [arg[1] for arg in arguments]
            arg_str = ", ".join(arg_list)
            
            import_str, dtype = self.handleImplicitImport(function_name)
            self.context.imports += import_str
            function_call = f"{function_name}.m_{function_name}({arg_str})"
        
        # Return types of user-defined functions are stored in lists (because they can possibly return multiple values).
        # If there is only a single value, extract it (it is then handled in visitFunctionCallAssignmentStatement).
        if isinstance(dtype, list) and len(dtype) == 1:
            dtype = dtype[0]
            
        return function_call, dtype

    def handle_assignment(self, target, function_call, dtype):
        assignment = f"{function_call};"
        if target is not None:
            if not self.context.in_thenBody and not self.context.in_elseBody:
                self.context.already_assigned.add(target)
            elif self.context.in_thenBody:
                self.context.then_assigns.add(target)
            elif self.context.in_elseBody:
                self.context.else_assigns.add(target)
                
            self.context.data_types[target] = dtype
            assignment = f"{target} = {function_call};"
            
        return assignment

    # -------------------------------------------------------------------------
    # Visitor functions
    # -------------------------------------------------------------------------

    # Translate function call with one return value.
    def visitFunctionCallAssignmentStatement(self, ctx: DmlParser.FunctionCallAssignmentStatementContext):
        target, function_name, arguments = self.retrieve_function_parameters(ctx)
        function_call, dtype = self.create_function_call(function_name, arguments)
        assignment = self.handle_assignment(target, function_call, dtype)

        return assignment

    # Translate function call with multiple return values.
    def visitFunctionCallMultiAssignmentStatement(self, ctx: DmlParser.FunctionCallMultiAssignmentStatementContext):
        targets = [self.visitDataIdentifier(di) for di in ctx.dataIdentifier() if di is not None]
        _, function_name, arguments = self.retrieve_function_parameters(ctx)
        function_call, dtypes = self.create_function_call(function_name, arguments)

        # Handle assignments (setting data types, adding variables to list of initialized variables).
        i = 0
        for target in targets:
            _ = self.handle_assignment(target, function_call, dtypes[i])
            self.context.data_types[target] = dtypes[i]
            i += 1

        # Format assignments.
        targets_str = ", ".join(targets)
        multi_assignment = f"{targets_str} = {function_call};"
            
        return multi_assignment

    # Translate built-in function call.
    def visitBuiltinFunctionExpression(self, ctx: DmlParser.BuiltinFunctionExpressionContext, type=False):
        target, function_name, arguments = self.retrieve_function_parameters(ctx)
        function_call, dtype = self.create_function_call(function_name, arguments)

        return dtype if type else function_call
        
    # TODO Allow more constructs as entry points of the translation (currently only functions).
    # Starting point of the translation.
    def visitProgramroot(self, ctx: DmlParser.ProgramrootContext):
        # Get the names of (1) all the functions defined inside the file and (2) of all imported functions.
        for child in ctx.children:
            if isinstance(child, DmlParser.InternalFunctionDefExpressionContext):
                function_name = child.ID().getText()
                self.context.function_names.append(function_name)
                
                self.context.return_types[function_name] = []
                for param in child.typedArgNoAssign():
                    self.context.return_types[function_name].append(self.visitTypedArgNoAssign(param, True))
                
                param_types[function_name] = []
                for param in child.typedArgAssign():
                    param_types[function_name].append(self.visitTypedArgAssign(param, True))
        
        # Init graph for modelling function calls.
        self.context.call_graph = {function: set() for function in self.context.function_names}

        # Translate the script.
        translated_functions = {}
        translated_imports = []
        for child in ctx.children:
            if isinstance(child, DmlParser.InternalFunctionDefExpressionContext):
                self.context.current_function = child.ID().getText()
                translated_functions[self.context.current_function] = self.visitInternalFunctionDefExpression(child)
                
        # Apply topological sort manually.
        visited = set()
        stack = []
        for node in self.context.call_graph:
            if node not in visited:
                self.topological_sort(node, visited, stack, self.context.call_graph)
        
        ordered_functions = reversed([translated_functions[function_name] for function_name in stack if function_name in translated_functions])
        
        translated_code = self.context.imports # implicit imports
        translated_code += "".join(translated_imports) # explicit imports
        translated_code += "".join(ordered_functions)
        
        return translated_code
        
    def topological_sort(self, node, visited, stack, graph):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                self.topological_sort(neighbor, visited, stack, graph)
        stack.insert(0, node)

    # Identify exact statement type and call responsible function.
    def visitStatement(self, ctx: DmlParser.StatementContext):
        # TODO ImportStatement
        # TODO PathStatement
        # TODO IfdefAssignmentStatement
        # TODO BlockStatement
        if isinstance(ctx, DmlParser.AssignmentStatementContext):
            return self.visitAssignmentStatement(ctx)
        elif isinstance(ctx, DmlParser.FunctionCallAssignmentStatementContext):
            return self.visitFunctionCallAssignmentStatement(ctx)
        elif isinstance(ctx, DmlParser.ForStatementContext):
            return self.visitForStatement(ctx)
        elif isinstance(ctx, DmlParser.IterablePredicateContext):
            return self.visitIterablePredicate(ctx)
        elif isinstance(ctx, DmlParser.IfStatementContext):
            return self.visitIfStatement(ctx)
        elif isinstance(ctx, DmlParser.FunctionCallMultiAssignmentStatementContext):
            return self.visitFunctionCallMultiAssignmentStatement(ctx)
        elif isinstance(ctx, DmlParser.WhileStatementContext):
            return self.visitWhileStatement(ctx)
        elif isinstance(ctx, DmlParser.ParForStatementContext):
            return self.visitParForStatement(ctx)
        elif isinstance(ctx, DmlParser.AccumulatorAssignmentStatementContext):
            return self.visitAccumulatorAssignmentStatement(ctx)
        else:
            raise ValueError(f"unknown statement {ctx}")
        
    # Identify exact expression type and call responsible function.
    def visitExpression(self, ctx: DmlParser.ExpressionContext):
        # TODO What about DML's MultiIdExpression?
        if isinstance(ctx, DmlParser.AddSubExpressionContext):
            return self.visitAddSubExpression(ctx)
        elif isinstance(ctx, DmlParser.DataIdExpressionContext):
            return self.visitDataIdExpression(ctx)
        elif isinstance(ctx, DmlParser.BuiltinFunctionExpressionContext):
            return self.visitBuiltinFunctionExpression(ctx)
        elif isinstance(ctx, DmlParser.MultDivExpressionContext):
            return self.visitMultDivExpression(ctx)
        elif isinstance(ctx, DmlParser.AtomicExpressionContext):
            return self.visitAtomicExpression(ctx)
        elif isinstance(ctx, DmlParser.MatrixMulExpressionContext):
            return self.visitMatrixMulExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstIntIdExpressionContext):
            return self.visitConstIntIdExpression(ctx)
        elif isinstance(ctx, DmlParser.RelationalExpressionContext):
            return self.visitRelationalExpression(ctx)
        elif isinstance(ctx, DmlParser.UnaryExpressionContext):
            return self.visitUnaryExpression(ctx)
        elif isinstance(ctx, DmlParser.BooleanAndExpressionContext):
            return self.visitBooleanAndExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstDoubleIdExpressionContext):
            return self.visitConstDoubleIdExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstFalseExpressionContext):
            return self.visitConstFalseExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstTrueExpressionContext):
            return self.visitConstTrueExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstStringIdExpressionContext):
            return self.visitConstStringIdExpression(ctx)
        elif isinstance(ctx, DmlParser.PowerExpressionContext):
            return self.visitPowerExpression(ctx)
        elif isinstance(ctx, DmlParser.BooleanNotExpressionContext):
            return self.visitBooleanNotExpression(ctx)
        elif isinstance(ctx, DmlParser.ModIntDivExpressionContext):
            return self.visitModIntDivExpression(ctx)
        elif isinstance(ctx, DmlParser.BooleanOrExpressionContext):
            return self.visitBooleanOrExpression(ctx)
        elif isinstance(ctx, DmlParser.ParameterizedExpressionContext):
            return self.visitParameterizedExpression(ctx)
        else:
            raise ValueError(f"unknown expression {ctx}")

    # Defining a function.
    def visitInternalFunctionDefExpression(self, ctx: DmlParser.InternalFunctionDefExpressionContext):
        function_name = ctx.ID().getText()

        # Get parameters.
        input_params = []
        for param in ctx.typedArgAssign():
            input_params.append(self.visitTypedArgAssign(param))
        input_params_string = ", ".join(input_params)

        # Get data/value types of return values.
        output_params = []
        for param in ctx.typedArgNoAssign():
            output_params.append(self.visitTypedArgNoAssign(param, True))
        output_params_string = ", ".join(output_params)

        # Get variable names of return values.
        output_names = []
        i = 0
        for param in ctx.typedArgNoAssign():
            return_value = self.visitTypedArgNoAssign(param, False)
            return_type = self.context.return_types[self.context.current_function][i]
            
            if return_type in ["si64", "f64", "bool", "str", "matrix<si64>", "matrix<f64>", "matrix<bool>", "matrix<str>"]:
                output_names.append(f"as.{return_type}({return_value})")
            else:
                output_names.append(return_value)
                
            i += 1
        output_names_string = ", ".join(output_names)
        
        # Get function body.
        function_body = []
        for statement in ctx.statement():
            function_body.append(self.visitStatement(statement))
        function_body_string = "\n".join(function_body)
        indented_function_body = self.indent(function_body_string, 1)

        # Construct the function.
        translated_function = f"def {function_name}({input_params_string}) -> {output_params_string}" + " {\n"
        translated_function += f"{INDENT}{indented_function_body}"
        translated_function += f"\n{INDENT}return {output_names_string};\n}}\n\n"
        
        return translated_function
        
    # Retrieve input parameters.
    def visitTypedArgAssign(self, ctx: DmlParser.TypedArgAssignContext, types=False):
        param_type = self.visitMl_type(ctx.ml_type())
        param_name = ctx.ID().getText()
        
        if types:
            return param_type
        
        # Return the parameter (optionally with initial value).
        if ctx.paramVal is not None:
            param_val = self.visitExpression(ctx.paramVal)
            # TODO Shouldn't such conversions be handled by float literal translation?
            if param_val == "NaN" or param_val == "Inf":
                param_val = param_val.lower()
            translated_param = f"{param_name}:{param_type} /*= {param_val}*/"
        else:
            translated_param = f"{param_name}:{param_type}"
        
        
        # Specify data type of variable in global dict.
        self.context.data_types[param_name] = param_type
        
        return translated_param

    # Retrieve output parameters.
    def visitTypedArgNoAssign(self, ctx: DmlParser.TypedArgNoAssignContext, types):
        param_type = self.visitMl_type(ctx.ml_type())
        param_name = ctx.ID().getText()
        
        # Return either type or name (depending on specified parameter "types").
        translated_param = param_type if types else param_name
        
        return translated_param
        
    # Retrieve DataType and ValueType.
    def visitMl_type(self, ctx:DmlParser.Ml_typeContext):
        if ctx.dataType() is not None: # e.g., "Matrix[Double] X"
            data_type = self.visitValueType(ctx.dataType())
            value_type = self.visitValueType(ctx.valueType())
            translated_type = f"{data_type}<{value_type}>"
        else: # e.g., "Double X"
            value_type = self.visitValueType(ctx.valueType())
            translated_type = value_type
            
        return translated_type
    
    # Retrieve data type.
    def visitValueType(self, ctx: DmlParser.DataTypeContext):
        dtype = ctx.getText().lower()
        if dtype == "integer":
            dtype = "si64"
        elif dtype == "string":
            dtype = "str"
        elif dtype == "boolean":
            dtype = "bool"
        elif dtype == "double":
            dtype = "f64"
            
        return dtype
    
    # Get name of target via visitDataIdentifier().
    def visitDataIdExpression(self, ctx: DmlParser.DataIdExpressionContext):
        data_identifier = self.visitDataIdentifier(ctx.dataIdentifier())
        
        return data_identifier

    # Return data identifier or data type.
    def visitDataIdentifier(self, ctx: DmlParser.DataIdentifierContext, type=False):
        # TODO This seems to be tailored to DML's IndexedExpression, also support
        # SimpleDataIdentifierExpression, CommandlineParamExpression, CommandlinePositionExpression.

        # TODO Why is the full text used here, why not further parse the kinds of expressions?
        identifier = ctx.getText()
        
        matrix_name = ""
        is_slice = False
    
        # Check if identifier contains indexing/slicing.
        if "[" in identifier and "]" in identifier:
            # Find the first and last bracket.
            first_bracket = identifier.index("[")
            last_bracket = identifier.rindex("]")
            
            # TODO By this custom parsing, the matrix_name could end with whitespaces.
            matrix_name = identifier[:first_bracket]
            indices = identifier[first_bracket + 1:last_bracket]
            rest = identifier[last_bracket + 1:]

            # Split the indices based on the last comma
            # TODO This is problematic, since nested expressions could contain commas.
            last_comma = indices.rindex(",") if "," in indices else -1
            if last_comma == -1:
                index = indices
                try:
                    index = str(int(index.strip()) - 1)
                except ValueError:
                    if index != "" and index != " ":
                        index = index + " - 1"
                        
                # Construct the identifier.
                identifier = f"{matrix_name}[{index}]{rest}"
            else:
                row_index = indices[:last_comma].strip()
                col_index = indices[last_comma + 1:].strip()
                
                # Handle slicing operation in row index.
                if ":" in row_index:
                    is_slice = True
                    start, end = row_index.split(":")
                    if start:
                        start = str(int(start) - 1) if start.isdigit() else start + " - 1"
                    else:
                        start = "0"
                    row_index = f"{start}:{end}"

                else:
                    try:
                        row_index = str(int(row_index.strip()) - 1)
                    except ValueError:
                        if row_index != "" and row_index != " ":
                            row_index = row_index + " - 1"

                # Handle slicing operation in column index.
                if ":" in col_index:
                    is_slice = True
                    start, end = col_index.split(":")
                    if start:
                        start = str(int(start) - 1) if start.isdigit() else start + " - 1"
                    else:
                        start = "0"
                    col_index = f"{start}:{end}"

                else:
                    try:
                        col_index = str(int(col_index.strip()) - 1)
                    except ValueError:
                        if col_index != "" and col_index != " ":
                            col_index = col_index + " - 1"
                
                # Construct the identifier.
                # TODO Can we safely omit the rest?
                identifier = f"{matrix_name}[{row_index}, {col_index}]{rest}"
        # TODO else: raise error?

        # Return the datatype instead of the string if type is true.
        if type:
            # Distinguish between slicing and indexing (e.g., matrix[1:5, 2:4] or matrix[0,1]).
            if is_slice:
                return self.context.data_types.get(matrix_name) # e.g., matrix<f64>
            else:
                # TODO In DaphneDSL (and DML), single-element access yields a 1x1 matrix.
                return self.getValueType(self.context.data_types.get(matrix_name)) # e.g., f64
        else:
            return identifier

    
    # Simply visit the sub-expression and return its translation in parenthesis.
    def visitAtomicExpression(self, ctx: DmlParser.AtomicExpressionContext):
        return "(" + self.visitExpression(ctx.expression()) + ")"
        
    # E.g., "x + y" or "x - y".
    def visitAddSubExpression(self, ctx: DmlParser.AddSubExpressionContext):
        # Visit the left and right expressions
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))
        
        # Get the operator.
        op = ctx.op.text
        
        # Construct the expression.
        translated_expression = f"{left} {op} {right}"
        
        return translated_expression
    
    # E.g., "X @ Y".
    def visitMatrixMulExpression(self, ctx: DmlParser.MatrixMulExpressionContext):
        # Visit the left and right expressions.
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))
        
        return f"{left} @ {right}"
        
    # E.g., "x * y" or "x / y".
    def visitMultDivExpression(self, ctx: DmlParser.MultDivExpressionContext):
        # Visit the left and right expressions.
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))
        
        # Get the operator.
        op = ctx.op.text
        
        # Construct the expression.
        translated_expression = f"{left} {op} {right}"
        
        return translated_expression
        
    # E.g., "x > y".
    def visitRelationalExpression(self, ctx: DmlParser.RelationalExpressionContext, type=False):
        # Visit the left and right expressions.
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))

        # Get the operator.
        op = ctx.op.text

        # Construct the expression.
        translated_expression = f"{left} {op} {right}"

        left_dtype = self.inferType(left, ctx.expression(0))
        right_dtype = self.inferType(right, ctx.expression(1))

        dtype = left_dtype
        if self.isMatrix(left_dtype) and not self.isMatrix(right_dtype):
            dtype = left_dtype
        elif not self.isMatrix(left_dtype) and self.isMatrix(right_dtype):
            # TODO Support scalar-matrix comparison in DaphneDSL and remove this workaround.
            # E.g., 0 < matrix is not allowed => translate to matrix > 0.
            if op == ">":
                op = "<"
            elif op == "<":
                op = ">"
            elif op == ">=":
                op = "<="
            elif op == "<=":
                op = ">="
            # TODO "==" and "!=": swap operands, but retain operator.
                
            translated_expression = f"{right} {op} {left}"
            dtype = right_dtype
        
        return dtype if type else translated_expression
        
    # E.g., translate "-x" to "0.0-x".
    def visitUnaryExpression(self, ctx: DmlParser.UnaryExpressionContext):
        # Visit the operand.
        operand = self.visitExpression(ctx.expression())
        
        # Get the operator.
        op = ctx.op.text
        
        # Construct the unary expression.
        try:
            # TODO Can this ever make sense?
            num = float(operand)
            if op == "-":
                num = 0.0 - num
            translated_expression = f"({num})"
        except ValueError:
            translated_expression = f"(0.0 {op} {operand})"
        
        return translated_expression
    
    # "x & y" and "x && y".
    def visitBooleanAndExpression(self, ctx: DmlParser.BooleanAndExpressionContext):
        # Visit the left and right expressions.
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))
        
        # Construct the expression (DaphneDSL: "&&", DML: "&" or "&&").
        translated_expression = f"{left} && {right}"
        
        return translated_expression
        
    # "x | y" and "x || y".
    def visitBooleanOrExpression(self, ctx: DmlParser.BooleanOrExpressionContext):
        # Visit the left and right expressions.
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))

        # Construct the expression (DaphneDSL: "||", DML: "|" or "||").
        translated_expression = f"{left} || {right}"
        
        return translated_expression
    
    # "x ^ y".
    def visitPowerExpression(self, ctx: DmlParser.PowerExpressionContext):
        # Visit the left and right expressions.
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))
        
        # Construct the expression.
        translated_expression = f"{left} ^ {right}"
        
        return translated_expression
        
    # DML supports "!x" but Daphne does not (thus, translating "!x" to "x == false").
    def visitBooleanNotExpression(self, ctx: DmlParser.BooleanNotExpressionContext):
        # Visit the expression.
        expr = self.visitExpression(ctx.expression())

        # Construct the translated expression.
        translated_expression = f"({expr} == false)"
        
        return translated_expression
        
    # Modulus and integer division.
    def visitModIntDivExpression(self, ctx: DmlParser.ModIntDivExpressionContext):
        # Visit the left and right expressions.
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))

        # Get the operator.
        op = ctx.op.text

        # Translate mod and integer division.
        if op == "%/%":
            translated_expression = f"floor({left} / {right})"
        elif op == "%%":
            translated_expression = f"{left} % {right}"
        else:
            raise RuntimeError(f"unknown operator: `{op}`")
        
        return translated_expression
        
    # Float constant.
    def visitConstDoubleIdExpression(self, ctx: DmlParser.ConstDoubleIdExpressionContext):
        # Get double constant.
        const = ctx.DOUBLE().getText()
        
        # Change from scientific notation to decimal values.
        num = float(const)
        # TODO Is 20 digits always enough?
        num = "{:.20f}".format(num)
        
        num = num.rstrip("0")
        
        if num[-1] == ".":
            num += "0"
        
        return num
        
    # Integer constant.
    def visitConstIntIdExpression(self, ctx: DmlParser.ConstIntIdExpressionContext):
        # Get the integer constant.
        const = ctx.INT().getText()
        return const
    
    # Boolean constant.
    def visitConstFalseExpression(self, ctx: DmlParser.ConstFalseExpressionContext):
        return "false"
    
    # Boolean constant.
    def visitConstTrueExpression(self, ctx: DmlParser.ConstTrueExpressionContext):
        return "true"
        
    # String constant.
    def visitConstStringIdExpression(self, ctx: DmlParser.ConstStringIdExpressionContext):
        # Get the string constant.
        dml_string = ctx.STRING().getText()
        
        # TODO Convert ' to ", since DML allows both, but DaphneDSL doesn't.
        return dml_string
        
    # Input parameters for a function.
    def visitParameterizedExpression(self, ctx: DmlParser.ParameterizedExpressionContext, init_vals = True):
        param_name = ctx.paramName.text if ctx.paramName is not None else None
        param_val = self.visitExpression(ctx.expression())
        
        # TODO Such conversions should be done centrally for float parsing.
        if param_val == "NaN" or param_val == "Inf":
            param_val = param_val.lower()
        
        # Check if the parameter has a name.
        if param_name is not None and init_vals == True:
            return param_name, param_val
        else:
            return None, param_val
        
    # Translate simple assignment statement.
    def visitAssignmentStatement(self, ctx: DmlParser.AssignmentStatementContext):
        target = self.visitDataIdentifier(ctx.dataIdentifier())
        source = self.visitExpression(ctx.expression())
        
        # TODO Such conversions should be done centrally for float parsing.
        if source == "NaN" or source == "Inf":
            source = source.lower()
            
        # Add target variable to fitting list (needed for initialization of variables before if-then-else).
        if not self.context.in_thenBody and not self.context.in_elseBody:
            self.context.already_assigned.add(target)
        elif self.context.in_thenBody:
            self.context.then_assigns.add(target)
        elif self.context.in_elseBody:
            self.context.else_assigns.add(target)
            
        dtype = self.inferType(source, ctx.expression())
        
        if "[" in target and "]" in target:
            # TODO More generally, convert scalars to matrices.
            if dtype == "si64":
                return f"{target} = as.matrix({source}.0);"
            
        dtype_target = self.context.data_types.get(target)
        if dtype_target:
            return f"{target} = as.{dtype_target}({source});"
        
        self.context.data_types[target] = dtype

        return f"{target} = {source};"
    
    # "x += y" (not supported in DaphneDSL; thus, needs to be translated to "x = x + y").
    def visitAccumulatorAssignmentStatement(self, ctx: DmlParser.AccumulatorAssignmentStatementContext):
        # Get the target.
        target = self.visitDataIdentifier(ctx.dataIdentifier())

        # Get the operator.
        op = ctx.op.text

        # Get the source.
        source = self.visitExpression(ctx.expression())

        # Create the assignment statement (turn a += b into a = a + b).
        assignment = f"{target} {op[1]} {target} {op[0]} {source};"

        return assignment
    
    # For-loop.
    def visitForStatement(self, ctx: DmlParser.ForStatementContext):
        # Get the iteration variable.
        iter_var = ctx.ID().getText()
        
        # Get the iterable predicate.
        iterable_pred = self.visitIterablePredicate(ctx.iterablePredicate())
        
        # Get the parameters.
        params = [param for param in (self.visitStrictParameterizedExpression(expr) for expr in ctx.strictParameterizedExpression()) if param is not None]
        params_string = ", ".join(params)
        
        # Get the body of the for-loop.
        body = [self.visitStatement(stmt) for stmt in ctx.statement()]
        body_string = "\n".join(body)
        
        # TODO The params are not supported in DaphneDSL at all.
        # Construct the translated for-loop.
        translated_for_loop = f"for({iter_var} in {iterable_pred}{params_string})" + " {\n"
        translated_for_loop += self.indent(f"{INDENT}{body_string}", 1)
        translated_for_loop += "\n}"
        
        return translated_for_loop
    
    # Parfor-Loop (does not exist in DAPHNE (yet, see #515); thus, translated to normal for-loop).
    def visitParForStatement(self, ctx: DmlParser.ParForStatementContext):
        # Get the iteration variable.
        iter_var = ctx.ID().getText()
        
        # Get the iterable predicate.
        iterable_pred = self.visitIterablePredicate(ctx.iterablePredicate())
        
        # Get the parameters.
        params = [param for param in (self.visitStrictParameterizedExpression(expr) for expr in ctx.strictParameterizedExpression()) if param is not None]
        params_string = ", ".join(params)
        
        # Get the body of the for-loop.
        body = [self.visitStatement(stmt) for stmt in ctx.statement()]
        body_string = "\n".join(body)
        
        # TODO The params are not supported in DaphneDSL at all.
        # Construct the translated for-loop.
        translated_for_loop = f"for({iter_var} in {iterable_pred}{params_string})" + " {\n"
        translated_for_loop += self.indent(f"{INDENT}{body_string}", 1)
        translated_for_loop += "\n}"
        
        return translated_for_loop
    
    # If-then-else.
    def visitIfStatement(self, ctx: DmlParser.IfStatementContext):
        # Get the predicate expression.
        predicate = self.visitExpression(ctx.expression())
        
        # Get the then body.
        self.context.in_thenBody = True
        then_body_statements = [self.visitStatement(stmt) for stmt in ctx.ifBody]
        self.context.in_thenBody = False
        
        # Get the else body.
        self.context.in_elseBody = True
        else_body_statements = [self.visitStatement(stmt) for stmt in ctx.elseBody]
        self.context.in_elseBody = False
        
        # TODO I think that doesn't work for nested if-statements, which would require arbitrary levels of nesting.
        # TODO I think the point is not which variables get assigned in common in then/else, but which are used
        # again later.
        # then_assigns, else_assigns, etc.
        # Initialize variables that are only initialized inside the if-then-else-block.
        common_assigns = self.context.then_assigns.intersection(self.context.else_assigns)
        common_assigns -= self.context.already_assigned
        common_assigns_str = "\n".join([f"{var} = {self.default_value(self.context.data_types[var.split('[')[0]])};" for var in common_assigns])
        
        # Turn then and else body into string.
        then_body = "\n".join(then_body_statements)
        else_body = "\n".join(else_body_statements)
        
        # Clear the variable list (otherwise the same variables would be initialized again before the next if-statement).
        self.context.then_assigns.clear()
        self.context.else_assigns.clear()
        
        # Construct the if statement.
        if_statement = f"if ({predicate})" + " {\n"
        if_statement += self.indent(f"{INDENT}{then_body}", 1)
        if_statement += "\n}"
        
        # If there is an else body, add it to the if statement.
        if else_body_statements:
            if_statement += " else {\n"
            if_statement += self.indent(f"{INDENT}{else_body}", 1)
            if_statement += "\n}"
            
        return common_assigns_str + "\n" + if_statement + "\n"
            
    # While-Loop.
    def visitWhileStatement(self, ctx: DmlParser.WhileStatementContext):
        # Get the expression.
        expression = self.visitExpression(ctx.expression())

        # Get the body of the while-loop.
        body_statements = [self.visitStatement(stmt) for stmt in ctx.statement()]
        body = "\n".join(body_statements)

        # Construct the translated while-loop.
        translated_while_loop = f"while ({expression})" + " {\n"
        translated_while_loop += self.indent(f"{INDENT}{body}", 1)
        translated_while_loop += "\n}"
        
        return translated_while_loop
        
    # Translate iterable predicate.
    def visitIterablePredicate(self, ctx: DmlParser.IterablePredicateContext):
        if isinstance(ctx, DmlParser.IterablePredicateColonExpressionContext):
            from_exp = self.visitExpression(ctx.expression(0))
            to_exp = self.visitExpression(ctx.expression(1))
            
            iterable_pred = f"{from_exp}:{to_exp}"
            
            return iterable_pred
        elif isinstance(ctx, DmlParser.IterablePredicateSeqExpressionContext):
            from_exp = self.visitExpression(ctx.expression(0))
            to_exp = self.visitExpression(ctx.expression(1))
            
            iterable_pred = f"{from_exp}:{to_exp}"
            if ctx.increment is not None:
                increment = self.visitExpression(ctx.increment)
                iterable_pred += f":{increment}"
            
            return iterable_pred
        else:
            raise ValueError(f"unknown expression {ctx} for iterable predicate")

# -------------------------------------------------------------------------
# Running the translator
# -------------------------------------------------------------------------

# Translator from DML to DaphneDSL.
def translate(dml_code):
    # Generate parse tree.
    lexer = DmlLexer(InputStream(dml_code))
    tokens = CommonTokenStream(lexer)
    parser = DmlParser(tokens)
    tree = parser.programroot()

    # Traverse parse tree and translate it to DaphneDSL.
    translator = Translator()
    daph_code = translator.visitProgramroot(tree)

    return daph_code

def get_daphne_filename(dml_filename):
    base_name = os.path.basename(dml_filename)
    base_name = base_name.replace(".dml", ".daph")
    
    return os.path.join(output_dir, base_name)

# Reading .dml script and writing .daph script.
if __name__ == "__main__":
    # Get DML file path via command line.
    parser = argparse.ArgumentParser(description="Translate DML file to DaphneDSL.")
    parser.add_argument("dml_filename", type=str, help="Path to the DML file to translate")
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "translated_files")
    os.makedirs(output_dir, exist_ok=True)

    with open(args.dml_filename, "r") as f:
        dml_code = f.read()

    daph_code = translate(dml_code)

    daph_filename = get_daphne_filename(args.dml_filename)
    with open(daph_filename, "w") as f:
        f.write(daph_code)
