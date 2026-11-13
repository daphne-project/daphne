/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef SRC_IR_DAPHNEIR_PASSES_TD
#define SRC_IR_DAPHNEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def AlgebraicSimplify : Pass<"daphne-algebraic-simplify", "::mlir::ModuleOp"> {
  let summary = "Simplifies DaphneIR using rewrite drivers keyed on algebraic op traits.";
  let constructor = "mlir::daphne::createAlgebraicSimplifyPass()";
}

def ConvertDaphneToLinalg : Pass<"convert-daphne-to-linalg"> {
  let summary = "Converts certain DaphneOps to ops from the Linalg dialect.";
  let description = [{
    Converts certain DaphneOps to ops from the Linalg dialect.
  }];
  let constructor = "daphne::createConvertDaphneToLinalgPass()";

}

def DistributeComputations : Pass<"distribute-computations", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createDistributeComputationsPass()";
}

def DistributePipelines : Pass<"distribute-pipelines", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createDistributePipelinesPass()";
}

def Inference: Pass<"inference", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createInferencePass()";
}

def AdaptTypesToKernels: Pass<"adapt-types-to-kernels", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createAdaptTypesToKernelsPass()";
}

def ManageObjRefs : Pass<"manage-obj-refs", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createManageObjRefsPass()";
}

def PrintIR : Pass<"daphne-print-ir", "::mlir::ModuleOp"> {
    let constructor = "mlir::daphne::createPrintIRPass()";
}

def RecordPropertiesPass : Pass<"record-properties", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createRecordPropertiesPass()";
}

def InsertPropertiesPass : Pass<"insert-properties", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createInsertPropertiesPass()";
    let options = [
        Option<"properties_file_path", "properties_file_path", "std::string", 
               /*default=*/"\"properties.json\"",
               "The path to the JSON file storing the data properties.">
    ];
}

def RewriteSqlOpPass : Pass<"rewrite-sqlop", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createRewriteSqlOpPass()";
}

def DaphneOpsOptPass : Pass<"opt-daphne", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createDaphneOptPass()";
}

def TransferDataPropertiesPass: Pass<"transfer-data-props", "::mlir::func::FuncOp"> {
    let constructor = "mlir::daphne::createTransferDataPropertiesPass()";
}

#endif // SRC_IR_DAPHNEIR_PASSES_TD
