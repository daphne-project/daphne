/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_IR_DAPHNEIR_DAPHNEDIALECT_TD
#define SRC_IR_DAPHNEIR_DAPHNEDIALECT_TD

include "mlir/IR/OpBase.td"

// ****************************************************************************
// Daphne dialect definition.
// ****************************************************************************

def Daphne_Dialect : Dialect {
    let name = "daphne";
    let summary = "An MLIR dialect for DAPHNE.";
    let description = [{
      This is an MLIR dialect for DAPHNE. Details coming soon.
    }];
    let cppNamespace = "::mlir::daphne";

    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 1;
}

#endif //SRC_IR_DAPHNEIR_DAPHNEDIALECT_TD
