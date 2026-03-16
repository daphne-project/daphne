#!/usr/bin/env python3

# Copyright 2021 The DAPHNE Consortium
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

"""
Generates the C++ code for the pre-compiled kernels library as well as a JSON
file for the kernel catalog.

This script generates C++ code instantiating the kernel templates that shall be
part of a pre-compiled kernel library. Each kernel instantiation is wrapped by
a shallow function that can be called from the JIT-compiled user program. An
input JSON-file specifies which kernel shall be instantiated with which
template arguments.

Kernels are sorted in descending order by the number of template
instantiations. The top N kernels are generated in separate files, while the
rest are generated in a single file. This is done so that kernels with large
amounts of code are compiled in parallel while the remaining kernels are short
enough to be compiled in a single file.

Furthermore, a JSON file is generated that contains information about the
pre-compiled kernels. This file is used to populate the kernel catalog at
system start-up.
"""

# TODO Note that this script currently makes strong assumptions about the
# signatures of possible kernels. It might not yet generalize well enough.

# TODO Note that the way the information on the kernel template is specified in
# the input JSON-file will be simplified significantly later on.

import io
import json
import platform
import re
import sys
from typing import List, Tuple
from pathlib import Path

INDENT = 4 * " "
DEFAULT_NEWRESPARAM = "res"


def toCppType(t):
    if isinstance(t, list):
        if len(t) == 2:
            return "{}<{}>".format(t[0], t[1])
        elif len(t) == 3:
            return "{}<{}<{}>>".format(t[0], t[1], t[2])
        else:
            raise RuntimeError("unexpected nesting level of template types: {}".format(t))
    else:
        return t

pStripNamespaces1 = re.compile(r"^(.*)<.*::(.*)>(.*)$") # matches the last "::" before the ">"
pStripNamespaces2 = re.compile(r"^.*::(.*)$") # matches the last "::"
def stripNamespaces(cppType):
    """Removes namespaces from the passed C++ type, if present.

    Examples:
    - `const DenseMatrix<std::string> *` -> `const DenseMatrix<string> *`
    - `const DenseMatrix<double> *` -> `const DenseMatrix<double> *`
    - `mlir::daphne::GroupEnum` -> `GroupEnum`
    - `int64_t` -> "int64_t`
    """

    # Limitation: if cppType is a template type with multiple template parameters, this function would not remove the
    # namespace for all template parameters individuall. However, this case is not important at the moment.

    res = cppType

    # Step 1: Remove namespace of template parameter.
    # Example: "const DenseMatrix<std::string> *" -> "const DenseMatrix<string> *" (removed namespace of value type)
    # Example: "const DenseMatrix<double> *" -> "const DenseMatrix<double> *" (unchanged)
    # Example: "int64_t" -> "int64_t" (unchanged, because doesn't match the pattern)
    res = pStripNamespaces1.sub(r"\1<\2>\3", res)

    # Step 2: Remove namespace of overall type.
    # Example: "mlir::daphne::GroupEnum" -> "GroupEnum" (removed namespace of overall type)
    # Example: "int64_t" -> "int64_t" (unchanged)
    res = pStripNamespaces2.sub(r"\1", res)

    return res

def generateKernelInstantiation(kernelTemplateInfo, templateValues, opCodes, outFile, catalogEntries, API):
    # Extract some information.
    opName = kernelTemplateInfo["opName"]
    returnType = kernelTemplateInfo["returnType"]
    templateParams = kernelTemplateInfo["templateParams"]
    runtimeParams = kernelTemplateInfo["runtimeParams"]
    opCodeAsTemplateParam = False
    if "opCodeAsTemplateParam" in kernelTemplateInfo:
        opCodeAsTemplateParam = True if kernelTemplateInfo["opCodeAsTemplateParam"] == 1 else False

    if len(templateParams) != len(templateValues):
        raise RuntimeError(
            f"kernel \"{opName}\" has {len(templateParams)} template parameters, but "
            f"{len(templateValues)} template values are supplied in an instantiation"
        )

    if opCodes is not None:
        # We assume that the op-code is the first run-time parameter.
        opCodeType = runtimeParams[0]["type"]
        runtimeParams = runtimeParams[1:]
    else:
        opCodeType = None

    # Create mapping from original template argument names to assigned C++
    # types.
    templateArgToCppType = {tp["name"]: toCppType(tv) for tp, tv in zip(templateParams, templateValues)}

    # ToDo: commented by mdokter - maybe remove. I think this would be too verbose
    # Comments indicating values assigned to original template arguments.
    # for tp in templateParams:
    #     outStr = INDENT + "// {} = {}\n".format(tp["name"], templateArgToCppType[tp["name"]])
    #     outFile.write(outStr)

    # The function wrapping the generated kernel instantiation always has
    # the return type void. If the considered kernel returns a scalar value,
    # we prepend an additional run-time parameter.
    extendedRuntimeParams = [
        {"name": DEFAULT_NEWRESPARAM, "type": "{} *".format(returnType), "isOutput": True}
    ] if (returnType != "void") else []
    # Add all run-time parameters of the kernel. We need to copy, because
    # we apply string replacements to the types.
    extendedRuntimeParams.extend([rp.copy() for rp in runtimeParams])
    # Replace occurences of original template arguments by their assigned
    # types.
    for rp in extendedRuntimeParams:
        for tpIdx, tp in enumerate(templateParams):
            if isinstance(templateValues[tpIdx], list):
                rp["type"] = rp["type"].replace("typename {}::VT".format(tp["name"]), templateValues[tpIdx][1])
            rp["type"] = rp["type"].replace(tp["name"], templateArgToCppType[tp["name"]])
        if rp["type"].endswith("*&"):
            rp["type"] = rp["type"][:-2] + "**"
            rp["isOutput"] = True
        if "isReturnValue" in rp:
            rp["isOutput"] = True
        if rp["type"].endswith("&"):
            rp["type"] = rp["type"][:-1]
            rp["isOutput"] = True
        elif "isOutput" not in rp:
            rp["isOutput"] = False

    isCreateDaphneContext = opName == "createDaphneContext"
    def isInstrumentedOp(op :str):
        # TODO: currently daphne ctx is deleted before ops when doing sparse
        # codegen, so instrumentation breaks
        if op in ["map", "createDaphneContext", "destroyDaphneContext"]:
            return False
        # TODO: KernelDispatchMapping currently does not support distributed ops
        if "distributed" in op:
            return False
        return True
    isInstrumented = isInstrumentedOp(opName)

    # typesForName = "__".join([("{}_{}".format(tv[0], tv[1]) if isinstance(tv, list) else tv) for tv in templateValues])
    typesForName = "__".join([
        stripNamespaces(rp["type"])
            .replace("const ", "")
            .replace(" **", "" if rp["isOutput"] else "_variadic")
            .replace(" *", "_variadic" if "isVariadic" in rp and rp["isVariadic"] else "")
            .replace("& ", "")
            .replace("<", "_")
            .replace(">", "")
            .replace(",", "_")
            .replace(" ", "_")
        for rp in extendedRuntimeParams
    ])
    if typesForName != "":
        typesForName = "__" + typesForName
    params = ", ".join(
        ["{} {}".format(rtp["type"], rtp["name"]) for rtp in
         extendedRuntimeParams] + ([] if not isInstrumented else ["int kId"]) + ([] if isCreateDaphneContext else ["DCTX(ctx)"])
    )

    def generateFunction(opCode):
        # Obtain the name of the function to be generated from the opName by
        # removing suffices "Sca"/"Mat"/"Obj" (they are not required here), and
        # potentially by inserting the opCode into the name.
        concreteOpName = opName
        while concreteOpName[-3:] in ["Sca", "Mat", "Obj"]:
            concreteOpName = concreteOpName[:-3]
        concreteOpName = concreteOpName.replace("::", "_")
        if opCode is not None:
            opCodeWord = opCodeType[:-len("OpCode")]
            sep_pos = (((opCodeWord.find("::") + 2) if "::" in opCodeWord else 0))
            opCodeWord = opCodeWord[sep_pos:]
            concreteOpName = concreteOpName.replace(opCodeWord, opCode[0].upper() + opCode[1:].lower())
            concreteOpName = concreteOpName.replace(opCodeWord.lower(), opCode.lower())

        if API != "CPP":
            funcName = API + "_" + concreteOpName
        else:
            funcName = "_" + concreteOpName


        # Signature of the function wrapping the kernel instantiation.
        outFile.write(INDENT + "void {}{}({}) {{\n".format(
            funcName,
            typesForName,
            # Run-time parameters, possibly including DaphneContext:
            params
        ))

        # List of parameters for the call.
        if opCode is None or opCodeAsTemplateParam:
            callParams = []
        else:
            callParams = ["{}::{}".format(opCodeType, opCode)]

        callParams.extend([
            # Dereference double pointer for output parameters.
            "{}{}".format("*" if (rp["type"].endswith("**") and rp["isOutput"]) else "", rp["name"])
            for rp in extendedRuntimeParams[(0 if returnType == "void" else 1):]
        ])

        # List of template parameters for the call.
        callTemplateParams = [toCppType(tv) for tv in templateValues]
        if opCodeAsTemplateParam and opCode is not None:
            opCodeWord = opCodeType[:-len("OpCode")]
            callTemplateParams = ["{}::{}".format(opCodeWord if API == "CPP" else API + "::" + opCodeWord, opCode)] + callTemplateParams

        # Body of that function: delegate to the kernel instantiation.
        outFile.write(2 * INDENT)
        if isInstrumented:
            outFile.write(f"try{{\n")
            outFile.write(3 * INDENT)
            outFile.write(f"preKernelInstrumentation(kId, ctx);\n")
            outFile.write(3 * INDENT)

        #  import pdb;pdb.set_trace()
        #  insertPreKernelInstrumentation(outFile)
        if returnType != "void":
            outFile.write("*{} = ".format(DEFAULT_NEWRESPARAM))

        kernelCallString = "{}{}::apply({});\n" if opCodeAsTemplateParam else "{}{}({});\n"

        outFile.write(kernelCallString.format(
            opName if API == "CPP" else (API + "::" + opName),
            # Template parameters, if the kernel is a template:
            "<{}>".format(", ".join(callTemplateParams)) if len(templateValues) else "",
            # Run-time parameters, possibly including DaphneContext:
            ", ".join(callParams + ([] if isCreateDaphneContext else ["ctx"])),
        ))
        if isInstrumented:
            outFile.write(3 * INDENT)
            outFile.write(f"postKernelInstrumentation(kId, ctx);\n")
            outFile.write(2 * INDENT)
            outFile.write(f"}} catch(std::exception &e) {{\n{3*INDENT}throw ErrorHandler::runtimeError(kId, e.what(), &(ctx->dispatchMapping));\n{2*INDENT}}}\n")
        outFile.write(INDENT + "}\n")

        argTypes = [rtp["type"].replace(" **", "").replace(" *", "").replace("const ", "") for rtp in extendedRuntimeParams if not rtp["isOutput"]]
        resTypes = [rtp["type"].replace(" **", "").replace(" *", "").replace("const ", "") for rtp in extendedRuntimeParams if     rtp["isOutput"]]

        argTypesTmp = []
        for t in argTypes:
            # TODO Don't hardcode these exceptions.
            if t in ["void", "mlir::daphne::GroupEnum", "CompareOperation"]:
                break
            argTypesTmp.append(t)
        argTypes = argTypesTmp

        catalogEntries.append({
            "opMnemonic": concreteOpName,
            "kernelFuncName": funcName + typesForName,
            "resTypes": resTypes,
            "argTypes": argTypes,
            "backend": API,
            # Assumes that the generated catalog file is saved in
            # the same directory as the kernels libraries.
            "libPath": ("libAllKernels.dylib" if platform.system() == "Darwin" else "libAllKernels.so") if API == "CPP" else (f"lib{API}Kernels.dylib" if platform.system() == "Darwin" else f"lib{API}Kernels.so")
        })

    # Generate the function(s).
    if opCodes is None:
        generateFunction(None)
    else:
        for opCode in opCodes:
            generateFunction(opCode)
    # outFile.write(INDENT + "\n")


def printHelp():
    print("Usage: python3 {} INPUT_SPEC_FILE OUTPUT_CPP_FILE OUTPUT_CATALOG_FILE API".format(sys.argv[0]))
    print(__doc__)

def getDefaultHeaders() -> str:
    return "#include <stdexcept>\n" \
        "#include <util/ErrorHandler.h>\n" \
        "#include <runtime/local/context/DaphneContext.h>\n" \
        "#include <runtime/local/instrumentation/KernelInstrumentation.h>\n"

def codegenKernelInfos(kernelInfos, catalog_entries) -> Tuple[bool, str]:
    ops_inst_str: str = ""
    header_str: str = getDefaultHeaders()
    didGenerateCode = False

    for kernelInfo in kernelInfos:
        kernelTemplateInfo = kernelInfo["kernelTemplate"]
        if "api" in kernelInfo:
            for api in kernelInfo["api"]:
                for name in api["name"]:
                    # print("Processing API: " + name)
                    # print("  OpName: " + kernelTemplateInfo["opName"])
                    # print("  Instantiations: " + str(api["instantiations"]))
                    # if "opCodes" in api:
                    #     print("  opCodes: " + str(api["opCodes"]))
                    if name == API:
                        # Comment reporting the kernel name.
                        ops_inst_str += INDENT + "// {}\n".format("-" * 76)
                        ops_inst_str += INDENT + "// {}\n".format(kernelTemplateInfo["opName"])
                        ops_inst_str += INDENT + "// {}\n".format("-" * 76)

                        # Include for the required header.
                        if API != "CPP":
                            header_str = header_str + "#include <runtime/local/kernels/{}/{}>\n".format(API, kernelTemplateInfo["header"])
                        else:
                            header_str = header_str + "#include <runtime/local/kernels/{}>\n".format(kernelTemplateInfo["header"])

                        outBuf = io.StringIO()
                        for instantiation in api["instantiations"]:
                            generateKernelInstantiation(kernelTemplateInfo, instantiation,
                                                        api.get("opCodes", None), outBuf, catalog_entries, API)
                        ops_inst_str += outBuf.getvalue()
                        didGenerateCode = True
        else:
            if API == "CPP":
                # Comment reporting the kernel name.
                ops_inst_str += INDENT + "// {}\n".format("-" * 76)
                ops_inst_str += INDENT + "// {}\n".format(kernelTemplateInfo["opName"])
                ops_inst_str += INDENT + "// {}\n".format("-" * 76)

                # Include for the required header.
                header_str = header_str + "#include <runtime/local/kernels/{}>\n".format(kernelTemplateInfo["header"])
                # One function per instantiation of the kernel.
                opCodes = kernelInfo.get("opCodes", None)
                outBuf = io.StringIO()
                for instantiation in kernelInfo["instantiations"]:
                    generateKernelInstantiation(kernelTemplateInfo, instantiation, opCodes, outBuf, catalog_entries, API)
                ops_inst_str += outBuf.getvalue()
                didGenerateCode = True

    file = header_str + '\nextern \"C\" {\n' + ops_inst_str + "}\n"
    return didGenerateCode, file


if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        printHelp()
        sys.exit(0)
    elif len(sys.argv) != 5:
        print("Wrong number of arguments.")
        print()
        printHelp()
        sys.exit(1)

    # Parse arguments.
    inSpecPath = sys.argv[1]
    outCppPath = sys.argv[2]
    outCatalogPath = sys.argv[3]
    API = sys.argv[4]
    cppCodegen: List[str] = []
    catalog_entries = []

    # Load the specification (which kernel template shall be instantiated
    # with which template arguments) from a JSON-file.
    with open(inSpecPath, "r") as inFile:
        kernelInfos = json.load(inFile)

        for kernelInfo in kernelInfos:
            didGenerateCode, file = codegenKernelInfos([kernelInfo], catalog_entries)
            if didGenerateCode:
                cppCodegen.append(file)

    cppFiles = []
    # Write the generated C++ code to separate files.
    for idx, file in enumerate(cppCodegen):
        fileName = outCppPath + "_" + str(idx) + ".cpp"
        cppFiles.append(fileName)
        with open(fileName, "w") as outFile:
            outFile.write(file)

    # Store kernel catalog info as JSON.

    catalogFile = Path(outCatalogPath)
    catalogFile.parent.mkdir(exist_ok=True, parents=True)
    with catalogFile.open('w') as outCatalog:
        print("writing catalog to " + outCatalogPath)
        outCatalog.write(json.dumps(catalog_entries, indent=2))

    #  Lists all generated *.cpp files.
    #  print(cppFiles)
