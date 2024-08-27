import os
import random

import onnx
import onnxparse
import cppgen
import onnxoptimizer as optimizer
import onnxsim

import onnx

# Load the ONNX model
model_path = '../tadma/bert-base-uncased/model.onnx'

p = onnxparse.ParsedModel(model_path)

output = open('generated.cu', 'w')

output.writelines([
    '#include "tadma/Tadma.hpp"\n\n',
    'using namespace tadma;\n',
    'using ALLOCATOR = Allocator<kCUDA>;\n\n',
    'struct Model {\n',
    '    std::ifstream bin;\n\n',
    '    Model(const std::string& weights) : bin(weights) { bin.close(); }\n\n'
] + cppgen.generate_members(p.simplified) + cppgen.generate_inference_function(p) + [
    '};\n\n'
])

