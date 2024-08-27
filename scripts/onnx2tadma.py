import onnx
import sys
import os
import struct

if len(sys.argv) != 2:
    print("Usage: python onnx2tadma.py <onnx_model_path> <output_file>")
    #sys.exit(1)
input_path = sys.argv[1]
output_path = sys.argv[2]

bin_path = output_path + '.bin'

output = open(output_path + '.cu', "w")
output_bin = open(bin_path, "wb")

output.write('#include "include/tadma/Tadma.hpp"\n\n')
output.write('using namespace tadma;\n')
output.write('using ALLOCATOR = Allocator<kCUDA>;\n\n')
output.write('struct Model {\n')
output.write('    std::ifstream bin;\n\n')


def onnx_type_to_cpp_typestr(type):
    # from https://onnx.ai/onnx/intro/concepts.html
    if type == 0:
        return 'float'
    elif type == 1:
        return 'float'
    elif type == 2:
        return 'uint8_t'
    elif type == 3:
        return 'int8_t'
    elif type == 4:
        return 'uint16_t'
    elif type == 5:
        return 'int16_t'
    elif type == 6:
        return 'int32_t'
    elif type == 7:
        return 'int64_t'
    elif type == 9:
        return 'bool'
    elif type == 10:
        return 'float16'
    elif type == 11:
        return 'double'
    elif type == 12:
        return 'uint32_t'
    elif type == 13:
        return 'uint64_t'
    elif type == 14:
        return 'std::complex<float>'
    elif type == 15:
        return 'std::complex<double>'
    #elif type == 16:
    #    return 'bfloat16'
    else:
        assert(False)

def raw_data_to_typed_array(data, type):
    result = []
    if type == 'float':
        result = list(struct.iter_unpack('f', data))
    elif type == 'int64_t':
        result = struct.iter_unpack('q', data)
    else:
        print(f"Unsupported type {type}")

    return [x[0] for x in result]

def constant_node_to_array(node):
    #assert(node.op_type == 'Constant' or node.op_type == 'ConstantOfShape')
    if hasattr(node, 'data_type') and hasattr(node, 'raw_data'):
        return raw_data_to_typed_array(node.raw_data, onnx_type_to_cpp_typestr(node.data_type))
    return raw_data_to_typed_array(node.attribute[0].t.raw_data, onnx_type_to_cpp_typestr(node.attribute[0].t.data_type))


def process_name(name):
    # make the symbols valid C++ variable names
    name = name.replace('.', '_').replace('::','_').replace('/','_')

    if name[0].isnumeric():
        name = '_' + name

    return name


def list_to_string(list):
    s = ''
    for i in list:
        s += str(i) + ','
    return s[:-1]


members = []
statements = []

def print_onnx_model_structure(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    global output

    num_references = {}
    nodes_by_name = {}
    values_by_name = {}

    file_offset = 0

    print("\nInitializers (Weights) in the Model:")
    for initializer in model.graph.initializer:
        if initializer.dims == []:
            type = onnx_type_to_cpp_typestr(initializer.data_type)
            val = raw_data_to_typed_array(initializer.raw_data, type)
            values_by_name[initializer.name] = val

            output.write(f'    static constexpr {type} {process_name(initializer.name)} = {val[0]};\n')

        else:
            dimstr = f'Sequence<{list_to_string([d for d in initializer.dims])}>'
            type = onnx_type_to_cpp_typestr(initializer.data_type)

            output_bin.write(initializer.raw_data)
            output.write(f'    Tensor<{type}, ALLOCATOR, {dimstr}> {process_name(initializer.name)} = Tensor<{type}, ALLOCATOR, {dimstr}>(bin, {file_offset}ULL);\n')
            #print(f"Initializer Name: {initializer.dims}")

            file_offset += len(initializer.raw_data)


    output.write('\nModel(const std::string& weights) : bin(weights) {}\n\n')
    output.write('auto forward(')

    inputs = []

    for input in model.graph.input:
        tmp = input.type.tensor_type
        type = onnx_type_to_cpp_typestr(tmp.elem_type)
        dimstr = 'Sequence<' + ''.join([str(d.dim_value) + ',' for d in tmp.shape.dim])

        dimstr = dimstr[:-1] + '>'
        inputs.append(f'Tensor<{type}, ALLOCATOR, {dimstr}> {process_name(input.name)}')

    output.write(', '.join(inputs) + ') {\n\n')

    for node in model.graph.node:
        for input in node.input:
            if input in num_references:
                num_references[input] += 1
            else:
                num_references[input] = 1


        for out in node.output:
            if out in nodes_by_name:
                print(f"Duplicate node name: {out}")
            nodes_by_name[out] = node

    for node in model.graph.initializer:
        nodes_by_name[node.name] = node

    print("\nNodes in the Model:")
    for node in model.graph.node:

        print (node)
        input_refs = [num_references[i] for i in node.input]
        can_inplace_0 = 'false' if len(input_refs) == 0 else 'true' if input_refs[0] == 1 else 'false'

        type = 'auto'
        init = ''

        if node.op_type == 'Constant' or node.op_type == 'ConstantOfShape':
            #init = f'Tensor<{onnx_type_to_cpp_typestr(node.attribute[0].t.data_type)}, ALLOCATOR, Sequence<{node.attribute[0].t.dims}>>({node.attribute[0].t.raw_data})'
            print(f'Constant dims: {node.attribute[0].t.dims}')

            arr = constant_node_to_array(node)

            if len(arr) == 1:
                init = arr[0]
                type = 'constexpr auto'
            else:
                type_ = onnx_type_to_cpp_typestr(node.attribute[0].t.data_type)
                init = f'Tensor<{type_}, ALLOCATOR, Sequence<{list_to_string(node.attribute[0].t.dims)}>>(std::array<{type_}, {len(arr)}> {{{list_to_string(arr)}}})'
        elif node.op_type == 'Sqrt': init = f'sqrt<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Add': init = f'{process_name(node.input[0])} + {process_name(node.input[1])}'
        elif node.op_type == 'Sub': init = f'{process_name(node.input[0])} - {process_name(node.input[1])}'
        elif node.op_type == 'Mul': init = f'{process_name(node.input[0])} * {process_name(node.input[1])}'
        elif node.op_type == 'Div': init = f'{process_name(node.input[0])} / {process_name(node.input[1])}'
        elif node.op_type == 'Exp': init = f'exp<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Log': init = f'log<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Erf': init = f'erf<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Softmax': init = f'softmax<{-1}, {can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Tanh': init = f'tanh<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'MatMul': init = f'matmul<1>({process_name(node.input[0])}, {process_name(node.input[1])})'
        elif node.op_type == 'Gemm': init = f'matmul<{node.attribute[0].f}, {node.attribute[1].f}>({process_name(node.input[0])}, {process_name(node.input[1])}, {process_name(node.input[2])})'
        elif node.op_type == 'Transpose': init = f'{process_name(node.input[0])}.transpose<{list_to_string(node.attribute[0].ints)}>()'
        elif node.op_type == 'Slice': init = f'{process_name(node.input[0])}.slice<{process_name(node.input[1])}, {process_name(node.input[2])}>()'
        elif node.op_type == 'Cast': init = f'{process_name(node.input[0])}.to<{onnx_type_to_cpp_typestr(node.attribute[0].type)}>()'
        elif node.op_type == 'Equal': init = f'{process_name(node.input[0])} == {process_name(node.input[1])}'
        elif node.op_type == 'LayerNormalization':
            init = f'layer_norm<{node.attribute[0].i}>({process_name(node.input[0])}, {process_name(node.input[1])}, {process_name(node.input[2])}, {node.attribute[1].f})'
        elif node.op_type == 'Reshape':
            print(node)
            tensor = node.input[0]
            shape = []

            if node.input[1] not in values_by_name:
                print(nodes_by_name[node.input[1]])
                shape = list_to_string(constant_node_to_array(nodes_by_name[node.input[1]]))
            else: shape = [values_by_name[node.input[1]]]

            init = f'{process_name(tensor)}.reshape<{shape}>()'
        elif node.op_type == 'Where':
            init = f'where({process_name(node.input[0])}, {process_name(node.input[1])}, {process_name(node.input[2])})'
        elif node.op_type == 'Shape':
            type = 'using'
            init = f'typename decltype({process_name(node.input[0])})::Dims'
        elif node.op_type == 'Gather':
            init = f'gather({process_name(node.input[0])}, {process_name(node.input[1])})'


        statements.append(f'    {type} {process_name(node.output[0])} = {init};\n')

        #assert(len(node.output) <= 1)



    if len(model.graph.output) > 1:
        statements.append('    return std::make_tuple(' + ', '.join([process_name(o.name) for o in model.graph.output]) + ');\n')
    else:
        statements.append(f'    return {process_name(model.graph.output[0].name)};\n')

    statements.append('}\n};\n\n')
    output.writelines(statements)

print_onnx_model_structure(input_path)



def __generate_model0(self, model):
    if os.path.exists(self.model_so):
        os.remove(self.model_so)

    if os.path.exists(self.model_bin):
        os.remove(self.model_bin)

    inputs = []
    statements = []
    num_references = {}
    nodes_by_name = {}

    for input in model.graph.input:
        tmp = input.type.tensor_type
        type = onnx_type_to_cpp_typestr(tmp.elem_type)
        shape = []

        for dim in tmp.shape.dim:
            if dim.HasField('dim_param'):
                shape.append(dim.dim_param)
                self.has_dynamic = True
                if dim.dim_param not in self.dynamic_dims:
                    self.dynamic_dims.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)

        inputs.append(f'Tensor<{type}, ALLOCATOR, Sequence<{list_to_string(shape)}>> {process_name(input.name)}')
        self.input_tensor_shapes[input.name] = shape

    self.generated_model.writelines([
        '#include "tadma/Tadma.hpp"\n\n',
        'using namespace tadma;\n',
        'using ALLOCATOR = Allocator<kCUDA>;\n\n',
        'struct Model {\n',
        '    std::ifstream bin;\n\n',
        '    Model(const std::string& weights) : bin(weights) { bin.close(); }\n\n'
    ])
    self.__generate_members(model)

    if self.has_dynamic:
        self.generated_model.write(f'\n    template<')
        first = True
        for dim in self.dynamic_dims:
            if not first: self.generated_model.write(', ')
            self.generated_model.write(f'int64_t {dim}')
            first = False
        self.generated_model.write('>\n')

    self.generated_model.write(f'    auto infer({', '.join(inputs)}) {{\n')

    for node in model.graph.node:
        for input in node.input:
            if input in num_references:
                num_references[input] += 1
            else:
                num_references[input] = 1

        for out in node.output:
            if out in nodes_by_name:
                print(f"Duplicate node name: {out}")
            nodes_by_name[out] = node

    for node in model.graph.initializer:
        nodes_by_name[node.name] = node

    print("\nNodes in the Model:")
    for node in model.graph.node:
        if node.op_type == 'Constant' or node.op_type == 'ConstantOfShape':
            continue

        if process_name(node.input[0]) in self.shape_nodes or node.op_type == 'Shape':
            statements.append(self.process_shape(node))
            continue

        input_refs = [num_references[i] for i in node.input]
        can_inplace_0 = 'false' if len(input_refs) == 0 else 'true' if input_refs[0] == 1 else 'false'

        type = 'auto'
        init = ''

        if node.op_type == 'Sqrt': init = f'sqrt<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Add': init = f'{process_name(node.input[0])} + {process_name(node.input[1])}'
        elif node.op_type == 'Sub': init = f'{process_name(node.input[0])} - {process_name(node.input[1])}'
        elif node.op_type == 'Mul': init = f'{process_name(node.input[0])} * {process_name(node.input[1])}'
        elif node.op_type == 'Div': init = f'{process_name(node.input[0])} / {process_name(node.input[1])}'
        elif node.op_type == 'Exp': init = f'exp<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Log': init = f'log<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Pow': init = f'pow<{can_inplace_0}>({process_name(node.input[0])}, {process_name(node.input[1])})'
        elif node.op_type == 'Erf': init = f'erf<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Softmax': init = f'softmax<{-1}, {can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Tanh': init = f'tanh<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'Sigmoid': init = f'sigmoid<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'MatMul': init = f'matmul<1>({process_name(node.input[0])}, {process_name(node.input[1])})'
        elif node.op_type == 'Gemm': init = f'matmul<{node.attribute[0].f}, {node.attribute[1].f}>({process_name(node.input[0])}, {process_name(node.input[1])}, {process_name(node.input[2])})'
        elif node.op_type == 'Transpose': init = f'{process_name(node.input[0])}.template transpose<{list_to_string(node.attribute[0].ints)}>()'
        elif node.op_type == 'Slice': init = f'{process_name(node.input[0])}.template slice<{process_name(node.input[1])}, {process_name(node.input[2])}>()'
        elif node.op_type == 'Expand': init = f'{process_name(node.input[0])}.template broadcastTo<{process_name(node.input[1])}>()'
        elif node.op_type == 'Cast': init = f'{process_name(node.input[0])}.template to<{onnx_type_to_cpp_typestr(node.attribute[0].type)}>()'
        elif node.op_type == 'Equal': init = f'{process_name(node.input[0])} == {process_name(node.input[1])}'
        elif node.op_type == 'Neg': init = f'negate<{can_inplace_0}>({process_name(node.input[0])})'
        elif node.op_type == 'LayerNormalization':
            init = f'layer_norm<{node.attribute[0].i}>({process_name(node.input[0])}, {process_name(node.input[1])}, {process_name(node.input[2])}, {node.attribute[1].f})'
        elif node.op_type == 'Reshape':
            tensor = node.input[0]

            #if node.input[1] not in values_by_name:
            #    print(nodes_by_name[node.input[1]])
            #    shape = list_to_string(constant_node_to_array(nodes_by_name[node.input[1]]))
            #else: shape = [values_by_name[node.input[1]]]

            init = f'{process_name(tensor)}.template reshape<{process_name(node.input[1])}>()'
        elif node.op_type == 'Where':
            init = f'where({process_name(node.input[0])}, {process_name(node.input[1])}, {process_name(node.input[2])})'
        elif node.op_type == 'Gather':
            init = f'gather({process_name(node.input[0])}, {process_name(node.input[1])})'

        elif node.op_type == 'ReduceMean':
            init = f'{process_name(node.input[0])}.template mean<>()'

        elif node.op_type == 'Squeeze':
            init = f'{process_name(node.input[0])}.template squeeze<>()'

        elif node.op_type == 'Unsqueeze':
            init = f'{process_name(node.input[0])}.template unsqueeze<>()'

        elif node.op_type == 'Split':
            init = f'{process_name(node.input[0])}.template split<{node.attribute[0].i}>()'

        elif node.op_type == 'Concat':
            init = f'concat<{node.attribute[0].i}>({", ".join([process_name(i) for i in node.input])})'


        if len(node.output) > 1:
            statements.append(f'    auto [{", ".join([process_name(o) for o in node.output])}] = {init};\n')
        else:
            statements.append(f'    {type} {process_name(node.output[0])} = {init};\n')

        #assert(len(node.output) <= 1)


    if len(model.graph.output) > 1:
        statements.append('    return std::make_tuple(' + ', '.join([process_name(o.name) for o in model.graph.output]) + ');\n')
    else:
        statements.append(f'    return {process_name(model.graph.output[0].name)};\n')

    self.generated_model.writelines(statements)
    self.generated_model.write('}\n')


    self.generated_model.write('\n};\n')
    self.generated_model.write('extern "C" Model* create() { return new Model("weights.bin"); }')



    cmd = [self.clang_path, '-std=c++23', '-x', 'cu', '-lcudart', '-lcublasLt', f'-I{self.tadma_headers}',
           '-shared', '-fPIC', self.model_cu, '-o', self.model_so]
    print(f'Running {' '.join(cmd)}')
    subprocess.Popen(cmd, stdout=sys.stdout).wait()
