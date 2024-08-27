import sys
import os
import onnx
import struct
import ctypes
import subprocess
import cupy as cp
from onnxsim import simplify

import onnx
from onnxconverter_common import float16



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

class TADMAInference:
    def __init__(self, onnx_path, workdir='tadma', clang_path='/usr/local/bin/clang++', tadma_headers = '/home/ognian/code/TADMA/include'):
        self.onnx_path = onnx_path
        self.clang_path = clang_path
        self.tadma_headers = tadma_headers
        self.dynamic_dims = []
        self.input_tensor_shapes = {}
        self.loaded_models = []
        self.shape_nodes = []
        self.has_dynamic = False

        if not os.path.exists(tadma_headers):
            print(f"Error: TADMA headers not found in {tadma_headers}")
            sys.exit(1)

        # get onnx filename
        self.filename = onnx_path.split('/')[-1][:-5]

        self.model_dir = workdir + '/' + self.filename
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_so = self.model_dir + '/model.so'
        self.model_bin = self.model_dir + '/model.bin'
        self.model_cu = self.model_dir + '/model.cu'

        self.generated_model = open(self.model_cu, 'w')

        self.__generate_model(onnx.load(onnx_path))

        self.generated_model.close()

        self.weights_library = ctypes.CDLL(self.model_so)
        self.weights_object = self.weights_library.create()



    def process_shape(self, node):
        output_name = process_name(node.output[0])
        input0_name = process_name(node.input[0])

        self.shape_nodes.append(output_name)

        if node.op_type == 'Shape':
            return f'    using {output_name} = typename TYPE({input0_name})::Dims;\n'
        elif node.op_type == 'Unsqueeze':
            return f'    using {output_name} = Sequence<{input0_name}>;\n'
        elif node.op_type == 'Gather':
            return f'    constexpr auto {output_name} = {input0_name}::Values({node.attribute[0].i});\n'
        elif node.op_type == 'Concat':
            input1_name = process_name(node.input[1])
            return f'    using {output_name} = typename {input0_name}::template Merge<{input1_name}>;\n'
        else:
            return f'    auto {output_name} = ...;\n'


    def __generate_model(self, model):
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

    def __get_model(self, dynamic_axes):
        if len(dynamic_axes) == 0:
            return self.loaded_models[0]

        dict_name = ''
        for i in self.dynamic_dims:
            dict_name += f'{i}_{dynamic_axes[i]}_'

        if dict_name in self.loaded_models:
            return self.loaded_models[dict_name]

        model_path = self.model_dir + f'/model_{dict_name}.cu'

        model = open(model_path, 'w')
        model.write('#include "model.cu"\n\n')


        model.write('extern "C" void* infer(Model* model')

        for i in range(len(self.input_tensor_shapes)):
            model.write(f', void* input{i}')

        model.write(') {\n')
        model.write(f'    return model->infer<{list_to_string([dynamic_axes[i] for i in self.dynamic_dims])}>'
                    f'({list_to_string([f'input{i}' for i in range(len(self.input_tensor_shapes))])}).disown_data();\n')

        model.write('}\n')
        model.close()

        cmd = [self.clang_path, '-std=c++23', '-x', 'cu', '-lcudart', '-lcublasLt', f'-I{self.tadma_headers}',
               '-shared', '-fPIC', model_path, '-o', f'{model_path[:-3]}_{str(dict_name)}.so']
        print(f'Running {' '.join(cmd)}')
        subprocess.Popen(cmd).wait()

    def forward(self, inputs: dict):
        dynamic_axes = {}

        for name, tensor in inputs.items():
            assert name in self.input_tensor_shapes
            # Same rank
            assert len(tensor.shape) == len(self.input_tensor_shapes[name])

            required_shape = self.input_tensor_shapes[name]

            for i in range(len(tensor.shape)):
                if type(required_shape[i]) == str:
                    if required_shape[i] not in dynamic_axes:
                        dynamic_axes[required_shape[i]] = tensor.shape[i]
                    else:
                        assert dynamic_axes[required_shape[i]] == tensor.shape[i]
                else:
                    assert tensor.shape[i] == required_shape[i]


        return self.__get_model(dynamic_axes).infer(*inputs)


infer_ctx = TADMAInference('/home/ognian/code/TADMA/tadma/bert-base-uncased/model_fp16_simplified_static.onnx')


infer_ctx.forward({'input_ids': cp.zeros((1, 384), dtype=cp.float32),
                   'attention_mask': cp.zeros((1, 384), dtype=cp.float32)})