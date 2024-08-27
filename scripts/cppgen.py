import struct
import onnxparse


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
    elif type == 'float16':
        result = list(struct.iter_unpack('e', data))
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

def generate_members(model):
    """
    :param model: loaded onnx

    Generate the local variables including weight tensors and constants
    Additionally generate the .bin file containing the model weights
    """

    values_by_name = {}
    file_offset = 0
    #output_bin = open(self.model_bin, 'wb')

    member_list = []

    print("\nInitializers (Weights) in the Model:")
    for initializer in model.graph.initializer:
        if initializer.dims == [] or initializer.dims == [1]:
            type = onnx_type_to_cpp_typestr(initializer.data_type)
            val = raw_data_to_typed_array(initializer.raw_data, type)
            values_by_name[initializer.name] = val

            member_list.append(f'    static constexpr {type} {process_name(initializer.name)} = {val[0]};\n')

        else:

            if len(initializer.dims) == 1 and initializer.dims[0] <= 128:
                type = onnx_type_to_cpp_typestr(initializer.data_type)
                member_list.append(f'    static constexpr Sequence<{list_to_string(raw_data_to_typed_array(initializer.raw_data, type))}> {process_name(initializer.name)};\n')
            else:
                dimstr = f'Sequence<{list_to_string([d for d in initializer.dims])}>'
                type = onnx_type_to_cpp_typestr(initializer.data_type)

                #output_bin.write(initializer.raw_data)
                member_list.append(f'    Tensor<{type}, ALLOCATOR, {dimstr}> {process_name(initializer.name)} = '
                                   f'Tensor<{type}, ALLOCATOR, {dimstr}>(bin, {file_offset}ULL);\n')
                #print(f"Initializer Name: {initializer.dims}")

                file_offset += len(initializer.raw_data)

    for node in model.graph.node:
        if node.op_type == 'Constant' or node.op_type == 'ConstantOfShape':
            arr = constant_node_to_array(node)

            if len(arr) == 1:
                init = arr[0]
                type = 'static constexpr auto'
            else:
                #dtype = onnx_type_to_cpp_typestr(node.attribute[0].t.data_type)
                #type = f'Tensor<{dtype}, ALLOCATOR, Sequence<{list_to_string(node.attribute[0].t.dims)}>>'
                #init = f'{type}(std::array<{dtype}, {len(arr)}> {{{list_to_string(arr)}}})'

                sequence_name = f'{process_name(node.output[0])}_sequence'
                member_list.append(f'    using {sequence_name} = Sequence<{list_to_string(arr)}>;\n')
                init = '{}'


                type = (f'Tensor<TYPE({sequence_name}::Array()[0]), ConstexprAllocator<{sequence_name}>, Sequence<{len(arr)}>>')


            member_list.append(f'    {type} {process_name(node.output[0])} = {init};\n')

    return member_list

def generate_inference_function(model : onnxparse.ParsedModel):
    statements = []

    # Generate declaration
    has_dynamic = len(model.dynamic_axes) > 0
    if has_dynamic:
        template = '\n    template<'
        first = True
        for axis in model.dynamic_axes:
            if not first:
                template += ', '
            template += f'int64_t {axis}'
            first = False
        template += '>\n'
        statements.append(template)

    inputs = []
    for input in model.simplified.graph.input:
        tmp = input.type.tensor_type
        type = onnx_type_to_cpp_typestr(tmp.elem_type)
        shape = []

        for dim in tmp.shape.dim:
            if dim.HasField('dim_param'):
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)

        inputs.append(f'Tensor<{type}, ALLOCATOR, Sequence<{list_to_string(shape)}>> {process_name(input.name)}')
    statements.append('    auto infer(' + ', '.join(inputs) + ') {\n')

    num_references = {}
    nodes_by_name = {}

    # Count references for each node input
    model = model.simplified
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
        # Constant nodes are already generated as members
        if node.op_type == 'Constant' or node.op_type == 'ConstantOfShape':
            continue


        input_refs = [num_references[i] for i in node.input]
        can_inplace_0 = 'false' if len(input_refs) == 0 else 'true' if input_refs[0] == 1 else 'false'

        type = 'auto'
        init = ''

        def get_second_arg():
            if process_name(node.output[0]) == 'bert_embeddings_word_embeddings_weight_transposed':
                print(node)
            if len(node.input) > 1:
                return f'{process_name(node.input[1])}'
            if len(node.attribute) > 0:
                return node.attribute[0].ints

            return []

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
        elif node.op_type == 'Transpose': init = f'{process_name(node.input[0])}.template transpose<{list_to_string(get_second_arg())}>()'
        elif node.op_type == 'Slice':
            init = f'{process_name(node.input[0])}.template slice<'
            for i in range(1, len(node.input)):
                init += process_name(node.input[i]) + ','
            init = init[:-1] + '>()'
        elif node.op_type == 'Expand': init = f'{process_name(node.input[0])}.template broadcastTo<TYPE({process_name(node.input[1])})>()'
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
            init = f'{process_name(node.input[0])}.template mean<{node.attribute[0].i if len(node.attribute) > 0 else process_name(node.input[1])}>()'

        elif node.op_type == 'Squeeze':
            init = f'{process_name(node.input[0])}.template squeeze<>()'

        elif node.op_type == 'Unsqueeze':
            init = f'unsqueeze<{process_name(node.input[1])}>({process_name(node.input[0])})'

        elif node.op_type == 'Split':
            init = f'{process_name(node.input[0])}.template split<{node.attribute[0].i}>()'

        elif node.op_type == 'Concat':
            init = f'concat<{node.attribute[0].i}>({", ".join([process_name(i) for i in node.input])})'
        elif node.op_type == 'Shape':
            init = f'shape({process_name(node.input[0])})'


        if len(node.output) > 1:
            statements.append(f'    auto [{", ".join([process_name(o) for o in node.output])}] = {init};\n')
        else:
            statements.append(f'    {type} {process_name(node.output[0])} = {init};\n')

        #assert(len(node.output) <= 1)


    if len(model.graph.output) > 1:
        statements.append('    return std::make_tuple(' + ', '.join([process_name(o.name) for o in model.graph.output]) + ');\n')
    else:
        statements.append(f'    return {process_name(model.graph.output[0].name)};\n}}')

    return statements
