import onnx
import onnxsim
import onnxoptimizer as optimizer
def get_next_prime(n):
    n = n + 1
    if n % 2 == 0:
        n += 1
    for i in range(n, 2 * n):
        for j in range(2, i):
            if i % j == 0:
                break
        else:
            return i

def get_op_shapes(model):
    shapes = {}
    for node in model.graph.node:
        for output in node.output:
            value_info = next((vi for vi in model.graph.value_info if vi.name == output), None)
            if value_info:
                shape = [dim.dim_value if dim.HasField('dim_value') else dim.dim_param for dim in value_info.type.tensor_type.shape.dim]
                shapes[output] = shape
    return shapes

"""
    ParsedModel class is used to parse the ONNX model and extract the shapes of the input and output tensors.
    importantly the inferred shapes can be placeholder values like batch_size, sequence_length etc.
"""
class ParsedModel:
    def __init__(self, model_path):
        model = onnx.load(model_path)
        graph = model.graph
        node = graph.node
        input = graph.input
        output = graph.output
        self.shape_dict = {}
        self.input_shapes = {}
        self.dynamic_axes = []
        self.inferred_shapes = {}
        self.inferred_constants = {}

        def get_input_shapes(axes_replacements):
            return {name: [axes_replacements[i] for i in axes] for name, axes in self.input_shapes.items()}

        for i in input:
            shapes = []
            for dim in i.type.tensor_type.shape.dim:
                if dim.HasField('dim_param'):
                    shapes.append(dim.dim_param)
                    if dim.dim_param not in self.dynamic_axes:
                        self.dynamic_axes.append(dim.dim_param)
                else:
                    shapes.append(dim.dim_value)
            self.input_shapes[i.name] = shapes
        prime = 13

        simplified1_axes = {}
        simplified2_axes = {}

        for i in self.dynamic_axes:
            simplified1_axes[i] = prime
            prime = get_next_prime(prime)
            simplified2_axes[i] = prime
            prime = get_next_prime(prime)

        #g1_shapes = get_input_shapes(simplified1_axes)
        #g2_shapes = get_input_shapes(simplified2_axes)

        #model = onnxsim.simplify(model)[0]

        self.simplified = model
