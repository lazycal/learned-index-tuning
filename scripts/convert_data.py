import struct
import sys
import numpy as np

# usage: python convert_data.py <input path> <output path> <structure> <L0 parameters>+
# format: 
# <VERSION_NUM:int64> <n:int64 /*number of layers*/> <layer_1_type:int64> ... <layer_n_type:int64>
# <m1:int64 /*number of models in layer 1*/> ... <mn:int64 /*number of models in layer n*/>
# <layer_1_param_array> <layer_2_param_array> <layer_3_param_array>
VERSION_NUM = 0
layer_type_ID = {
    "linear": 0,
    "loglinear": 1,
    "cubic": 2,
    "radix": 3,
}

def get_num_models(layer_type, n, is_last):
    if layer_type == layer_type_ID["linear"]:
        n_params = 2
    elif layer_type == layer_type_ID["loglinear"]:
        n_params = 2
    elif layer_type == layer_type_ID["cubic"]:
        n_params = 4
    elif layer_type == layer_type_ID["radix"]:
        raise NotImplementedError
    if is_last: n_params += 1
    assert n % (n_params * 8) == 0
    return n // (n_params * 8)

inp_path = sys.argv[1]
out_path = sys.argv[2]
layer_types = list(map(lambda name: layer_type_ID[name], sys.argv[3].split(',')))
n_layers = len(layer_types)
assert n_layers == 2
L0_params = np.array(list(map(float, sys.argv[4:])), dtype='<f8')
print('L0_params=', L0_params)
with open(inp_path, 'rb') as fin:
    data = fin.read()
with open(out_path, 'wb') as fout:
    fout.write(struct.pack('<Q', VERSION_NUM))
    fout.write(struct.pack('<Q', n_layers))
    for layer_type in layer_types:
        fout.write(struct.pack('<Q', layer_type))
    #m1 ... mn
    fout.write(struct.pack('<Q', 1))
    fout.write(struct.pack('<Q', get_num_models(layer_type, len(data), True)))
    fout.write(L0_params.tobytes())
    fout.write(data)