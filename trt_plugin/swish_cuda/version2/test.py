'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-06-23 22:10:19
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-06-23 23:16:18
FilePath: /mobilenet/swish_cuda/version2/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np

def _load_tensor(file):     
    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."
    
    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    else:
        assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"
        
    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)


def load_tensor(file):
    if file.endswith("npz"):
        return np.load(file)['data']
    elif file.endswith("npy"):
        return np.load(file)
    else:
        return _load_tensor(file)

def swish(x):
    return x / (1 + np.exp(-x))

def test():
    q_tensor = load_tensor('input_tensor.npz')
    out_tensor = load_tensor('out_tensor.npz')

    out = swish(q_tensor)
    
    # B,S,seq_len,seq_len
    print(np.abs(out - out_tensor).max())


if __name__ == "__main__":
    test()
