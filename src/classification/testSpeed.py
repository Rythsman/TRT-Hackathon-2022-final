from cuda import cudart
import tensorrt as trt
import ctypes
import numpy as np
import os
import torch
from glob import glob
from time import time_ns
import sys
sys.path.insert(0, "../ml-cvnets")
from cvnets import get_model
from options.opts import get_eval_arguments
from utils.common_utils import device_setup, create_directories
import argparse

planFilePath = './'
soFileList = glob(planFilePath + "*.so")
logFile = './log/log.txt'
pt_filePath = '../ml-cvnets/pretrained_models/mobilevit_s.pt'

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

for plugin_creator in PLUGIN_CREATORS:
    print(plugin_creator.name, plugin_creator.plugin_namespace, plugin_creator.plugin_version)


def deserialize_engine(trtFile):
    with open(trtFile, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    return engine

tableHead = \
"""
bs: Batch Size
sl: Sequence Length
lt: Latency (ms)
tp: throughput (word/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
----+----+--------+---------+---------+---------+-------------
  bs|  sl|      lt|       tp|       a0|       r0| output check
----+----+--------+---------+---------+---------+-------------
"""

def check(a, b, weak=False, epsilon = 1e-5):
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    diff0 = np.max(np.abs(a - b))
    diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    return res,diff0,diff1

@torch.no_grad()
def test_engine_speed(trt_engine, torch_model):
    
    f = open(logFile, 'w+')
    nInput = np.sum([ trt_engine.binding_is_input(i) for i in range(trt_engine.num_bindings) ])
    nOutput = trt_engine.num_bindings - nInput
    context = trt_engine.create_execution_context()
    ori_device = next(torch_model.parameters()).device
    
    dst_dir = "../ml-cvnets/npz"
    batch_sizes = [1, 16, 32, 64, 128]
    # batch_sizes = [1]
    print(tableHead)
    torch_model.to('cpu')
    for b in batch_sizes:
        data_filepath = os.path.join(dst_dir, '%d-3-256-256.npz' % b)
        data = torch.as_tensor(np.load(data_filepath)['data'])
        real_output = torch_model(data.cpu()).cpu().detach().numpy()
        
        data_numpy = data.numpy()
        context.set_binding_shape(0, data_numpy.shape)
        
        bufferH = []
        bufferH.append( data_numpy.astype(np.float32).reshape(-1) )

        for i in range(nInput, nInput + nOutput):                
            bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(trt_engine.get_binding_dtype(i))) )

        bufferD = []
        for i in range(nInput + nOutput):                
            bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        context.execute_v2(bufferD)

        for i in range(nInput, nInput + nOutput):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # warm up
        for i in range(10):
            context.execute_v2(bufferD)

        # test infernece time
        t0 = time_ns()
        for i in range(30):
            context.execute_v2(bufferD)
        t1 = time_ns()
        timePerInference = (t1-t0)/1000/1000/30

        indexOutput = trt_engine.get_binding_index('output')
        
        check0 = check(bufferH[indexOutput], real_output, True, 1e-3)

        batchSize = data.shape[0]
        sequenceLength = 1

        string = "%4d,%4d,%8.3f,%9.3e,%9.3e,%9.3e, %s"%(batchSize,
                                                                    sequenceLength,
                                                                    timePerInference,
                                                                    batchSize*sequenceLength/timePerInference*1000,
                                                                    check0[1],
                                                                    check0[2],
                                                                    "Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 else "Bad")

        
        print(string)
        f.write(string + '\n')

        for i in range(nInput + nOutput):                
            cudart.cudaFree(bufferD[i])
    f.close()
    torch_model.to(ori_device)

def main(opts, **kwargs):
    device = torch.device('cpu')
    model = get_model(opts)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(pt_filePath, map_location='cpu'))
    engine_filepath = opts.engine_filepath
    test_engine_speed(deserialize_engine(engine_filepath), model)

def main_worker(**kwargs):
    parser = argparse.ArgumentParser(description='Parser', add_help=True)
    parser.add_argument('--engine_filepath', type=str, default='', required=True, help="engine filepath")
    parser.add_argument('--common.config-file', type=str, default='', required=True, help="config file")
    
    config = parser.parse_args()
    
    start_index = None
    for i,item in enumerate(sys.argv):
        item = item.strip()
        if item == "--engine_filepath":
            start_index = i
            break
    next_index = start_index + 1
    if next_index < len(sys.argv):
        if not sys.argv[next_index].startswith('-'):
            del sys.argv[next_index]
            del sys.argv[start_index]
        else:
            del sys.argv[start_index]
    else:
        del sys.argv[start_index]
    
    opts = get_eval_arguments()
    opts = device_setup(opts)
    setattr(opts, 'engine_filepath', config.engine_filepath)

    main(opts=opts, **kwargs)

if __name__ == "__main__":
    main_worker()