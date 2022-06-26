
import torch
import multiprocessing
import sys
import os
import argparse


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import onnx
import numpy as np

from cuda import cudart
import tensorrt as trt
import onnx_graphsurgeon as gs

from glob import glob
import ctypes


trt_logger = trt.Logger(trt.Logger.VERBOSE)
epsilon = 1.0e-2
np.random.seed(97)
eps = 1e-6
use_fp16 = False

def GiB(val):
    return val * 1 << 30
trtFile = "./mobilevit_s.plan"


def buildTrtEngine(max_batch_size, onnx_filepath, trtFile, use_fp16):
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        print("Succeeded loading engine!")
    else:
        with trt.Builder(trt_logger) as builder, builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            ) as network, trt.OnnxParser(network, trt_logger) as parser:
            
            builder.max_batch_size = max_batch_size
            profile = builder.create_optimization_profile()
            profile.set_shape('input', min=(1, 3, 256, 256), opt=(max_batch_size, 3, 256, 256), max=(max_batch_size, 3, 256, 256))

            config = builder.create_builder_config()
            config.max_workspace_size = GiB(18) # 8G
            with open(onnx_filepath, 'rb') as model:
                parser.parse(model.read())
                for i in range(parser.num_errors):
                    print(parser.get_error(i))

            config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
            if builder.platform_has_fast_fp16 and use_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                network.get_output(0).dtype = trt.float16

            config.add_optimization_profile(profile)
            engineString = builder.build_serialized_network(network, config)
            if engineString == None:
                print("Failed building engine!")
                return
            print("Succeeded building engine!")
            with open(trtFile, 'wb') as f:
                f.write(engineString)
            engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engineString)
    return engine



def export_engine():
    parser = argparse.ArgumentParser(description='Training arguments', add_help=True)
    parser.add_argument('--engine_filepath', type=str, default='', required=True, help="engine filepath")
    parser.add_argument('--onnx_filepath', type=str, default='', required=True, help="onnx filepath")
    parser.add_argument('--use_fp16', default=False, action="store_true", help="use fp16")
    parser.add_argument('--max_batch_size', default=128, type=int, help="max batch size")
    config = parser.parse_args()

    use_fp16 = config.use_fp16
    engine_filepath = config.engine_filepath
    onnx_filepath = config.onnx_filepath
    max_batch_size = config.max_batch_size
    buildTrtEngine(max_batch_size, onnx_filepath, engine_filepath, use_fp16)

if __name__ == "__main__":
    export_engine()
