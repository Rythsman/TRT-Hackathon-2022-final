
import os
import numpy as np
from cuda import cudart
import tensorrt as trt
import torch.nn.functional as F
import torch
import ctypes

soFile = "./swishPlugin.so"
trtFile = "./swishTest.plan"
epsilon = 1.0e-2
np.random.seed(97)


num_heads = 3
batch_size = 7
seq_len = 136
embedding_dim = 42

def swish(x):
    return x / (1 + np.exp(-x))

def getSwishPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'Swish' and c.plugin_version == "1":
            parameterList = []
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

use_fp16 = False

def run():

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')

    ctypes.cdll.LoadLibrary(soFile)

    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        
        config.max_workspace_size = 8 << 30
        config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32)) # 关闭TF32
        if builder.platform_has_fast_fp16 and use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if use_fp16:
            input = network.add_input('input', trt.DataType.HALF, [num_heads, batch_size, seq_len, embedding_dim])
        else:
            input = network.add_input('input', trt.DataType.FLOAT, [num_heads, batch_size, seq_len, embedding_dim])
        # profile.set_shape(input.name, [seq_len, 128, num_heads, 3, embedding_dim], [seq_len, 256, num_heads, 3, embedding_dim], [seq_len, 512, num_heads, 3, embedding_dim])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([input], getSwishPlugin())
        
        # pluginLayer.
        network.mark_output(pluginLayer.get_output(0))
        if use_fp16:
            pluginLayer.precision = trt.float16
            pluginLayer.set_output_type(0, trt.float16)
            network.get_output(0).dtype = trt.float16
        print('type', network.get_output(0).dtype)
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()

    shape = (num_heads, batch_size, seq_len, embedding_dim)

    context.set_binding_shape(0, shape)
    # context.set_binding_shape(1, [batch_size, seq_len, seq_len])

    _, stream = cudart.cudaStreamCreate()

    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput

    bufferH = []
    data_type = np.float16 if use_fp16 else np.float32
    data = np.random.rand(np.prod(shape)).astype(data_type).reshape(shape)
    print("min, max:", data.min(), data.max())

    bufferH.append(data)

    print('nOutput:', nOutput)
    for i in range(nOutput):
        print('context.get_binding_shape(nInput + %d)' % i, context.get_binding_shape(nInput + i))
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMallocAsync(bufferH[i].nbytes, stream)[1])

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    context.execute_async_v2(bufferD, stream)

    for i in range(nOutput):
        cudart.cudaMemcpyAsync(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaStreamSynchronize(stream)

    output = bufferH[-1]
    print(output.shape)

    ret = swish(data)

    diff = abs(ret - output)
    print("diff max: ", diff.max())
    print("diff sum: ", diff.sum())

    cudart.cudaStreamDestroy(stream)
    for buffer in bufferD:
        cudart.cudaFree(buffer)
    # print("Test", testCase, "finish!")

if __name__ == '__main__':
    os.system('rm ' + trtFile)
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run()

    print("test finish!")