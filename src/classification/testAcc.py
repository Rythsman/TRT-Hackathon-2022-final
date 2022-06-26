import os
import sys
import torch
import argparse
import tensorrt as trt
from cuda import cudart
import numpy as np
sys.path.insert(0, os.path.abspath('../ml-cvnets'))
from data import create_eval_loader
from functools import partial
from options.opts import get_eval_arguments
from utils.common_utils import device_setup

trt_logger = trt.Logger(trt.Logger.VERBOSE)

def test_acc(val_loader, trt_model):
    
    cuda_device = torch.device('cuda')
    trt_correct = 0

    for batch_id, batch in enumerate(val_loader):
        input_img, target_label = batch['image'], batch['label']

        # move data to device
        input_img = input_img.to(cuda_device)
        target_label = target_label.to(cuda_device)

        trt_pred_label = trt_model(input_img)
        trt_pred = trt_pred_label.argmax(dim=1, keepdim=True)
        trt_correct += trt_pred.eq(target_label.view_as(trt_pred)).sum().item()

    print('\nTest set: TensorRT Accuracy: {:.3f}% \n'.format(
        100. * trt_correct / len(val_loader.dataset)
    ))


def trt_model(torch_data, engine):

    data = torch_data.cpu().numpy()
    if hasattr(trt_model, 'context'):
        context = getattr(trt_model, 'context')
    else:
        context = engine.create_execution_context()
        setattr(trt_model, 'context', context)

    context.set_binding_shape(0, tuple(data.shape))
    _, stream = cudart.cudaStreamCreate()
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])

    nOutput = engine.num_bindings - nInput

    bufferH = []
    bufferH.append(data)

    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    
    output = bufferH[-1]

    cudart.cudaStreamDestroy(stream)
    for buffer in bufferD:
        cudart.cudaFree(buffer)

    return torch.as_tensor(output).to(torch_data.device)


def deserialize_engine(trtFile):
    with open(trtFile, 'rb') as f:
        engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
    return engine


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
    num_gpus = 1
    n_cpus = os.cpu_count()

    setattr(opts, "dataset.workers", n_cpus)
    # adjust the batch size
    train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
    val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
    setattr(opts, "dataset.train_batch_size0", train_bsize)
    setattr(opts, "dataset.val_batch_size0", val_bsize)
    setattr(opts, "dev.device_id", None)

    val_loader = create_eval_loader(opts)
    
    engine = deserialize_engine(config.engine_filepath)
    test_acc(val_loader, partial(trt_model, engine=engine))


if __name__ == "__main__":
    main_worker()