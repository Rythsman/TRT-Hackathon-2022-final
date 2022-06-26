import torch
import sys
import os
import tensorrt as trt
import numpy as np
from cuda import cudart
sys.path.insert(0, os.path.abspath('../ml-cvnets'))
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# import cvnets.models.classification.mobilevit as mobilevit
from data import create_eval_loader

class EasyDict(dict):
    
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            def get_EasyDict_list(value):
                return [EasyDict(x) if isinstance(x, dict) else x if not isinstance(x, (list, tuple)) else get_EasyDict_list(x) for x in value]
            value = get_EasyDict_list(value)
        else:
            value = EasyDict(value) if isinstance(value, dict) else value
        super(EasyDict, self).__setattr__(name, value)
        self[name] = value

    def __getattr__(self, name):
        """
        如果 key 不存在返回 None
        """
        try:
            return super(EasyDict, self).__getattr__(name)
        except AttributeError as e:
            if name in self:
                return self[name]
            else:
                raise e

config = EasyDict()

setattr(config, "dataset.root_train", "/mnt/imagenet/training")
setattr(config, "dataset.root_val", "/root/trt2022_src/mobilenet/new_imagenet/val")
setattr(config, "dataset.root_test", "")
setattr(config, "dataset.name", "imagenet")
setattr(config, "dataset.category", "classification")
setattr(config, "dataset.train_batch_size0", 128)
setattr(config, "dataset.val_batch_size0", 32)
setattr(config, "dataset.eval_batch_size0", 32)
setattr(config, "dataset.workers", 0)
setattr(config, "dataset.persistent_workers", False)
setattr(config, "dataset.pin_memory", True)
setattr(config, "dataset.augmentation.gauss_noise_var", None)
setattr(config, "dataset.augmentation.jpeg_q_range", None)
setattr(config, "dataset.augmentation.gamma_corr_range", None)
setattr(config, "dataset.augmentation.blur_kernel_range", None)
setattr(config, "dataset.augmentation.translate_factor", None)
setattr(config, "dataset.augmentation.rotate_angle", None)

setattr(config, "dataset.pascal.use_coco_data", False)
setattr(config, "dataset.pascal.coco_root_dir", None)


class ImageNetEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = dataloader
        self.data_iter = iter(dataloader)
        self.dataset_length = len(self.data.dataset)
        self.batch_size = int(dataloader.batch_sampler.batch_size_gpu0)
        self.current_index = 0
        self.buffer_size = trt.volume([self.batch_size, 3, 256 , 256]) * trt.float32.itemsize
        # Allocate enough memory for a whole batch
        _, self.device_input = cudart.cudaMalloc(self.buffer_size)
    def __del__(self):
        cudart.cudaFree(self.device_input)

    def get_batch_size(self):
        return self.batch_size
    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.dataset_length:
            return None
        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))
        batch = next(self.data_iter)['image']
        # print(batch)
        batch = np.ascontiguousarray(batch.cpu().numpy().astype(np.float32).ravel())

        cudart.cudaMemcpy(self.device_input, batch.ctypes.data, self.buffer_size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
    def get_algorithm(self):
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

def GiB(val):
    return val * 1 << 30

def buildTrtEngine(calib):
    output_onnx = './mobilevit_s.onnx'
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    max_batch_size = 128
    trtFile = "./mobilevit_s_int8.plan"

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
            profile.set_shape('input', min=(1, 3, 256, 256), opt=(32, 3, 256, 256), max=(max_batch_size, 3, 256, 256))

            config = builder.create_builder_config()
            config.max_workspace_size = GiB(18) # 8G
            with open(output_onnx, 'rb') as model:
                parser.parse(model.read())
                for i in range(parser.num_errors):
                    print(parser.get_error(i))

            config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib

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

def inferUseTrt(data, calib):
    engine = buildTrtEngine(calib)
    context = engine.create_execution_context()
    context.set_binding_shape(0, tuple(data.shape))
    _, stream = cudart.cudaStreamCreate()
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    print(f"nInput: {nInput}")
    nOutput = engine.num_bindings - nInput

    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i))

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

    return output, engine




def main():
    
    val_loader = create_eval_loader(config)
    calibration_cache = "imagenet_calibration.cache"
    calib = ImageNetEntropyCalibrator(val_loader, cache_file=calibration_cache)

    input_tensor = torch.rand(1, 3, 256, 256)
    data = input_tensor.numpy()
    outputs, engine = inferUseTrt(data, calib)

    print("trt input shape: ", data.shape, data.dtype)
    print("trt output shape: ", outputs.shape, outputs.dtype)


main()