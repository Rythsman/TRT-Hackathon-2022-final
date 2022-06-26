# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from statistics import mode
import time
import argparse
import datetime
import numpy as np
# from sqlalchemy import false

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import tensorrt as trt
import ctypes
import ml_collections
import onnxruntime as ort
import onnx
import onnx_graphsurgeon as gs

import sys
# sys.path.insert(0, "../../pytorch/vit/ViT-quantization/ViT-pytorch")

# from models.modeling import VisionTransformer, CONFIGS
from plugin_loader import ViTPluginLoader


test_time = 100
warmup_time = 10

# pretrained_dir = "../ml-cvnets/test1.npz"
# plugin_path = "../../../relesae/lib/libmobileVit_plugin.so"
# batch_size = 128
# seq_len = 256
# # embded_dim = 144
# use_fp16 = false
# seed = 42

# def setup_torch(args):
#     # Prepare model
#     config = CONFIGS[args.model_type]
#     print(config)

#     model = VisionTransformer(config, args.img_size, zero_head=False, num_classes=1000)
#     model.load_from(np.load(args.pretrained_dir))
#     model.to(args.device)
#     return config, model

def setup_trt(args, config):
    p_loader = ViTPluginLoader(args.plugin_path)
    p_loader.load_model_config(config, args)
    engine = p_loader.build_network(args.pretrained_dir)
    return engine, p_loader

def parse_option():
    parser = argparse.ArgumentParser('ViT evaluation script', add_help=False)

    parser.add_argument("--pretrained_dir", type=str, default="/target/ml-cvnets/test1.npz",
                        help="Where to search for pretrained ViT models.")

    # easy config modification
    parser.add_argument('--plugin_path', type=str, default="../../../release/lib/libmobileVit_plugin.so", help='path to plugin lib')

    parser.add_argument('--batch-size', type=int, default=4, help="batch size for single GPU")

    parser.add_argument('--fp16', action='store_true', 
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args, unparsed = parser.parse_known_args()


    return args

def get_mobilevit_config():
    config = ml_collections.ConfigDict()
    # config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 144
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 288
    config.transformer.num_heads = 4
    config.transformer.num_layers = 2
    return config

def main(args):

    # config, model = setup_torch(args)

    config = get_mobilevit_config()
    engine, p_loader = setup_trt(args, config)

    validate_with_random_data_for_mvit(p_loader, engine)
    # validate_with_random_data(p_loader, model, engine)


@torch.no_grad()
def run_trt_plugin(plugin_loader:ViTPluginLoader, input_tensor, engine):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    seq_len = plugin_loader.seq_len_
    embed_dim = plugin_loader.embed_dim_
    max_batch = plugin_loader.max_batch_
    # img_size = plugin_loader.img_size_
    # in_chans = plugin_loader.in_chans_
    is_fp16 = plugin_loader.is_fp16_

    dtype_trt = trt.float16 if is_fp16 else trt.float32
    dtype_np = np.float16 if is_fp16 else np.float32
    dtype_torch = torch.float16 if is_fp16 else torch.float32

    with engine.create_execution_context() as context:

        context.active_optimization_profile = 0

        stream = torch.cuda.Stream()

        context.set_binding_shape(0, (int(input_tensor.size(0)), seq_len, embed_dim))
        output_shape = tuple(context.get_binding_shape(1))
        print(output_shape)

        # Copy input h2d
        d_inputs = [input_tensor]
        d_output = torch.empty(output_shape, dtype=torch.float32).cuda()

        # warm up
        for i in range(warmup_time):
            context.execute_async_v2([d_inp.data_ptr() for d_inp in d_inputs] + [d_output.data_ptr()], stream.cuda_stream)

        #ignore the last fc layer
        torch.cuda.synchronize()
        op_end = time.time()
        for i in range(test_time):
            context.execute_async_v2([d_inp.data_ptr() for d_inp in d_inputs] + [d_output.data_ptr()], stream.cuda_stream)
        stream.synchronize()

        torch.cuda.synchronize()
        print("plugin time : ", (time.time() - op_end)/test_time*1000.0, "ms")

        return d_output.cpu().numpy()


@torch.no_grad()
def validate_with_random_data_for_mvit(plugin_loader:ViTPluginLoader, engine):
    
    dtype_torch = torch.float16 if plugin_loader.is_fp16_ else torch.float32

    data_type = np.float16 if plugin_loader.is_fp16_ else np.float32

    shape = (1, 3, 256, 256)
    images = np.random.rand(np.prod(shape)).astype(data_type).reshape(shape)
    
    # onnxRuntime
    model = onnx.load("/root/trt2022_src/mobilenet/ml-cvnets/mobilevit_s.onnx")
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    
    # graph = gs.import_onnx(model)
    # for node in graph.nodes:
    #     if node.name == "MatMul_63":
    #         matmul_tensor = node.inputs[1].values

    ort_session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])


    ort_output1, ffn_output, ort_output2 = ort_session.run(["input.184", "input.224", "patches"], {"input": images.astype(np.float32)})


    # # trt_plugin
    trt_input_tensor = torch.as_tensor(ort_output1, dtype=dtype_torch)
    trt_input_tensor = trt_input_tensor.cuda(non_blocking=False)
    plugin_output = run_trt_plugin(plugin_loader, trt_input_tensor, engine)

    # a0 = np.load("/target/ml-cvnets/weights/trt_input.npy")
    # a1 = np.load("/target/mobilebit_trt/FasterTransformer/build/weights/output.npy")
    # print('a1.shape', a1.shape)
    # print('ort_output2.shape', ort_output2.shape)
    # diff = (ort_output2.reshape(-1, ort_output2.shape[-1]) - a1)
    # print("trt attn output mean diff: ", diff.mean(), "max diff : ", diff.max())
    # a2 = np.load("/target/ml-cvnets/weights/trt_attn_out.npy")

    # # print("shape:", ort_output1.shape, a0.shape)
    # trt_input_tensor = trt_input_tensor.cpu().numpy()
    # diff0 = abs(trt_input_tensor - a0.reshape(trt_input_tensor.shape))
    # print("ort_output0 vs plugin_output0 , avg diff : ", diff0.mean(), "max diff : ", diff0.max())

    # diff1 = abs(ort_output_attn - a2.reshape(ort_output_attn.shape))
    # print("ort_output1 vs plugin_output1 , avg diff : ", diff1.mean(), "max diff : ", diff1.max())

    # diff2 = abs(ort_output_ln - a1.reshape(ort_output_ln.shape))
    # print("ort_output2 vs plugin_output2 , avg diff : ", diff2.mean(), "max diff : ", diff2.max())

    # 比较qkv weights是否一致， 结果ok
    # a3 = np.load("/target/ml-cvnets/weights/trt_attn_q_weight.npy")
    # a4 = np.load("/target/ml-cvnets/weights/trt_attn_k_weight.npy")
    # a5 = np.load("/target/ml-cvnets/weights/trt_attn_v_weight.npy")
    # a_concat = np.concatenate((a3, a4, a5), axis=0).T
    # diff2 = abs(a_concat - matmul_tensor)
    # print("qkv weights : ", diff2.mean(), "max diff : ", diff2.max())

    # 比较qkv 结果
    # a3 = np.load("/target/ml-cvnets/weights/trt_q_buf.npy").reshape(144, 4, 256)
    # a4 = np.load("/target/ml-cvnets/weights/trt_k_buf.npy").reshape(144, 4, 256)
    # a5 = np.load("/target/ml-cvnets/weights/trt_v_buf.npy").reshape(144, 4, 256)
    # # print(a3[0][0][0])
    # # print(ort_output_attn[0][0][0])
    # a_concat = np.concatenate((a3, a4, a5), axis=0).transpose(1,2,0)
    # diffa = abs(ort_output_attn - a_concat.reshape(ort_output_attn.shape))
    # print("qkv merge , avg diff : ", diffa.mean(), "max diff : ", diffa.max())
    
    
    diff = abs(ort_output2 - plugin_output)
    print("ort_output vs plugin_output , avg diff : ", diff.mean(), "max diff : ", diff.max())


if __name__ == '__main__':
    args = parse_option()

    seed = args.seed + int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    main(args)
