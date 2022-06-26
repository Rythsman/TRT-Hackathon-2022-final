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

import ctypes
import numpy as np
import os
import os.path
import tensorrt as trt
from scipy import ndimage

def load_weights(weight_path:str):
    suffix = weight_path.split('.')[-1]
    if suffix != 'npz':
        print("Unsupport weight file: Unrecognized format %s " % suffix)
        exit(-1)
    return np.load(weight_path)

class ViTPluginLoader:
    def __init__(self, plugin_path) -> None:

        handle = ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
        if not handle:
            raise RuntimeError("Fail to load plugin library: %s" % plugin_path)

        self.logger_ = trt.Logger(trt.Logger.VERBOSE)
        trt.init_libnvinfer_plugins(self.logger_, "")
        plg_registry = trt.get_plugin_registry()

        self.plg_creator = plg_registry.get_plugin_creator("CustomMobileVisionTransformerPlugin", "1", "")

    def load_model_config(self, config, args):
        self.num_heads_         = config.transformer.num_heads
        self.layer_num_         = config.transformer.num_layers
        self.inter_size_        = config.transformer.mlp_dim
        self.embed_dim_         = config.hidden_size
        self.max_batch_         = 4
        self.seq_len_           = 256
        self.is_fp16_           = args.fp16
        self.serial_name_       = "mobileViTEngine_{}_{}_{}_{}_{}_{}_{}".format(self.num_heads_ ,
                                                                      self.layer_num_ ,
                                                                      self.inter_size_,
                                                                      self.embed_dim_ ,
                                                                      self.max_batch_ ,
                                                                      self.seq_len_,
                                                                      int(self.is_fp16_))
        self.value_holder = []


    def build_plugin_field_collection(self, weights):
        field_type = trt.PluginFieldType.FLOAT16 if self.is_fp16_ else trt.PluginFieldType.FLOAT32
        arr_type = np.float16 if self.is_fp16_ else np.float32

        self.value_holder = [np.array([self.max_batch_ ]).astype(np.int32),
                             np.array([self.embed_dim_ ]).astype(np.int32),
                             np.array([self.num_heads_ ]).astype(np.int32),
                             np.array([self.seq_len_ ]).astype(np.int32),
                             np.array([self.inter_size_ ]).astype(np.int32),
                             np.array([self.layer_num_ ]).astype(np.int32)     
        ]

        max_batch   = trt.PluginField("max_batch",  self.value_holder[0], trt.PluginFieldType.INT32)
        embed_dim   = trt.PluginField("embed_dim",  self.value_holder[1], trt.PluginFieldType.INT32)
        num_heads   = trt.PluginField("num_heads",  self.value_holder[2], trt.PluginFieldType.INT32)
        seq_len    = trt.PluginField("seq_len",  self.value_holder[3], trt.PluginFieldType.INT32)
        inter_size  = trt.PluginField("inter_size", self.value_holder[4], trt.PluginFieldType.INT32)
        layer_num   = trt.PluginField("layer_num",  self.value_holder[5], trt.PluginFieldType.INT32)
        
        part_fc = []
        for name in weights.files:
            self.value_holder.append(weights[name].astype(arr_type))
            part_fc.append(trt.PluginField(name, self.value_holder[-1], field_type))

        return trt.PluginFieldCollection([max_batch, embed_dim, num_heads, seq_len, inter_size, layer_num] + part_fc)


    def build_network(self, weights_path):
        file_path = os.path.join('./', self.serial_name_)
        if os.path.exists(file_path):
            return self.deserialize_engine()

        trt_dtype = trt.float16 if self.is_fp16_ else trt.float32
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        weights = load_weights(weights_path)

        with trt.Builder(self.logger_) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
            builder_config.max_workspace_size = 8 << 30
            if self.is_fp16_:
                builder_config.set_flag(trt.BuilderFlag.FP16)
                builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            # Create the network
            input_tensor = network.add_input(name="input_img", dtype=trt_dtype, shape=(-1, self.seq_len_, self.embed_dim_))
   
            # Specify profiles 
            profile = builder.create_optimization_profile()
            min_shape = (1, self.seq_len_, self.embed_dim_)
            ##TODO: There is a bug in TRT when opt batch is large
            max_shape = (self.max_batch_, self.seq_len_, self.embed_dim_)
            profile.set_shape("input_img", min=min_shape, opt=min_shape, max=max_shape)
            builder_config.add_optimization_profile(profile)

            #import pdb;pdb.set_trace()
            print("Generate plugin field collection...")
            pfc = self.build_plugin_field_collection(weights)
            fn = self.plg_creator.create_plugin("mobile_vision_transformer", pfc)
            inputs = [input_tensor]
            vit = network.add_plugin_v2(inputs, fn)

            output_tensor = vit.get_output(0)
            output_tensor.name = "mobile_vision_transformer_output"

            if self.is_fp16_:
                vit.precision = trt.float16 
                vit.set_output_type(0, trt.float16)
            network.mark_output(output_tensor)

            print("Building TRT engine....")
            engine = builder.build_engine(network, builder_config)
            self.serialize_engine(engine)
            return engine

    def serialize_engine(self, engine, file_folder='./'):
        if not os.path.isdir(file_folder):
            self.logger_.log(self.logger_.VERBOSE, "%s is not a folder." % file_folder)
            exit(-1)

        file_path = os.path.join(file_folder, self.serial_name_)

        self.logger_.log(self.logger_.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()

        self.logger_.log(self.logger_.INFO, "Saving Engine to {:}".format(file_path))
        with open(file_path, "wb") as fout:
            fout.write(serialized_engine)
        self.logger_.log(self.logger_.INFO, "Done.")

    # 20220604 外部貌似无调用的地方
    def deserialize_engine(self, file_folder='./'):
        if not os.path.isdir(file_folder):
            self.logger_.log(self.logger_.VERBOSE, "%s is not a folder." % file_folder)
            exit(-1)

        file_path =os.path.join(file_folder, self.serial_name_)
        if not os.path.isfile(file_path):
            self.logger_.log(self.logger_.VERBOSE, "%s not exists. " % file_path)
            return None
        
        filename = os.path.basename(file_path)
        info = filename.split('_')
        # self.patch_size_ = int(info[1])
        self.num_heads_  = int(info[1])
        self.layer_num_  = int(info[2])
        self.inter_size_ = int(info[3])
        self.embed_dim_  = int(info[4])
        self.max_batch_  = int(info[5])
        self.seq_len_    = int(info[6])
        self.is_fp16_    = bool(int(info[7]) == 1)
        # print(info)
        # print('self.is_fp16_', self.is_fp16_)
        # self.in_chans_   = 3
        with open(file_path, 'rb') as f: 
            runtime = trt.Runtime(self.logger_)
            return runtime.deserialize_cuda_engine(f.read())


        

