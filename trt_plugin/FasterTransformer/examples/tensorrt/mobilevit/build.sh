#!/bin/bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python3 infer_mobileVisiontransformer_plugin.py --fp16 --plugin_path=../../../release/lib/libmobileVit_plugin.so --pretrained_dir=/root/trt2022_src/mobilenet/ml-cvnets/test.npz
