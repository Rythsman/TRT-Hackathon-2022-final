clear

# 生成onnx
python3 export_onnx.py --common.config-file ../ml-cvnets/pretrained_models/mobilevit_s.yaml
# python3 main_export.py --common.config-file ../ml-cvnets/mobilevit_s.yaml

# 生成engine
python3 export_engine.py --engine_filepath mobilevit_s_fp16.engine --onnx_filepath ./mobilevit_s.onnx  --use_fp16

# 速度测试
python3 testSpeed.py --common.config-file ../ml-cvnets/pretrained_models/mobilevit_s.yaml --engine_filepath mobilevit_s_fp16.engine

# 精度测试
python3 testAcc.py --common.config-file ../ml-cvnets/pretrained_models/mobilevit_s.yaml --engine_filepath mobilevit_s_fp16.engine