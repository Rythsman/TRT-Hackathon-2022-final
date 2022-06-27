clear

# 生成onnx
python3 main_export_detection.py --common.config-file ../ml-cvnets/pretrained_models/ssd_mobilevit_s_320.yaml --evaluation.detection.mode validation_set

# 生成engine并测试精度与速度(FP32)，需要将--ano_filepath 后的路径改为当前环境中coco数据集相应文件路径
make -j64 > ./log/fp32_log.txt
python3 coco_eval.py --ano_filepath /root/trt2022_src/mobilenet/dataset/fast-ai-coco/annotations/instances_val2017.json --result_filepath mobilevit_ssd_det_FP32.json

# 生成engine并测试精度与速度(FP16),需要先修改mobileVitDet.cu 1517:FP32为FP16
# make clean && make -j64 > ./log/fp16_log.txt
# python3 coco_eval.py --ano_filepath /root/trt2022_src/mobilenet/dataset/fast-ai-coco/annotations/instances_val2017.json --result_filepath mobilevit_ssd_det_FP16.json