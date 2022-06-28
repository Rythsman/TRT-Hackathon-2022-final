clear

usefp16=0
testAcc=0
testAccPath=" "

while getopts ":a:t:p:h" optname
do
    case "$optname" in
        'a')
        usefp16=$OPTARG
        ;;
        't')
        testAcc=$OPTARG
        ;;
        'p')
        testAccPath=$OPTARG
        ;;
        'h')
        echo "示例输入：autobuild.sh -a 1 -t 1 -p \"cocopath\"; -a 代表是否使用fp16，-t 代表是否使用coco进行精度测试, -p 代表测试集路径"
        exit 0;;
        ':')
        echo "未输入 $OPTARG 对应的值"
        exit 1;;
        '?')
        echo "无法识别当前输入 $OPTARG"
        exit 2;;
        *)
        echo "未知问题"
        exit 3;;
    esac
done

if [ ! -d "../ml-cvnets/npz" ]; then
    echo "--------Creating npz dataset-------"
    python3 ../ml-cvnets/gen_npz.py
fi

# 生成onnx
echo "--------generating onnx model-------"
python3 main_export_detection.py --common.config-file ../ml-cvnets/pretrained_models/ssd_mobilevit_s_320.yaml --evaluation.detection.mode validation_set

# 生成engine并测试精度与速度
make clean && make -j64 
if [ "$usefp16" -eq "1" ];then
    echo "-------- generate trt engine & test using fp16 -------"
else
    echo "-------- generate trt engine & test using fp32 -------"
fi
./mobileVitDet usefp16 testAcc testAccPath > ./log/log.txt

if [ "$testAcc" -eq "1"]; then
    jsonPath1="${testAccPath}/annotations/instances_val2017.json"
    jsonPath2="mobilevit_ssd_det_FP32.json"
    if [ "$usefp16" -eq "1"]; then
        jsonPath2="mobilevit_ssd_det_FP16.json"
    fi
    # python3 coco_eval.py --ano_filepath /root/trt2022_src/mobilenet/dataset/fast-ai-coco/annotations/instances_val2017.json --result_filepath mobilevit_ssd_det_FP32.json
    python3 coco_eval.py --ano_filepath ${jsonPath1} --result_filepath ${jsonPath2}
fi

echo ""
echo "--------------- 结果保存在./log/log.txt 中 ------------------"
echo "----------------------- Done -------------------------------"