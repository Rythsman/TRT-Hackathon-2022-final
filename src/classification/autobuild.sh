clear

# -a 是否使用fp16
# -t 是否使用imagenet进行精度测试（需自行下载数据集）

usefp16=0
testAcc=0

while getopts ":a:t:h" optname
do
    case "$optname" in
        'a')
        usefp16=$OPTARG
        ;;
        't')
        testAcc=$OPTARG
        ;;
        'h')
        echo "示例输入：autobuild.sh -a 1 -t 1 ; -a 代表是否使用fp16，-t 代表是否使用imagenet进行精度测试（需自行下载数据集）"
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
python3 export_onnx.py --common.config-file ../ml-cvnets/pretrained_models/mobilevit_s.yaml

# 生成engine
if [ "$usefp16" -eq "1" ];then
    echo "------generating trt engine use fp16-------"
    python3 export_engine.py --engine_filepath mobilevit_s.engine --onnx_filepath ./mobilevit_s.onnx  --use_fp16
else
    echo "------generating trt engine use fp32-------"
    python3 export_engine.py --engine_filepath mobilevit_s.engine --onnx_filepath ./mobilevit_s.onnx
fi

# 速度测试
if [ "$usefp16" -eq "1" ];then
    echo "--------testing speed use fp16------------------"
else
    echo "--------testing speed use fp32------------------"
fi
python3 testSpeed.py --common.config-file ../ml-cvnets/pretrained_models/mobilevit_s.yaml --engine_filepath mobilevit_s.engine

# 精度测试
if [ "$testAcc" -eq "1" ]; then
    echo "--------testing accuracy---------------"
    python3 testAcc.py --common.config-file ../ml-cvnets/pretrained_models/mobilevit_s.yaml --engine_filepath mobilevit_s.engine
fi

echo ""
echo "--------------- 结果保存在./log/log.txt 中 ------------------"
echo "----------------------- Done -------------------------------"
