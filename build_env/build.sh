pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install nvidia-pyindex -i https://mirrors.aliyun.com/pypi/simple/
pip3 install onnx-graphsurgeon -i https://mirrors.aliyun.com/pypi/simple/

# opencv
apt update
apt install -y libopencv-dev

# install cuda-python
if [ ! -f "v11.7.0.tar.gz" ]; then
    wget https://github.com/NVIDIA/cuda-python/archive/refs/tags/v11.7.0.tar.gz
fi

tar -xvf ./v11.7.0.tar.gz
cd ./cuda-python-11.7.0 && pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ 
python setup.py build_ext --inplace