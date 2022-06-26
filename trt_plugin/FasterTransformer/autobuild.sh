clear
if [ -d "./release" ]
then
    rm -rf "./release"
fi
mkdir -p release && cd release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j64