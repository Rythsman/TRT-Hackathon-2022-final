#include "bertCommon.h"
#include "trt_tensor.hpp"

#include <cudnn.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>


#define checkCUDNN(expression)                               \
{                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
        std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                               \
    }                                                        \
}

int test_swish() {
    ::srand(::time(0));
    std::cout << ::rand() << std::endl;

    constexpr int batch_size = 4;
    constexpr int channel_in = 3;
    constexpr int height_in = 112;
    constexpr int width_in = 112;

    constexpr int channel_out = channel_in;
    constexpr int height_out = 112;
    constexpr int width_out = 112;

    TRT::Tensor input_tensor(std::vector<int>{batch_size, channel_in, height_in, width_in});
    TRT::Tensor out_tensor(std::vector<int>{batch_size, channel_in, height_out, width_out});
    
    const float alpha1 = 1;
    const float alpha2 = 0;

    auto input_ptr_cpu = input_tensor.cpu<float>();

    for(int i = 0; i < input_tensor.numel(); ++i) 
    {
        input_ptr_cpu[i] = float(rand() % 100000) / 100000;
    }
    input_tensor.save_to_file("input_tensor.npz");

    cudaStream_t stream = out_tensor.get_stream();
    
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    checkCUDNN(cudnnSetStream(cudnn, stream));

    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                            /*mode=*/CUDNN_ACTIVATION_SWISH,
                                            /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                            /*relu_coef=*/0));
    // checkCUDNN(cudnnSetActivationDescriptorSwishBeta(activation_descriptor, 0));

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/batch_size,
                                          /*channels=*/channel_in,
                                          /*image_height=*/height_in,
                                          /*image_width=*/width_in));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/channel_out,
                                      /*image_height=*/height_out,
                                      /*image_width=*/width_out));

    auto input_tensor_gpu = input_tensor.to_gpu(true).gpu<float>();
    auto outptr_gpu = out_tensor.to_gpu().gpu<float>();

    checkCUDNN(cudnnActivationForward(
        cudnn,
        activation_descriptor,
        &alpha1,
        input_descriptor,
        input_tensor_gpu,
        &alpha2,
        output_descriptor,
        outptr_gpu));

    out_tensor.to_cpu(true);
    out_tensor.save_to_file("out_tensor.npz");

    return 0;
}

int main() {

    test_swish();

    return 0;
}