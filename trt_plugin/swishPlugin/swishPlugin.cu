/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <numeric>
#include <stdexcept>
#include "swishPlugin.h"
#include "swish.cuh"

using namespace nvinfer1;
using nvinfer1::plugin::swishPlugin;
using nvinfer1::plugin::swishPluginCreator;

#define CHECK_CUDNN(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            printf("error: %s:%d, %d\n", __FILE__, __LINE__, status);                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

namespace
{
constexpr const char* PLUGIN_VERSION{"1"};
constexpr const char* PLUGIN_NAME{"Swish"};
} // namespace

// // Static class fields initialization
PluginFieldCollection swishPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> swishPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(swishPluginCreator);

swishPlugin::swishPlugin()
{

}

int swishPlugin::initialize() noexcept
{
    return 0;
}

swishPlugin::swishPlugin(const void* data, size_t length)
{

}

const char* swishPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* swishPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int swishPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs swishPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    assert(nbInputs == 1);
    assert(index == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

void swishPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    _cudnn_handle = cudnnContext;
}

// Detach the plugin object from its execution context.
void swishPlugin::detachFromContext() noexcept
{
    // desc = nullptr;
    // bnDesc = nullptr;
}

int swishPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // Get the input dimensions
    nvinfer1::Dims input_dims = inputDesc[0].dims;
   
    int elem_cnt = input_dims.nbDims == 0 ? 0 : 1;
    for (int i=0; i < input_dims.nbDims; ++i)
        elem_cnt *= input_dims.d[i];

    bool use_fp16 = inputDesc[0].type == DataType::kHALF;

    const float alpha1 = 1;
    const float alpha2 = 0;

    cudnnActivationDescriptor_t activation_descriptor;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                            /*mode=*/CUDNN_ACTIVATION_SWISH,
                                            /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                            /*relu_coef=*/0));
    
    cudnnTensorDescriptor_t input_descriptor;

    if(use_fp16) 
    {
        
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                            /*format=*/CUDNN_TENSOR_NCHW,
                                            /*dataType=*/CUDNN_DATA_HALF,
                                            /*batch_size=*/1,
                                            /*channels=*/1,
                                            /*image_height=*/1,
                                            /*image_width=*/elem_cnt));

        CHECK_CUDNN(cudnnActivationForward(
            _cudnn_handle,
            activation_descriptor,
            &alpha1,
            input_descriptor,
            (half*)(inputs[0]),
            &alpha2,
            input_descriptor,
            (half*)(outputs[0])
            ));

        ///! 还需要理解elementwish.cuh的pack方法
        // oneflow::cuda::elementwise::Unary< ACTIVATION::SwishFunctor<half>, half, half >(
        // ACTIVATION::SwishFunctor<half>(), elem_cnt, reinterpret_cast<half*>(outputs[0]), reinterpret_cast<const half*>(inputs[0]), stream);
    } 
    else 
    {

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                            /*format=*/CUDNN_TENSOR_NCHW,
                                            /*dataType=*/CUDNN_DATA_FLOAT,
                                            /*batch_size=*/1,
                                            /*channels=*/1,
                                            /*image_height=*/1,
                                            /*image_width=*/elem_cnt));

        CHECK_CUDNN(cudnnActivationForward(
            _cudnn_handle,
            activation_descriptor,
            &alpha1,
            input_descriptor,
            (float*)(inputs[0]),
            &alpha2,
            input_descriptor,
            (float*)(outputs[0])
            ));

        // oneflow::cuda::elementwise::Unary< ACTIVATION::SwishFunctor<float>, float, float >(
            // ACTIVATION::SwishFunctor<float>(), elem_cnt, reinterpret_cast<float*>(outputs[0]), reinterpret_cast<const float*>(inputs[0]), stream);
    }

    return 0;
}

size_t swishPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void swishPlugin::serialize(void* buffer) const noexcept
{

}

bool swishPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(inOut && pos < (nbInputs + nbOutputs));
    // for(int i = 0; i < nbInputs; ++i) {
    //     printf("inOut[%d].type = %d\n", i, inOut[i].type);
    // }
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

void swishPlugin::terminate() noexcept
{
    // cudaFree(mean);
    // mean = nullptr;
    // cudaFree(inv_variance);
    // inv_variance = nullptr;
}

void swishPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* swishPlugin::clone() const noexcept
{
    auto* plugin = new swishPlugin();
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void swishPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{

    // for (int i = 0; i < nbInputs; i++)
    // {
    //   for (int j = 0; j < in[0].desc.dims.nbDims; j++)
    //   {
    //     // Do not support dynamic dimensions
    //     printf("in[0].desc.dims.d[%d] = %d\n", j, in[0].desc.dims.d[j]);
    //     assert(in[0].desc.dims.d[j] != -1);
    //   }
    // }

    // int batchSize = in[0].desc.dims.d[0] < 0 ? 256 : in[0].desc.dims.d[0];
    // int nbChannels = in[0].desc.dims.d[1] < 0 ? 256 : in[0].desc.dims.d[1];


    // // cudaMalloc(&mean, batchSize * nbChannels * sizeof(float));
    // // cudaMalloc(&inv_variance, batchSize * nbChannels * sizeof(float));

    // // printf("batchSize: %d\n", batchSize);
    // // printf("nbChannels: %d\n", nbChannels);
    // // Allocate device memory and initialize scale and bias values
    // int need_size = batchSize * nbChannels;
    // if(mean_len < need_size) {
    //     if(mean != nullptr) {
    //         cudaFree(mean);
    //         mean = nullptr;
    //     }
    //     mean_len = need_size;
    //     cudaMalloc(&mean, mean_len * sizeof(float));
    // }
    // if(inv_variance_len < need_size) {
    //     if(inv_variance != nullptr) {
    //         cudaFree(inv_variance);
    //         inv_variance = nullptr;
    //     }
    //     inv_variance_len = need_size;
    //     cudaMalloc(&inv_variance, inv_variance_len * sizeof(float));
    // }
}

nvinfer1::DataType swishPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t swishPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void swishPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* swishPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

swishPluginCreator::swishPluginCreator()
{
    mPluginAttributes.clear();
    // mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    // mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* swishPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* swishPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* swishPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* swishPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void swishPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* swishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // Set default values
    // int nbGroups{1};
    // float epsilon{0.00001F};
    // for (int i = 0; i < fc->nbFields; i++)
    // {
    //     std::string field_name(fc->fields[i].name);
    //     if (field_name.compare("eps") == 0)
    //     {
    //         epsilon = *static_cast<const float*>(fc->fields[i].data);
    //     }
    //     if (field_name.compare("num_groups") == 0)
    //     {
    //         nbGroups = *static_cast<const int*>(fc->fields[i].data);
    //     }
    // }

    swishPlugin* plugin = new swishPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* swishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    swishPlugin* plugin = new swishPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}
