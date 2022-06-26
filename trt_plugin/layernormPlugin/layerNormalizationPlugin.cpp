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
#include "layerNormalizationPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::GroupNormalizationPlugin;
using nvinfer1::plugin::GroupNormalizationPluginCreator;

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
constexpr const char* GROUP_NORM_VERSION{"1"};
constexpr const char* GROUP_NORM_NAME{"lnplugin"};
} // namespace

// // Static class fields initialization
PluginFieldCollection GroupNormalizationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GroupNormalizationPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GroupNormalizationPluginCreator);

GroupNormalizationPlugin::GroupNormalizationPlugin(float epsilon, int nbGroups)
    : mEpsilon(epsilon),
      mNbGroups(nbGroups)
{
    // Number of groups should be positive
    assert(nbGroups > 0);
}

int GroupNormalizationPlugin::initialize() noexcept
{
    return 0;
}

GroupNormalizationPlugin::GroupNormalizationPlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mEpsilon);
    deserialize_value(&data, &length, &mNbGroups);
}

const char* GroupNormalizationPlugin::getPluginType() const noexcept
{
    return GROUP_NORM_NAME;
}

const char* GroupNormalizationPlugin::getPluginVersion() const noexcept
{
    return GROUP_NORM_VERSION;
}

int GroupNormalizationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs GroupNormalizationPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Input (from previous layer), scale and bias are the three inputs to the plugin.
    assert(nbInputs == 3);
    assert(index == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

void GroupNormalizationPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    _cudnn_handle = cudnnContext;
    // printf("设置_cudnn_handle\n");
    /*
    if(desc == nullptr) {
        cudnnCreateTensorDescriptor(&desc);
    }
    if(bnDesc == nullptr) {
        cudnnCreateTensorDescriptor(&bnDesc);
    }
    */
}

// Detach the plugin object from its execution context.
void GroupNormalizationPlugin::detachFromContext() noexcept
{
    // cudnnDestroyTensorDescriptor(desc);
    // cudnnDestroyTensorDescriptor(bnDesc);
    desc = nullptr;
    bnDesc = nullptr;
}

int GroupNormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // Get the input dimensions
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int batchSize = input_dims.d[0];
    int nbChannels = input_dims.d[1];
    bool use_fp16 = inputDesc[0].type == DataType::kHALF;

    // printf("use_fp160=%d\n", (bool)(inputDesc[0].type == DataType::kHALF));
    // printf("use_fp161=%d\n", (bool)(inputDesc[1].type == DataType::kHALF));
    // Calculate size of each group
    // mNbGroups = nbChannels;
    // int groupSize = 1;
    
    // if(!bnScale) {
    //     cudaError_t err = cudaSuccess;
    //     printf("bnScale alloc\n");
    //     err = cudaMalloc(&bnScale, batchSize * nbChannels * sizeof(float));
    //     if( err != cudaSuccess) {
    //         printf("CUDA error: %s\n", cudaGetErrorString(err));
    //         return -1;
    //     }
    //     printf("bnScale = 0x%x, bnBias = 0x%x \n", reinterpret_cast<intptr_t>(bnScale), reinterpret_cast<intptr_t>(bnBias));
    //     // allot ones and zeros to bn parameters
    //     std::vector<float> ones(batchSize * nbChannels, 1.F);
    //     cudaMemcpy(bnScale, ones.data(), batchSize * nbChannels * sizeof(float), cudaMemcpyHostToDevice);
    // }

    // if(!bnBias) {
    //     printf("bnBias alloc\n");
    //     cudaMalloc(&bnBias, batchSize * nbChannels * sizeof(float));
    //     printf("bnScale = 0x%x, bnBias = 0x%x \n", reinterpret_cast<intptr_t>(bnScale), reinterpret_cast<intptr_t>(bnBias));
    //     std::vector<float> zeroes(batchSize * nbChannels, 0.F);
    //     cudaMemcpy(bnBias, zeroes.data(), batchSize * nbChannels * sizeof(float), cudaMemcpyHostToDevice);
    // }
    
    mChannelVolume = std::accumulate(input_dims.d + 2, input_dims.d + inputDesc[0].dims.nbDims, 1, std::multiplies<int>());
    // printf("use_fp16 = %d, batchSize = %d, nbChannels = %d mChannelVolume = %d\n", (int)use_fp16, batchSize, nbChannels, mChannelVolume);
    // printf("output: use_fp16 = %d, batchSize = %d, nbChannels = %d mChannelVolume = %d\n", (int)(outputDesc[0].type == DataType::kHALF), 
            // outputDesc[0].dims.d[0], outputDesc[0].dims.d[1], outputDesc[0].dims.d[2]);
    // printf("mChannelVolume = %d\n", mChannelVolume);
    // if(use_fp16) {
    //     printf("use_fp16\n");
    // } else {
    //     printf("use_fp32\n");
    // }
    // CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc, // descriptor
    //     CUDNN_TENSOR_NCHW,                 // tensor format
    //     CUDNN_DATA_FLOAT, // type
    //     1,                                 // Batchsize
    //     batchSize * nbChannels,             // Channels
    //     1,                         // Height
    //     mChannelVolume                     // Width
    //     ));
    

    // cudaMalloc(&mean, batchSize * nbChannels * sizeof(float));
    // cudaMalloc(&inv_variance, batchSize * nbChannels * sizeof(float));
    int need_size = batchSize * nbChannels;
    if(mean_len < need_size) {
        if(mean != nullptr) {
            cudaFree(mean);
            mean = nullptr;
        }
        mean_len = need_size;
        cudaMalloc(&mean, mean_len * sizeof(float));
    }
    if(inv_variance_len < need_size) {
        if(inv_variance != nullptr) {
            cudaFree(inv_variance);
            inv_variance = nullptr;
        }
        inv_variance_len = need_size;
        cudaMalloc(&inv_variance, inv_variance_len * sizeof(float));
    }

    if(use_fp16) {
        // layer_norm::LayerNormForwardGpu<__half>(stream, const int64_t num_instances, const int64_t norm_size,
        //                     const double epsilon, const __half* x_ptr, const __half* gamma_ptr,
        //                     const __half* beta_ptr, __half* y_ptr, __half* mean,
        //                     __half* inv_variance);
        // LayerNormForwardGpu<__half>(stream, mChannelVolume, batchSize * nbChannels,
        //                 mEpsilon, reinterpret_cast<const __half*>(inputs[0]), reinterpret_cast<const __half*>(inputs[1]),
        //                 reinterpret_cast<const __half*>(inputs[2]), reinterpret_cast<__half*>(outputs[0]),reinterpret_cast<__half*>(mean),
        //                 reinterpret_cast<__half*>(inv_variance));
        LayerNormForwardGpu(stream, batchSize * nbChannels, mChannelVolume,
                        mEpsilon, reinterpret_cast<const __half*>(inputs[0]), reinterpret_cast<const __half*>(inputs[1]),
                        reinterpret_cast<const __half*>(inputs[2]), reinterpret_cast<__half*>(outputs[0]), mean,
                        inv_variance);

        // return scaleShiftChannelsInplace(output, batchSize, nbChannels, mChannelVolume, static_cast<const __half*>(inputs[2]), static_cast<const __half*>(inputs[1]), stream); //mBetaDev, mGammaDev,    
    } else {
        LayerNormForwardGpu(stream, batchSize * nbChannels, mChannelVolume,
                        mEpsilon, reinterpret_cast<const float*>(inputs[0]), reinterpret_cast<const float*>(inputs[1]),
                        reinterpret_cast<const float*>(inputs[2]), reinterpret_cast<float*>(outputs[0]),mean,
                        inv_variance);

        // return scaleShiftChannelsInplace(output, batchSize, nbChannels, mChannelVolume, static_cast<const float*>(inputs[2]), static_cast<const float*>(inputs[1]), stream); //mBetaDev, mGammaDev,    
    }

    return 0;
}

size_t GroupNormalizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNbGroups) + sizeof(mEpsilon);
}

void GroupNormalizationPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNbGroups);
}

bool GroupNormalizationPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(inOut && pos < (nbInputs + nbOutputs));
    // for(int i = 0; i < nbInputs; ++i) {
    //     printf("inOut[%d].type = %d\n", i, inOut[i].type);
    // }
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

void GroupNormalizationPlugin::terminate() noexcept
{
    cudaFree(mean);
    mean = nullptr;
    cudaFree(inv_variance);
    inv_variance = nullptr;
}

void GroupNormalizationPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* GroupNormalizationPlugin::clone() const noexcept
{
    auto* plugin = new GroupNormalizationPlugin(mEpsilon, mNbGroups);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void GroupNormalizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
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

    int batchSize = in[0].desc.dims.d[0] < 0 ? 256 : in[0].desc.dims.d[0];
    int nbChannels = in[0].desc.dims.d[1] < 0 ? 256 : in[0].desc.dims.d[1];


    // cudaMalloc(&mean, batchSize * nbChannels * sizeof(float));
    // cudaMalloc(&inv_variance, batchSize * nbChannels * sizeof(float));

    // printf("batchSize: %d\n", batchSize);
    // printf("nbChannels: %d\n", nbChannels);
    // Allocate device memory and initialize scale and bias values
    int need_size = batchSize * nbChannels;
    if(mean_len < need_size) {
        if(mean != nullptr) {
            cudaFree(mean);
            mean = nullptr;
        }
        mean_len = need_size;
        cudaMalloc(&mean, mean_len * sizeof(float));
    }
    if(inv_variance_len < need_size) {
        if(inv_variance != nullptr) {
            cudaFree(inv_variance);
            inv_variance = nullptr;
        }
        inv_variance_len = need_size;
        cudaMalloc(&inv_variance, inv_variance_len * sizeof(float));
    }
}

nvinfer1::DataType GroupNormalizationPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t GroupNormalizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void GroupNormalizationPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* GroupNormalizationPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

GroupNormalizationPluginCreator::GroupNormalizationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GroupNormalizationPluginCreator::getPluginName() const noexcept
{
    return GROUP_NORM_NAME;
}

const char* GroupNormalizationPluginCreator::getPluginVersion() const noexcept
{
    return GROUP_NORM_VERSION;
}

const PluginFieldCollection* GroupNormalizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* GroupNormalizationPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GroupNormalizationPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* GroupNormalizationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // Set default values
    int nbGroups{1};
    float epsilon{0.00001F};
    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("eps") == 0)
        {
            epsilon = *static_cast<const float*>(fc->fields[i].data);
        }
        if (field_name.compare("num_groups") == 0)
        {
            nbGroups = *static_cast<const int*>(fc->fields[i].data);
        }
    }

    GroupNormalizationPlugin* plugin = new GroupNormalizationPlugin(epsilon, nbGroups);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* GroupNormalizationPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    GroupNormalizationPlugin* plugin = new GroupNormalizationPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}
