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

#include "ViTPlugin.h"
#include "NvInfer.h"
#include "cnpy.h"
#include <cuda.h>
#include <math.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

using namespace nvinfer1;
using namespace std;

namespace fastertransformer {

// Static class fields initialization
PluginFieldCollection MobileVisionTransformerPluginCreator::mFC{};
std::vector<PluginField> MobileVisionTransformerPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MobileVisionTransformerPluginCreator);

template<typename T>
MobileVisionTransformerPlugin<T>::MobileVisionTransformerPlugin(const std::string& name,
                                                    const int max_batch,
                                                    const int embed_dim,
                                                    const int seq_len,
                                                    const int num_heads,
                                                    const int inter_size,
                                                    const int layer_num,
                                                    const float q_scaling,
                                                    const std::vector<const T*>& w):
    layer_name_(name)
{

    settings_.max_batch_size = max_batch;
    // settings_.img_size = img_size;
    // settings_.chn_num = in_chans;
    // settings_.patch_size = patch_size;
    settings_.embed_dim = embed_dim;
    settings_.head_num = num_heads;
    settings_.inter_size = inter_size;
    settings_.num_layer = layer_num;
    // settings_.with_cls_token = with_cls_token;
    settings_.sm = getSMVersion();
    settings_.q_scaling = q_scaling;
    settings_.seq_len = seq_len;
    //settings_.seq_len = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);
    settings_.attention_type = getAttentionType<T>(embed_dim / num_heads, settings_.sm, true, settings_.seq_len);

    Init(w);
}

template<typename T>
MobileVisionTransformerPlugin<T>::MobileVisionTransformerPlugin(const std::string& name, const void* data, size_t length):
    layer_name_(name)
{
    ::memcpy(&settings_, data, sizeof(settings_));
    const char* w_buffer = static_cast<const char*>(data) + sizeof(settings_);

    std::vector<const T*> dummy;
    Init(dummy);

    params_->deserialize(w_buffer);
}

template<typename T>
MobileVisionTransformerPlugin<T>::MobileVisionTransformerPlugin(const MobileVisionTransformerPlugin<T>& plugin):
    layer_name_(plugin.layer_name_), settings_(plugin.settings_)
{
    std::vector<const T*> dummy;
    Init(dummy);
    *params_ = *plugin.params_;
}

template<typename T>
void MobileVisionTransformerPlugin<T>::Init(const std::vector<const T*>& w)
{
        // ViTWeight(const int embed_dim,
        //       const int inter_size,
        //       const int num_layer,
        //       const bool hold_buffer = true):

    params_ = new ViTWeight<T>(settings_.embed_dim,
                               settings_.inter_size,
                               settings_.num_layer);

    if (w.size() > 0) {
        size_t weight_num = params_->GetWeightCount();

        if (weight_num != w.size()) {
            printf("[ERROR][MobileVisionTransformerPlugin] weights number %lu does not match expected number %lu!\n",
                   w.size(),
                   weight_num);
            exit(-1);
        }
        const T* const* pp_buf = &w[0];
        params_->CopyWeightsFromHostBuffers(pp_buf);
    }

    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    checkCUDNN(cudnnCreate(&cudnn_handle_));

    cublasAlgoMap_ = new cublasAlgoMap("igemm.config", "");
    cublasWrapperMutex_ = new std::mutex();
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());

    cublas_wrapper_ =
        new cublasMMWrapper(cublas_handle_, cublaslt_handle_, nullptr, cublasAlgoMap_, cublasWrapperMutex_, allocator_);
    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }

    vit_transformer_ = new MobileViTTransformer<T>(settings_.max_batch_size,
                                             settings_.embed_dim,
                                             settings_.head_num,
                                             settings_.inter_size,
                                             settings_.num_layer,
                                             settings_.seq_len,
                                             settings_.sm,
                                             settings_.q_scaling,
                                             0,
                                             cudnn_handle_,
                                             cublas_wrapper_,
                                             allocator_,
                                             false,
                                             settings_.attention_type);
}

template<typename T>
MobileVisionTransformerPlugin<T>::~MobileVisionTransformerPlugin()
{
    check_cuda_error(cublasDestroy(cublas_handle_));
    check_cuda_error(cublasLtDestroy(cublaslt_handle_));
    checkCUDNN(cudnnDestroy(cudnn_handle_));
    delete cublasWrapperMutex_;
    delete cublasAlgoMap_;
    delete vit_transformer_;
    delete allocator_;
    delete params_;
}

// IPluginV2DynamicExt Methods
template<typename T>
nvinfer1::IPluginV2DynamicExt* MobileVisionTransformerPlugin<T>::clone() const noexcept
{

    MobileVisionTransformerPlugin* ret = new MobileVisionTransformerPlugin<T>(*this);
    return ret;
}

///! #todo
template<typename T>
DimsExprs MobileVisionTransformerPlugin<T>::getOutputDimensions(int outputIndex,
                                                          const DimsExprs* inputs,
                                                          int nbInputs,
                                                          IExprBuilder& exprBuilder) noexcept
{
    // Input is B*in_chans*H*W, output should be B*seq_len*embed_dim*1
    assert(outputIndex == 0);
    DimsExprs output(*inputs);
    // output.nbDims = 3;
    // output.d[0] = inputs[0].d[0];
    // output.d[1] = exprBuilder.constant(settings_.seq_len);
    // output.d[2] = exprBuilder.constant(settings_.embed_dim);
    return output;
}

template<typename T>
bool MobileVisionTransformerPlugin<T>::supportsFormatCombination(int pos,
                                                           const PluginTensorDesc* inOut,
                                                           int nbInputs,
                                                           int nbOutputs) noexcept
{
    bool res = false;
    assert(pos >= 0 && pos < 2);
    assert(nbInputs == 1);
    switch (pos) {
        case 0:  // input
        case 1:  // output
            res = (inOut[pos].type
                   == (std::is_same<T, half>::value ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT))
                  && (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
            break;
        default:
            break;
    }

    return res;
}

template<typename T>
void MobileVisionTransformerPlugin<T>::configurePlugin(const DynamicPluginTensorDesc* in,
                                                 int nbInputs,
                                                 const DynamicPluginTensorDesc* out,
                                                 int nbOutputs) noexcept
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
}

template<typename T>
size_t MobileVisionTransformerPlugin<T>::getWorkspaceSize(const PluginTensorDesc* inputs,
                                                    int nbInputs,
                                                    const PluginTensorDesc* outputs,
                                                    int nbOutputs) const noexcept
{
    return 0;
}

// IPluginV2Ext Methods
template<typename T>
nvinfer1::DataType MobileVisionTransformerPlugin<T>::getOutputDataType(int index,
                                                                 const nvinfer1::DataType* inputTypes,
                                                                 int nbInputs) const noexcept
{
    assert(index == 0);
    assert(inputTypes[0] == nvinfer1::DataType::kFLOAT || inputTypes[0] == nvinfer1::DataType::kHALF);
    return inputTypes[0];
}

// IPluginV2 Methods
template<typename T>
const char* MobileVisionTransformerPlugin<T>::getPluginType() const noexcept
{
    return VIT_PLUGIN_NAME;
}

template<typename T>
const char* MobileVisionTransformerPlugin<T>::getPluginVersion() const noexcept
{
    return VIT_PLUGIN_VERSION;
}

template<typename T>
int MobileVisionTransformerPlugin<T>::getNbOutputs() const noexcept
{
    return 1;
}

template<typename T>
int MobileVisionTransformerPlugin<T>::initialize() noexcept
{
    return 0;
}

template<typename T>
void MobileVisionTransformerPlugin<T>::terminate() noexcept
{
}

template<typename T>
size_t MobileVisionTransformerPlugin<T>::getSerializationSize() const noexcept
{

    size_t size = sizeof(int) + sizeof(settings_);
    size += params_->GetSerializeSize();
    return size;
}

template<typename T>
void MobileVisionTransformerPlugin<T>::serialize(void* buffer) const noexcept
{
    printf("start serialize vit...\n");

    int type_id = 0;
    if (std::is_same<T, half>::value) {
        type_id = 1;
    }
    ::memcpy(buffer, &type_id, sizeof(type_id));
    char* serial_buffer = (char*)buffer + sizeof(type_id);
    ::memcpy(serial_buffer, &settings_, sizeof(settings_));
    serial_buffer += sizeof(settings_);
    params_->serialize(serial_buffer);
}

template<typename T>
void MobileVisionTransformerPlugin<T>::destroy() noexcept
{
    delete this;
}

template<typename T>
void MobileVisionTransformerPlugin<T>::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

template<typename T>
const char* MobileVisionTransformerPlugin<T>::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template<typename T>
int MobileVisionTransformerPlugin<T>::enqueue(const PluginTensorDesc* inputDesc,
                                        const PluginTensorDesc* outputDesc,
                                        const void* const* inputs,
                                        void* const* outputs,
                                        void* workspace,
                                        cudaStream_t stream) noexcept
{
    ///! input: Batch_size, seq_len, embed_dim
    int batch_size = inputDesc->dims.d[0];
    assert(batch_size <= settings_.max_batch_size);
    // assert(settings_.chn_num == inputDesc->dims.d[1]);
    // assert(settings_.img_size == inputDesc->dims.d[2]);
    // assert(settings_.img_size == inputDesc->dims.d[3]);
    assert(settings_.seq_len == inputDesc->dims.d[1]);
    assert(settings_.embed_dim == inputDesc->dims.d[2]);

    int sm_ptr[1] = {sm_};
    std::vector<Tensor> input_tensors = std::vector<Tensor>{Tensor{
        MEMORY_GPU,
        getTensorType<T>(),
        std::vector<size_t>{
            (size_t)batch_size, (size_t)settings_.seq_len, (size_t)settings_.embed_dim},
        (const T*)(inputs[0])}};
    
    // debug
    // input_tensors[0].save<T>("/target/ml-cvnets/weights/trt_input.npy");

    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU,
               getTensorType<T>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)settings_.seq_len, (size_t)settings_.embed_dim},
               (T*)(outputs[0])}};

    cublas_wrapper_->setStream(stream);
    vit_transformer_->setStream(stream);
    // std::printf("plugin stream = %p \n", stream);
    
    vit_transformer_->forward(&output_tensors, &input_tensors, params_);
    return 0;
}


MobileVisionTransformerPluginCreator::MobileVisionTransformerPluginCreator()
{
    /*
        std::map<std::string, int*> name2pint = {{"max_batch", &max_batch},
                                             {"seq_len", &seq_len},
                                             {"embed_dim", &embed_dim},
                                             {"num_heads", &num_heads},
                                             {"inter_size", &inter_size},
                                             {"layer_num", &layer_num}};
    */

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("max_batch", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("seq_len", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("embed_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("inter_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("layer_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("npz_path", nullptr, PluginFieldType::kCHAR, 1));
    mPluginAttributes.emplace_back(PluginField("npz_path_len", nullptr, PluginFieldType::kINT32, 1));


    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MobileVisionTransformerPluginCreator::getPluginName() const noexcept
{
    return VIT_PLUGIN_NAME;
}

const char* MobileVisionTransformerPluginCreator::getPluginVersion() const noexcept
{
    return VIT_PLUGIN_VERSION;
}

const PluginFieldCollection* MobileVisionTransformerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}



nvinfer1::PluginFieldType getFieldCollectionType(std::string name, const nvinfer1::PluginFieldCollection* fc)
{
    for (int i = 0; i < fc->nbFields; i++) {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare(name) == 0) {
            return fc->fields[i].type;
        }
    }
    return nvinfer1::PluginFieldType::kUNKNOWN;
}


// Creator
#define L_ROOT "Transformer/encoderblock_%d"
#define ATT_Q "MultiHeadDotProductAttention_1/query"
#define ATT_K "MultiHeadDotProductAttention_1/key"
#define ATT_V "MultiHeadDotProductAttention_1/value"
#define ATT_OUT "MultiHeadDotProductAttention_1/out"
#define ATT_NORM "LayerNorm_0"
#define FFN_NORM "LayerNorm_2"
#define FFN_IN "MlpBlock_3/Dense_0"
#define FFN_OUT "MlpBlock_3/Dense_1"

const std::vector<const char*> layer_weight_names = {L_ROOT "/" ATT_NORM "/scale",
                                                    L_ROOT "/" ATT_NORM "/bias",
                                                    L_ROOT "/" ATT_Q "/kernel",
                                                    L_ROOT "/" ATT_Q "/bias",
                                                    L_ROOT "/" ATT_K "/kernel",
                                                    L_ROOT "/" ATT_K "/bias",
                                                    L_ROOT "/" ATT_V "/kernel",
                                                    L_ROOT "/" ATT_V "/bias",
                                                    L_ROOT "/" ATT_OUT "/kernel",
                                                    L_ROOT "/" ATT_OUT "/bias",
                                                    L_ROOT "/" FFN_NORM "/scale",
                                                    L_ROOT "/" FFN_NORM "/bias",
                                                    L_ROOT "/" FFN_IN "/kernel",
                                                    L_ROOT "/" FFN_IN "/bias",
                                                    L_ROOT "/" FFN_OUT "/kernel",
                                                    L_ROOT "/" FFN_OUT "/bias"};


const std::vector<std::string> post_layer_weight_names = {"Transformer/encoder_norm/scale",
                                                        "Transformer/encoder_norm/bias"};

template<typename T>
void loadWeightsPtr(std::vector<const T*>& w,
                    const nvinfer1::PluginFieldCollection* fc,
                    int layer_num)
{



    int idx = 0;

    for (int i = 0; i < layer_num; i++){
        for (auto& name : layer_weight_names) {
            char str_buf[1024];
            sprintf(str_buf, name, i);
            for (int j = 0; j < fc->nbFields; j++) {
                std::string field_name(fc->fields[j].name);
                if (field_name.compare(str_buf) == 0) {
                    w[idx++] = (const T*)fc->fields[j].data;
                }
            }
        }
    }

    for (auto& name : post_layer_weight_names) {
        // std::cout << "name: " << name << std::endl;
        for (int i = 0; i < fc->nbFields; i++) {
            std::string field_name(fc->fields[i].name);
            // std::cout << "field_name: " << field_name << std::endl;
            if (field_name.compare(name) == 0) {
                w[idx++] = (const T*)fc->fields[i].data;
            }
        }
    }
    // std::printf("idx = %d\n", idx);

    FT_CHECK(idx == w.size());
}

IPluginV2* MobileVisionTransformerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{


    int max_batch;
    int embed_dim;
    int seq_len;
    int num_heads;
    int inter_size;
    int layer_num;


    std::map<std::string, int*> name2pint = {{"max_batch", &max_batch},
                                             {"seq_len", &seq_len},
                                             {"embed_dim", &embed_dim},
                                             {"num_heads", &num_heads},
                                             {"inter_size", &inter_size},
                                             {"layer_num", &layer_num}};

    std::string npz_name("npz_path");
    std::vector<PluginField> fieldCollections;
    cnpy::npz_t data_npz;
    
    int npz_path_len = 0;
    std::string npz_path_len_name("npz_path_len");
    for(int i = 0; i < fc->nbFields; i++) {
        if(npz_path_len_name.compare(fc->fields[i].name) == 0) {
            npz_path_len = *((const int*)fc->fields[i].data);
        }
    }

    int type_id = 0;
    std::string type_id_name("type_id");
    for(int i = 0; i < fc->nbFields; i++) {
        if(type_id_name.compare(fc->fields[i].name) == 0) {
            type_id = *((const int*)fc->fields[i].data);
        }
    }
    // std::cout << "type_id: " << type_id << std::endl;

    for(int i = 0; i < fc->nbFields; i++) {
        if(npz_name.compare(fc->fields[i].name) == 0) {
            string npz_path((const char*)fc->fields[i].data, npz_path_len);

            data_npz = cnpy::npz_load(npz_path);
            for(auto& iter: data_npz) {
                auto& item = iter.second;
                if(0 == type_id) {
                    fieldCollections.emplace_back(PluginField(iter.first.c_str(), item.data<float>(), PluginFieldType::kFLOAT32, item.num_vals));
                } else if(1 == type_id) {
                    fieldCollections.emplace_back(PluginField(iter.first.c_str(), item.data<half>(), PluginFieldType::kFLOAT16, item.num_vals));
                }
            }
        } else {
            fieldCollections.emplace_back(fc->fields[i]);
        }
    }

    PluginFieldCollection* new_fc = new PluginFieldCollection();
    new_fc->nbFields = fieldCollections.size();
    new_fc->fields = fieldCollections.data();

    for (int i = 0; i < new_fc->nbFields; i++) {
        auto iter = name2pint.find(new_fc->fields[i].name);
        if (iter != name2pint.end()) {
            *(iter->second) = *((int*)new_fc->fields[i].data);
            continue;
        } 
    }


    size_t weights_num =
        post_layer_weight_names.size() + layer_num * layer_weight_names.size();

    auto weights_type = getFieldCollectionType(post_layer_weight_names[0], new_fc);

    std::vector<const half*> w_fp16;
    std::vector<const float*> w_fp32;
    IPluginV2* p;
    switch (weights_type) {
        case nvinfer1::PluginFieldType::kFLOAT16:
            w_fp16.resize(weights_num);
            loadWeightsPtr(w_fp16, new_fc, layer_num);
            p = new MobileVisionTransformerPlugin<half>(name,
                                                  max_batch,
                                                  embed_dim,
                                                  seq_len,
                                                  num_heads,
                                                  inter_size,
                                                  layer_num,
                                                  1.0,
                                                  w_fp16);

            break;
        case nvinfer1::PluginFieldType::kFLOAT32:
            w_fp32.resize(weights_num);
            loadWeightsPtr(w_fp32, new_fc, layer_num);
            p = new MobileVisionTransformerPlugin<float>(name,
                                                   max_batch,
                                                   embed_dim,
                                                   seq_len,
                                                   num_heads,
                                                   inter_size,
                                                   layer_num,
                                                   1.0,
                                                   w_fp32);
            break;
        default:
            delete new_fc;
            std::printf("[ERROR][MobileVisionTransformerPluginCreator::createPlugin] unsupport datatype: %d.\n", (int)weights_type);
            exit(-1);
    }
    delete new_fc;
    return p;
}

IPluginV2* MobileVisionTransformerPluginCreator::deserializePlugin(const char* name,
                                                             const void* serialData,
                                                             size_t serialLength) noexcept
{
    int type_id;
    ::memcpy(&type_id, serialData, sizeof(int));
    char* modelData = (char*)serialData + sizeof(int);

    // This object will be deleted when the network is destroyed, which will
    // call MobileVisionTransformerPlugin::destroy()
    if (type_id == 0)
        return new MobileVisionTransformerPlugin<float>(name, modelData, serialLength);
    else if (type_id == 1)
        return new MobileVisionTransformerPlugin<half>(name, modelData, serialLength);
    else {
        printf("[ERROR][MobileVisionTransformerPluginCreator::deserializePlugin] unsupport data type %d\n", type_id);
        exit(-1);
    }
}

void MobileVisionTransformerPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    namespace_ = libNamespace;
}

const char* MobileVisionTransformerPluginCreator::getPluginNamespace() const noexcept
{
    return namespace_.c_str();
}

template class MobileVisionTransformerPlugin<half>;
template class MobileVisionTransformerPlugin<float>;

}  // namespace fastertransformer
