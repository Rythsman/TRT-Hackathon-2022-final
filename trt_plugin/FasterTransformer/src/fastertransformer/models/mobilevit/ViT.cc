/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/mobilevit/ViT.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/vit_kernels.h"

#define SAFE_FREE(x)                                                                                                   \
    if (x) {                                                                                                           \
        delete x;                                                                                                      \
        x = nullptr;                                                                                                   \
    }

namespace fastertransformer {

template<typename T>
void MobileViTTransformer<T>::initialize()
{
    FT_LOG_DEBUG("batch_size:%lu \n"
                 "seq_len   :%lu, embed_dim: %lu\n"
                 "head_num  :%lu, head_dim : %lu\n"
                 "inter_size:%lu, num_layer: %lu\n"
                 "att_type  : %d, \n",
                 max_batch_size_,
                 max_seq_len_,
                 embed_dim_,
                 head_num_,
                 head_dim_,
                 inter_size_,
                 num_layer_,
                 int(attention_type_));

    if (head_num_ * head_dim_ != embed_dim_) {
        std::ostringstream buffer;
        buffer << "[FT][ERROR] Embed size and head number mismatch. Embed_dim=" << embed_dim_
               << "; head_num*head_dim = "
               << "(" << head_num_ << "*" << head_dim_ << ")=" << head_num_ * head_dim_ << std::endl;
        throw std::runtime_error(buffer.str());
    }

    max_seq_len_ = request_seq_len_;
    if ((attention_type_ == AttentionType::FUSED_MHA) && std::is_same<T, half>::value == true && max_seq_len_ <= 384) {

        attention_layer_ = new FusedAttentionLayer<T>(max_batch_size_,
                                                      max_seq_len_,
                                                      head_num_,
                                                      head_dim_,
                                                      sm_,
                                                      q_scaling_,
                                                      stream_,
                                                      cublas_wrapper_,
                                                      allocator_,
                                                      is_free_buffer_after_forward_,
                                                      false);
    }
    else if (attention_type_ == AttentionType::UNFUSED_MHA) {
        if (request_seq_len_ % 8 != 0 && std::is_same<half, T>::value) {
            max_seq_len_ = (request_seq_len_ + 7) / 8 * 8;
            FT_LOG_DEBUG("Request sequence length(%lu) is odd with unfused mha. Padding to %lu\n",
                         request_seq_len_,
                         max_seq_len_);
        }
        attention_layer_ = new UnfusedAttentionLayer<T>(max_batch_size_,
                                                        max_seq_len_,
                                                        head_num_,
                                                        head_dim_,
                                                        q_scaling_,
                                                        stream_,
                                                        cublas_wrapper_,
                                                        allocator_,
                                                        is_free_buffer_after_forward_,
                                                        false);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] Invalid attention type or sequence length\n"));
    }

    ffn_layer_ = new SwishFfnLayer<T>(max_batch_size_,
                                     max_seq_len_,
                                     head_num_,
                                     head_dim_,
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_,
                                     false);
}

template<typename T>
MobileViTTransformer<T>::MobileViTTransformer(size_t max_batch_size,
                                  size_t embed_dim,
                                  size_t head_num,
                                  size_t inter_size,
                                  size_t num_layer,
                                  size_t seq_len,
                                  int sm,
                                  float q_scaling,
                                  cudaStream_t stream,
                                  cudnnHandle_t cudnn_handle,
                                  cublasMMWrapper* cublas_wrapper,
                                  IAllocator* allocator,
                                  bool is_free_buffer_after_forward,
                                  AttentionType attention_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    request_seq_len_(seq_len),
    max_seq_len_(0),
    embed_dim_(embed_dim),
    head_num_(head_num),
    head_dim_(embed_dim / head_num),
    inter_size_(inter_size),
    num_layer_(num_layer),
    sm_(sm),
    q_scaling_(q_scaling),
    attention_type_(attention_type),
    cudnn_handle_(cudnn_handle)
{
    initialize();
}

template<typename T>
MobileViTTransformer<T>::MobileViTTransformer(MobileViTTransformer<T> const& vit):
    BaseLayer(vit),
    max_batch_size_(vit.max_batch_size_),
    max_seq_len_(vit.max_seq_len_),
    request_seq_len_(vit.request_seq_len_),
    embed_dim_(vit.embed_dim_),
    head_num_(vit.head_num_),
    head_dim_(vit.head_dim_),
    inter_size_(vit.inter_size_),
    num_layer_(vit.num_layer_),
    sm_(vit.sm_),
    q_scaling_(vit.q_scaling_),
    attention_type_(vit.attention_type_),
    cudnn_handle_(vit.cudnn_handle_)
{
    initialize();
}

template<typename T>
MobileViTTransformer<T>::~MobileViTTransformer()
{
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void MobileViTTransformer<T>::setStream(cudaStream_t stream)
{
    BaseLayer::setStream(stream);
    if (attention_layer_)
        attention_layer_->setStream(stream);
    if (ffn_layer_)
        ffn_layer_->setStream(stream);
}

template<typename T>
void MobileViTTransformer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        embed_buf_1_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        embed_buf_2_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        embed_buf_3_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * embed_dim_, false);
        mask_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_, false);
        padding_offset_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_ * max_seq_len_, false);
        token_num_ = (size_t*)allocator_->malloc(sizeof(size_t) * 1, false);

        trt_mha_padding_offset_ = (int*)allocator_->malloc(sizeof(int) * (2 * max_batch_size_ + 1), false);
        seq_len_vec_ = (int*)allocator_->malloc(sizeof(int) * max_batch_size_, false);

        setSeqLenVec(max_batch_size_);
        setDefaultMask(max_batch_size_);
        setDefaultPaddingOffset(max_batch_size_);

        is_allocate_buffer_ = true;
    }
}

#define REMALLOC(var, size)                                                                                            \
    {                                                                                                                  \
        var = (decltype(var))allocator_->reMalloc(var, size, false);                                                   \
        FT_LOG_DEBUG("ReMalloc %s, ptr:%x, size:%lu", #var, var, size);                                                \
    }

template<typename T>
void MobileViTTransformer<T>::allocateBuffer(size_t batch_size)
{
    if (is_allocate_buffer_ && batch_size <= max_batch_size_) {
        return;
    }

    batch_size = batch_size > max_batch_size_ ? batch_size : max_batch_size_;
    REMALLOC(embed_buf_1_, sizeof(T) * batch_size * max_seq_len_ * embed_dim_);
    REMALLOC(embed_buf_2_, sizeof(T) * batch_size * max_seq_len_ * embed_dim_);
    REMALLOC(embed_buf_3_, sizeof(T) * batch_size * max_seq_len_ * embed_dim_);
    REMALLOC(mask_buf_, sizeof(T) * batch_size * max_seq_len_ * max_seq_len_);
    REMALLOC(padding_offset_, sizeof(int) * batch_size * max_seq_len_);
    REMALLOC(token_num_, sizeof(size_t) * 1);
    REMALLOC(trt_mha_padding_offset_, sizeof(int) * (2 * batch_size + 1));
    REMALLOC(seq_len_vec_, sizeof(int) * batch_size);
    resetBatch(batch_size);
    setSeqLenVec(batch_size);
    setDefaultPaddingOffset(batch_size);
    setDefaultMask(batch_size);

    is_allocate_buffer_ = true;
}

template<typename T>
void MobileViTTransformer<T>::freeBuffer()
{
    allocator_->free(embed_buf_1_);
    allocator_->free(embed_buf_2_);
    allocator_->free(embed_buf_3_);
    allocator_->free(mask_buf_);
    allocator_->free(trt_mha_padding_offset_);
    allocator_->free(seq_len_vec_);
    allocator_->free(padding_offset_);
    allocator_->free(token_num_);

    is_allocate_buffer_ = false;
}

template<typename T>
void MobileViTTransformer<T>::forward(std::vector<Tensor>* output_tensors,
                                const std::vector<Tensor>* input_tensors,
                                const ViTWeight<T>* weights)
{


// input_tensors:
    //      input_img, BCHW [batch, seq_len, embed_dim]
    // output tensors:
    //      output feature_map [batch, seq_len, embed_dim]

    const size_t input_batch_size = input_tensors->at(0).shape[0];
    const size_t input_seq_len = input_tensors->at(0).shape[1];

    size_t seq_len = input_seq_len;

    FT_CHECK(input_seq_len == request_seq_len_);
    FT_CHECK(input_tensors->size() == 1);
    FT_CHECK(input_tensors->at(0).shape.size() == 3);
    FT_CHECK(output_tensors->size() == 1);
    FT_CHECK(output_tensors->at(0).shape.size() == 3);
    allocateBuffer(input_batch_size);

    const T* input = (const T*)input_tensors->at(0).data;
    T* output = (T*)output_tensors->at(0).data;
    
    // T* encoder_input_ptr = const_cast<T*>(input);
    T* encoder_input_ptr = embed_buf_1_;
    
    cudaMemcpyAsync(encoder_input_ptr, input, sizeof(T) * input_batch_size * seq_len * embed_dim_, cudaMemcpyDeviceToDevice, stream_);
    DataType data_type = getTensorType<T>();

    // input_tensors:
    //      input_img, BCHW [batch, seq_len, embed_dim]   => B,S,E
    // output tensors:
    //      output feature_map [batch, seq_len, embed_dim]



    size_t h_token_num = input_batch_size * seq_len;
    // get offsets
    Tensor* offset_tensor_ptr;
    if (attention_type_ == AttentionType::FUSED_MHA) {
        invokeGetTrtPaddingOffset(trt_mha_padding_offset_, seq_len_vec_, input_batch_size, stream_);
        offset_tensor_ptr =
            new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{input_batch_size + 1}, trt_mha_padding_offset_);
    }
    else {
        offset_tensor_ptr = new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{0}, nullptr);
    }

    T* from_buf = encoder_input_ptr;
    T* norm_out_buf = embed_buf_2_;
    T* attn_out_buf = embed_buf_3_;
    T* encoder_out_buf = from_buf;

    for (uint i = 0; i < num_layer_; i++) {

        invokeGeneralLayerNorm(norm_out_buf,
                               from_buf,
                               weights->vit_layer_weights[i].attn_layernorm_weights.gamma,
                               weights->vit_layer_weights[i].attn_layernorm_weights.beta,
                               h_token_num,
                               embed_dim_,
                               stream_);

        // Attention
        {

            std::vector<Tensor> attn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, embed_dim_}, norm_out_buf},
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{input_batch_size, 1, seq_len, seq_len}, mask_buf_},
                *offset_tensor_ptr};
            std::vector<Tensor> attn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, embed_dim_}, attn_out_buf}};

            attention_layer_->forward(
                &attn_output_tensors, &attn_input_tensors, &weights->vit_layer_weights[i].attention_weights);

        }

        invokeGeneralAddBiasResidualPreLayerNorm(
            from_buf,
            norm_out_buf,
            attn_out_buf,
            weights->vit_layer_weights[i].ffn_layernorm_weights.gamma,
            weights->vit_layer_weights[i].ffn_layernorm_weights.beta,
            weights->vit_layer_weights[i].attention_weights.attention_output_weight.bias,
            h_token_num,
            embed_dim_,
            stream_);


        // FFN
        {
            std::vector<Tensor> ffn_input_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, embed_dim_}, norm_out_buf}};
            std::vector<Tensor> ffn_output_tensors{
                Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, embed_dim_}, attn_out_buf}};
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &weights->vit_layer_weights[i].ffn_weights);
        }
        
        

        invokeAddBiasResidual(from_buf,
                              attn_out_buf,
                              weights->vit_layer_weights[i].ffn_weights.output_weight.bias,
                              h_token_num,
                              embed_dim_,
                              stream_);
        

        sync_check_cuda_error();
    }



    invokeGeneralLayerNorm(output,
                           from_buf,
                           weights->post_transformer_layernorm_weights.gamma,
                           weights->post_transformer_layernorm_weights.beta,
                           h_token_num,
                           embed_dim_,
                           stream_);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();

    SAFE_FREE(offset_tensor_ptr);
}

template<typename T>
bool MobileViTTransformer<T>::resetBatch(size_t batch_size)
{
    if (max_batch_size_ < batch_size) {
        max_batch_size_ = batch_size;
    }

    return true;
}

template<typename T>
bool MobileViTTransformer<T>::setSeqLenVec(size_t batch_size)
{
    int* seq_len_vec = new int[batch_size];
    for (int i = 0; i < batch_size; i++) {
        seq_len_vec[i] = request_seq_len_;
    }
    cudaMemcpy(seq_len_vec_, seq_len_vec, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    delete seq_len_vec;
    return true;
}

template<typename T>
void MobileViTTransformer<T>::setDefaultMask(size_t batch_size)
{
    invokeBuildEncoderAttentionMask(mask_buf_, seq_len_vec_, batch_size, max_seq_len_, stream_);
}

template<typename T>
void MobileViTTransformer<T>::setDefaultPaddingOffset(size_t batch_size)
{
    invokeGetPaddingOffset(
        &nopad_token_num_, token_num_, padding_offset_, seq_len_vec_, batch_size, max_seq_len_, stream_);
}

template<typename T>
void MobileViTTransformer<T>::patchEmbed(T* output,
                                   const T* input,
                                   const int batch,
                                   const int seq_len,
                                   const int embed_dim)
{
    // Todo padding
    T* tmp_buf = output;
}

template class MobileViTTransformer<float>;
template class MobileViTTransformer<half>;

}  // namespace fastertransformer
