#include "swish.cuh"
#include "bertCommon.h"
#include "trt_tensor.hpp"


#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>


int test_swish() {
    ::srand(::time(0));
    std::cout << ::rand() << std::endl;
    // std::cout << "test for cuda" << std::endl;

    int batch_size = 1;
    int seq_len = 63;
    int hidden_dim = 11;
    int num_head = 23;


    TRT::Tensor q_tensor(std::vector<int>{seq_len, batch_size, num_head, hidden_dim});
    TRT::Tensor out_tensor(std::vector<int>{seq_len, batch_size, num_head, hidden_dim});

    auto qptr_cpu = q_tensor.cpu<float>();

    for(int i = 0; i < q_tensor.numel(); ++i) {
        qptr_cpu[i] = float(rand() % 100000) / 100000;
    }

    q_tensor.save_to_file("q_tensor.npz");


    auto qptr_gpu = q_tensor.to_gpu(true).gpu<float>();
    auto outptr_gpu = out_tensor.to_gpu().gpu<float>();
    int elem_cnt = out_tensor.numel();

    cudaStream_t stream = out_tensor.get_stream();
    
    CUASSERT((oneflow::cuda::elementwise::Unary< ACTIVATION::SwishFunctor<float>, float, float >(
      ACTIVATION::SwishFunctor<float>(), elem_cnt, outptr_gpu, qptr_gpu, stream)));
    CUASSERT(cudaStreamSynchronize(stream));

    out_tensor.to_cpu(true);
    out_tensor.save_to_file("out_tensor.npz");
    

    return 0;
}

int main() {

    test_swish();

    return 0;
}