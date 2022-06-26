#ifndef SWISH_H_
#define SWISH_H_

#include <cuda_fp16.h>
#include "elementwise.cuh"

namespace ACTIVATION {

template <typename T> 
struct SwishFunctor {
    __device__ T operator()(T x) const;
};



}

#endif