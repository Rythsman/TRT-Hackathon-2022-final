#ifndef SWISH_H_
#define SWISH_H_

#include "elementwise.cuh"
#include <cuda_fp16.h>



namespace ACTIVATION {


template<typename T>
__device__ __forceinline__ T swish(T x)
{
    return x / (static_cast<T>(1.0) + static_cast<T>(expf(-x)));
}

template<>
__device__ __forceinline__ half2 swish(half2 x)
{
    float2 tmp = __half22float2(x);
    tmp.x = tmp.x / (1.0f + expf(-tmp.x));
    tmp.y = tmp.y / (1.0f + expf(-tmp.y));
    return __float22half2_rn(tmp);
}

#ifdef ENABLE_BF16
template<>
__device__ __forceinline__ __nv_bfloat162 swish(__nv_bfloat162 x)
{
    float2 tmp = bf1622float2(x);

    tmp.x = tmp.x / (1.0f + expf(-tmp.x));
    tmp.y = tmp.y / (1.0f + expf(-tmp.y));

    return __floats2bfloat162_rn(tmp.x, tmp.y);
}
#endif


template <typename T> 
struct SwishFunctor 
{
    __device__ __forceinline__ T operator()(T x) const 
    {
        return swish<T>(x);
    }
};

// template<>
// struct SwishFunctor<half> 
// {
//     __device__ __forceinline__ half2 operator()(half2 x) const 
//     {
//         return swish<half>(x);
//     }

//     __device__ void Apply2(half* to, const From* from) const {
//         float2 f2;
//         f2.x = static_cast<float>(from[0]);
//         f2.y = static_cast<float>(from[1]);
//         *reinterpret_cast<half2*>(to) = __float22half2_rn(f2);
//     }
// };

}


#endif
