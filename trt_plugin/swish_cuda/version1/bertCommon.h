/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cublas_v2.h>
#include <sstream>
#include <iostream>

#ifndef CUDA_VERSION
#define CUDA_VERSION 10020
#endif

#if CUDA_VERSION >= 10010

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#ifndef BERT_COMMON_H
#define BERT_COMMON_H

#define CHECK(status)                                                                                              \
do                                                                                                                 \
{                                                                                                                  \
    if (status != 0)                                                                                               \
        abort();                                                                                                   \
} while (0)

class LogStream : public std::ostream
{
    class Buf : public std::stringbuf
    {
    public:
        int sync() override;
    };

    Buf buffer;

public:
    LogStream() : std::ostream(&buffer) {};
};

int LogStream::Buf::sync()
{
    std::string s = str();
    while (!s.empty() && s.back() == '\n')
    {
        s.pop_back();
    }
    
    std::cout << "[L]:" << s << std::endl;
    
    str("");
    return 0;
}

LogStream gLogError;

class TRTException : public std::exception
{
public:
    TRTException(const char* fl, const char* fn, int ln, int st, const char* msg, const char* nm)
        : file(fl)
        , function(fn)
        , line(ln)
        , status(st)
        , message(msg)
        , name(nm)
    {
    }
    virtual void log(std::ostream& logStream) const;
    void setMessage(const char* msg) { message = msg; }

protected:
    const char* file{nullptr};
    const char* function{nullptr};
    int line{0};
    int status{0};
    const char* message{nullptr};
    const char* name{nullptr};
};

void TRTException::log(std::ostream& logStream) const
{
    logStream << file << " (" << line << ") - " << name << " Error in " << function << ": " << status;
    if (message != nullptr)
    {
        logStream << " (" << message << ")";
    }
    logStream << std::endl;
}

class CublasError : public TRTException
{
public:
    CublasError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "cuBLAS")
    {
    }
};

void throwCublasError(const char* file, const char* function, int line, int status, const char* msg = nullptr)
{
    if (msg == nullptr)
    {
        auto s_ = static_cast<cublasStatus_t>(status);
        switch (s_)
        {
        case CUBLAS_STATUS_SUCCESS: msg = "CUBLAS_STATUS_SUCCESS"; break;
        case CUBLAS_STATUS_NOT_INITIALIZED: msg = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
        case CUBLAS_STATUS_ALLOC_FAILED: msg = "CUBLAS_STATUS_ALLOC_FAILED"; break;
        case CUBLAS_STATUS_INVALID_VALUE: msg = "CUBLAS_STATUS_INVALID_VALUE"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH: msg = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
        case CUBLAS_STATUS_MAPPING_ERROR: msg = "CUBLAS_STATUS_MAPPING_ERROR"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: msg = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR: msg = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
        case CUBLAS_STATUS_NOT_SUPPORTED: msg = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
        case CUBLAS_STATUS_LICENSE_ERROR: msg = "CUBLAS_STATUS_LICENSE_ERROR"; break;
        }
    }
    CublasError error(file, function, line, status, msg);
    error.log(gLogError);
    throw error;
}

#define CUBLASASSERTMSG(status_, msg)                                                                                  \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            throwCublasError(__FILE__, FN_NAME, __LINE__, s_, msg);                                  \
        }                                                                                                              \
    }

#define CUBLASASSERT(status_)                                                                                          \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            throwCublasError(__FILE__, FN_NAME, __LINE__, s_);                                       \
        }                                                                                                              \
    }

class CudaError : public TRTException
{
public:
    CudaError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Cuda")
    {
    }
};

// break-pointable
void throwCudaError(const char* file, const char* function, int line, int status, const char* msg)
{
    CudaError error(file, function, line, status, msg);
    error.log(gLogError);
    throw error;
}

#define CUASSERT(status_)                                                                                              \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            const char* msg = cudaGetErrorString(s_);                                                                  \
            throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg);                                    \
        }                                                                                                              \
    }

#include "cublas_v2.h"
#include "cuda_fp16.h"

#include <algorithm>
#include <cassert>
#include <cuda_runtime_api.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#define TRT_UNUSED (void)

#define BERT_PRINT_DEBUG_MSG 1

#if BERT_PRINT_DEBUG_MSG
#define BERT_DEBUG_MSG(msg) (gLogVerbose << (msg) << std::endl)
#define BERT_DEBUG_VALUE(key, value) (gLogVerbose << key << value << std::endl)
#else
#define BERT_DEBUG_MSG(msg) TRT_UNUSED (msg)
#define BERT_DEBUG_VALUE(key, value) TRT_UNUSED (key); TRT_UNUSED (value)
#endif

using half = __half;



constexpr int32_t kSM_53 = 53;
constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;

// For full mask mode, we must produce the compressed mask format expected by the fused attention path. Currently, only
// two sequence lengths are supported. We hard code the sizes here.
// The number of threads per CTA: warps_m * warps_n * warps_k * 32;
constexpr size_t threadsPerCta128 = 2 * 2 * 32;
constexpr size_t threadsPerCta384 = 1 * 8 * 32;

// The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M dimension: (s + 16*warps_m - 1)
// / (16*warps_m);
constexpr size_t xmmasM128 = 4;
constexpr size_t xmmasM384 = 24;

// Packed mask size per batch. Layout is XMMAS_M * THREADS_PER_CTA.
constexpr size_t unfusedMaskSize = 1;
constexpr size_t packedMaskSize64 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize96 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize128 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize384 = xmmasM384 * threadsPerCta384;

namespace bert
{

inline int getSMVersion()
{
    int device{-1};
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CHECK(cudaGetDeviceProperties(&props, device));
    return props.major * 10 + props.minor;
}



template <typename IntType>
constexpr IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}
template <typename IntType>
constexpr IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

template <typename T>
inline T* deserToDev(const char*& buffer, size_t nbElem)
{
    void* dev{nullptr};
    const size_t len = sizeof(T) * nbElem;
    CUASSERT(cudaMalloc(&dev, len));
    CUASSERT(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

    buffer += len;
    return static_cast<T*>(dev);
}

template <typename T>
inline void serFromDev(char*& buffer, const T* data, size_t nbElem)
{
    const size_t len = sizeof(T) * nbElem;
    CUASSERT(cudaMemcpy(buffer, static_cast<const void*>(data), len, cudaMemcpyDeviceToHost));
    buffer += len;
}

template <typename T>
inline T* devToDev(const T* data, size_t nbElem)
{
    void* dev{nullptr};
    const size_t len = sizeof(T) * nbElem;
    CUASSERT(cudaMalloc(&dev, len));
    CUASSERT(cudaMemcpy(dev, static_cast<const void*>(data), len, cudaMemcpyDeviceToDevice));
    return static_cast<T*>(dev);
}

template <typename T>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const T alpha, const T* A, int lda, const T* B, int ldb, const T beta, T* C, int ldc);

template <>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const float alpha, const float* A, int lda, const float* B, int ldb, const float beta, float* C,
    int ldc)
{

    return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const half alpha, const half* A, int lda, const half* B, int ldb, const half beta, half* C, int ldc)
{
    return cublasHgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <typename T>
cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const T alpha, const T* A, int lda, long long int strideA,
    const T* B, int ldb, long long int strideB, const T beta, T* C, int ldc, long long int strideC, int batchCount,
    cublasGemmAlgo_t algo);

template <>
cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const float alpha, const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC,
    int batchCount, cublasGemmAlgo_t algo)
{

    return ::cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA, B,
        CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC, batchCount, CUDA_R_32F, algo);
}

template <>
cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const half alpha, const half* A, int lda, long long int strideA,
    const half* B, int ldb, long long int strideB, const half beta, half* C, int ldc, long long int strideC,
    int batchCount, cublasGemmAlgo_t algo)
{
    return ::cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_16F, lda, strideA, B,
        CUDA_R_16F, ldb, strideB, &beta, C, CUDA_R_16F, ldc, strideC, batchCount, CUDA_R_16F, algo);
}

template <typename T>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const T alpha, const T* A, int lda, long long int strideA,
    const T* B, int ldb, long long int strideB, const T beta, T* C, int ldc, long long int strideC, int batchCount);

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const float alpha, const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC,
    int batchCount)
{

    return cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const half alpha, const half* A, int lda, long long int strideA,
    const half* B, int ldb, long long int strideB, const half beta, half* C, int ldc, long long int strideC,
    int batchCount)
{
    return cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

struct CublasConfigHelper
{
    cublasPointerMode_t pm;
    cublasMath_t mm;
    cublasHandle_t cublas;
    CublasConfigHelper(cublasHandle_t cublas_)
        : cublas(cublas_)
    {
        cublasGetPointerMode(cublas, &pm);
        cublasGetMathMode(cublas, &mm);
        cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
        cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);
    }
    ~CublasConfigHelper()
    {
        cublasSetMathMode(cublas, mm);
        cublasSetPointerMode(cublas, pm);
    }
};




} // namespace bert
#endif // BERT_COMMON_H

#endif // CUDA_VERSION >= 10010
