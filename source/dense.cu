#include "dense.hpp"
#include <assert.h>
#include "coo.hpp"

void DenseTensor::freeAllArrays() {
    free(values_h);
    cudaErrorCheck(cudaFree(values_d));
}

// safely uploads to gpu
void DenseTensor::uploadToDevice() {
    cudaErrorCheck(cudaFree(values_d));
    cudaErrorCheck(cudaMalloc((void **) &values_d, sizeof(float) * width*height*depth));
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(float) * width*height*depth, cudaMemcpyHostToDevice));
}

// safely downloads from gpu
void DenseTensor::downloadToHost() {
    free(values_h);
    values_h = (float*)malloc(sizeof(float) * width*height*depth);
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(float) * width*height*depth, cudaMemcpyDeviceToHost));
}


void DenseMatrix::freeAllArrays() {
    free(values_h);
    cudaErrorCheck(cudaFree(values_d));
    values_h = nullptr;
    values_d = nullptr;
}

// safely uploads to gpu
void DenseMatrix::uploadToDevice() {
    cudaErrorCheck(cudaFree(values_d));
    cudaErrorCheck(cudaMalloc((void **) &values_d, sizeof(float) * width*height));
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(float) * width*height, cudaMemcpyHostToDevice));
}

// safely downloads from gpu
void DenseMatrix::downloadToHost() {
    free(values_h);
    values_h = (float*)malloc(sizeof(float) * width*height);
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(float) * width*height, cudaMemcpyDeviceToHost));
}


CooTensorManager DenseTensor::toCoo() {
    CooTensorManager ret;
    assert(0);
    return ret;
}


DenseMatrixManager DenseTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    DenseMatrix a = ret;
    assert(values_h != nullptr);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    int I = this->width, J = d.height, K = this->height, L = this->depth;
    assert(d.width  == L);
    assert(c.width  == K);
    assert(c.height == J);

    a.setSize(I, J);

    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; j++) {
            for(int k = 0; k < K; k++) {
              for(int l = 0; l < L; l++) {
                  a.access(i, j) += access(i,k,l) * d.access(l,j) * c.access(k,j);
              }
            }
        }
    }


    return ret;
}

DenseMatrixManager DenseTensor::mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(values_d != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager DenseTensor::mttkrp_fast(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}
