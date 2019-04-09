#include "dense.hpp"
#include <assert.h>

void DenseTensor::freeAllArrays() {
    free(values_h);
    cudaErrorCheck(cudaFree(values_d));
}

// safely uploads to gpu
void DenseTensor::uploadToDevice() {
    cudaErrorCheck(cudaFree(values_d));
    cudaErrorCheck(cudaMalloc((void **) &d_weight, sizeof(float)));
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(DensePoint) * num_elements, cudaMemcpyHostToDevice));
}

// safely downloads from gpu
void DenseTensor::downloadToHost() {
    free(values_h);
    values_h = malloc(sizeof(DensePoint) * num_elements);
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(DensePoint) * num_elements, cudaMemcpyDeviceToHost));
}


void DenseMatrix::freeAllArrays() {
    free(values_h);
    cudaErrorCheck(cudaFree(values_d));
}

// safely uploads to gpu
void DenseMatrix::uploadToDevice() {
    cudaErrorCheck(cudaFree(values_d));
    cudaErrorCheck(cudaMalloc((void **) &d_weight, sizeof(float)));
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(CooPoint) * num_elements, cudaMemcpyHostToDevice));
}

// safely downloads from gpu
void DenseMatrix::downloadToHost() {
    free(values_h);
    values_h = malloc(sizeof(CooPoint) * num_elements);
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(CooPoint) * num_elements, cudaMemcpyDeviceToHost));
}

DenseMatrixManager DenseTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(values_h != nullptr);

    // TODO - remalloc arrays
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; l++) {
            for(int k = 0; k < K; k++) {
              for(int l = 0; l < L; l++) {
                  ret.access(i, j) += B.access(i,j,k) * D.access(l,j) * C.access(k,j);
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
