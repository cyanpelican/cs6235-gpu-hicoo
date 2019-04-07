#include "dense.hpp"

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

    // TODO

    return ret;
}

DenseMatrixManager DenseTensor::mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO

    return ret;
}

DenseMatrixManager DenseTensor::mttkrp_fast(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO

    return ret;
}
