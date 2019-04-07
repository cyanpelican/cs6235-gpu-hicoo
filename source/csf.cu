#include "csf.hpp"

void CsfTensor::freeAllArrays() {
    free(points_h);
    free(fiberAddresses_h);
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaFree(fiberAddresses_d));
}

void CsfTensor::uploadToDevice() {
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaMalloc((void **) &d_weight, sizeof(float)));
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(CsfPoint) * num_elements, cudaMemcpyHostToDevice));
}

void CsfTensor::downloadToHost() {
    free(points_h);
    points_h = malloc(sizeof(CsfPoint) * num_elements);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(CsfPoint) * num_elements, cudaMemcpyDeviceToHost));
}

DenseMatrixManager CsfTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO

    return ret;
}

DenseMatrixManager CsfTensor::mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO

    return ret;
}

DenseMatrixManager CsfTensor::mttkrp_fast(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO

    return ret;
}
