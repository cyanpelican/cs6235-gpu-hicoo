#include "coo.hpp"

void CooTensor::freeAllArrays() {
    free(points_h);
    cudaErrorCheck(cudaFree(points_d));
}

void CooTensor::uploadToDevice() {
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaMalloc((void **) &d_weight, sizeof(float)));
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(CooPoint) * num_elements, cudaMemcpyHostToDevice));
}

void CooTensor::downloadToHost() {
    free(points_h);
    points_h = malloc(sizeof(CooPoint) * num_elements);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(CooPoint) * num_elements, cudaMemcpyDeviceToHost));
}

DenseMatrixManager CooTensor::mttkrp(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO

    return ret;
}
