#include "hicoo.hpp"

void HicooTensor::freeAllArrays() {
    free(points_h);
    free(blocks_h);
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaFree(blocks_d));
}

void HicooTensor::uploadToDevice() {
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaMalloc((void **) &d_weight, sizeof(float)));
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(HicooPoint) * num_elements, cudaMemcpyHostToDevice));
}

void HicooTensor::downloadToHost() {
    free(points_h);
    points_h = malloc(sizeof(HicooPoint) * num_elements);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(HicooPoint) * num_elements, cudaMemcpyDeviceToHost));
}

DenseMatrixManager HicooTensor::mttkrp(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO

    return ret;
}
