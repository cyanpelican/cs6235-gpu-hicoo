#include "hicoo.hpp"
#include <assert.h>

void HicooTensor::freeAllArrays() {
    free(points_h);
    free(blocks_h);
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaFree(blocks_d));
}

void HicooTensor::uploadToDevice() {
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaMalloc((void **) &d_weight, sizeof(HicooPoint) * num_elements));
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(HicooPoint) * num_elements, cudaMemcpyHostToDevice));
}

void HicooTensor::downloadToHost() {
    free(points_h);
    points_h = malloc(sizeof(HicooPoint) * num_elements);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(HicooPoint) * num_elements, cudaMemcpyDeviceToHost));
}

DenseMatrixManager HicooTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(points_h != nullptr);
    assert(blocks_h != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager HicooTensor::mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(values_d != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager HicooTensor::mttkrp_fast(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}
