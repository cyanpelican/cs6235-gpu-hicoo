#include "hicoo.hpp"
#include "coo.hpp"


void HicooTensor::freeAllArrays() {
    DEBUG_PRINT("HT: free all arrays\n");
    freeHostArrays();
    freeDeviceArrays();
}
void HicooTensor::freeHostArrays() {
    DEBUG_PRINT("HT: free host arrays\n");
    DEBUG_PRINT("    - points_h = %p\n", points_h);
    DEBUG_PRINT("    - blocks_h = %p\n", blocks_h);
    free(points_h);
    free(blocks_h);
    points_h = nullptr;
    blocks_h = nullptr;
}
void HicooTensor::freeDeviceArrays() {
    DEBUG_PRINT("HT: free device arrays\n");
    DEBUG_PRINT("    - points_d = %p\n", points_d);
    DEBUG_PRINT("    - blocks_d = %p\n", blocks_d);
    if(points_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(points_d));
    if(blocks_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(blocks_d));
    points_d = nullptr;
    blocks_d = nullptr;
}

void HicooTensor::uploadToDevice() {
    DEBUG_PRINT("HT: upload to device\n");
    freeDeviceArrays();

    cudaErrorCheck(cudaMalloc((void **) &points_d, sizeof(HicooPoint) * numPoints));
    assert(points_d != nullptr);
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(HicooPoint) * numPoints, cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc((void **) &blocks_d, sizeof(HicooBlock) * (numBlocks+1)));
    assert(blocks_d != nullptr);
    cudaErrorCheck(cudaMemcpy(blocks_d, blocks_h, sizeof(HicooBlock) * (numBlocks+1), cudaMemcpyHostToDevice));
}

void HicooTensor::downloadToHost() {
    DEBUG_PRINT("HT: download to host\n");
    freeHostArrays();

    points_h = (HicooPoint*)malloc(sizeof(HicooPoint) * numPoints);
    assert(points_h != nullptr);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(HicooPoint) * numPoints, cudaMemcpyDeviceToHost));

    blocks_h = (HicooBlock*)malloc(sizeof(HicooBlock) * (numBlocks+1));
    assert(blocks_h != nullptr);
    cudaErrorCheck(cudaMemcpy(blocks_h, blocks_d, sizeof(HicooBlock) * (numBlocks+1), cudaMemcpyDeviceToHost));
}


CooTensorManager HicooTensor::toCoo() {
    DEBUG_PRINT("HT: to coo\n");
    CooTensorManager ret;
    assert(0); // TODO
    return ret;
}


DenseMatrixManager HicooTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    /*
     * for each block (except the last)
     *      for each element starting at block address and ending at next block address
     *          l = blockX * blockWidth + pointX
     *          k = blockY * blockHeight + pointY
     *          i = blockZ * blockDepth + pointZ
     *
     *          for j = 1..j
     *              A(i,j) += point.val * C(k,j) + D(l,j)
     * return A
     */

    //Naive: each thread is a non-zero
    //optimization: each thread does a few R's

    //Naive implementation:

    DEBUG_PRINT("HICOO: mttkrp naive cpu\n");

    DenseMatrixManager ret;
    DenseMatrix& a = ret;

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    int I = this->depth, J = d.width, K = this->height, L = this->width;
    DEBUG_PRINT("    - I = %d, J = %d, K = %d, L = %d\n", I, J, K, L);
    assert(d.height == L);
    assert(c.height == K);
    assert(c.width  == J);


    a.setSize(I, J);

    //for each non-zero
    DEBUG_PRINT("    - performing operation\n");

    for (int b = 0; b < this->numBlocks - 1; b++) {
        HicooBlock block = this->access_block(b);
        unsigned long long startBlockAddress = block.blockAddress;
        unsigned long long endBlockAddress = this->access_block(b+1).blockAddress;
        for (unsigned long long index = startBlockAddress; index < endBlockAddress; index++) {
            HicooPoint point = access_point(index);

            int l = block.blockX * this->blockWidth + point.x;
            int k = block.blockY * this->blockHeight + point.y;
            int i = block.blockZ * this->blockDepth + point.z;

            for (int j = 0; j < J; j++) {
                a.access(i,j) += point.value * d.access(l,j) * c.access(k,j);
            }
        }
    }

    return ret;
}

DenseMatrixManager HicooTensor::mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(points_d != nullptr);
    assert(blocks_d != nullptr);

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
