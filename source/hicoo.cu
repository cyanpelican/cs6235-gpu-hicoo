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
    assert(points_h != nullptr);
    assert(blocks_h != nullptr);
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
    assert(points_d != nullptr);
    assert(blocks_d != nullptr);
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


DenseMatrixManager HicooTensor::mttkrp_naive_cpu(DenseMatrixManager D, DenseMatrixManager C) {
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

    DenseMatrixManager ret;
    DenseMatrix& a = ret;
    DenseMatrix& c = C;
    DenseMatrix& d = D;
    assert(this->points_h != nullptr);
    assert(points_h != nullptr);
    assert(blocks_h != nullptr);

    //Naive: each thread is a non-zero
    //optimization: each thread does a few R's

    //Naive implementation:

    DEBUG_PRINT("HICOO: mttkrp naive cpu\n");

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    int I = this->depth, J = d.width, K = this->height, L = this->width;
    DEBUG_PRINT("    - I = %d, J = %d, K = %d, L = %d\n", I, J, K, L);
    assert(d.height == L);
    assert(c.height == K);
    assert(c.width  == J);


    a.setSize(I, J);

    //for each non-zero
    DEBUG_PRINT("    - performing operation\n");

    for (int b = 0; b < this->numBlocks; b++) {
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
__global__ void mttkrp_naive_gpu_kernel(HicooTensor hicooTensor, DenseMatrix d, DenseMatrix c, DenseMatrix ret);

//wrapper function for the sake of convenience
DenseMatrixManager HicooTensor::mttkrp_naive_gpu(DenseMatrixManager D, DenseMatrixManager C) {
    DEBUG_PRINT("HT: naive mttkrp gpu\n");
    this->uploadToDevice();

    DenseMatrixManager ret;
    DenseMatrix& a = ret;
    DenseMatrix& c = C;
    DenseMatrix& d = D;
    assert(points_h != nullptr);
    assert(blocks_h != nullptr);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    int I = this->depth, J = d.width, K = this->height, L = this->width;
    DEBUG_PRINT("    - I = %d, J = %d, K = %d, L = %d\n", I, J, K, L);
    assert(d.height == L);
    assert(c.height == K);
    assert(c.width  == J);


    DEBUG_PRINT("    - uploadToDevice\n");
    this->uploadToDevice();
    d.uploadToDevice();
    c.uploadToDevice();
    a.setSize_d(I, J);

    assert(this->points_d != nullptr);
    assert(this->blocks_d != nullptr);

    //todo: split up the blocks & blocks per threads appropriately
    mttkrp_naive_gpu_kernel<<<ceil(this->numBlocks/64.0), 64>>>(*this, d, c, ret);
    cudaDeviceSynchronize();

    ret.tensor->tensor.downloadToHost();
    DEBUG_PRINT("    - done\n");

    return ret;
}

//Not declared as part of the class... Cuda doesn't like it's kernels as part of OOP
__global__ void mttkrp_naive_gpu_kernel(HicooTensor hicooTensor, DenseMatrix d, DenseMatrix c, DenseMatrix ret) {
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

    DenseMatrix& a = ret;

    //Naive: each thread is a block
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < hicooTensor.numBlocks) {
        // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
        int J = d.width;// K = hicooTensor.height, L = hicooTensor.width, I = hicooTensor.depth


        //each thread gets a block
        HicooBlock block = hicooTensor.access_block(index);
        unsigned long long startBlockAddress = block.blockAddress;
        unsigned long long endBlockAddress = hicooTensor.access_block(index + 1).blockAddress;
        for (unsigned long long index = startBlockAddress; index < endBlockAddress; index++) {
            HicooPoint point = hicooTensor.access_point(index);

            int l = block.blockX * hicooTensor.blockWidth + point.x;
            int k = block.blockY * hicooTensor.blockHeight + point.y;
            int i = block.blockZ * hicooTensor.blockDepth + point.z;

            for (int j = 0; j < J; j++) {
                float val = point.value * d.access(l, j) * c.access(k, j);
                atomicAdd(&a.access(i,j), val);
            }
        }
    }
}

DenseMatrixManager HicooTensor::mttkrp_guy1(DenseMatrixManager D, DenseMatrixManager C) {
    DenseMatrixManager ret;
    DenseMatrix& c = C;
    DenseMatrix& d = D;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager HicooTensor::mttkrp_james1(DenseMatrixManager D, DenseMatrixManager C) {
    DenseMatrixManager ret;
    DenseMatrix& c = C;
    DenseMatrix& d = D;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

__global__ void hicoo_kevin1_kernel(DenseMatrix a, HicooTensor b, DenseMatrix d, DenseMatrix c) {
    HicooBlock& ba = b.access_block(blockIdx.x);
    HicooBlock& bb = b.access_block(blockIdx.x+1);

    unsigned int bx = ba.blockX * b.blockWidth;
    unsigned int by = ba.blockY * b.blockHeight;
    unsigned int bz = ba.blockZ * b.blockDepth;

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    for(int e = ba.blockAddress; e < bb.blockAddress; e++) {
        HicooPoint& p = b.access_point(e);
        for(int j = threadIdx.x; j < a.width; j+=32) {
            float val = p.value * d.access(p.x+bx,j) * c.access(p.y+by,j);
            atomicAdd(&a.access(p.z+bz, j), val);
        }
    }
}

DenseMatrixManager HicooTensor::mttkrp_kevin1(DenseMatrixManager D, DenseMatrixManager C) {
    // Has each thread block mapped to a hicoo block (parallelizing blocks across J)
    DEBUG_PRINT("HT: mttkrp kevin1\n");
    DEBUG_PRINT("    - asserts, initialization\n");
    DenseMatrixManager ret;
    DenseMatrix& a = ret;
    DenseMatrix& c = C;
    DenseMatrix& d = D;
    assert(points_h != nullptr);
    assert(blocks_h != nullptr);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    int I = this->depth, J = d.width, K = this->height, L = this->width;
    DEBUG_PRINT("    - I = %d, J = %d, K = %d, L = %d\n", I, J, K, L);
    assert(d.height == L);
    assert(c.height == K);
    assert(c.width  == J);


    DEBUG_PRINT("    - uploadToDevice\n");
    this->uploadToDevice();
    d.uploadToDevice();
    c.uploadToDevice();

    DEBUG_PRINT("    - malloc output matrix\n");
    a.setSize_d(I, J);

    DEBUG_PRINT("    - do compute on gpu\n");
    hicoo_kevin1_kernel<<<numBlocks, 32>>>(a, *this, d, c);

    DEBUG_PRINT("    - downloading to host\n");
    a.downloadToHost();

    DEBUG_PRINT("    - done\n");
    return ret;
}
