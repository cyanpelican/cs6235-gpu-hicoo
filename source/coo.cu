#include <map>
#include <math.h>
#include "coo.hpp"
#include "hicoo.hpp"
#include <map>
#include "common.hpp"
#include <list>


void CooTensor::freeAllArrays() {
    DEBUG_PRINT("CT: freeing all arrays\n");
    freeHostArrays();
    freeDeviceArrays();
}
void CooTensor::freeHostArrays() {
    DEBUG_PRINT("CT: freeing host arrays\n");
    DEBUG_PRINT("    - points_h = %p\n", points_h);
    free(points_h);
    points_h = nullptr;
}
void CooTensor::freeDeviceArrays() {
    DEBUG_PRINT("CT: freeing device arrays\n");
    DEBUG_PRINT("    - points_d = %p\n", points_d);
    if(points_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(points_d));
    points_d = nullptr;
}

void CooTensor::uploadToDevice() {
    DEBUG_PRINT("CT: uploading to device\n");
    assert(points_h != nullptr);
    freeDeviceArrays();
    assert(numElements != 0);
    cudaErrorCheck(cudaMalloc((void **) &points_d, sizeof(CooPoint) * numElements));
    assert(points_d != nullptr);
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(CooPoint) * numElements, cudaMemcpyHostToDevice));
}

// todo: is this ever necessary? The contents of the tensor are never changed. We should probably just free the device
//  memory
void CooTensor::downloadToHost() {
    DEBUG_PRINT("CT: downloading to host\n");
    assert(points_d != nullptr);
    freeHostArrays();
    points_h = (CooPoint*)malloc(sizeof(CooPoint) * numElements);
    assert(points_h != nullptr);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(CooPoint) * numElements, cudaMemcpyDeviceToHost));
}



// for std::map / std::set insertion
bool operator<(const HicooBlock& a, const HicooBlock& b) {
    if(a.blockX < b.blockX) {
        return true;
    } else if(a.blockX > b.blockX) {
        return false;
    }
    if(a.blockY < b.blockY) {
        return true;
    } else if(a.blockY > b.blockY) {
        return false;
    }
    if(a.blockZ < b.blockZ) {
        return true;
    } else if(a.blockZ > b.blockZ) {
        return false;
    }

    return false;
}
HicooTensorManager CooTensor::toHicoo(int blockDepth, int blockHeight, int blockWidth) {
    DEBUG_PRINT("CT: to hicoo (bd %d, bh %d, bw %d)\n", blockDepth, blockHeight, blockWidth);
    HicooTensorManager ret;
    HicooTensor& retTensor = ret;

    // build an std::map of everything
    DEBUG_PRINT("    - building map\n");
    std::map<HicooBlock, std::list<HicooPoint>> hicooMap;
    for(int i = 0; i < numElements; i++) {
        CooPoint p = access(i);

        HicooBlock block = {/*blockAddress =*/ 0,
            /*blockX =*/ (p.x)/blockWidth, /*blockX =*/ (p.y)/blockHeight, /*blockX =*/ (p.z)/blockDepth,
            /*UNUSED =*/ 0};

        HicooPoint hp = {/*x =*/ (unsigned char)((p.x)%blockWidth), /*y =*/ (unsigned char)((p.y)%blockHeight), /*z =*/ (unsigned char)((p.z)%blockDepth),
            /*UNUSED =*/ 0,
            /*value =*/ p.value};

        hicooMap[block].push_back(hp);
    }

    // put everything into the tensor
    DEBUG_PRINT("    - realloc ret tensor\n");
    retTensor.setSize(hicooMap.size(), numElements, depth, height, width, blockDepth, blockHeight, blockWidth);

    unsigned int blockIndex = 0;
    unsigned long long pointIndex = 0;
    DEBUG_PRINT("    - insert to ret tensor\n");
    for(const std::pair<HicooBlock, std::list<HicooPoint>>& pair : hicooMap) {
        retTensor.blocks_h[blockIndex] = pair.first;
        retTensor.blocks_h[blockIndex].blockAddress = pointIndex;
        for(HicooPoint p : pair.second) {
            retTensor.points_h[pointIndex++] = p;
        }
        blockIndex++;
    }

    // final element off the end of the list
    retTensor.blocks_h[blockIndex].blockAddress = pointIndex;
    retTensor.blocks_h[blockIndex].blockX = 0xFFFFFFFF;
    retTensor.blocks_h[blockIndex].blockY = 0xFFFFFFFF;
    retTensor.blocks_h[blockIndex].blockZ = 0xFFFFFFFF;
    retTensor.blocks_h[blockIndex].UNUSED = 0xFFFFFFFF;


    return ret;
}
DenseTensorManager CooTensor::toDense() {
    DEBUG_PRINT("CT: to dense\n");
    DEBUG_PRINT("    - realloc\n");
    DenseTensorManager ret;
    DenseTensor& retTensor = ret;
    retTensor.setSize(depth, height, width);

    DEBUG_PRINT("    - insertion\n");
    for(int i = 0; i < numElements; i++) {
        CooPoint p = access(i);
        retTensor.access(p.z, p.y, p.x) = p.value;
    }

    return ret;
}


DenseMatrixManager CooTensor::mttkrp_naive_cpu(DenseMatrixManager D, DenseMatrixManager C) {
    /*
      * for each non-zero
      *      i = nnz.i, l = nnz.l, k = nnz.k
      *      for j = 1..j
      *          A(i,j) += val(nnz) * C(k,j) * D (l,j)
      *
      * return A
      */

    //Naive: each thread is a non-zero
    //optimization: each thread does a few R's

    //Naive implementation:

    DEBUG_PRINT("COO: mttkrp naive cpu\n");

    DenseMatrixManager ret;
    DenseMatrix& a = ret;
    DenseMatrix& c = C;
    DenseMatrix& d = D;

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    int I = this->depth, J = d.width, K = this->height, L = this->width;
    DEBUG_PRINT("    - I = %d, J = %d, K = %d, L = %d\n", I, J, K, L);
    assert(d.height == L);
    assert(c.height == K);
    assert(c.width  == J);


    a.setSize(I, J);

    //for each non-zero
    DEBUG_PRINT("    - performing operation\n");
    for (int index = 0; index < this->numElements; index++) {
        CooPoint point = this->access(index);
        int l = point.x;
        int k = point.y;
        int i = point.z;

        for (int j = 0; j < J; j++) {
            a.access(i,j) += point.value * d.access(l,j) * c.access(k,j);
        }
    }

    return ret;
}


//Not declared as part of the class... Cuda doesn't like it's kernels as part of OOP
__global__ void mttkrp_naive_gpu_kernel(CooTensor cooTensor, DenseMatrix d, DenseMatrix c, DenseMatrix ret) {
    /*
     * for each non-zero
     *      i = nnz.i, l = nnz.l, k = nnz.k
     *      for j = 1..j
     *          A(i,j) += val(nnz) * C(k,j) * D (l,j)
     *
     * return A
     */

    //Naive: each thread is a non-zero
    //optimization: each thread does a few R's

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    // int I = cooTensor.depth, J = d.width, K = cooTensor.height, L = cooTensor.width;
    int J = d.width;

    //for each non-zero
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < cooTensor.numElements) {
        CooPoint point = cooTensor.access(index);
        int l = point.x;
        int k = point.y;
        int i = point.z;

        for (int j = 0; j < J; j++) {
            float val = point.value * c.access(k, j) * d.access(l, j);
            atomicAdd(&ret.access(i, j), val);
        }
    }
}


//wrapper function for the sake of convenience
DenseMatrixManager CooTensor::mttkrp_naive_gpu(DenseMatrixManager D, DenseMatrixManager C) {
    this->uploadToDevice();

    DenseMatrixManager ret;
    DenseMatrix& c = C;
    DenseMatrix& d = D;

    int I = this->depth, J = d.width, K = this->height, L = this->width;
    DEBUG_PRINT("    - I = %d, J = %d, K = %d, L = %d\n", I, J, K, L);
    assert(d.height == L);
    assert(c.height == K);
    assert(c.width  == J);
    a.setSize_d(I, J);
    d.uploadToDevice();
    c.uploadToDevice();

    assert(this->points_d != nullptr);
    //check for compatible dimensions
    assert(this->width == d.width);
    assert(this->depth == c.width);

    //todo: split up the blocks & blocks per threads appropriately
    mttkrp_naive_gpu_kernel<<<ceil(this->numElements/64.0), 64>>>(*this, d, c, ret);
    cudaDeviceSynchronize();

    ret.tensor->tensor.downloadToHost();

    return ret;
}


DenseMatrixManager CooTensor::mttkrp_guy1(DenseMatrixManager d, DenseMatrixManager c) {
    DEBUG_PRINT("CT: fast mttkrp gpu\n");
    DenseMatrixManager ret;

    // TODO or DELTEME
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager CooTensor::mttkrp_james1(DenseMatrixManager d, DenseMatrixManager c) {
    DEBUG_PRINT("CT: fast mttkrp gpu\n");
    DenseMatrixManager ret;

    // TODO or DELTEME
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}


__global__ void coo_mttkrp_kevin1_kernel(DenseMatrix a, CooTensor b, DenseMatrix d, DenseMatrix c) {
    CooPoint point = b.access(blockIdx.x);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    for(int j = threadIdx.x; j < a.width; j += 32) {
        float val = point.value * d.access(point.x, j) * c.access(point.y, j);
        atomicAdd(&a.access(point.z, j), val);
    }
}

DenseMatrixManager CooTensor::mttkrp_kevin1(DenseMatrixManager D, DenseMatrixManager C) {
    // Has each thread block mapped to a point (parallelizing blocks across J)
    DEBUG_PRINT("CT: naive mttkrp gpu\n");
    DEBUG_PRINT("    - asserts, initialization\n");
    DenseMatrixManager ret;
    DenseMatrix& a = ret;
    DenseMatrix& c = C;
    DenseMatrix& d = D;

    assert(points_h != nullptr);

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
    coo_mttkrp_kevin1_kernel<<<numElements, 32>>>(a, *this, d, c);



    return ret;
}

void CooTensorManager::create(char *tensorFileName) {
    DEBUG_PRINT("CT: parsing file %s\n", tensorFileName);
    DEBUG_PRINT("    - file validations, etc\n");
    std::vector<CooPoint> tensorPoints;

    size_t nonZeroes = 0;
    std::string line;
    std::ifstream myfile(tensorFileName);
    assert(myfile.good()); // assert file exists, etc

    //put all the points into a vector
    DEBUG_PRINT("    - load all points into vector\n");
    int maxX = 0, maxY = 0, maxZ = 0;
    while (std::getline(myfile, line)) {
        if(line.length() < 4 || line[0] == '#') {
            // uselessly-short line or comment
            continue;
        }
        ++nonZeroes;
        CooPoint currentPoint;
        std::stringstream ss(line); // Turn the string into a stream.
        ss >> currentPoint.x;
        ss >> currentPoint.y;
        ss >> currentPoint.z;
        ss >> currentPoint.value;

        if(currentPoint.x > maxX) maxX = currentPoint.x;
        if(currentPoint.y > maxY) maxY = currentPoint.y;
        if(currentPoint.z > maxZ) maxZ = currentPoint.z;

        //This assumes there are 3 dimensions followed by one value
        tensorPoints.push_back(currentPoint);
    }

    tensorPoints.shrink_to_fit();

    //construct the COO object
    DEBUG_PRINT("    - rebuild tensor from input\n");
    tensor->tensor.setSize(nonZeroes, maxZ, maxY, maxX);
    memcpy(tensor->tensor.points_h, tensorPoints.data(), sizeof(CooPoint) * tensorPoints.size());

    DEBUG_PRINT("    - done; size = %d; %d x %d x %d\n", nonZeroes, maxZ, maxY, maxX);
}
