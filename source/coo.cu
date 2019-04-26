#include <map>
#include <math.h>
#include "coo.hpp"
#include "hicoo.hpp"
#include "common.hpp"


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
    std::map<HicooBlock, std::vector<HicooPoint>> hicooMap;
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
    for(std::pair<HicooBlock, std::vector<HicooPoint>> pair : hicooMap) {
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
        retTensor.access(p.x, p.y, p.z) = p.value;
    }

    return ret;
}


DenseMatrixManager CooTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    //order of dimensions goes height, width , depth
    //todo: double check this. It might be x,y,z: width, height, depth

    //assert(this->points_h != nullptr);
    //check for compatible dimensions
    //assert(this->width == d.width);
    //assert(this->depth == c.width);



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

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    int I = this->depth, J = d.width, K = this->height, L = this->width;
    assert(d.height == L);
    assert(c.height == K);
    assert(c.width  == J);


    a.setSize(J, I);

    //for each non-zero
    for (int index = 0; index < this->numElements; index++) {
	CooPoint point = this->access(index);
        int l = point.x;
        int k = point.y;
        int i = point.z;

        for (int j = 0; j < J; j++) {
            //float val = point.value * d.access(j, l) * c.access(j, k);
            //ret.tensor->tensor.access(j, i) += val;
	    a.access(j,i) += point.value * d.access(j, l) * c.access(j,k);
        }
    }

    return ret;
//    for (unsigned int i = 0; i < this->height; i++) {
//        for (unsigned int k = 0; k < this->width; k++) {
//            for (unsigned int l = 0; l < this->depth; l++) {
//                for (unsigned int j = 0; j < d.height; j++) {
//                    ret.tensor->tensor.access(i,j) = ret.tensor->tensor.access(i,j) + this->access(i,k,l) * d.access(l,j) * c.access(k,j);
//                }
//            }
//        }
//    }

}


//Not declared as part of the class... Cuda doesn't like it's kernels as part of OOP
__global__ void mttkrp_naive_gpu(CooTensor cooTensor, DenseMatrix d, DenseMatrix c, DenseMatrix ret) {
//    if(blockDim.x * blockIdx.x + threadIdx.x < cooTensor.height * d.height) {
//        //https://stackoverflow.com/a/29148148
//        int index = blockDim.x * blockIdx.x + threadIdx.x;
//
//        unsigned int i = index % cooTensor.height;
//        unsigned int j = ((index - i) / cooTensor.height) % d.height;
//
//
//        // We parallelize the 'i' and 'j' loops
//        for (unsigned int k = 0; k < cooTensor.width; k++) {
//            for (unsigned int l = 0; l < cooTensor.depth; l++) {
//                //access will differentiate between host and device on its own
//                ret.access(i, j) = ret.access(i, j) + cooTensor.access(i, k, l) * d.access(l, j) * c.access(k, j);
//            }
//        }
//
//    }
//
//    __syncthreads();


    // -----------------------
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

//    DEBUG_PRINT("COO: mttkrp naive gpu\n");

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
//    int I = cooTensor.depth, J = d.width, K = cooTensor.height, L = cooTensor.width;
    int J = d.width;

    //for each non-zero
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    CooPoint point = cooTensor.access(index);
    int i = point.x;
    int l = point.y;
    int k = point.z;

    for (int j = 0; j < J; j++) {
        float val = point.value * c.access(k,j) * d.access(l, j);
        atomicAdd(&ret.access(j, i), val);
    }

//    __syncthreads();
}


//wrapper function for the sake of convenience
DenseMatrixManager CooTensor::mttkrp_naive_gpu_wrapper(DenseMatrix d, DenseMatrix c) {
    this->uploadToDevice();

    DenseMatrixManager ret;

    ret.tensor->tensor.setSize_d(d.height, this->height);
    ret.tensor->tensor.uploadToDevice();
    d.uploadToDevice();
    c.uploadToDevice();

    assert(this->points_d != nullptr);
    //check for compatible dimensions
    assert(this->width == d.width);
    assert(this->depth == c.width);

    //todo: split up the blocks & blocks per threads appropriately
    mttkrp_naive_gpu<<<ceil(this->numElements/64.0), 64>>>(*this, d, c, ret.tensor->tensor);
    cudaDeviceSynchronize();

    ret.tensor->tensor.downloadToHost();

    return ret;
}


DenseMatrixManager CooTensor::mttkrp_fast(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

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
    tensor->tensor.setSize(nonZeroes, maxX, maxY, maxZ);
    memcpy(tensor->tensor.points_h, tensorPoints.data(), sizeof(CooPoint) * tensorPoints.size());

    DEBUG_PRINT("    - done; size = %d; %d x %d x %d\n", nonZeroes, maxZ, maxY, maxX);
}
