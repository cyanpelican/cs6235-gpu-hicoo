#include <map>
#include <math.h>
#include "coo.hpp"
#include "csf.hpp"
#include "hicoo.hpp"


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
CsfTensorManager CooTensor::toCsf() {
    DEBUG_PRINT("CT: to csf\n");
    CsfTensorManager ret;
    assert(0);
    return ret;
}


DenseMatrixManager CooTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    //order of dimensions goes height, width , depth
    //todo: double check this. It might be x,y,z: width, height, depth

    assert(this->points_h != nullptr);
    //check for compatible dimensions
    assert(this->width == d.width);
    assert(this->depth == c.width);

    DenseMatrixManager ret;
    ret.tensor->tensor.setSize(d.height, this->height);

    // http://tensor-compiler.org/codegen.html
    // https://sc17.supercomputing.org/SC17%20Archive/tech_poster/poster_files/post213s2-file2.pdf

    //the poster and the taco algorithm don't quite line up. This is the translated version:
    // tensor: i x k x l
    // matrix1: j x k
    // matrix2: j x l
    // result: j x i



    // Generated by the Tensor Algebra Compiler (tensor-compiler.org)
    // for (int32_t pA = 0; pA < (A1_dimension * A2_dimension); pA++) {
    //     A_vals[pA] = 0.0;
    // }
    // for (int32_t pB1 = B1_pos[0]; pB1 < B1_pos[1]; pB1++) {
    //     int32_t iB = B1_coord[pB1];
    //     for (int32_t pB2 = B2_pos[pB1]; pB2 < B2_pos[(pB1 + 1)]; pB2++) {
    //         int32_t kB = B2_coord[pB2];
    //         for (int32_t pB3 = B3_pos[pB2]; pB3 < B3_pos[(pB2 + 1)]; pB3++) {
    //             int32_t lB = B3_coord[pB3];
    //             double tl = B_vals[pB3];
    //             for (int32_t jC = 0; jC < C2_dimension; jC++) {
    //                 int32_t pC2 = kB * C2_dimension + jC;
    //                 int32_t pD2 = lB * D2_dimension + jC;
    //                 int32_t pA2 = iB * A2_dimension + jC;
    //                 A_vals[pA2] = A_vals[pA2] + tl * C_vals[pC2] * D_vals[pD2];
    //                 //my version:
    //                 *ret.tensor->tensor.access(i,j) = *ret.tensor->tensor.access(i,j) + this->access(i,k,l) * d.access(l,j) * c.access(k,j);
    //             }
    //         }
    //     }
    // }


   //A(i,j) = B(i,k,l) * C(k,j) * D(l,j)

    for (unsigned int i = 0; i < this->height; i++) {
        for (unsigned int k = 0; k < this->width; k++) {
            for (unsigned int l = 0; l < this->depth; l++) {
                for (unsigned int j = 0; j < d.height; j++) {
                    ret.tensor->tensor.access(i,j) = ret.tensor->tensor.access(i,j) + this->access(i,k,l) * d.access(l,j) * c.access(k,j);
                }
            }
        }
    }

    return ret;
}


//Not declared as part of the class... Cuda doesn't like it's kernels as part of OOP
__global__ void mttkrp_naive_gpu(CooTensor cooTensor, DenseMatrix d, DenseMatrix c, DenseMatrix ret) {
    assert(cooTensor.points_d != nullptr);
    //check for compatible dimensions
    assert(cooTensor.width == d.width);
    assert(cooTensor.depth == c.width);

    if(blockDim.x * blockIdx.x + threadIdx.x < cooTensor.height * d.height) {
        //https://stackoverflow.com/a/29148148
        int index = blockDim.x * blockIdx.x + threadIdx.x;

        unsigned int i = index % cooTensor.height;
        unsigned int j = ((index - i) / cooTensor.height) % d.height;


        // We parallelize the 'i' and 'j' loops
        for (unsigned int k = 0; k < cooTensor.width; k++) {
            for (unsigned int l = 0; l < cooTensor.depth; l++) {
                //access will differentiate between host and device on its own
                ret.access(i, j) = ret.access(i, j) + cooTensor.access(i, k, l) * d.access(l, j) * c.access(k, j);
            }
        }

    }

    __syncthreads();
    
    // ret has already been written to. Now just return
}


//wrapper function for the sake of convenience
DenseMatrixManager CooTensor::mttkrp_naive_gpu_wrapper(DenseMatrix d, DenseMatrix c) {
    this->uploadToDevice();

    DenseMatrixManager ret;
    ret.tensor->tensor.setSize(d.height, this->height);
    ret.tensor->tensor.uploadToDevice();
    d.uploadToDevice();
    c.uploadToDevice();

    mttkrp_naive_gpu<<<ceil(this->numElements/d.height), d.height>>>(*this, d, c, ret.tensor->tensor);
    cudaDeviceSynchronize();

    ret.tensor->tensor.downloadToHost();
    //todo: free the mem on the gpu?

//    this->downloadToHost(); //but like, why? The result is already in ret.
    return ret;
}


DenseMatrixManager CooTensor::mttkrp_fast(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

void CooTensorManager::create(const char *tensorFileName) {
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
