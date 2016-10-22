#include <stdio.h>
#include "bvh.h"

__global__ void hello()
{
    printf("Hello world! I'm a thread in block %d\n", blockIdx.x);

}
// Expands a 10-bit integer into 30 bits
// // by inserting 2 zeros after each bit.
__device__
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__
unsigned int morton3D(float x, float y, float z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

__device__ __inline__
Vec3f computeCentroid(const BoundingBox& BBox){
    return (BBox.getMin() + BBox.getMax()) / 2.0f;
};

__global__ 
void computeMortonCodesKernel(unsigned int* mortonCodes, unsigned int* object_ids, 
        BoundingBox* BBoxs, int numTriangles);

void BVH_d::computeMortonCodes(){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
    computeMortonCodesKernel<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, object_ids, BBoxs, numTriangles);
    
}
void bvh(void)
{
    //launch the kernel
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    //force the printf()s to flush
    cudaDeviceSynchronize();

    printf("That's all!\n");

}

// This kernel just computes the object id and morton code for the centroid of each bounding box
__global__ 
void computeMortonCodesKernel(unsigned int* mortonCodes, unsigned int* object_ids, 
        BoundingBox* BBoxs, int numTriangles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > numTriangles)
        return;
    
    object_ids[idx] = idx;
    Vec3f centroid = computeCentroid(BBoxs[idx]);
    mortonCodes[idx] = morton3D(centroid.x, centroid.y, centroid.z);

};
