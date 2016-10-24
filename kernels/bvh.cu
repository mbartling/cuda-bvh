#include <stdio.h>
#include "bvh.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

__global__ void hello()
{
    printf("Hello world! I'm a thread in block %d\n", blockIdx.x);

}
// Expands a 10-bit integer into 30 bits
// // by inserting 2 zeros after each bit.
__device__
unsigned int expandBits(unsigned int v);

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__
unsigned int morton3D(float x, float y, float z);

__device__ __inline__
Vec3f computeCentroid(const BoundingBox& BBox){
    return (BBox.getMin() + BBox.getMax()) / 2.0f;
};
__device__
int findSplit(  unsigned int* sortedMortonCodes,
        int first,
        int last);

__device__
int2 determineRange(unsigned int* sortedMortonCodes, int numTriangles, int idx);

__global__ 
void computeMortonCodesKernel(unsigned int* mortonCodes, unsigned int* object_ids, 
        BoundingBox* BBoxs, int numTriangles);
__global__ 
void setupLeafNodesKernel(unsigned int* sorted_object_ids, 
        LeafNode* leafNodes, int numTriangles);

__global__ 
void generateHierarchyKernel(unsigned int* mortonCodes,
        unsigned int* sorted_object_ids, 
        InternalNode* internalNodes,
        LeafNode* leafNodes, int numTriangles);

void BVH_d::computeMortonCodes(){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
    computeMortonCodesKernel<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, object_ids, BBoxs, numTriangles);

}
void BVH_d::sortMortonCodes(){
    thrust::device_ptr<unsigned int> dev_mortonCodes(mortonCodes);
    thrust::device_ptr<unsigned int> dev_object_ids(object_ids);

    // Let thrust do all the work for us
    thrust::sort_by_key(dev_mortonCodes, dev_mortonCodes + numTriangles, dev_object_ids);
}

void BVH_d::setupLeafNodes(){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
    setupLeafNodesKernel<<<blocksPerGrid, threadsPerBlock>>>(object_ids, leafNodes, numTriangles);

}
void BVH_d::buildTree(){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles - 1 + threadsPerBlock - 1) / threadsPerBlock;
    setupLeafNodesKernel<<<blocksPerGrid, threadsPerBlock>>>(object_ids, leafNodes, numTriangles);

}
void bvh(void)
{
    //launch the kernel
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    //force the printf()s to flush
    cudaDeviceSynchronize();

    printf("That's all!\n");

}

//===========Begin KERNELS=============================
//===========Begin KERNELS=============================
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

__global__ 
void setupLeafNodesKernel(unsigned int* sorted_object_ids, 
        LeafNode* leafNodes, int numTriangles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > numTriangles)
        return;
    leafNodes[idx].isLeaf = true;
    leafNodes[idx].object_id = sorted_object_ids[idx];
    leafNodes[idx].childA = nullptr;
    leafNodes[idx].childB = nullptr;
}

    __global__ 
void generateHierarchyKernel(unsigned int* sortedMortonCodes,
        unsigned int* sorted_object_ids, 
        InternalNode* internalNodes,
        LeafNode* leafNodes, int numTriangles)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > numTriangles - 1 )
        return;

    internalNodes[idx].isLeaf = false ;

    int2 range = determineRange(sortedMortonCodes, numTriangles, idx);
    int first = range.x;
    int last = range.y;

    //Determine where to split the range.

    int split = findSplit(sortedMortonCodes, first, last);

    // Select childA.

    Node* childA;
    if (split == first)
        childA = &leafNodes[split];
    else
        childA = &internalNodes[split];

    // Select childB.

    Node* childB;
    if (split + 1 == last)
        childB = &leafNodes[split + 1];
    else
        childB = &internalNodes[split + 1];

    // Record parent-child relationships.

    internalNodes[idx].childA = childA;
    internalNodes[idx].childB = childB;
    childA->parent = &internalNodes[idx];
    childB->parent = &internalNodes[idx];

}
//===========END KERNELS=============================
//===========END KERNELS=============================

    __device__
int findSplit( unsigned int* sortedMortonCodes,
        int first,
        int last)
{
    // Identical Morton codes => split the range in the middle.
    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    }
    while (step > 1);

    return split;
}

__device__
int2 determineRange(unsigned int* sortedMortonCodes, int numTriangles, int idx)
{
   return make_int2(0,0);
}
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
