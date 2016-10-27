#include <stdio.h>
#include "bvh.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "scene.h"

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
        BoundingBox* BBoxs, int numTriangles, Vec3f mMin, Vec3f mMax);
__global__ 
void setupLeafNodesKernel(unsigned int* sorted_object_ids, 
        LeafNode* leafNodes, int numTriangles);

__global__ 
void generateHierarchyKernel(unsigned int* mortonCodes,
        unsigned int* sorted_object_ids, 
        InternalNode* internalNodes,
        LeafNode* leafNodes, int numTriangles);

void BVH_d::computeMortonCodes(Vec3f& mMin, Vec3f& mMax){
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
    computeMortonCodesKernel<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, object_ids, BBoxs, numTriangles, mMin , mMax);

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
    generateHierarchyKernel<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, object_ids, internalNodes , leafNodes , numTriangles);

}

void bvh(Scene_h& scene_h)
{
    Scene_d scene_d;
    scene_d = scene_h;
    
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
        BoundingBox* BBoxs, int numTriangles, Vec3f mMin , Vec3f mMax){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles)
        return;

    object_ids[idx] = idx;
    Vec3f centroid = computeCentroid(BBoxs[idx]);
    centroid.x = (centroid.x - mMin.x)/(mMax.x - mMin.x);
    centroid.y = (centroid.y - mMin.y)/(mMax.y - mMin.y);
    centroid.z = (centroid.z - mMin.z)/(mMax.z - mMin.z);
    //map this centroid to unit cube
    mortonCodes[idx] = morton3D(centroid.x, centroid.y, centroid.z);
    printf("in computeMortonCodesKernel: idx->%d , mortonCode->%d, centroid(%0.6f,%0.6f,%0.6f)\n", idx, mortonCodes[idx], centroid.x, centroid.y, centroid.z);

};

__global__ 
void setupLeafNodesKernel(unsigned int* sorted_object_ids, 
        LeafNode* leafNodes, int numTriangles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numTriangles)
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

    if (idx > numTriangles - 2 ) //there are n - 1 internal nodes
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
   //determine the range of keys covered by each internal node (as well as its children)
    //direction is found by looking at the neighboring keys ki-1 , ki , ki+1
    //the index is either the beginning of the range or the end of the range
    int direction = 0;
    int common_prefix_with_left = 0;
    int common_prefix_with_right = 0;

    common_prefix_with_right = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx + 1]);
    if(idx == 0){
        common_prefix_with_left = -1;
    }
    else
    {
        common_prefix_with_left = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - 1]);

    }

    direction = ( (common_prefix_with_right - common_prefix_with_left) > 0 ) ? 1 : -1;
    int min_prefix_range = 0;

    if(idx == 0)
    {
        min_prefix_range = -1;

    }
    else
    {
        min_prefix_range = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - direction]); 
    }

    int lmax = 2;
    int next_key = idx + lmax*direction;

    while((next_key >= 0) && (next_key <  numTriangles) && (__clz(sortedMortonCodes[idx] ^ sortedMortonCodes[next_key]) > min_prefix_range))
    {
        lmax *= 2;
        next_key = idx + lmax*direction;
    }
    //find the other end using binary search
    unsigned int l = 0;

    do
    {
        lmax = (lmax + 1) >> 1; // exponential decrease
        int new_val = idx + (l + lmax)*direction ; 

        if(new_val >= 0 && new_val < numTriangles )
        {
            unsigned int Code = sortedMortonCodes[new_val];
            int Prefix = __clz(sortedMortonCodes[idx] ^ Code);
            if (Prefix > min_prefix_range)
                l = l + lmax;
        }
    }
    while (lmax > 1);

    int j = idx + l*direction;

    int left = 0 ; 
    int right = 0;
    
    if(idx < j){
        left = idx;
        right = j;
    }
    else
    {
        left = j;
        right = idx;
    }

    printf("idx : (%d) returning range (%d, %d) \n" , idx , left, right);

    return make_int2(left,right);
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
