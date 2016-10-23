#include "bvh.h"

void BVH_d::setUp(Vec3f* mvertices, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int mnumTriangles){
    numTriangles = mnumTriangles;
    vertices = mvertices;
    BBoxs = mBBoxs;
    t_indices = mt_indices;

    cudaMalloc(&mortonCodes, numTriangles*sizeof(unsigned int));
    cudaMalloc(&object_ids, numTriangles*sizeof(unsigned int));
    
    cudaMalloc(&leafNodes, numTriangles*sizeof(LeafNode));
    cudaMalloc(&internalNodes, (numTriangles - 1)*sizeof(InternalNode));

    // Set up for the BVH Build
    computeMortonCodes();
    sortMortonCodes();

    // Build the BVH

}

BVH_d::~BVH_d(){
    cudaFree(mortonCodes);
    cudaFree(object_ids);
    cudaFree(leafNodes);
    cudaFree(internalNodes);
}
