#include "bvh.h"

void BVH_d::setUp(Vec3f* mvertices, Vec3f* mnormals, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int mnumTriangles, Material* mmaterials, Vec3f mMin, Vec3f mMax){
    numTriangles = mnumTriangles;
    normals = mnormals;
    vertices = mvertices;
    BBoxs = mBBoxs;
    t_indices = mt_indices;
    materials = mmaterials;

    cudaMalloc(&mortonCodes, numTriangles*sizeof(unsigned int));
    cudaMalloc(&object_ids, numTriangles*sizeof(unsigned int));
    
    cudaMalloc(&leafNodes, numTriangles*sizeof(LeafNode));
    cudaMalloc(&internalNodes, (numTriangles - 1)*sizeof(InternalNode));

    // Set up for the BVH Build
    computeMortonCodes(mMin, mMax);
    sortMortonCodes();
    buildTree();

    // Build the BVH

}

BVH_d::~BVH_d(){
    cudaFree(mortonCodes);
    cudaFree(object_ids);
    cudaFree(leafNodes);
    cudaFree(internalNodes);
}
