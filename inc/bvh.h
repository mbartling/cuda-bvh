#pragma once
#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1
#include "tinyobjloader.h"
#include "tris.h"

//Device BVH
class BVH_d {
    private:
        unsigned int* mortonCodes;
        unsigned int* object_ids;

        // These are stored in the scene
        int numTriangles;
        Vec3f* vertices; 
        BoundingBox* BBoxs;
        TriangleIndices* t_indices;


    public:

        void setUp(Vec3f* mvertices, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int numTriangles);
        ~BVH_d();
        

};

void bvh(void);
