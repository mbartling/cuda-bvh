#pragma once
#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1
#include "tiny_obj_loader.h"
#include "tris.h"
#include "bbox.h"
//#include "scene.h"

class Scene_d;

struct Node{
    Node* childA;
    Node* childB;
    Node* parent;
    bool isLeaf;
    BoundingBox BBox;

    __device__ 
    Node() : isLeaf(false) {}
};
struct LeafNode : public Node {
    unsigned int object_id;
    
    __device__
    LeafNode() {
        this->isLeaf = true;
    }
};
struct InternalNode : public Node {
    __device__
    InternalNode() {
        this->isLeaf = false;
    }
};
//Device BVH
class BVH_d {
    private:
        unsigned int* mortonCodes;
        unsigned int* object_ids;

        LeafNode*       leafNodes; //numTriangles
        InternalNode*   internalNodes; //numTriangles - 1
        
        // These are stored in the scene
        int numTriangles;
        Vec3f* vertices; 
        BoundingBox* BBoxs;
        TriangleIndices* t_indices;


    public:

        void setUp(Vec3f* mvertices, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int numTriangles);
        ~BVH_d();
        void computeMortonCodes(); //Also Generates the objectIds
        void sortMortonCodes();

        void setupLeafNodes();
        void buildTree();

        __device__
        bool intersect(const ray& r, isect& i, Scene_d* scene);
        
        

};
class Scene_h;
void bvh(Scene_h& scene_h);
