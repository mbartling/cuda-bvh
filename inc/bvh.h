#pragma once
#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1
#include "tiny_obj_loader.h"
#include "tris.h"
#include "bbox.h"
#include "material.h"
#include "isect.h"
#include "ray.h"
class Scene_d;

struct Node{
    Node* childA;
    Node* childB;
    Node* parent;
    int flag;
    bool isLeaf;
    BoundingBox BBox;

    __device__ 
        Node() : isLeaf(false) , flag(0), parent(nullptr) {}
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
        Vec3f* normals;
        BoundingBox* BBoxs;
        TriangleIndices* t_indices;
        Material* materials;
        


    public:

        void setUp(Vec3f* mvertices, Vec3f* mnormals, BoundingBox* mBBoxs, TriangleIndices* mt_indices, int mnumTriangles, Material* mmaterials, Vec3f mMin , Vec3f mMax);
        ~BVH_d();
        void computeMortonCodes(Vec3f& mMin, Vec3f& mMax); //Also Generates the objectIds
        void sortMortonCodes();

        void setupLeafNodes();
        void buildTree();

        __device__
            bool intersectTriangle(const ray& r, isect&  i, int object_id){

                TriangleIndices ids = t_indices[object_id];

                Vec3f a = vertices[ids.a.vertex_index];
                Vec3f b = vertices[ids.b.vertex_index];
                Vec3f c = vertices[ids.c.vertex_index];

                /*
                   -DxAO = AOxD
                   AOx-D = -(-DxAO)
                   |-D AB AC| = -D*(ABxAC) = -D*normal = 1. 1x
                   |AO AB AC| = AO*(ABxAC) = AO*normal = 1. 
                   |-D AO AC| = -D*(AOxAC) = 1. 1x || AC*(-DxAO) = AC*M = 1. 1x
                   |-D AB AO| = -D*(ABxAO) = 1. 1x || (AOx-D)*AB = (DxAO)*AB = -M*AB = 1.
                   */
                float mDet;
                float mDetInv;
                float alpha;
                float beta;
                float t;
                Vec3f rDir = r.getDirection();
                //Moller-Trombore approach is a change of coordinates into a local uv space
                // local to the triangle
                Vec3f AB = b - a;
                Vec3f AC = c - a;

                // if (normal * -r.getDirection() < 0) return false;
                Vec3f P = rDir ^ AC;
                mDet = AB * P;
                if(fabsf(mDet) < RAY_EPSILON ) return false;

                mDetInv = 1/mDet;
                Vec3f T = r.getPosition() - a;
                alpha = T * P * mDetInv;    
                if(alpha < 0 || alpha > 1) return false;

                Vec3f Q = T ^ AB;
                beta = rDir * Q * mDetInv;
                if(beta < 0 || alpha + beta > 1) return false;
                t = AC * Q * mDetInv;

                if(fabsf(t) < RAY_EPSILON) return false; // Jaysus this sucked
                i.bary = Vec3f(1 - (alpha + beta), alpha, beta);
                i.t = t;


                // std::cout << traceUI->smShadSw() << std::endl; 
                // if(traceUI->smShadSw() && !parent->floatCheck()){
                //Smooth Shading
                Vec3f aN = normals[ids.a.normal_index];
                Vec3f bN = normals[ids.b.normal_index];
                Vec3f cN = normals[ids.c.normal_index];
                i.N = (1 - (alpha + beta))*aN + \
                      alpha*bN + \
                      beta*cN;

                //i.N = normal;

                normalize(i.N);

                i.object_id = object_id;
                //if(!parent->materials.empty() && parent->hasVertexMaterials()){
                Material aM;
                //TODO Be able to uncomment the following lines
                //int material_id = material_ids[object_id];
                //aM += (1 - (alpha + beta))*(materials[ids.a]); 
                //aM +=                alpha*(materials[ids.b]); 
                //aM +=                beta* (materials[ids.c]); 

                i.material = aM;

                return true;


            }
            __device__
                bool intersect(const ray& r, isect& i){

                }



            };

           class Scene_h;
           void bvh(Scene_h& scene_h);

