#pragma once
#include "vec.h"
#include "tinyobjloader.h"
#include <vector>
#include "tris.h"

using namespace tinyobj;
using std::vector;
using std::cout;
using std::endl;



//Forward Declarations
class Scene_d;
class Scene_h;

// This is a host side scene class
// It holds all of the triangles/mesh_t from the obj file
// We can copy a scene to and from the host/device using copy assignment
class Scene_h{
    private:
        // Host Side
        attrib_t mAttributes;
        vector<Vec4f> image;
        vector<TriangleIndices> t_indices;
        vector<int> material_ids;
        int imageWidth;
        int imageHeight;
        
        friend class Scene_d;

    public:
        Scene_h(): imageWidth(512), imageHeight(512) {}
        Scene_h(int imageWidth, int imageHeight): imageWidth(imageWidth), imageHeight(imageHeight) {}

        void LoadObj(string filename);

        Scene_h& operator = (const Scene_d& deviceScene); //Copy image from the device


};

//This is the device side scene.
// 5*sizeof(pointer) in size
class Scene_d{
    private:
        int numVertices;
        int numTriangles;
        int imageWidth;
        int imageHeight;

        Vec3f* vertices;
        Vec3f* normals;
        Vec3f* texcoords;
        BoundingBox* BBoxs; //Per Triangle Bounding Box

        TriangleIndices* t_indices;

        Vec4f* image;
        BVH_d bvh;

        friend class Scene_h;
    public:

        Scene_d& operator = (const Scene_h& hostScene); //Copy Triangles, materials, etc to the device
    
        void computeBoundingBoxes();

        ~Scene_d();

};

//Our mesh is always 3 vertices per face
// A trimesh just knows the indices of its vertices/normals/
// This trimesh is probably not needed
class TriMesh{
    private:
        //On Device data
        TriangleIndices* indices; //Includes Vertex Normals for smooth shading
        int*     material_ids;

    public:

        __host__
        TriMesh(const tinyobj::attrib_t& attrib,
                const std::vector<index_t>& t_indices,
                const std::vector<int>& m_material_ids){

            //Allocate some memory for these 
            cudaMalloc(indices, t_indices.size()*sizeof(index_t));
            cudaMalloc(material_ids, m_material_ids.size()*sizeof(int));

            // Copy the Indices and material Ids from the host to device
            // Note: C++11 
            cudaMemcpy(indices, t_indices.data(), t_indices.size()*sizeof(index_t), cudaMemcpyHostToDevice);
            cudaMemcpy(material_ids, m_material_ids.data(), m_material_ids.size()*sizeof(int), cudaMemcpyHostToDevice);
        }

};
