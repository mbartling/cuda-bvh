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
    public:
        int numVertices;
        int numTriangles;
        int imageWidth;
        int imageHeight;

        Vec3f* vertices;
        Vec3f* normals;
        Vec3f* texcoords;

        Material* materials;
        BoundingBox* BBoxs; //Per Triangle Bounding Box

        TriangleIndices* t_indices;

        Vec4f* image;
        BVH_d bvh;

        friend class Scene_h;
    public:

        Scene_d& operator = (const Scene_h& hostScene); //Copy Triangles, materials, etc to the device
    
        void computeBoundingBoxes();

        __device__
        bool intersect(const ray& r, isect& i); //Find the closest point of intersection

        ~Scene_d();

};

