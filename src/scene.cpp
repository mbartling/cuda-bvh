#include "scene.h"


// Load the OBJ and add all the triangles to a linear array
void Scene_h::LoadObj(string filename){
    vector<shape_t> shapes;
    vector<material_t> materials;
    string err;

    bool ret = tinyobj::LoadObj(&mAttributes, &shapes, &materials, &err, filename.c_str());

    if (!err.empty()) { // `err` may contain warning message.
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    //For each shape
    for(size_t s = 0; s < shapes.size(); s++){

        //For each Triangle Add the 
        size_t index_offset = 0;
        for(size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++){
            TriangleIndices index;
            index.a = shapes[s].mesh.indices[index_offset + 0];
            index.b = shapes[s].mesh.indices[index_offset + 1];
            index.c = shapes[s].mesh.indices[index_offset + 2];

            t_indices.push_back(index);
            material_ids.push_back(shapes[s].mesh.material_ids[f]);
        }
        
        index_offset += 3;

    }
}

Scene_h& Scene_h::operator = (const Scene_d& deviceScene){
    cudaMemcpy(&image, deviceScene.image, imageWidth*imageHeight*sizeof(Vec4f), cudaMemcpyDeviceToHost);
}

Scene_d& Scene_d::operator = (const Scene_h& hostScene){
    numVertices = hostScene.mAttributes.vertices.size();
    numTriangles = hostScene.t_indices.size();
    imageWidth = hostScene.imageWidth;
    imageHeight = hostScene.imageHeight;

    //Allocate Space for everything
    cudaMalloc(&vertices, numVertices*sizeof(float));
    cudaMalloc(&normals, numVertices*sizeof(float));

    cudaMalloc(&BBoxs, numTriangles*sizeof(BoundingBox));
    cudaMalloc(&t_indices, numTriangles*sizeof(TriangleIndices));

    cudaMalloc(&image, imageWidth*imageHeight*sizeof(float4));

    //Copy stuff
    cudaMemcpy(vertices, hostScene.mAttributes.vertices.data(), numVertices*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(normals, hostScene.mAttributes.normals.data(), numVertices*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t_indices, hostScene.t_indices.data(), numTriangles*sizeof(TriangleIndices), cudaMemcpyHostToDevice);

    computeBoundingBoxes();

    bvh.setUp(vertices, BBoxs, t_indices, numTriangles);


    return *this;
}

Scene_d::~Scene_d(){
    cudaFree(vertices);
    cudaFree(normals);
    cudaFree(BBoxs);
    cudaFree(t_indices);
    cudaFree(image);
}

