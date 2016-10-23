#include "scene.h"
#include <stdio.h>

__global__ 
void computeBoundingBoxes_kernel(int numTriangles, Vec3f* vertices, TriangleIndices* t_indices, BoundingBox* BBoxs);
__device__
BoundingBox computeTriangleBoundingBox(const Vec3f& a, const Vec3f& b, const Vec3f& c);

void Scene_d::computeBoundingBoxes(){
    // Invoke kernel
    int N = numTriangles;
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;
    computeBoundingBoxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(numTriangles, vertices, t_indices, BBoxs);
    //cudaDeviceSynchronize();

}


__global__ 
void computeBoundingBoxes_kernel(int numTriangles, Vec3f* vertices, TriangleIndices* t_indices, BoundingBox* BBoxs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > numTriangles) return;

    TriangleIndices t_idx = t_indices[idx];
    printf("idx(%d), a(%d, %d, %d)\n" , idx, vertices[t_idx.a.vertex_index].x,
                                     vertices[t_idx.a.vertex_index].y,
                                     vertices[t_idx.a.vertex_index].z);

    BBoxs[idx] = computeTriangleBoundingBox(vertices[t_idx.a.vertex_index],vertices[t_idx.b.vertex_index],vertices[t_idx.c.vertex_index]);

    return;
}

__device__ 
BoundingBox computeTriangleBoundingBox(const Vec3f& a, const Vec3f& b, const Vec3f& c){
    BoundingBox bbox;
    BoundingBox localbounds;
    localbounds.setMax(maximum( a, b));
    localbounds.setMin(minimum( a, b));

    localbounds.setMax(maximum( c, localbounds.getMax()));
    localbounds.setMin(minimum( c, localbounds.getMin()));
    return localbounds;
}


