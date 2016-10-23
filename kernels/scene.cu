#include "scene.h"

struct minAccessor{
    
    __host__ __device__
    Vec3f operator () (const BoundingBox& a){
        return a.bmin;
    }
};

struct minFunctor{
    __host__ __device__
    Vec3f operator () (const Vec3f& a, const Vec3f& b){
        return minimum(a,b);
    }
};
struct maxAccessor{
    
    __host__ __device__
    Vec3f operator () (const BoundingBox& a){
        return a.bmax;
    }
};

struct maxFunctor{
    __host__ __device__
    Vec3f operator () (const Vec3f& a, const Vec3f& b){
        return maximum(a,b);
    }
};
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
}


void Scene_d::findMinMax(Vec3f& mMin, Vec3f mMax){

    thrust::device_ptr<BoundingBox> dvp(BBoxs);
    mMin = thrust::transform_reduce(dvp, dvp + numTriangles, minAccessor(), Vec3f(1e9, 1e9, 1e9), minFunctor);
    mMax = thrust::transform_reduce(dvp, dvp + numTriangles, maxAccessor(),Vec3f(-1e9, -1e9, -1e9), maxFunctor);
}


__global__ 
void computeBoundingBoxes_kernel(int numTriangles, Vec3f* vertices, TriangleIndices* t_indices, BoundingBox* BBoxs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > numTriangles) return;

    TriangleIndices t_idx = t_indices[idx];

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


