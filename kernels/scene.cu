#include "scene.h"
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

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
// Declarations
__global__ 
void computeBoundingBoxes_kernel(int numTriangles, Vec3f* vertices, TriangleIndices* t_indices, BoundingBox* BBoxs);

__device__
BoundingBox computeTriangleBoundingBox(const Vec3f& a, const Vec3f& b, const Vec3f& c);

__global__
void AverageSuperSamplingKernel(Vec3f* smallImage, Vec3f* deviceImage, int imageWidth, int imageHeight, int superSampling);

//============================
//
void Scene_d::computeBoundingBoxes(){
    // Invoke kernel
    int N = numTriangles;
    int threadsPerBlock = 256;
    int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock;
    computeBoundingBoxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(numTriangles, vertices, t_indices, BBoxs);
    //cudaDeviceSynchronize();

}

void AverageSuperSampling(Vec3f* smallImage, 
                          Vec3f* deviceImage, 
                          int imageWidth, 
                          int imageHeight, 
                          int superSampling)
{
    int blockSize = 32;
    dim3 blockDim(blockSize, blockSize); //A thread block is 32x32 pixels
    dim3 gridDim(imageWidth/blockDim.x, imageHeight/blockDim.y);
    AverageSuperSamplingKernel<<<gridDim, blockDim>>>(smallImage, deviceImage, imageWidth, imageHeight, superSampling);
}

void Scene_d::findMinMax(Vec3f& mMin, Vec3f& mMax){

    thrust::device_ptr<BoundingBox> dvp(BBoxs);
    mMin = thrust::transform_reduce(dvp, 
            dvp + numTriangles, 
            minAccessor(), 
            Vec3f(1e9, 1e9, 1e9), 
            minFunctor());
    mMax = thrust::transform_reduce(dvp, 
            dvp + numTriangles, 
            maxAccessor(),
            Vec3f(-1e9, -1e9, -1e9),
            maxFunctor());
}


__global__ 
void computeBoundingBoxes_kernel(int numTriangles, Vec3f* vertices, TriangleIndices* t_indices, BoundingBox* BBoxs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;

    TriangleIndices t_idx = t_indices[idx];
    printf("idx(%d), a(%0.6f, %0.6f, %0.6f)\n" , idx, vertices[t_idx.a.vertex_index].x,
                                     vertices[t_idx.a.vertex_index].y,
                                     vertices[t_idx.a.vertex_index].z);

    printf("idx(%d), b(%0.6f, %0.6f, %0.6f)\n" , idx, vertices[t_idx.b.vertex_index].x,
                                     vertices[t_idx.b.vertex_index].y,
                                     vertices[t_idx.b.vertex_index].z);

    printf("idx(%d), c(%0.6f, %0.6f, %0.6f)\n" , idx, vertices[t_idx.c.vertex_index].x,
                                     vertices[t_idx.c.vertex_index].y,
                                     vertices[t_idx.c.vertex_index].z);

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

//__device__
//bool Scene_d::intersect(const ray& r, isect& i){
//    return bvh.intersect(r, i, this);
//}

__global__
void AverageSuperSamplingKernel(Vec3f* smallImage, Vec3f* deviceImage, int imageWidth, int imageHeight, int superSampling)
{
    int pixelX = blockIdx.x*blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y*blockDim.y + threadIdx.y;
    int pixelIdx = pixelY*imageWidth + pixelX;
    
    Vec3f mSum;
    for(int i = 0; i < superSampling; i++){
         for(int j = 0; j < superSampling; j++){
            int idxX = pixelX*superSampling + j;
            int idxY = pixelY*superSampling + i;
            int idx = idxY*superSampling*imageWidth + idxX;
            mSum += deviceImage[idx];
        }
    }

    mSum /= float(superSampling);
    smallImage[pixelIdx] = mSum;
}

