#include "raytracer.h"
struct RayStack{
    ray r;
    isect i;
    Vec3f colorC;
};

__global__ 
void runRayTracerKernel(Scene_d scene, int depth);

void RayTracer::run(){
    int blockSize = 32;
    dim3 blockDim(blockSize, blockSize); //A thread block is 32x32 pixels
    dim3 gridDim(deviceScene.imageWidth/blockDim.x, deviceScene.imageHeight/blockDim.y);
    int stackDepth ( 1 << depth) - 1;
    runRayTracerKernel<<<gridDim, blockDim, stackDepth*sizeof(RayStack)>>>(deviceScene, depth);
}

__global__ 
void runRayTracerKernel(Scene_d scene, int depth){
    extern __shared__ RayStack rayStack[];
    Scene_d* scenePtr = &scene;
    int stackI = 0;
    while(true){
        ray& r = &rayStack[stackI].r;
        if(scene.intersect(,
        
    }
}
