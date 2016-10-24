#include "raytracer.h"
struct RayStack{
    ray r;
    isect i;
    Vec3f colorC;
    int state;
};

__global__ 
void runRayTracerKernel(Scene_d scene, int depth);

void RayTracer::run(){
    int blockSize = 32;
    dim3 blockDim(blockSize, blockSize); //A thread block is 32x32 pixels
    dim3 gridDim(deviceScene.imageWidth/blockDim.x, deviceScene.imageHeight/blockDim.y);
    int stackDepth = ( 1 << depth) - 1;
    runRayTracerKernel<<<gridDim, blockDim, stackDepth*sizeof(RayStack)>>>(deviceScene, depth);
}

__global__ 
void runRayTracerKernel(Scene_d scene, int depth){
    extern __shared__ RayStack rayStack[];
    RayStack* stackPtr = rayStack;
    int curDepth = 0;

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = py*scene.imageWidth + px;

    float x = float(px)/float(scene.imageWidth);
    float y = float(py)/float(scene.imageHeight);

    //Get view from the camera
    //perturb
    //x += randx; //in [0,1]
    //y += randy; //in [0,1]
    //scene.camera.rayThrough(x, y, stackPtr->r);


    while(true){
        ray& r = stackPtr->r;
        isect& i = stackPtr->i;
        Vec3f& colorC = stackPtr->colorC;
        int& state = stackPtr->state;

        if(state == 0) //Check for intersection
        {
            if(scene.intersect(r, i)){
                Vec3f q = r.at(i.t);
                colorC = i.material.shade(&scene, r, i);
                if(curDepth >= depth){state = 5;} //Exit
                else
                    state = 1;
            }else{
                colorC = Vec3f(0.0,0.0,0.0);
            }
        }
        if(state == 1) //Check for reflection
        {
            if(!i.material.Refl())
                state = 3;
            else{
                Vec3f Rdir = -2.0*(r.getDirection()*i.N)*i.N + r.getDirection();
                normalize(Rdir);
                
                //Put DRT stuff HERE
                
                state = 2; //Select next state for my stack frame return
                Vec3f q = r.at(i.t);

                //Push
                stackPtr++;
                curDepth++;

                stackPtr->r = ray(q, Rdir);
                stackPtr->state = 0;
                continue; //Handle the stack push
            }
        }
        if(state == 2) //Post reflection
        {
            colorC += i.material.kr % (stackPtr+1)->colorC;
            state = 3;
        }
        if(state == 3) //Check for refraction
        {
            if(!i.material.Trans())
                state = 5; // Done
            else{
                Vec3f n = i.N;
                Vec3f rd = r.getDirection();
                Vec3f rcos = n*(-rd*n);
                Vec3f rsin = rcos + rd;
                float etai = 1.0;
                float etat = i.material.ior;
                Vec3f tcos, tsin;
                float eta;
                if(rd*n < 0){
                    eta = etai/etat;
                    n = -n;
                } else{
                    eta = etat/etai;
                }
                tsin = eta*rsin;
                float TIR = 1 - tsin*tsin;
                if(TIR >= 0){
                    tcos = n*sqrt(TIR);
                    Vec3f Tdir = tcos + tsin;
                    Vec3f q = r.at(i.t);
                    normalize(Tdir);
                    
                    //Put DRT stuff HERE

                    //Recusive part
                    state = 4;

                //Push
                stackPtr++;
                curDepth++;

                stackPtr->r = ray(q, Tdir);
                stackPtr->state = 0;
                continue; //Handle the stack push
                }
            }
        }
        if(state == 4) //Post refraction
        {

            colorC += i.material.kt % (stackPtr+1)->colorC;
            state = 5;
        }
        // There is no state 5 on purpose
        if(curDepth == 0) //Hit nothing and am at root of stack
            break;
        else{
            stackPtr--; //Pop
            curDepth--;
        }

    }

    scene.image[idx] = rayStack[0].colorC;
}
