#pragma once
#include "scene.h"

class RayTracer{
    private:
        int imageWidth;
        int imageHeight;
        int superSampling; //1x 2x 4x 16x 

        int depth;

        Vec3f* image; //Host side
        Scene_h hostScene;
        Scene_d deviceScene;
    
    public:

        RayTracer(): imageWidth(512), imageHeight(512), superSampling(1), hostScene(imageWidth, imageHeight, superSampling), depth(1)
        {
            image = new Vec3f[imageWidth*imageHeight];
        }
        
        RayTracer(int imageWidth, int imageHeight, int superSampling): imageWidth(512), imageHeight(512), superSampling(1), hostScene(imageWidth, imageHeight, superSampling), depth(1)
        {
            image = new Vec3f[imageWidth*imageHeight];
        }

        void LoadObj(string filename){ hostScene.LoadObj(filename); }
        void setUpDevice(){ deviceScene = hostScene; }
        
        void pullRaytracedImage(){ hostScene = deviceScene; }
        void writeImage(string filename) {} //TODO

        void run();
        

        ~RayTracer(){ delete[] image; }


};
