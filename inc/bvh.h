#pragma once
#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1
#include "tinyobjloader.h"
#include "scene.h"

//Device BVH
class BVH_d {
    private:
        const Scene_d& mScene;

    public:
        BVH_d(const Scene_d& mScene) : mScene(mScene) {}

        

};

void bvh(void);
