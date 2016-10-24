#pragma once
#include "vec.h"
#include "material.h"
class isect {
    public:
        float t;
        Vec3f N;
        Vec3f bary;
        int object_id;
        Material material; // For smooth shading


};
