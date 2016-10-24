#pragma once
#include <iostream>
#include "vec.h"
#include "tiny_obj_loader.h"
#include "common.h"
using namespace tinyobj;

// A triangle is 3 Vertex Indices
// This just makes it easier to access
struct TriangleIndices{
        index_t a;
        index_t b;
        index_t c;
};

class ray;
class isect;
class Scene_d;

