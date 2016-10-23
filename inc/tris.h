#pragma once

using namespace tinyobj;

// A triangle is 3 Vertex Indices
// This just makes it easier to access
struct TriangleIndices{
        index_t a;
        index_t b;
        index_t c;
}

class ray;
class isect;
class Scene_d;

__device__
bool intersectTriangle(const ray& r, isect&  i, Scene_d* scene, int object_id);
