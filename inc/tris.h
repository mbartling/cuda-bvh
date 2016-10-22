#pragma once

using namespace tinyobj;
using std::vector;
using std::cout;
using std::endl;

// A triangle is 3 Vertex Indices
// This just makes it easier to access
struct TriangleIndices{
        index_t a;
        index_t b;
        index_t c;
}
