#include "tris.h"

__device__
bool intersectTriangle(const ray& r, isect&  i, Scene_d* scene, int object_id){

    TriangleIndices ids = scene->t_indices[object_id];

    Vec3f a = scene->vertices[ids.a];
    Vec3f b = scene->vertices[ids.b];
    Vec3f c = scene->vertices[ids.c];

    /*
       -DxAO = AOxD
       AOx-D = -(-DxAO)
       |-D AB AC| = -D*(ABxAC) = -D*normal = 1. 1x
       |AO AB AC| = AO*(ABxAC) = AO*normal = 1. 
       |-D AO AC| = -D*(AOxAC) = 1. 1x || AC*(-DxAO) = AC*M = 1. 1x
       |-D AB AO| = -D*(ABxAO) = 1. 1x || (AOx-D)*AB = (DxAO)*AB = -M*AB = 1.
       */
    float mDet;
    float mDetInv;
    float alpha;
    float beta;
    float t;
    Vec3f rDir = r.getDirection();
    //Moller-Trombore approach is a change of coordinates into a local uv space
    // local to the triangle
    Vec3f AB = b - a;
    Vec3f AC = c - a;

    // if (normal * -r.getDirection() < 0) return false;
    Vec3f P = rDir ^ AC;
    mDet = AB * P;
    if(fabsf(mDet) < RAY_EPSILON ) return false;

    mDetInv = 1/mDet;
    Vec3f T = r.getPosition() - a;
    alpha = T * P * mDetInv;    
    if(alpha < 0 || alpha > 1) return false;

    Vec3f Q = T ^ AB;
    beta = rDir * Q * mDetInv;
    if(beta < 0 || alpha + beta > 1) return false;
    t = AC * Q * mDetInv;

    if(fabsf(t) < RAY_EPSILON) return false; // Jaysus this sucked
    i.bary = Vec3f(1 - (alpha + beta), alpha, beta);
    i.t = t;


    // std::cout << traceUI->smShadSw() << std::endl; 
    // if(traceUI->smShadSw() && !parent->floatCheck()){
    //Smooth Shading
    Vec3f aN = scene->normals[ids.a];
    Vec3f bN = scene->normals[ids.b];
    Vec3f cN = scene->normals[ids.c];
    i.N = (1 - (alpha + beta))*aN + \
          alpha*bN + \
          beta*cN;

    //i.N = normal;

    i.N.normalize();

    i.object_id = object_id;
    //if(!parent->materials.empty() && parent->hasVertexMaterials()){
    Material aM;
    aM += (1 - (alpha + beta))*(scene->materials[ids.a]); 
    aM +=                alpha*(scene->materials[ids.b]); 
    aM +=                beta* (scene->materials[ids.c]); 

    i.material = aM;

    return true;


}
