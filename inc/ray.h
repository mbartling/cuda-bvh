//
// ray.h
//
// The low-level classes used by ray tracing: the ray and isect classes.
//

#ifndef __RAY_H__
#define __RAY_H__

// who the hell cares if my identifiers are longer than 255 characters:
#pragma warning(disable : 4786)

#include "vec.h"
//#include "mat.h"
#include "common.h"

// A ray has a position where the ray starts, and a direction (which should
// always be normalized!)

class ray {
    public:

        __device__
        ray(const Vec3f &pp, const Vec3f &dd)
            : p(pp), d(dd) {}
        __device__
        ray(const ray& other) : p(other.p), d(other.d){}
        __device__
        ~ray() {}
        // virtual ~ray(){}
        __device__
        ray& operator =( const ray& other ) 
        { p = other.p; d = other.d; return *this; }

        __device__
        Vec3f at( double t ) const
        { return p + (t*d); }

        __device__
        Vec3f getPosition() const { return p; }
        __device__
        Vec3f getDirection() const { return d; }

    public:
        Vec3f p;
        Vec3f d;
};

// The description of an intersection point.


Vec3f CosWeightedRandomHemiDir2(Vec3f n);


#endif // __RAY_H__
