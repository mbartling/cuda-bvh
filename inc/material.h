#pragma once

#include "vec.h"

//Forward Decs
class Scene_d;
class ray;
class isect;

struct Material {
    Vec3f ka; //Ambient
    Vec3f kd; //diffuse
    Vec3f ks; //specular
    Vec3f kt; //Transmittance
    Vec3f ke; //Emmission
    Vec3f kr; //reflectance == specular
    float shininess;
    float ior;
    float dissolve; // 1 == opaque; 0 == fully transparent

    bool _refl;								  // specular reflector?
    bool _trans;							  // specular transmitter?
    bool _spec;								  // any kind of specular?
    bool _both;								  // reflection and transmission


    void setBools() {
        _refl  = isZero(kr);
        _trans = isZero(kt);
        _spec  = _refl || isZero(ks);
        _both  = _refl && _trans;
    }

    __device__
        Vec3f shade(Scene_d* scene, const ray& r, const isect& i) const; 

    __device__
        Material& operator += (const Material& rhs){
            _ke += m._ke;
            _ka += m._ka;
            _ks += m._ks;
            _kd += m._kd;
            _kr += m._kr;
            _kt += m._kt;
            _index += m._index;
            _shininess += m._shininess;
            setBools();
            return *this;

        }

    friend __device__ __inline__
        Material operator*(float d, Material m);


};

__device__ __inline__
Material operator*(float d, Material m){
    m._ke *= d;
    m._ka *= d;
    m._ks *= d;
    m._kd *= d;
    m._kr *= d;
    m._kt *= d;
    m._index *= d;
    m._shininess *= d;
    return m;

}
