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


    __host__ __device__
        void setBools() {
            _refl  = isZero(kr);
            _trans = isZero(kt);
            _spec  = _refl || isZero(ks);
            _both  = _refl && _trans;
        }

    __host__ __device__
        bool Refl() const {
            return _refl;
        }
    __host__ __device__
        bool Trans() const {
            return _trans;
        }

    __device__
        Vec3f shade(Scene_d* scene, const ray& r, const isect& i){
 //           Vec3f I = ke + (ka % scene->ambient());
            Vec3f I = kd;

 //           Vec3f V = scene->getCamera().getEye() - r.at(i.t) ;
 //           V = -V;
 //           normalize(V);

 //           AreaLight* pLight  = scene->getAreaLight();
 //           Vec3f lightDir = pLight->getDirection(r.at(i.t));
 //           ray toLightR(r.at(i.t), lightDir);


 //           Vec3f atten    = pLight->distanceAttenuation(r.at(i.t)) * pLight->shadowAttenuation(toLightR, r.at(i.t));
 //           // Vec3f atten    = Vec3f() * pLight->shadowAttenuation(toLightR, r.at(i.t));
 //           float blah = i.N*lightDir;
 //           if(blah< 0) blah = 0;
 //           Vec3f diffuseTerm  = blah*kd;
 //           // Vec3f diffuseTerm  = maximum(Vec3f(0,0,0), i.N*lightDir*kd);


 //           Vec3f Rdir = -2.0*(lightDir*i.N)*i.N + lightDir;
 //           normalize(Rdir); 
 //           float tmp = Rdir*V;

 //           tmp =  powf(max(0.0, tmp), shininess);
 //           Vec3f specularTerm = tmp * ks;
 //           I += atten % ( diffuseTerm + specularTerm) % pLight->getColor();
            return I;
            // return kd;
        }
    __device__
        Material& operator += (const Material& m){
            ke += m.ke;
            ka += m.ka;
            ks += m.ks;
            kd += m.kd;
            kr += m.kr;
            kt += m.kt;
            ior += m.ior;
            shininess += m.shininess;
            setBools();
            return *this;

        }

    friend __device__ __inline__
        Material operator*(float d, Material m);


};

__device__ __inline__
Material operator*(float d, Material m){
    m.ke *= d;
    m.ka *= d;
    m.ks *= d;
    m.kd *= d;
    m.kr *= d;
    m.kt *= d;
    m.ior *= d;
    m.shininess *= d;
    return m;

}
