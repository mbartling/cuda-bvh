#include "material.h"

__device__
Vec3f Material::shade(Scene_d* scene, const ray& r, const isect& i){
    Vec3d I = ke + (ka % scene->ambient());

    Vec3d V = scene->getCamera().getEye() - r.at(i.t) ;
    V = -V;
    normalize(V);

    AreaLight* pLight  = scene->getAreaLight();
    Vec3d lightDir = pLight->getDirection(r.at(i.t));
    ray toLightR(r.at(i.t), lightDir);


    Vec3d atten    = pLight->distanceAttenuation(r.at(i.t)) * pLight->shadowAttenuation(toLightR, r.at(i.t));
    // Vec3d atten    = Vec3d() * pLight->shadowAttenuation(toLightR, r.at(i.t));
    float blah = i.N*lightDir;
    if(blah< 0) blah = 0;
    Vec3d diffuseTerm  = blah*kd;
    // Vec3d diffuseTerm  = maximum(Vec3d(0,0,0), i.N*lightDir*kd);


    Vec3d Rdir = -2.0*(lightDir*i.N)*i.N + lightDir;
    normalize(Rdir); 
    float tmp = Rdir*V;

    tmp =  powf(max(0.0, tmp), shininess);
    Vec3d specularTerm = tmp * ks;
    I += atten % ( diffuseTerm + specularTerm) % pLight->getColor();
}
return I;
// return kd;
}
