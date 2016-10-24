#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class Camera
{
    public:
        Camera();
        void rayThrough( float x, float y, ray &r );
        void setEye( const Vec3f &eye );
        void setLook( float, float, float, float );
        void setLook( const Vec3f &viewDir, const Vec3f &upDir );
        void setFOV( float );
        void setAspectRatio( float );

        float getAspectRatio() { return aspectRatio; }

        const Vec3f& getEye() const         { return eye; }
        const Vec3f& getLook() const        { return look; }
        const Vec3f& getU() const           { return u; }
        const Vec3f& getV() const           { return v; }
    private:
        Mat3f m;                     // rotation matrix
        float normalizedHeight;    // dimensions of image place at unit dist from eye
        float aspectRatio;

        void update();              // using the above three values calculate look,u,v

        Vec3f eye;
        Vec3f look;                  // direction to look
        Vec3f u,v;                   // u and v in the 
};

#endif
