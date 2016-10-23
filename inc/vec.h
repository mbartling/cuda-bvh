#pragma once
#include <cuda_runtime.h>
#include <math.h>

class Vec3f{
    public:
        float x;
        float y;
        float z;

        // Constructors
        __host__ __device__
        Vec3f(float x, float y, float z): x(x), y(y), z(z){}
        
        __host__ __device__
        Vec3f(float x, float y): x(x), y(y) {}

        __host__ __device__
        explicit Vec3f(float a): x(a), y(a), z(a) {}
        
        __host__ __device__
        Vec3f(float* a) : x(a[0]), y(a[1]), z(a[2]) {}

        __host__ __device__
        Vec3f& operator = (float a){ x = a; y = a; z = a; return *this; }
        
        __host__ __device__
        Vec3f operator - (){ return Vec3f(-x, -y, -z); }
        
        __host__ __device__
        Vec3f& operator *= (float a){ x *= a; y *= a; z *= a; return *this; }
        
        __host__ __device__
        Vec3f& operator %= (const Vec3f& a){ x *= a.x; y *= a.y; z *= a.z; return *this; }


};

__host__ __device__ __inline__
float operator * (const Vec3f& a, const Vec3f b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ __inline__

Vec3f operator + (const Vec3f& a, const Vec3f b){
    return Vec3f(a.x + b.x , a.y + b.y , a.z + b.z);
}

__host__ __device__ __inline__
Vec3f operator - (const Vec3f& a, const Vec3f b){
    return Vec3f(a.x - b.x , a.y - b.y , a.z - b.z);
}

__host__ __device__ __inline__
Vec3f operator % (const Vec3f& a, const Vec3f b){
    return Vec3f(a.x * b.x , a.y * b.y , a.z * b.z);
}
//Cross Product
__host__ __device__ __inline__
Vec3f operator ^ (const Vec3f& a, const Vec3f& b){
    return Vec3f(a.y*b.z - a.z*b.y,
                 a.z*b.x - a.x*b.z,
                 a.x*b.y - a.y*b.x);
}

__host__ __device__ __inline__
void normalize(Vec3f& a){
    float sqinv = rnorm3df(a.x, a.y, a.z);
    a *= sqinv;
}
__host__ __device__ __inline__
float norm(const Vec3f& a){
    return norm3df(a.x, a.y, a.z);
}

__host__ __device__ __inline__
bool isZero(const Vec3f& a){
    retrun a.x == 0 && a.y == 0 && a.z == 0;
}
__device__ __inline__
Vec3f maximum(const Vec3f& a, const Vec3f& b){
    return Vec3f(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __inline__
Vec3f minimum(const Vec3f& a, const Vec3f& b){
    return Vec3f(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

class Vec4f{
    public:
        float x;
        float y;
        float z;
        float w;
        
        // Constructors
        __host__ __device__
        Vec4f(float x, float y, float z, float w): x(x), y(y), z(z), w(w){}

        __host__ __device__
        Vec4f(float x, float y, float z): x(x), y(y), z(z){}
        
        __host__ __device__
        Vec4f(float x, float y): x(x), y(y) {}

        __host__ __device__
        Vec4f(float a): x(a), y(a), z(a), w(a) {}

        __host__ __device__
        Vec4f& operator = (float a){ x = a; y = a; z = a; w = a; return *this; }
        
        __host__ __device__
        Vec4f operator - (){ return Vec4f(-x, -y, -z, -w); }
        
        __host__ __device__
        Vec4f& operator *= (float a){ x *= a; y *= a; z *= a; w *= a; return *this; }
        
        __host__ __device__
        Vec4f& operator %= (const Vec4f& a){ x *= a.x; y *= a.y; z *= a.z; w *= a.w; return *this; }

};

__host__ __device__ __inline__
float operator * (const Vec4f& a, const Vec4f b){
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__host__ __device__ __inline__

Vec4f operator + (const Vec4f& a, const Vec4f b){
    return Vec4f(a.x + b.x , a.y + b.y , a.z + b.z, a.w + b.w);
}

__host__ __device__ __inline__
Vec4f operator - (const Vec4f& a, const Vec4f b){
    return Vec4f(a.x - b.x , a.y - b.y , a.z - b.z, a.w - b.w);
}

__host__ __device__ __inline__
Vec4f operator % (const Vec4f& a, const Vec4f b){
    return Vec4f(a.x * b.x , a.y * b.y , a.z * b.z, a.w*b.w);
}

__host__ __device__ __inline__
void normalize(Vec4f& a){
    float sqinv = 1.0f/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w);
    a *= sqinv;
}
__host__ __device__ __inline__
float norm(const Vec4f& a){
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w);
}

// x,y,z are rows
class Mat3f{
    public:
        Vec3f x;
        Vec3f y;
        Vec3f z;

        __host__ __device__
        Mat3f(float xx, float xy, float xz,
              float yx, float yy, float yz,
              float zx, float zy, float zz): x(xx,yy,zz), y(yx, yy, yz), z(zx, zy, zz) {}
        
        __host__ __device__
        Mat3f() : x(), y(), z() {}
        
        __host__ __device__
        Mat3f(Vec3f x, Vec3f y, Vec3f z) x(x), y(y), z(z) {}

};

__host__ __device__ __inline__
Vec3f multAT_x(const Mat3f& A, const Vec3f& f){
    return Vec3f(A.x.x * f.x + A.y.x * f.y + A.z.x * f.z, 
                 A.x.y * f.x + A.y.y*f.y + A.z.y*f.z, 
                 A.x.z * f.x + A.y.z * f.y + A.z.z * f.z);

__host__ __device__ __inline__
Vec3f multxT_A(const Vec3f& f, const Mat3f& A){
    return multAT_x(A, f); // Same value except this one should be treated as transposed
}
