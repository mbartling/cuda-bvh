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
        explicit Vec3f(float* a) : x(a[0]), y(a[1]), z(a[2]) {}

        __host__ __device__
        Vec3f(void): x(0.0), y(0.0), z(0.0) {}

        __host__ __device__
        Vec3f& operator = (float a){ x = a; y = a; z = a; return *this; }

        __host__ __device__
        Vec3f& operator /= (float a){ x /= a; y /= a; z /= a; return *this; }
        
        __host__ __device__
        Vec3f operator - (){ return Vec3f(-x, -y, -z); }
        
        __host__ __device__
        Vec3f& operator *= (float a){ x *= a; y *= a; z *= a; return *this; }
        
        __host__ __device__
        Vec3f& operator %= (const Vec3f& a){ x *= a.x; y *= a.y; z *= a.z; return *this; }
        
        __host__ __device__
        Vec3f& operator -= (const Vec3f& a){ x -= a.x; y -= a.y; z -= a.z; return *this; }
        __host__ __device__
        Vec3f& operator += (const Vec3f& a){ x += a.x; y += a.y; z += a.z; return *this; }


};

__host__ __device__ __inline__
float operator * (const Vec3f& a, const Vec3f& b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ __inline__

Vec3f operator + (const Vec3f& a, const Vec3f& b){
    return Vec3f(a.x + b.x , a.y + b.y , a.z + b.z);
}

__host__ __device__ __inline__
Vec3f operator - (const Vec3f& a, const Vec3f& b){
    return Vec3f(a.x - b.x , a.y - b.y , a.z - b.z);
}

__host__ __device__ __inline__
Vec3f operator * (const Vec3f& a, float b){
    return Vec3f(a.x*b, a.y*b, a.z*b);
}

__host__ __device__ __inline__
Vec3f operator - (const Vec3f& a, float b){
    return Vec3f(a.x - b, a.y -b, a.z -b);
}

__host__ __device__ __inline__
Vec3f operator + (const Vec3f& a, float b){
    return Vec3f(a.x + b, a.y+b, a.z+b);
}

__host__ __device__ __inline__
Vec3f operator * (float b, const Vec3f& a){
    return Vec3f(a.x*b, a.y*b, a.z*b);
}

__host__ __device__ __inline__
Vec3f operator - (float b, const Vec3f& a){
    return Vec3f(a.x - b, a.y -b, a.z -b);
}

__host__ __device__ __inline__
Vec3f operator + (float b, const Vec3f& a){
    return Vec3f(a.x + b, a.y+b, a.z+b);
}
    __host__ __device__ __inline__
Vec3f operator / (const Vec3f& a, float b){
    return Vec3f(a.x/b, a.y/b, a.z/b);
}

__host__ __device__ __inline__
Vec3f operator % (const Vec3f& a, const Vec3f& b){
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
    float sqinv = 1.0/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z); //rnorm3df(a.x, a.y, a.z); TODO where is this defined
    a *= sqinv;
}

__host__ __device__ __inline__
float norm(const Vec3f& a){
    //return norm3df(a.x, a.y, a.z);
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

__host__ __device__ __inline__
float rnorm(const Vec3f& a){
    float sqinv = 1.0/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z); //rnorm3df(a.x, a.y, a.z); TODO where is this defined
    return sqinv;
}

__host__ __device__ __inline__
bool isZero(const Vec3f& a){
    return (a.x == 0) && (a.y == 0) && (a.z == 0);
}
__host__ __device__ __inline__
Vec3f maximum(const Vec3f& a, const Vec3f& b){
    return Vec3f(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ __inline__
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
        Vec4f(): x(0.0), y(0.0), z(0.0), w(0.0) {}

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
        Mat3f(void) : x(0.0), y(0.0), z(0.0) {}
        
        __host__ __device__
        Mat3f(Vec3f x, Vec3f y, Vec3f z) : x(x), y(y), z(z) {}

};

__host__ __device__ __inline__
Vec3f multAT_x(const Mat3f& A, const Vec3f& f){
    return Vec3f(A.x.x * f.x + A.y.x * f.y + A.z.x * f.z, 
                 A.x.y * f.x + A.y.y*f.y + A.z.y*f.z, 
                 A.x.z * f.x + A.y.z * f.y + A.z.z * f.z);
}

__host__ __device__ __inline__
Vec3f multxT_A(const Vec3f& f, const Mat3f& A){
    return multAT_x(A, f); // Same value except this one should be treated as transposed
}
__host__ __device__ __inline__
Vec3f operator * (const Mat3f& a, const Vec3f& f){
    return Vec3f(a.x*f, a.y*f, a.z*f);
}
class Mat4f{
    public:
        Vec4f x; //Row 1
        Vec4f y; //Row 2
        Vec4f z;
        Vec4f w;

        __host__ __device__
        Mat4f(float xx, float xy, float xz, float xw,
              float yx, float yy, float yz, float yw,
              float zx, float zy, float zz, float zw): x(xx,yy,zz,zw), y(yx, yy, yz,yw), z(zx, zy, zz, zw) {}
        
        
        __host__ __device__
        Mat4f(const Vec4f& x, const Vec4f& y, const Vec4f& z, const Vec4f& w) : x(x), y(y), z(z), w(w) {}
        
        __host__ __device__
        Mat4f(): x(), y(), z(), w() {}
        
        __host__ __device__
        Mat4f& operator *= (float d){
            x *= d;
            y *= d;
            z *= d;
            w *= d;
            return *this;
        }

};

__host__ __device__ __inline__
float operator * (const Vec4f& a, const Vec3f& b){
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w;
}
__host__ __device__ __inline__
Vec3f operator * (const Mat4f& a, const Vec3f& b){
    return Vec3f(a.x*b, a.y*b, a.z*b);
}

//__host__ __device__
//Mat4f getInverse(const Mat4f& a)
//{
//    float s0 = a.x.x * a.y.y - a.y.x * a.x.y;
//    float s1 = a.x.x * a.y.z - a.y.x * a.x.z;
//    float s2 = a.x.x * a.y.w - a.y.x * a.x.w;
//    float s3 = a.x.y * a.y.z - a.y.y * a.x.z;
//    float s4 = a.x.y * a.y.w - a.y.y * a.x.w;
//    float s5 = a.x.z * a.y.w - a.y.z * a.x.w;
//
//    float c5 = a.z.z * a.w.w - a.w.z * a.z.w;
//    float c4 = a.z.y * a.w.w - a.w.y * a.z.w;
//    float c3 = a.z.y * a.w.z - a.w.y * a.z.z;
//    float c2 = a.z.x * a.w.w - a.w.x * a.z.w;
//    float c1 = a.z.x * a.w.z - a.w.x * a.z.z;
//    float c0 = a.z.x * a.w.y - a.w.x * a.z.y;
//
//    // Should check for 0 determinant
//    float invdet = 1.0 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
//
//    Mat4f b;
//
//    b.x = Vec4f(( a.y.y * c5 - a.y.z * c4 + a.y.w * c3) 
//          ,(-a.x.y * c5 + a.x.z * c4 - a.x.w * c3) 
//          ,( a.w.y * s5 - a.w.z * s4 + a.w.w * s3) 
//          ,(-a.z.y * s5 + a.z.z * s4 - a.z.w * s3)) ;
//
//    b.y = Vec4f((-a.y.x * c5 + a.y.z * c2 - a.y.w * c1) 
//          ,( a.x.x * c5 - a.x.z * c2 + a.x.w * c1) 
//          ,(-a.w.x * s5 + a.w.z * s2 - a.w.w * s1) 
//          ,( a.z.x * s5 - a.z.z * s2 + a.z.w * s1)) ;
//
//    b.z = Vec4f(( a.y.x * c4 - a.y.y * c2 + a.y.w * c0) 
//          ,(-a.x.x * c4 + a.x.y * c2 - a.x.w * c0) 
//          ,( a.w.x * s4 - a.w.y * s2 + a.w.w * s0) 
//          ,(-a.z.x * s4 + a.z.y * s2 - a.z.w * s0)) ;
//
//    b.w = Vec4f((-a.y.x * c3 + a.y.y * c1 - a.y.z * c0) 
//          ,( a.x.x * c3 - a.x.y * c1 + a.x.z * c0) 
//          ,(-a.w.x * s3 + a.w.y * s1 - a.w.z * s0) 
//          ,( a.z.x * s3 - a.z.y * s1 + a.z.z * s0)) ;
//    
//    b*= invdet;
//    return b;
//}
