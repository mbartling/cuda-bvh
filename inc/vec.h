#pragma once

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
        Vec3f(float a): x(a), y(a), z(a) {}

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
    return a.x*b*x + a.y*b*y + a.z*b*z;
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

__host__ __device__ __inline__
void normalize(Vec3f& a){
    float sqinv = 1.0f/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
    a *= sqinv;
}
__host__ __device__ __inline__
float norm(const Vec3f& a){
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

class Vec4f{
    public:
        float x;
        float y;
        float z;
        float w;
        
        // Constructors
        __host__ __device__
        Vec3f(float x, float y, float z, float w): x(x), y(y), z(z), w(w){}

        __host__ __device__
        Vec3f(float x, float y, float z): x(x), y(y), z(z){}
        
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
