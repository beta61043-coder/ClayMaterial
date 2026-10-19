#pragma once
#include <cmath>
#include <limits>
#include <random>

const float PI = 3.14159265358979f;
const float INF = std::numeric_limits<float>::max();

// ==================== МАТЕМАТИКА ====================
struct Vec2 { float x, y; Vec2():x(0),y(0){} Vec2(float x,float y):x(x),y(y){} Vec2 operator+(const Vec2& o) const { return {x+o.x, y+o.y}; } };
struct Vec3 {
    float x, y, z;
    Vec3():x(0),y(0),z(0){} Vec3(float x,float y,float z):x(x),y(y),z(z){}
    Vec3 operator+(const Vec3& o)const{return{x+o.x,y+o.y,z+o.z};}
    Vec3 operator-(const Vec3& o)const{return{x-o.x,y-o.y,z-o.z};}
    Vec3 operator*(float s)const{return{x*s,y*s,z*s};}
    Vec3 operator*(const Vec3& o)const{return{x*o.x,y*o.y,z*o.z};}
    Vec3& operator+=(const Vec3& o){x+=o.x;y+=o.y;z+=o.z;return *this;}
    Vec3& operator-=(const Vec3& o){x-=o.x;y-=o.y;z-=o.z;return *this;}
    float dot(const Vec3& o)const{return x*o.x+y*o.y+z*o.z;}
    Vec3 cross(const Vec3& o)const{return{y*o.z-z*o.y,z*o.x-x*o.z,x*o.y-y*o.x};}
    Vec3 normalize()const{float l=std::sqrt(x*x+y*y+z*z);if(l<1e-8f)return{0,0,0};return{x/l,y/l,z/l};}
    float length()const{return std::sqrt(x*x+y*y+z*z);}
};

struct Ray { Vec3 orig, dir; Ray(){} Ray(const Vec3& o,const Vec3& d):orig(o),dir(d){} };
