#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "Math.h"
#include "stb_image.h"

struct Image {
    int w=0, h=0;
    std::vector<uint8_t> rgba; // RGBA8
    bool load(const std::string& path) {
        int ch=0; unsigned char* p = stbi_load(path.c_str(), &w, &h, &ch, 4);
        if (!p) return false;
        rgba.assign(p, p + w*h*4);
        stbi_image_free(p);
        return true;
    }
    // Билинейная выборка с обёрткой UV
    Vec3 sample(float u, float v) const {
        if (w==0) return {0.5f,0.5f,0.5f};
        u = u - std::floor(u); v = v - std::floor(v);
        float fx = u * w - 0.5f, fy = (1.0f - v) * h - 0.5f;
        int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
        float dx = fx - x0, dy = fy - y0;
        auto get = [&](int x, int y) {
            x = ((x % w) + w) % w; y = ((y % h) + h) % h;
            int i = (y*w + x) * 4;
            return Vec3(rgba[i]/255.f, rgba[i+1]/255.f, rgba[i+2]/255.f);
        };
        Vec3 c00=get(x0,y0), c10=get(x0+1,y0), c01=get(x0,y0+1), c11=get(x0+1,y0+1);
        Vec3 top = c00*(1-dx) + c10*dx;
        Vec3 bot = c01*(1-dx) + c11*dx;
        return top*(1-dy) + bot*dy;
    }
    float sample_lum(float u, float v) const {
        Vec3 c = sample(u, v);
        return c.x*0.2126f + c.y*0.7152f + c.z*0.0722f;
    }
};

struct Mesh {
    std::vector<Vec3> verts;
    std::vector<std::vector<int>> faces;
    std::vector<std::vector<Vec2>> face_uvs;
    std::vector<Vec3> normals;       // smooth normals (displaced)
    std::vector<Vec3> orig_normals;  // базовые (до displacement)
    // Триангулированный (после subdivision) для трассировки
    struct Tri {
        Vec3 v0, v1, v2, n0, n1, n2;       // displaced
        Vec3 op0, op1, op2, on0, on1, on2; // original positions & normals
        Vec2 uv0, uv1, uv2;
    };
    std::vector<Tri> tris;
};

struct SubdivResult { std::vector<Vec3> verts; std::vector<std::vector<int>> faces; std::vector<std::vector<Vec2>> face_uvs; }

Mesh init_cube();
Mesh parse_obj(const std::string& path);
