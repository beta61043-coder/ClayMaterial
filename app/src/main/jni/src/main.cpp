#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"
#include <SDL.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <thread>
#include <atomic>
#include <numeric>
#include <limits>
#include <random>
#include <map>
#include <fstream>
#include <sstream>
#include <chrono>
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>

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

// ==================== ИЗОБРАЖЕНИЕ ====================
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

// ==================== МЕШ ====================
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

// Инициализация куба (как _init_default_cube)
Mesh init_cube() {
    Mesh m;
    m.verts = {
        {-0.5f,-0.5f,-0.5f},{0.5f,-0.5f,-0.5f},{0.5f,0.5f,-0.5f},{-0.5f,0.5f,-0.5f},
        {-0.5f,-0.5f, 0.5f},{0.5f,-0.5f, 0.5f},{0.5f,0.5f, 0.5f},{-0.5f,0.5f, 0.5f},
    };
    m.faces = {
        {0,3,2,1},{4,5,6,7},{0,4,7,3},{1,2,6,5},{0,1,5,4},{3,7,6,2}
    };
    m.face_uvs = {
        {{1,1},{1,0},{0,0},{0,1}},{{0,1},{1,1},{1,0},{0,0}},
        {{0,1},{0,0},{1,0},{1,1}},{{1,1},{1,0},{0,0},{0,1}},
        {{0,1},{1,1},{1,0},{0,0}},{{0,0},{0,1},{1,1},{1,0}}
    };
    return m;
}

// ==================== CATMULL-CLARK ====================
struct SubdivResult { std::vector<Vec3> verts; std::vector<std::vector<int>> faces; std::vector<std::vector<Vec2>> face_uvs; };

SubdivResult catmull_clark(const Mesh& src) {
    SubdivResult out;
    int NF = src.faces.size();
    int NV = src.verts.size();
    std::vector<Vec3> face_points(NF);
    std::vector<Vec2> face_point_uvs(NF);
    for (int fi = 0; fi < NF; ++fi) {
        const auto& f = src.faces[fi];
        Vec3 s(0,0,0); Vec2 us(0,0);
        for (int v : f) s += src.verts[v];
        for (const auto& uv : src.face_uvs[fi]) us = us + uv;
        face_points[fi] = s * (1.0f / f.size());
        face_point_uvs[fi] = Vec2(us.x / std::max(1,(int)src.face_uvs[fi].size()), us.y / std::max(1,(int)src.face_uvs[fi].size()));
    }

    struct EdgeInfo { int a, b; std::vector<int> adj; };
    std::vector<EdgeInfo> edges;
    std::vector<int> edge_index(NF);
    std::vector<std::vector<int>> edge_ids; // edge id per face-side
    edge_ids.resize(NF);
    auto find_edge = [&](int a, int b) -> int {
        for (int i = 0; i < (int)edges.size(); ++i) if (edges[i].a == a && edges[i].b == b) return i;
        return -1;
    };
    auto find_or_create_edge = [&](int a, int b) -> int {
        if (a > b) std::swap(a,b);
        int idx = find_edge(a,b);
        if (idx >= 0) { edges[idx].adj.push_back(-1); return idx; }
        edges.push_back({a, b, {-1}});
        return (int)edges.size()-1;
    };
    for (int fi = 0; fi < NF; ++fi) {
        int n = src.faces[fi].size();
        edge_ids[fi].resize(n);
        for (int i = 0; i < n; ++i) {
            int a = src.faces[fi][i], b = src.faces[fi][(i+1)%n];
            if (a > b) std::swap(a,b);
            int idx = find_edge(a,b);
            if (idx < 0) {
                edges.push_back({a, b, {}});
                idx = (int)edges.size()-1;
            }
            edges[idx].adj.push_back(fi);
            edge_ids[fi][i] = idx;
        }
    }

    std::vector<Vec3> edge_points(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
        auto& e = edges[i];
        if (e.adj.size() >= 2) {
            edge_points[i] = (src.verts[e.a] + src.verts[e.b] + face_points[e.adj[0]] + face_points[e.adj[1]]) * 0.25f;
        } else {
            edge_points[i] = (src.verts[e.a] + src.verts[e.b]) * 0.5f;
        }
    }

    std::vector<std::vector<int>> vert_edges(NV), vert_faces(NV);
    for (int ei = 0; ei < (int)edges.size(); ++ei) {
        vert_edges[edges[ei].a].push_back(ei);
        vert_edges[edges[ei].b].push_back(ei);
    }
    for (int fi = 0; fi < NF; ++fi) for (int v : src.faces[fi]) vert_faces[v].push_back(fi);

    std::vector<Vec3> new_pos(NV);
    for (int vi = 0; vi < NV; ++vi) {
        int n = vert_edges[vi].size();
        if (n < 3) { new_pos[vi] = src.verts[vi]; continue; }
        Vec3 F(0,0,0); for (int fi : vert_faces[vi]) F += face_points[fi];
        F = F * (1.0f / vert_faces[vi].size());
        Vec3 R(0,0,0);
        for (int ei : vert_edges[vi]) R += (src.verts[edges[ei].a] + src.verts[edges[ei].b]) * 0.5f;
        R = R * (1.0f / n);
        new_pos[vi] = (F + R*2.0f + src.verts[vi]*((float)(n-3))) * (1.0f/(float)n);
    }

    out.verts.reserve(NV + edges.size() + NF);
    for (auto& v : new_pos) out.verts.push_back(v);
    int edge_offset = out.verts.size();
    for (auto& ep : edge_points) out.verts.push_back(ep);
    int face_offset = out.verts.size();
    for (auto& fp : face_points) out.verts.push_back(fp);

    for (int fi = 0; fi < NF; ++fi) {
        const auto& f = src.faces[fi];
        const auto& uvs = src.face_uvs[fi];
        int n = f.size();
        int fp_idx = face_offset + fi;
        Vec2 fp_uv = face_point_uvs[fi];
        for (int k = 0; k < n; ++k) {
            int vk = f[k], vn = f[(k+1)%n], vp = f[(k-1+n)%n];
            Vec2 uk = uvs[k], un = uvs[(k+1)%n], up = uvs[(k-1+n)%n];
            int e_next = edge_offset + edge_ids[fi][k];
            int e_prev = edge_offset + edge_ids[fi][(k-1+n)%n];
            out.faces.push_back({vk, e_next, fp_idx, e_prev});
            out.face_uvs.push_back({uk, Vec2((uk.x+un.x)*0.5f,(uk.y+un.y)*0.5f), fp_uv, Vec2((up.x+uk.x)*0.5f,(up.y+uk.y)*0.5f)});
        }
    }
    return out;
}

// ==================== НОРМАЛИ (smooth) ====================
void compute_normals(Mesh& m) {
    m.normals.assign(m.verts.size(), Vec3(0,0,0));
    for (size_t fi = 0; fi < m.faces.size(); ++fi) {
        auto& f = m.faces[fi];
        Vec3 n(0,0,0);
        for (size_t i = 0; i < f.size(); ++i) {
            const Vec3& c = m.verts[f[i]];
            const Vec3& nx = m.verts[f[(i+1)%f.size()]];
            n.x += (c.y - nx.y) * (c.z + nx.z);
            n.y += (c.z - nx.z) * (c.x + nx.x);
            n.z += (c.x - nx.x) * (c.y + nx.y);
        }
        for (int v : f) m.normals[v] += n;
    }
    for (auto& n : m.normals) n = n.normalize();
}



// ==================== ПОДГОТОВКА ТЕКСТУРЫ (как в Godot) ====================
void box_blur_rgba(Image& img, int radius); // forward
void prepare_displacement_image(Image& img) {
    if (img.w == 0 || img.h == 0) return;
    int max_dim = 256;
    // 1) Уменьшаем до 256x256 с билинейной интерполяцией
    if (img.w > max_dim || img.h > max_dim) {
        float s = (float)max_dim / std::max(img.w, img.h);
        int nw = std::max(1, (int)(img.w * s));
        int nh = std::max(1, (int)(img.h * s));
        std::vector<uint8_t> dst(nw * nh * 4);
        for (int y = 0; y < nh; ++y) {
            for (int x = 0; x < nw; ++x) {
                float src_x = (x + 0.5f) / s - 0.5f;
                float src_y = (y + 0.5f) / s - 0.5f;
                int x0 = std::max(0, std::min(img.w - 1, (int)src_x));
                int y0 = std::max(0, std::min(img.h - 1, (int)src_y));
                int x1 = std::max(0, std::min(img.w - 1, x0 + 1));
                int y1 = std::max(0, std::min(img.h - 1, y0 + 1));
                float fx = src_x - x0;
                float fy = src_y - y0;
                for (int ch = 0; ch < 4; ++ch) {
                    float v00 = img.rgba[(y0 * img.w + x0) * 4 + ch];
                    float v10 = img.rgba[(y0 * img.w + x1) * 4 + ch];
                    float v01 = img.rgba[(y1 * img.w + x0) * 4 + ch];
                    float v11 = img.rgba[(y1 * img.w + x1) * 4 + ch];
                    float top = v00 * (1.0f - fx) + v10 * fx;
                    float bot = v01 * (1.0f - fx) + v11 * fx;
                    dst[(y * nw + x) * 4 + ch] = (uint8_t)std::clamp(top * (1.0f - fy) + bot * fy, 0.0f, 255.0f);
                }
            }
        }
        img.w = nw; img.h = nh;
        img.rgba = dst;
    }
    // 2) Двойной box_blur(2), как в _prepare_displacement_image
    box_blur_rgba(img, 2);
    box_blur_rgba(img, 2);
}

Image make_highpass(const Image& raw, int blur_radius) {
    Image blurred = raw;
    box_blur_rgba(blurred, blur_radius);
    Image hp = raw;
    for (size_t i = 0; i < hp.rgba.size(); ++i) {
        int v = (int)raw.rgba[i] - (int)blurred.rgba[i] + 128;
        hp.rgba[i] = (uint8_t)std::clamp(v, 0, 255);
    }
    return hp;
}

// ==================== БЛЮР ТЕКСТУРЫ ====================
void box_blur_rgba(Image& img, int radius) {
    if (img.w == 0 || radius <= 0 || img.rgba.empty()) return;
    int W = img.w, H = img.h;
    std::vector<uint8_t> tmp(img.rgba.size());
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int rs=0,gs=0,bs=0,as=0,cnt=0;
            for (int k = -radius; k <= radius; ++k) {
                int xx = std::clamp(x+k,0,W-1);
                int i = (y*W+xx)*4;
                rs+=img.rgba[i]; gs+=img.rgba[i+1]; bs+=img.rgba[i+2]; as+=img.rgba[i+3]; cnt++;
            }
            int j = (y*W+x)*4;
            tmp[j]=rs/cnt; tmp[j+1]=gs/cnt; tmp[j+2]=bs/cnt; tmp[j+3]=as/cnt;
        }
    }
    std::vector<uint8_t> out(img.rgba.size());
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int rs=0,gs=0,bs=0,as=0,cnt=0;
            for (int k = -radius; k <= radius; ++k) {
                int yy = std::clamp(y+k,0,H-1);
                int i = (yy*W+x)*4;
                rs+=tmp[i]; gs+=tmp[i+1]; bs+=tmp[i+2]; as+=tmp[i+3]; cnt++;
            }
            int j = (y*W+x)*4;
            out[j]=rs/cnt; out[j+1]=gs/cnt; out[j+2]=bs/cnt; out[j+3]=as/cnt;
        }
    }
    img.rgba.swap(out);
}

// ==================== НОРМАЛИ ИЗ ПРОИЗВОЛЬНЫХ ВЕРШИН ====================
std::vector<Vec3> compute_normals_for(const std::vector<Vec3>& verts,
                                      const std::vector<std::vector<int>>& faces) {
    std::vector<Vec3> normals(verts.size(), Vec3(0,0,0));
    for (const auto& f : faces) {
        Vec3 n(0,0,0);
        for (size_t i = 0; i < f.size(); ++i) {
            const Vec3& cur = verts[f[i]];
            const Vec3& nxt = verts[f[(i+1)%f.size()]];
            n.x += (cur.y - nxt.y) * (cur.z + nxt.z);
            n.y += (cur.z - nxt.z) * (cur.x + nxt.x);
            n.z += (cur.x - nxt.x) * (cur.y + nxt.y);
        }
        for (int v : f) normals[v] += n;
    }
    for (auto& n : normals) n = n.normalize();
    return normals;
}

// ==================== ПАРАМЕТРЫ ПРОЦЕДУРНОЙ ТЕКСТУРЫ ====================
struct DisplaceParams {
    bool enabled = false;
    float strength = 0.2f;
    float uv_scale = 1.0f;
    float midlevel = 0.5f;
    int direction = 0; // 0=normal,1=X,2=Y,3=Z
    float rotate_x = 90.0f, rotate_y = -180.0f, rotate_z = 0.0f;
    bool mirror_x = false, mirror_y = true, mirror_z = false;
    Vec3 empty_pos { 0, 0, 0 };
    bool invert = true;
    float threshold = 0.05f;
    float sharpness = 1.0f;
    float contrast = 0.0f;
    // Вторая процедурная текстура (overlay)
    bool enabled2 = false;
    float strength2 = 0.10f;
    float alpha2 = 0.5f;
    float midlevel2 = 0.5f;
    bool invert2 = false;
    float threshold2 = 0.05f;
};

struct Material {
    Vec3 albedo_color { 0.82f, 0.82f, 0.88f };
    float metallic = 0.05f, roughness = 0.35f, specular = 0.6f;
    float bump_strength = 1.5f;
    float sss_strength = 0.0f;
    Vec3 sss_color { 0.785f, 0.0f, 0.152f };
    float procedural_alpha = 1.0f;
    float procedural_overlay = 0.5f;
};

// ==== Трипленарная выборка (порт sample_triplanar) ====
Vec2 rotate_uv(Vec2 uv, float deg) {
    if (std::abs(deg) < 0.0001f) return uv;
    float cu = uv.x - 0.5f, cv = uv.y - 0.5f;
    float ca = std::cos(deg*PI/180.f), sa = std::sin(deg*PI/180.f);
    return Vec2(std::fmod(cu*ca - cv*sa + 0.5f + 10.0f, 1.0f),
                std::fmod(cu*sa + cv*ca + 0.5f + 10.0f, 1.0f));
}

float triplanar_height(Vec3 p, Vec3 n, const Image& tex, float inv_scale, Vec3 off3, const DisplaceParams& dp) {
    float ax = std::abs(n.x), ay = std::abs(n.y), az = std::abs(n.z);
    // TRIPLANAR_BLEND_SHARPNESS = 1.0 в Godot, но мы оставляем настройку
    float wx = std::pow(ax, dp.sharpness);
    float wy = std::pow(ay, dp.sharpness);
    float wz = std::pow(az, dp.sharpness);
    float wsum = wx + wy + wz;
    if (wsum < 1e-6f) return 0.5f;
    wx /= wsum; wy /= wsum; wz /= wsum;

    Vec2 ux = Vec2(std::fmod((p.y+0.5f)*inv_scale + off3.y + 10.f, 1.f),
                   std::fmod((p.z+0.5f)*inv_scale + off3.z + 10.f, 1.f));
    if (dp.mirror_x) ux.x = std::fmod(1.0f - ux.x + 10.f, 1.f);
    ux = rotate_uv(ux, dp.rotate_x);

    Vec2 uy = Vec2(std::fmod((p.x+0.5f)*inv_scale + off3.x + 10.f, 1.f),
                   std::fmod((p.z+0.5f)*inv_scale + off3.z + 10.f, 1.f));
    if (dp.mirror_y) uy.x = std::fmod(1.0f - uy.x + 10.f, 1.f);
    uy = rotate_uv(uy, dp.rotate_y);

    Vec2 uz = Vec2(std::fmod((p.x+0.5f)*inv_scale + off3.x + 10.f, 1.f),
                   std::fmod((p.y+0.5f)*inv_scale + off3.y + 10.f, 1.f));
    if (dp.mirror_z) uz.x = std::fmod(1.0f - uz.x + 10.f, 1.f);
    uz = rotate_uv(uz, dp.rotate_z);

    float hx = tex.sample_lum(ux.x, ux.y);
    float hy = tex.sample_lum(uy.x, uy.y);
    float hz = tex.sample_lum(uz.x, uz.y);
    return hx*wx + hy*wy + hz*wz;
}

Vec3 sample_triplanar(Vec3 p, Vec3 n, const Image& tex, float inv_scale, Vec3 off3, float sharpness) {
    float ax = std::abs(n.x), ay = std::abs(n.y), az = std::abs(n.z);
    float wx = std::pow(ax, sharpness);
    float wy = std::pow(ay, sharpness);
    float wz = std::pow(az, sharpness);
    float wsum = wx + wy + wz;
    if (wsum < 1e-6f) return {0.5f,0.5f,0.5f};
    wx /= wsum; wy /= wsum; wz /= wsum;

    Vec2 ux = Vec2(std::fmod((p.y+0.5f)*inv_scale + off3.y + 10.f, 1.f),
                   std::fmod((p.z+0.5f)*inv_scale + off3.z + 10.f, 1.f));
    Vec2 uy = Vec2(std::fmod((p.x+0.5f)*inv_scale + off3.x + 10.f, 1.f),
                   std::fmod((p.z+0.5f)*inv_scale + off3.z + 10.f, 1.f));
    Vec2 uz = Vec2(std::fmod((p.x+0.5f)*inv_scale + off3.x + 10.f, 1.f),
                   std::fmod((p.y+0.5f)*inv_scale + off3.y + 10.f, 1.f));
    Vec3 cx = tex.sample(ux.x, ux.y);
    Vec3 cy = tex.sample(uy.x, uy.y);
    Vec3 cz = tex.sample(uz.x, uz.y);
    return cx*wx + cy*wy + cz*wz;
}

// Bump нормаль из градиента высоты (порт bump_normal_local)
Vec3 bump_normal(Vec3 p, Vec3 N_base, Vec3 N_weights,
                 const Image& tex, float inv_scale, Vec3 off3,
                 float strength, const DisplaceParams& dp) {
    float eps = 0.005f;
    float h  = triplanar_height(p, N_weights, tex, inv_scale, off3, dp);
    float hx = triplanar_height(p + Vec3(eps,0,0), N_weights, tex, inv_scale, off3, dp);
    float hy = triplanar_height(p + Vec3(0,eps,0), N_weights, tex, inv_scale, off3, dp);
    float hz = triplanar_height(p + Vec3(0,0,eps), N_weights, tex, inv_scale, off3, dp);
    Vec3 grad((hx-h)/eps, (hy-h)/eps, (hz-h)/eps);
    Vec3 grad_t = grad - N_base * grad.dot(N_base);
    return (N_base - grad_t * strength).normalize();
}

Vec3 bump_normal_pair(Vec3 p, Vec3 N_base, Vec3 N_weights,
                      const Image* t1, const Image* t2, bool has2,
                      float inv_scale, Vec3 off3, const DisplaceParams& dp,
                      float bump_s1, float bump_s2) {
    float eps = 0.005f;
    auto H = [&](Vec3 pp) -> float {
        float g1 = triplanar_height(pp, N_weights, *t1, inv_scale, off3, dp) - 0.5f;
        float val = g1 * bump_s1;
        if (has2 && t2) {
            float g2 = triplanar_height(pp, N_weights, *t2, inv_scale, off3, dp) - 0.5f;
            val += g2 * dp.alpha2 * bump_s2;
        }
        return val;
    };
    float h  = H(p);
    float hx = H(p + Vec3(eps,0,0));
    float hy = H(p + Vec3(0,eps,0));
    float hz = H(p + Vec3(0,0,eps));
    Vec3 grad((hx-h)/eps, (hy-h)/eps, (hz-h)/eps);
    Vec3 grad_t = grad - N_base * grad.dot(N_base);
    Vec3 n_res = N_base - grad_t;
    
    // Защита от "выворачивания" нормали внутрь объекта (чёрные пятна)
    if (n_res.dot(N_base) < 0.1f) {
        n_res = N_base; 
    }
    
    return n_res.normalize();
}


// ==================== OBJ IMPORT ====================
Mesh parse_obj(const std::string& path) {
    Mesh m;
    std::ifstream file(path);
    if (!file.is_open()) return m;

    std::vector<Vec3> verts;
    std::vector<Vec2> uvs;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;
        if (prefix == "v") {
            float x, y, z;
            if (ss >> x >> y >> z) verts.push_back(Vec3(x, y, z));
        } else if (prefix == "vt") {
            float u, v;
            if (ss >> u >> v) uvs.push_back(Vec2(u, 1.0f - v));
        } else if (prefix == "f") {
            std::vector<int> face_v;
            std::vector<Vec2> face_uv;
            std::string v_str;
            while (ss >> v_str) {
                size_t slash1 = v_str.find('/');
                size_t slash2 = v_str.find('/', slash1 + 1);
                int vi = -1;
                if (slash1 != std::string::npos) {
                    vi = std::stoi(v_str.substr(0, slash1)) - 1;
                } else {
                    vi = std::stoi(v_str) - 1;
                }
                if (vi >= 0 && vi < (int)verts.size()) face_v.push_back(vi);
                
                if (slash1 != std::string::npos && slash2 != std::string::npos && slash2 > slash1 + 1) {
                    int ti = std::stoi(v_str.substr(slash1 + 1, slash2 - slash1 - 1)) - 1;
                    if (ti >= 0 && ti < (int)uvs.size()) face_uv.push_back(uvs[ti]);
                    else face_uv.push_back(Vec2(0,0));
                } else {
                    face_uv.push_back(Vec2(0,0));
                }
            }
            if (face_v.size() >= 3) {
                m.faces.push_back(face_v);
                m.face_uvs.push_back(face_uv);
            }
        }
    }

    if (!verts.empty()) {
        Vec3 mn(INF, INF, INF), mx(-INF, -INF, -INF);
        for (const auto& v : verts) {
            mn.x = std::min(mn.x, v.x); mn.y = std::min(mn.y, v.y); mn.z = std::min(mn.z, v.z);
            mx.x = std::max(mx.x, v.x); mx.y = std::max(mx.y, v.y); mx.z = std::max(mx.z, v.z);
        }
        Vec3 center = (mn + mx) * 0.5f;
        float max_dim = std::max({mx.x - mn.x, mx.y - mn.y, mx.z - mn.z});
        if (max_dim < 1e-6f) max_dim = 1.0f;
        for (auto& v : verts) v = (v - center) * (2.0f / max_dim);
    }
    m.verts = verts;
    return m;
}

// ==================== ВЫЧИСЛЕНИЕ DISPLACED-МЕША ====================
void rebuild_triangles(Mesh& m,
                       const DisplaceParams& dp,
                       const Image* proc_tex,
                       const Image* proc_tex2,
                       const Material& mat) {
    compute_normals(m);
    m.orig_normals = m.normals;

    std::vector<Vec3> displaced = m.verts;
    bool t2_ok = proc_tex2 && proc_tex2->w > 0 && dp.enabled2;
    if (dp.enabled && proc_tex && proc_tex->w > 0) {
        float inv_scale = 1.0f / std::max(dp.uv_scale, 0.001f);
        Vec3 off3(dp.empty_pos.x * 0.5f, dp.empty_pos.y * 0.5f, dp.empty_pos.z * 0.5f);
        for (int vi = 0; vi < (int)m.verts.size(); ++vi) {
            Vec3 p_orig = m.verts[vi];
            Vec3 n = m.normals[vi];
            Vec3 dir = (dp.direction == 1) ? Vec3(1,0,0) :
                       (dp.direction == 2) ? Vec3(0,1,0) :
                       (dp.direction == 3) ? Vec3(0,0,1) : n;
            Vec3 p_sample = p_orig;
            for (int it = 0; it < 3; ++it) {
                // Текстура 1
                float gray1 = triplanar_height(p_sample, n, *proc_tex, inv_scale, off3, dp);
                float delta1 = gray1 - dp.midlevel;
                if (dp.invert) delta1 = -delta1;
                if (std::abs(delta1) < dp.threshold) delta1 = 0.0f;
                float disp1 = delta1 * dp.strength;
                
                // Текстура 2 (overlay)
                float disp2 = 0.0f;
                if (t2_ok) {
                    float gray2 = triplanar_height(p_sample, n, *proc_tex2, inv_scale, off3, dp);
                    float delta2 = gray2 - dp.midlevel2;
                    if (dp.invert2) delta2 = -delta2;
                    if (std::abs(delta2) < dp.threshold2) delta2 = 0.0f;
                    disp2 = delta2 * dp.strength2 * dp.alpha2;
                }
                
                p_sample = p_orig + dir * (disp1 + disp2);
            }
            displaced[vi] = p_sample;
        }
    }

    std::vector<Vec3> disp_normals = compute_normals_for(displaced, m.faces);

    m.tris.clear();
    for (size_t fi = 0; fi < m.faces.size(); ++fi) {
        auto& f = m.faces[fi];
        auto& uvs = m.face_uvs[fi];
        int n = f.size();
        if (n < 3) continue;
        for (int j = 1; j < n - 1; ++j) {
            int i0 = f[0], i1 = f[j], i2 = f[j+1];
            Mesh::Tri t;
            t.v0 = displaced[i0]; t.v1 = displaced[i1]; t.v2 = displaced[i2];
            t.n0 = disp_normals[i0]; t.n1 = disp_normals[i1]; t.n2 = disp_normals[i2];
            t.op0 = m.verts[i0]; t.op1 = m.verts[i1]; t.op2 = m.verts[i2];
            t.on0 = m.orig_normals[i0]; t.on1 = m.orig_normals[i1]; t.on2 = m.orig_normals[i2];
            t.uv0 = uvs[0];
            t.uv1 = (uvs.size() > (size_t)j) ? uvs[j] : Vec2(0,0);
            t.uv2 = (uvs.size() > (size_t)(j+1)) ? uvs[j+1] : Vec2(0,0);
            m.tris.push_back(t);
        }
    }
}

// ==================== BVH ====================
struct AABB {
    Vec3 mn { INF, INF, INF }, mx { -INF, -INF, -INF };
    void expand(const Vec3& p) {
        mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
        mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
    }
    bool hit(const Ray& r, float& tmin, float tmax) const {
        float t1=(mn.x-r.orig.x)/(r.dir.x+1e-8f), t2=(mx.x-r.orig.x)/(r.dir.x+1e-8f);
        float t3=(mn.y-r.orig.y)/(r.dir.y+1e-8f), t4=(mx.y-r.orig.y)/(r.dir.y+1e-8f);
        float t5=(mn.z-r.orig.z)/(r.dir.z+1e-8f), t6=(mx.z-r.orig.z)/(r.dir.z+1e-8f);
        float tmn = std::max({std::min(t1,t2), std::min(t3,t4), std::min(t5,t6)});
        float tmx = std::min({std::max(t1,t2), std::max(t3,t4), std::max(t5,t6)});
        if (tmx < 0 || tmn > tmx) return false;
        tmin = tmn > 0 ? tmn : tmx;
        return true;
    }
};

struct BVHNode {
    AABB b;
    int left=-1, right=-1, start=0, count=0;
    bool leaf() const { return left==-1 && right==-1; }
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<Mesh::Tri> tris;

    void build(const std::vector<Mesh::Tri>& src) {
        tris = src;
        nodes.clear();
        if (tris.empty()) return;
        nodes.push_back(BVHNode());
        rec(0, 0, tris.size());
    }

    void rec(int ni, int s, int c) {
        AABB b;
        for (int i=0;i<c;++i) {
            auto& t = tris[s+i];
            b.expand(t.v0); b.expand(t.v1); b.expand(t.v2);
        }
        nodes[ni].b = b;
        if (c <= 4) { nodes[ni].start = s; nodes[ni].count = c; return; }
        Vec3 e = b.mx - b.mn;
        int ax = (e.x > e.y && e.x > e.z) ? 0 : (e.y > e.z ? 1 : 2);
        std::sort(tris.begin()+s, tris.begin()+s+c,
            [ax](const Mesh::Tri& A, const Mesh::Tri& B){
                float ca = (ax==0?A.v0.x:(ax==1?A.v0.y:A.v0.z));
                float cb = (ax==0?B.v0.x:(ax==1?B.v0.y:B.v0.z));
                return ca < cb;
            });
        int mid = c/2;
        int li = nodes.size(); nodes.push_back(BVHNode());
        int ri = nodes.size(); nodes.push_back(BVHNode());
        nodes[ni].left = li; nodes[ni].right = ri;
        rec(li, s, mid); rec(ri, s+mid, c-mid);
    }

    // Hit info
    struct Hit { float t; int tri; Vec3 p, n; Vec3 op, on; Vec2 uv; };

    bool intersect(const Ray& ray, Hit& h) const {
        if (nodes.empty()) return false;
        float tmin;
        if (!nodes[0].b.hit(ray, tmin, INF)) return false;
        float bestT = INF;
        int bestTri = -1;
        float bestU=0, bestV=0;

        int stack[128]; int sp=0;
        stack[sp++] = 0;
        while (sp) {
            int ni = stack[--sp];
            auto& node = nodes[ni];
            if (!node.b.hit(ray, tmin, bestT)) continue;
            if (node.leaf()) {
                for (int i=0;i<node.count;++i) {
                    auto& t = tris[node.start+i];
                    Vec3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0;
                    Vec3 pv = ray.dir.cross(e2);
                    float det = e1.dot(pv);
                    if (std::abs(det) < 1e-8f) continue;
                    float inv = 1.0f/det;
                    Vec3 tv = ray.orig - t.v0;
                    float u = tv.dot(pv) * inv;
                    if (u < 0 || u > 1) continue;
                    Vec3 qv = tv.cross(e1);
                    float v = ray.dir.dot(qv) * inv;
                    if (v < 0 || u+v > 1) continue;
                    float tHit = e2.dot(qv) * inv;
                    if (tHit > 1e-4f && tHit < bestT) {
                        bestT = tHit; bestTri = node.start+i; bestU=u; bestV=v;
                    }
                }
            } else {
                stack[sp++] = node.left; stack[sp++] = node.right;
            }
        }
        if (bestTri < 0) return false;
        auto& t = tris[bestTri];
        h.t = bestT;
        h.tri = bestTri;
        h.p = ray.orig + ray.dir * bestT;
        float w0 = 1 - bestU - bestV;
        h.n = (t.n0*w0 + t.n1*bestU + t.n2*bestV).normalize();
        h.op = (t.op0*w0 + t.op1*bestU + t.op2*bestV);
        h.on = (t.on0*w0 + t.on1*bestU + t.on2*bestV).normalize();
        h.uv = Vec2(t.uv0.x*w0 + t.uv1.x*bestU + t.uv2.x*bestV,
                    t.uv0.y*w0 + t.uv1.y*bestU + t.uv2.y*bestV);
        return true;
    }

    bool shadow(const Ray& ray, float maxT) const {
        if (nodes.empty()) return false;
        float tmin;
        if (!nodes[0].b.hit(ray, tmin, maxT)) return false;
        int stack[128]; int sp=0; stack[sp++] = 0;
        while (sp) {
            int ni = stack[--sp];
            auto& node = nodes[ni];
            if (!node.b.hit(ray, tmin, maxT)) continue;
            if (node.leaf()) {
                for (int i=0;i<node.count;++i) {
                    auto& t = tris[node.start+i];
                    Vec3 e1 = t.v1-t.v0, e2 = t.v2-t.v0;
                    Vec3 pv = ray.dir.cross(e2);
                    float det = e1.dot(pv);
                    if (std::abs(det) < 1e-8f) continue;
                    float inv = 1.0f/det;
                    Vec3 tv = ray.orig - t.v0;
                    float u = tv.dot(pv)*inv;
                    if (u < 0 || u > 1) continue;
                    Vec3 qv = tv.cross(e1);
                    float v = ray.dir.dot(qv)*inv;
                    if (v < 0 || u+v > 1) continue;
                    float tHit = e2.dot(qv)*inv;
                    if (tHit > 1e-4f && tHit < maxT) return true;
                }
            } else {
                stack[sp++] = node.left; stack[sp++] = node.right;
            }
        }
        return false;
    }
};

// ==================== ПАРАМЕТРЫ СЦЕНЫ ====================
struct Scene {
    bool interactive = false;
    bool high_quality = false;
    Vec3 target {0,0,0};
    Mesh mesh;
    Image proc_tex;       // prepared (256px + blur) для displacement + модулятор
    Image proc_tex_raw;   // оригинал без изменений (для bump)
    Image proc_tex_bump;  // bump-текстура (raw или high-pass)
    Image uv_tex;         // UV albedo
    bool has_proc = false, has_uv = false;
    bool has_proc_bump = false;
    bool bump_highpass = true;
    // Вторая процедурная текстура (overlay)
    Image proc_tex2;
    Image proc_tex2_raw;
    Image proc_tex2_bump;
    bool has_proc2 = false;
    bool has_proc2_bump = false;
    bool bump_highpass2 = true;
    float bump_strength2 = 1.0f;
    float procedural_alpha2 = 0.5f;
    float procedural_overlay2 = 0.5f;
    BVH bvh;
    int subdivision_level = 0;
    DisplaceParams dp;
    Material mat;
    Vec3 light_pos { 3, 5, 2 };
    // Камера
    Vec3 eye, forward, right, up;
    float fov = 45.f * PI / 180.f;
    int W, H;
};

// ==================== МАТЕРИАЛ: SHADE ====================
// Порт fragment() из Godot шейдера + PBR + SSS
Vec3 shade_hit(const Scene& s, const BVH::Hit& h, const Vec3& eye, std::mt19937& gen) {
    Vec3 N_disp    = h.n;
    Vec3 N_weights = h.on;
    Vec3 p_tex     = h.op;

    Vec3 uv_col(1,1,1);
    if (s.has_uv) uv_col = s.uv_tex.sample(h.uv.x, h.uv.y);

    Vec3 proc_raw(1,1,1);
    Vec3 proc_raw2(1,1,1);
    bool t2_ok = s.has_proc2 && s.proc_tex2.w > 0;

    if (s.has_proc) {
        float inv_scale = 1.0f / std::max(s.dp.uv_scale, 0.001f);
        Vec3 off3(s.dp.empty_pos.x*0.5f, s.dp.empty_pos.y*0.5f, s.dp.empty_pos.z*0.5f);
        proc_raw = sample_triplanar(p_tex, N_weights, s.proc_tex, inv_scale, off3, s.dp.sharpness);
        
        if (t2_ok) {
            proc_raw2 = sample_triplanar(p_tex, N_weights, s.proc_tex2, inv_scale, off3, s.dp.sharpness);
        }

        // Bump: суммарный градиент от обеих текстур
        if (s.mat.bump_strength > 0.001f && s.mat.procedural_alpha > 0.001f) {
            float bs1 = s.mat.bump_strength * s.mat.procedural_alpha;
            float bs2 = (t2_ok && s.dp.enabled2) ? s.bump_strength2 * s.dp.alpha2 : 0.0f;
            const Image* bt1 = s.has_proc_bump ? &s.proc_tex_bump : &s.proc_tex;
            const Image* bt2 = (s.has_proc2_bump && t2_ok) ? &s.proc_tex2_bump : (t2_ok ? &s.proc_tex2 : nullptr);
            N_disp = bump_normal_pair(p_tex, N_disp, N_weights,
                                      bt1, bt2, t2_ok,
                                      inv_scale, off3, s.dp, bs1, bs2);
        }
    }

    Vec3 albedo;
    if (s.has_uv && s.has_proc) {
        // Объединяем proc1 и proc2: multiply-blend (вторая ЗАТЕНЯЕТ первую)
        Vec3 combined = proc_raw;
        if (t2_ok) {
            // proc2 * 2.0 → нейтрально в 0.5, темнее — темнее, светлее — светлее
            Vec3 mul(proc_raw2.x * 2.0f, proc_raw2.y * 2.0f, proc_raw2.z * 2.0f);
            float a2 = s.dp.alpha2;
            combined = Vec3(
                proc_raw.x * (1.0f - a2 + mul.x * a2),
                proc_raw.y * (1.0f - a2 + mul.y * a2),
                proc_raw.z * (1.0f - a2 + mul.z * a2)
            );
            combined = Vec3(std::clamp(combined.x,0.0f,1.0f),
                            std::clamp(combined.y,0.0f,1.0f),
                            std::clamp(combined.z,0.0f,1.0f));
        }
        float lum = combined.x*0.2126f + combined.y*0.7152f + combined.z*0.0722f;
        float mod_full = std::clamp(lum * 2.0f, 0.0f, 2.0f);
        float strength = s.mat.procedural_alpha * s.mat.procedural_overlay;
        float mod = 1.0f * (1.0f - strength) + mod_full * strength;
        albedo = Vec3(std::clamp(uv_col.x*mod,0.0f,1.0f),
                      std::clamp(uv_col.y*mod,0.0f,1.0f),
                      std::clamp(uv_col.z*mod,0.0f,1.0f));
    } else if (s.has_uv) {
        albedo = uv_col;
    } else if (s.has_proc) {
        Vec3 combined = proc_raw;
        if (t2_ok) {
            Vec3 mul(proc_raw2.x*2.0f, proc_raw2.y*2.0f, proc_raw2.z*2.0f);
            float a2 = s.dp.alpha2;
            combined = Vec3(
                std::clamp(proc_raw.x*(1.0f-a2+mul.x*a2), 0.0f, 1.0f),
                std::clamp(proc_raw.y*(1.0f-a2+mul.y*a2), 0.0f, 1.0f),
                std::clamp(proc_raw.z*(1.0f-a2+mul.z*a2), 0.0f, 1.0f)
            );
        }
        float a = s.mat.procedural_alpha;
        albedo = Vec3(1*(1-a)+combined.x*a, 1*(1-a)+combined.y*a, 1*(1-a)+combined.z*a);
    } else {
        albedo = Vec3(1,1,1);
    }
    albedo = albedo * s.mat.albedo_color;

    Vec3 V = (eye - h.p).normalize();
    Vec3 L = (s.light_pos - h.p).normalize();
    float NdotL = std::max(0.0f, N_disp.dot(L));
    float NdotV = std::max(0.001f, N_disp.dot(V));

    float shadow = 0.0f;
    if (NdotL > 0.001f) {
        if (s.interactive) {
            shadow = 1.0f;
        } else {
            Ray sr(h.p + N_disp * 0.01f, L);
            shadow = s.bvh.shadow(sr, 100.0f) ? 0.0f : 1.0f;
        }
    }

    Vec3 Hh = (L + V).normalize();
    float NdotH = std::max(0.0f, N_disp.dot(Hh));
    float VdotH = std::max(0.0f, V.dot(Hh));
    float rough = std::max(0.04f, s.mat.roughness);
    float a = rough * rough; float a2 = a * a;
    float denom = NdotH*NdotH*(a2 - 1.0f) + 1.0f;
    float D = a2 / (PI * denom * denom + 1e-8f);
    float k = (rough + 1) * (rough + 1) / 8.0f;
    float G = (NdotL / (NdotL*(1-k)+k)) * (NdotV / (NdotV*(1-k)+k));
    Vec3 F0 = Vec3(s.mat.specular*(1-s.mat.metallic) + albedo.x*s.mat.metallic,
                   s.mat.specular*(1-s.mat.metallic) + albedo.y*s.mat.metallic,
                   s.mat.specular*(1-s.mat.metallic) + albedo.z*s.mat.metallic);
    Vec3 F = F0 + Vec3(1-F0.x, 1-F0.y, 1-F0.z) * std::pow(1-VdotH, 5.0f);
    Vec3 spec = F * (D * G / (4*NdotV*NdotL + 1e-4f));
    Vec3 kd = Vec3(1-F.x, 1-F.y, 1-F.z) * (1 - s.mat.metallic);

    Vec3 direct = (kd * albedo * (1.0f/PI) + spec) * (3.0f * shadow * NdotL);
    Vec3 ambient = albedo * 0.15f * (0.5f + 0.5f*N_disp.y);

    Vec3 sss_emit(0,0,0);
    if (s.mat.sss_strength > 0.001f) {
        float fres = std::pow(1.0f - std::max(0.0f, N_disp.dot(V)), 2.5f);
        float wrap_ = std::clamp(N_disp.dot(L)*0.5f + 0.5f, 0.0f, 1.0f);
        float sss_term = fres*0.75f + (1.0f - wrap_)*0.25f;
        sss_emit = s.mat.sss_color * (s.mat.sss_strength * sss_term * 0.9f);
    }

    return direct + ambient + sss_emit;
}

// ==================== RENDERING ====================
struct RenderCtx {
    int W, H, rW, rH, scale;
    std::vector<Vec3> accum;
    int sample_count = 0;
    bool needs_reset = true;
    std::vector<uint32_t> pixel_buf;
    RenderCtx(int w, int h, int s) : W(w), H(h), scale(std::max(1,s)) {
        rW = w/scale; rH = h/scale;
        accum.assign(rW*rH, Vec3(0,0,0));
        pixel_buf.resize(w*h);
    }
    void set_scale(int s) {
        s = std::max(1, std::min(4, s));
        if (s == scale && !accum.empty()) return;
        scale = s;
        rW = W/scale; rH = H/scale;
        accum.assign(rW*rH, Vec3(0,0,0));
        sample_count = 0;
        needs_reset = true;
    }
    void reset() {
        if (accum.size() != (size_t)rW*rH) accum.assign(rW*rH, Vec3(0,0,0));
        else std::fill(accum.begin(), accum.end(), Vec3(0,0,0));
        sample_count = 0;
    }
    void render_frame(const Scene& s, int threads) {
        if (needs_reset) { reset(); needs_reset = false; }
        sample_count++;
        std::atomic<int> next_row{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t](){
                std::mt19937 gen(1234 + t + sample_count);
                while (true) {
                    int y = next_row.fetch_add(1);
                    if (y >= rH) break;
                    for (int x = 0; x < rW; ++x) {
                        float px = x + 0.5f, py = y + 0.5f;
                        float tanH = std::tan(s.fov*0.5f);
                        float aspect = (float)rW/rH;
                        float ndcX = (2.0f*px/rW - 1.0f) * aspect * tanH;
                        float ndcY = (1.0f - 2.0f*py/rH) * tanH;
                        Ray r;
                        r.orig = s.eye;
                        r.dir = (s.forward + s.right*ndcX + s.up*ndcY).normalize();
                        BVH::Hit hit;
                        Vec3 c(0.12f, 0.12f, 0.14f);
                        if (s.bvh.intersect(r, hit)) {
                            c = shade_hit(s, hit, s.eye, gen);
                        }
                        accum[y*rW + x] += c;
                    }
                }
            });
        }
        for (auto& th : pool) th.join();

        // Tonemap + билинейный апскейл
        float inv = 1.0f / sample_count;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                Vec3 c;
                if (scale == 1) {
                    c = accum[y*W + x] * inv;
                } else {
                    float fx = (float)x / scale - 0.5f;
                    float fy = (float)y / scale - 0.5f;
                    int x0 = std::max(0, std::min(rW - 1, (int)std::floor(fx)));
                    int y0 = std::max(0, std::min(rH - 1, (int)std::floor(fy)));
                    int x1 = std::min(rW - 1, x0 + 1);
                    int y1 = std::min(rH - 1, y0 + 1);
                    float dx = std::clamp(fx - x0, 0.0f, 1.0f);
                    float dy = std::clamp(fy - y0, 0.0f, 1.0f);
                    Vec3 c00 = accum[y0*rW + x0] * inv;
                    Vec3 c10 = accum[y0*rW + x1] * inv;
                    Vec3 c01 = accum[y1*rW + x0] * inv;
                    Vec3 c11 = accum[y1*rW + x1] * inv;
                    Vec3 top = c00*(1-dx) + c10*dx;
                    Vec3 bot = c01*(1-dx) + c11*dx;
                    c = top*(1-dy) + bot*dy;
                }
                c.x = c.x / (1.0f + c.x);
                c.y = c.y / (1.0f + c.y);
                c.z = c.z / (1.0f + c.z);
                uint8_t r = (uint8_t)(std::pow(std::clamp(c.x,0.0f,1.0f), 1.f/2.2f) * 255);
                uint8_t g = (uint8_t)(std::pow(std::clamp(c.y,0.0f,1.0f), 1.f/2.2f) * 255);
                uint8_t b = (uint8_t)(std::pow(std::clamp(c.z,0.0f,1.0f), 1.f/2.2f) * 255);
                pixel_buf[y*W + x] = (0xFFu<<24)|(r<<16)|(g<<8)|b;
            }
        }
    }
};


// ==================== FILE BROWSER DIALOG ====================
struct FileBrowserDialog {
    bool is_open = false;
    std::string current_dir;
    std::string selected;
    std::string title;
    std::string callback_tag;
    std::vector<std::string> filters;
    std::vector<std::string> dirs;
    std::vector<std::string> files;
    char path_buf[512] = "";

    void open(const std::string& t, const std::vector<std::string>& ext, const std::string& tag) {
        is_open = true; title = t; filters = ext; callback_tag = tag;
        selected.clear();
        if (current_dir.empty()) { current_dir = "/storage/emulated/0/"; // Путь к памяти телефона }
        refresh();
    }

    void refresh() {
        dirs.clear(); files.clear();
        DIR* d = opendir(current_dir.c_str());
        if (!d) { current_dir = "."; d = opendir("."); if (!d) return; }
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            std::string name = e->d_name;
            if (name == "." || name == "..") { dirs.push_back(name); continue; }
            std::string full = current_dir + "/" + name;
            struct stat st;
            if (stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) dirs.push_back(name);
            else {
                size_t dot = name.find_last_of('.');
                if (dot != std::string::npos) {
                    std::string ext = name.substr(dot+1);
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    for (auto& f : filters) if (ext == f) { files.push_back(name); break; }
                }
            }
        }
        closedir(d);
        std::sort(dirs.begin(), dirs.end());
        std::sort(files.begin(), files.end());
        strncpy(path_buf, current_dir.c_str(), sizeof(path_buf)-1);
        path_buf[sizeof(path_buf)-1] = 0;
    }

    std::string draw() {
        if (!is_open) return "";
        std::string result;
        bool visible = true;
        ImGui::SetNextWindowSize(ImVec2(680, 540), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(150, 60), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str(), &visible, ImGuiWindowFlags_NoSavedSettings)) {
            ImGui::InputText("##dir", path_buf, sizeof(path_buf));
            ImGui::SameLine();
            if (ImGui::Button("Go")) { current_dir = path_buf; refresh(); }
            ImGui::SameLine();
            if (ImGui::Button("Home")) { current_dir = "/storage/emulated/0/"; // Путь к памяти телефона refresh(); }
            ImGui::SameLine();
            if (ImGui::Button("Refresh")) refresh();
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu папок, %zu файлов)", dirs.size(), files.size());
            ImGui::Separator();
            ImGui::BeginChild("file_list", ImVec2(0, -55), true);
            for (auto& d : dirs) {
                if (d == ".") continue;
                if (d == "..") {
                    if (ImGui::Selectable("[..] вверх")) {
                        size_t p = current_dir.find_last_of('/');
                        if (p != std::string::npos && p > 0) current_dir = current_dir.substr(0, p);
                        else current_dir = ".";
                        refresh();
                    }
                    continue;
                }
                if (ImGui::Selectable(("[DIR] " + d).c_str())) {
                    current_dir = current_dir + "/" + d;
                    refresh();
                }
            }
            for (auto& f : files) {
                std::string full = current_dir + "/" + f;
                bool is_sel = (selected == full);
                if (ImGui::Selectable(f.c_str(), is_sel)) selected = full;
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                    selected = full;
                    result = "SELECT:" + full;
                }
            }
            ImGui::EndChild();
            if (!selected.empty())
                ImGui::TextColored(ImVec4(0.4f,1,0.4f,1), "Selected: %s", selected.c_str());
            else
                ImGui::TextDisabled("Double click: open, single: select");
            if (ImGui::Button("Open", ImVec2(120,0)) && !selected.empty()) result = "SELECT:" + selected;
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120,0))) { is_open = false; }
        }
        ImGui::End();
        if (!visible) is_open = false;
        if (!result.empty()) is_open = false;
        return result;
    }
};

// ==================== UI STATE ====================
struct UIState {
    char proc_path[512] = "";
    char proc2_path[512] = "";
    char uv_path[512] = "";
    char obj_path[512] = "";
    bool show_ui = true;
    // Камера
    float yaw = 0.7f, pitch = 0.4f, dist = 4.5f;
    // Показ текстуры-превью
    void* proc_tex_id = nullptr;
    void* uv_tex_id = nullptr;
};


// ==================== WIREFRAME ====================
void draw_line(std::vector<uint32_t>& buf, int W, int H, int x0, int y0, int x1, int y1, uint32_t color) {
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (true) {
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) buf[y0 * W + x0] = color;
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void draw_wireframe(std::vector<uint32_t>& buf, int W, int H, const Scene& scene) {
    std::fill(buf.begin(), buf.end(), 0xFF2E2E2E); // Серый фон
    if (scene.mesh.tris.empty()) return;

    float tanH = std::tan(scene.fov * 0.5f);
    float aspect = (float)W / H;

    auto proj = [&](const Vec3& p, float& x, float& y) -> bool {
        Vec3 d = p - scene.eye;
        float z = d.dot(scene.forward);
        if (z <= 0.1f) return false;
        float sx = d.dot(scene.right) / (z * tanH * aspect);
        float sy = d.dot(scene.up) / (z * tanH);
        x = (sx * 0.5f + 0.5f) * W;
        y = (1.0f - (sy * 0.5f + 0.5f)) * H;
        return true;
    };

    uint32_t line_color = 0xFFDDDDDD;
    for (const auto& t : scene.mesh.tris) {
        float x0, y0, x1, y1, x2, y2;
        if (proj(t.v0, x0, y0) && proj(t.v1, x1, y1) && proj(t.v2, x2, y2)) {
            draw_line(buf, W, H, (int)x0, (int)y0, (int)x1, (int)y1, line_color);
            draw_line(buf, W, H, (int)x1, (int)y1, (int)x2, (int)y2, line_color);
            draw_line(buf, W, H, (int)x2, (int)y2, (int)x0, (int)y0, line_color);
        }
    }
}

// ==================== MAIN ====================
int main(int argc, char** argv) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) return 1;
    const int W = 2000, H = 1200;
    SDL_Window* win = SDL_CreateWindow("Sculpting 3D - C++ port",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H, SDL_WINDOW_SHOWN);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
    SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING, W, H);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplSDL2_InitForSDLRenderer(win, ren);
    ImGui_ImplSDLRenderer2_Init(ren);

    // ---- Сцена ----
    Scene scene;
    scene.W = W; scene.H = H;
    scene.mesh = init_cube();
    scene.subdivision_level = 2; // 2 уровня сразу для гладкости
    for (int i = 0; i < scene.subdivision_level; ++i) {
        auto r = catmull_clark(scene.mesh);
        scene.mesh.verts = r.verts;
        scene.mesh.faces = r.faces;
        scene.mesh.face_uvs = r.face_uvs;
    }

    RenderCtx rctx(W, H, 1);
    UIState ui;
    FileBrowserDialog fb;
    int num_threads = std::max(2u, std::thread::hardware_concurrency());

    auto rebuild = [&]() {
        rebuild_triangles(scene.mesh, scene.dp,
            scene.has_proc ? &scene.proc_tex : nullptr,
            scene.has_proc2 ? &scene.proc_tex2 : nullptr,
            scene.mat);
        scene.bvh.build(scene.mesh.tris);
        rctx.needs_reset = true;
    };
    rebuild();

    auto compute_bump_from_raw = [&]() {
        if (scene.proc_tex_raw.w == 0) return;
        if (scene.bump_highpass) {
            scene.proc_tex_bump = make_highpass(scene.proc_tex_raw, 4);
        } else {
            scene.proc_tex_bump = scene.proc_tex_raw;
        }
        scene.has_proc_bump = true;
        rctx.needs_reset = true;
    };
    auto compute_bump_from_raw2 = [&]() {
        if (scene.proc_tex2_raw.w == 0) return;
        if (scene.bump_highpass2) {
            scene.proc_tex2_bump = make_highpass(scene.proc_tex2_raw, 4);
        } else {
            scene.proc_tex2_bump = scene.proc_tex2_raw;
        }
        scene.has_proc2_bump = true;
        rctx.needs_reset = true;
    };
    auto load_proc_tex2 = [&](const std::string& path) {
        Image img;
        if (img.load(path)) {
            scene.proc_tex2_raw = img;
            Image disp_img = img;
            prepare_displacement_image(disp_img);
            scene.proc_tex2 = disp_img;
            scene.has_proc2 = true;
            scene.dp.enabled2 = true;
            compute_bump_from_raw2();
            rebuild();
            rctx.needs_reset = true;
        }
    };
    auto load_proc_tex = [&](const std::string& path) {
        Image img;
        if (img.load(path)) {
            scene.proc_tex_raw = img;
            Image disp_img = img;
            prepare_displacement_image(disp_img);
            scene.proc_tex = disp_img;
            scene.has_proc = true;
            compute_bump_from_raw();
            rctx.needs_reset = true;
        }
    };
    auto load_uv_tex = [&](const std::string& path) {
        Image img;
        if (img.load(path)) {
            scene.uv_tex = img;
            scene.has_uv = true;
            rctx.needs_reset = true;
        }
    };
    auto rebuild_all = [&]() {
        // Полная пересборка с пересчётом subdivision
        scene.mesh = init_cube();
        for (int i = 0; i < scene.subdivision_level; ++i) {
            auto r = catmull_clark(scene.mesh);
            scene.mesh.verts = r.verts;
            scene.mesh.faces = r.faces;
            scene.mesh.face_uvs = r.face_uvs;
        }
        rebuild();
    };

    bool running = true;
    std::map<SDL_FingerID, ImVec2> touches;
    ImVec2 last_mid(0,0);
    float last_pinch = 0;
    bool is_panning_mouse = false;
    auto last_input = std::chrono::steady_clock::now();
    int last_render_ms = 0;
    while (running) {
        SDL_Event e;
        bool event_input = false;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_MOUSEBUTTONDOWN || e.type == SDL_MOUSEBUTTONUP ||
                e.type == SDL_MOUSEWHEEL || e.type == SDL_KEYDOWN) {
                event_input = true;
            }
            if (e.type == SDL_MOUSEMOTION && (e.motion.state & SDL_BUTTON_LMASK)) {
                event_input = true;
            }
            if (e.type == SDL_FINGERDOWN) {
                touches[e.tfinger.fingerId] = ImVec2(e.tfinger.x * W, e.tfinger.y * H);
                if (touches.size() == 2) {
                    auto it = touches.begin();
                    ImVec2 p1 = it->second; ++it; ImVec2 p2 = it->second;
                    last_mid = ImVec2((p1.x+p2.x)*0.5f, (p1.y+p2.y)*0.5f);
                    last_pinch = std::sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
                }
                event_input = true;
            }
            if (e.type == SDL_FINGERUP) {
                touches.erase(e.tfinger.fingerId);
                if (touches.size() < 2) last_pinch = 0;
                event_input = true;
            }
            if (e.type == SDL_FINGERMOTION && !io.WantCaptureMouse) {
                ImVec2 pos(e.tfinger.x * W, e.tfinger.y * H);
                if (touches.find(e.tfinger.fingerId) != touches.end()) {
                    ImVec2 prev = touches[e.tfinger.fingerId];
                    ImVec2 delta(pos.x - prev.x, pos.y - prev.y);
                    touches[e.tfinger.fingerId] = pos;
                    
                    if (touches.size() == 1) {
                        ui.yaw -= delta.x * 0.003f;
                        ui.pitch += delta.y * 0.003f;
                        ui.pitch = std::clamp(ui.pitch, -1.55f, 1.55f);
                        rctx.needs_reset = true;
                    } else if (touches.size() == 2) {
                        auto it = touches.begin();
                        ImVec2 p1 = it->second; ++it; ImVec2 p2 = it->second;
                        ImVec2 mid((p1.x+p2.x)*0.5f, (p1.y+p2.y)*0.5f);
                        float pinch = std::sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
                        if (last_pinch > 0) {
                            float zoom_delta = last_pinch - pinch;
                            ui.dist *= (1.0f - zoom_delta * 0.01f);
                            ui.dist = std::clamp(ui.dist, 0.5f, 50.0f);
                            
                            ImVec2 pan_delta(mid.x - last_mid.x, mid.y - last_mid.y);
                            scene.target -= scene.right * (pan_delta.x * ui.dist * 0.002f);
                            scene.target += scene.up * (pan_delta.y * ui.dist * 0.002f);
                            rctx.needs_reset = true;
                        }
                        last_mid = mid;
                        last_pinch = pinch;
                    }
                }
                event_input = true;
            }
            if (e.type == SDL_MOUSEBUTTONDOWN && !io.WantCaptureMouse) {
                if (e.button.button == SDL_BUTTON_RIGHT || e.button.button == SDL_BUTTON_MIDDLE) is_panning_mouse = true;
            }
            if (e.type == SDL_MOUSEBUTTONUP) {
                if (e.button.button == SDL_BUTTON_RIGHT || e.button.button == SDL_BUTTON_MIDDLE) is_panning_mouse = false;
            }
            if (e.type == SDL_MOUSEMOTION && !io.WantCaptureMouse) {
                if (is_panning_mouse) {
                    scene.target -= scene.right * (e.motion.xrel * ui.dist * 0.002f);
                    scene.target += scene.up * (e.motion.yrel * ui.dist * 0.002f);
                    rctx.needs_reset = true;
                    event_input = true;
                } else if (e.motion.state & SDL_BUTTON_LMASK) {
                    ui.yaw += e.motion.xrel * 0.003f;
                    ui.pitch += e.motion.yrel * 0.003f;
                    ui.pitch = std::clamp(ui.pitch, -1.55f, 1.55f);
                    rctx.needs_reset = true;
                    event_input = true;
                }
            }
            if (e.type == SDL_MOUSEWHEEL && !io.WantCaptureMouse) {
                ui.dist *= (1.0f - e.wheel.y * 0.1f);
                ui.dist = std::clamp(ui.dist, 1.5f, 15.0f);
                rctx.needs_reset = true;
            }
        }

        // Позиция камеры (орбитальная)
        scene.eye = scene.target + Vec3(std::cos(ui.pitch)*std::sin(ui.yaw), std::sin(ui.pitch), std::cos(ui.pitch)*std::cos(ui.yaw)) * ui.dist;
        Vec3 up_ref(0,1,0);
        if (std::abs(ui.pitch) > 1.5f) up_ref = Vec3(0,0,1); // Избегаем коллинеарности при взгляде в полюс
        scene.forward = (scene.target - scene.eye).normalize();
        scene.right = scene.forward.cross(up_ref).normalize();
        scene.up = scene.right.cross(scene.forward);

        // ===== АДАПТИВНЫЙ РЕНДЕР =====
        if (event_input) last_input = std::chrono::steady_clock::now();
        auto now_t = std::chrono::steady_clock::now();
        auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_t - last_input).count();
        bool user_active = (idle_ms < 200);

        int target_samples;
        bool want_hq = scene.high_quality;
        if (want_hq && !user_active) {
            // HQ режим: полное разрешение, 32 сэмпла, тени + GI
            scene.interactive = false;
            rctx.set_scale(1);
            target_samples = 32;
        } else if (user_active) {
            // Взаимодействие: scale=3, 1 сэмпл, без теней
            scene.interactive = true;
            rctx.set_scale(3);
            target_samples = 1;
        } else {
            // Простой без HQ: scale=2, 6 сэмплов
            scene.interactive = false;
            rctx.set_scale(2);
            target_samples = 6;
        }

        if (!scene.high_quality) {
            // Быстрый каркас (60 FPS)
            draw_wireframe(rctx.pixel_buf, W, H, scene);
            SDL_UpdateTexture(tex, nullptr, rctx.pixel_buf.data(), W * sizeof(uint32_t));
            last_render_ms = 0;
        } else {
            // Тяжёлая трассировка (Path Tracing)
            bool need_render = (rctx.sample_count < target_samples) || rctx.needs_reset;
            if (need_render) {
                auto rt0 = std::chrono::steady_clock::now();
                rctx.render_frame(scene, num_threads);
                last_render_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - rt0).count();
                SDL_UpdateTexture(tex, nullptr, rctx.pixel_buf.data(), W * sizeof(uint32_t));
            }
        }

        // UI
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(W - 340, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340, H), ImGuiCond_FirstUseEver);
        ImGui::Begin("Sculpting 3D - C++");

        ImGui::Text("Subdivision level: %d", scene.subdivision_level);
        if (ImGui::Button("+ Subdivide")) {
            if (scene.subdivision_level < 6) {
                scene.subdivision_level++;
                auto r = catmull_clark(scene.mesh);
                scene.mesh.verts = r.verts;
                scene.mesh.faces = r.faces;
                scene.mesh.face_uvs = r.face_uvs;
                rebuild();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("- Subdivide")) {
            if (scene.subdivision_level > 0) {
                scene.subdivision_level--;
                rebuild_all();
            }
        }

        ImGui::Separator();

        if (ImGui::CollapsingHeader("Textures", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Procedural (bump / displace):");
            ImGui::InputText("##proc", ui.proc_path, sizeof(ui.proc_path));
            ImGui::SameLine();
            if (ImGui::Button("Browse...##proc")) fb.open("Select procedural texture", {"png","jpg","jpeg","bmp","tga","webp"}, "proc");
            if (ImGui::Button("Load procedural")) { load_proc_tex(ui.proc_path); rebuild(); }
            ImGui::Separator();
            ImGui::Text("Procedural #2 (overlay / detail):");
            ImGui::InputText("##proc2", ui.proc2_path, sizeof(ui.proc2_path));
            ImGui::SameLine();
            if (ImGui::Button("Browse...##proc2")) fb.open("Select texture #2", {"png","jpg","jpeg","bmp","tga","webp"}, "proc2");
            if (ImGui::Button("Load procedural #2")) { load_proc_tex2(ui.proc2_path); }
            ImGui::Separator();
            ImGui::Text("UV albedo:");
            ImGui::InputText("##uv", ui.uv_path, sizeof(ui.uv_path));
            ImGui::SameLine();
            if (ImGui::Button("Browse...##uv")) fb.open("Select UV texture", {"png","jpg","jpeg","bmp","tga","webp"}, "uv");
            if (ImGui::Button("Load UV")) { load_uv_tex(ui.uv_path); rctx.needs_reset = true; }
            ImGui::Separator();
            ImGui::Text("Import Geometry:");
            ImGui::InputText("##obj", ui.obj_path, sizeof(ui.obj_path));
            ImGui::SameLine();
            if (ImGui::Button("Browse...##obj")) fb.open("Import OBJ", {"obj"}, "obj");
            if (ImGui::Button("Import OBJ")) {
                scene.mesh = parse_obj(ui.obj_path);
                if (!scene.mesh.verts.empty()) {
                    scene.subdivision_level = 0;
                    rebuild();
                }
            }
        }


        if (ImGui::CollapsingHeader("Displacement", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Enable displacement", &scene.dp.enabled)) rebuild();
            float v;
            v = scene.dp.strength;
            if (ImGui::SliderFloat("Strength", &v, 0.0f, 1.0f)) { scene.dp.strength = v; rebuild(); }
            v = scene.dp.uv_scale;
            if (ImGui::SliderFloat("UV Scale", &v, 0.1f, 10.0f)) { scene.dp.uv_scale = v; rebuild(); rctx.needs_reset = true; }
            v = scene.dp.midlevel;
            if (ImGui::SliderFloat("Midlevel", &v, 0.0f, 1.0f)) { scene.dp.midlevel = v; rebuild(); }

            const char* dirs[] = {"Normal","X","Y","Z"};
            if (ImGui::Combo("Direction", &scene.dp.direction, dirs, 4)) rebuild();

            v = scene.dp.empty_pos.x;
            if (ImGui::SliderFloat("Empty X", &v, -2, 2)) { scene.dp.empty_pos.x = v; rebuild(); }
            v = scene.dp.empty_pos.y;
            if (ImGui::SliderFloat("Empty Y", &v, -2, 2)) { scene.dp.empty_pos.y = v; rebuild(); }
            v = scene.dp.empty_pos.z;
            if (ImGui::SliderFloat("Empty Z", &v, -2, 2)) { scene.dp.empty_pos.z = v; rebuild(); }
            ImGui::Separator();
            if (ImGui::Checkbox("Invert (black outward)", &scene.dp.invert)) rebuild();
            v = scene.dp.threshold;
            if (ImGui::SliderFloat("Threshold", &v, 0.0f, 0.3f)) { scene.dp.threshold = v; rebuild(); }
            v = scene.dp.sharpness;
            if (ImGui::SliderFloat("Blend Sharpness", &v, 1.0f, 16.0f)) { scene.dp.sharpness = v; rebuild(); }
            
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.9f,0.7f,0.4f,1), "Texture #2 (overlay)");
            if (ImGui::Checkbox("Enable 2nd texture", &scene.dp.enabled2)) rebuild();
            v = scene.dp.alpha2;
            if (ImGui::SliderFloat("Alpha #2", &v, 0.0f, 1.0f)) { scene.dp.alpha2 = v; rebuild(); rctx.needs_reset = true; }
            v = scene.dp.strength2;
            if (ImGui::SliderFloat("Strength #2", &v, 0.0f, 1.0f)) { scene.dp.strength2 = v; rebuild(); }
            v = scene.dp.midlevel2;
            if (ImGui::SliderFloat("Midlevel #2", &v, 0.0f, 1.0f)) { scene.dp.midlevel2 = v; rebuild(); }
            if (ImGui::Checkbox("Invert #2", &scene.dp.invert2)) rebuild();
            v = scene.dp.threshold2;
            if (ImGui::SliderFloat("Threshold #2", &v, 0.0f, 0.3f)) { scene.dp.threshold2 = v; rebuild(); }
        }

        if (ImGui::CollapsingHeader("Rotation / Mirror", ImGuiTreeNodeFlags_DefaultOpen)) {
            float v;
            v = scene.dp.rotate_x;
            if (ImGui::SliderFloat("Rotate X", &v, -180, 180)) { scene.dp.rotate_x = v; rebuild(); }
            if (ImGui::Checkbox("Mirror X", &scene.dp.mirror_x)) rebuild();
            v = scene.dp.rotate_y;
            if (ImGui::SliderFloat("Rotate Y", &v, -180, 180)) { scene.dp.rotate_y = v; rebuild(); }
            if (ImGui::Checkbox("Mirror Y", &scene.dp.mirror_y)) rebuild();
            v = scene.dp.rotate_z;
            if (ImGui::SliderFloat("Rotate Z", &v, -180, 180)) { scene.dp.rotate_z = v; rebuild(); }
            if (ImGui::Checkbox("Mirror Z", &scene.dp.mirror_z)) rebuild();
        }

        if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
            float col[3] = {scene.mat.albedo_color.x, scene.mat.albedo_color.y, scene.mat.albedo_color.z};
            if (ImGui::ColorEdit3("Color", col)) {
                scene.mat.albedo_color = Vec3(col[0], col[1], col[2]);
                rctx.needs_reset = true;
            }
            float v;
            v = scene.mat.metallic;
            if (ImGui::SliderFloat("Metallic", &v, 0.0f, 1.0f)) { scene.mat.metallic = v; rctx.needs_reset = true; }
            v = scene.mat.roughness;
            if (ImGui::SliderFloat("Roughness", &v, 0.0f, 1.0f)) { scene.mat.roughness = v; rctx.needs_reset = true; }
            v = scene.mat.specular;
            if (ImGui::SliderFloat("Specular", &v, 0.0f, 1.0f)) { scene.mat.specular = v; rctx.needs_reset = true; }
            v = scene.mat.bump_strength;
            if (ImGui::SliderFloat("Bump Strength", &v, 0.0f, 3.0f)) { scene.mat.bump_strength = v; rctx.needs_reset = true; }
            if (ImGui::Checkbox("Bump: Line Extraction (high-pass) #1", &scene.bump_highpass)) {
                compute_bump_from_raw();
            }
            v = scene.bump_strength2;
            if (ImGui::SliderFloat("Bump Strength #2", &v, 0.0f, 3.0f)) { scene.bump_strength2 = v; rctx.needs_reset = true; }
            if (ImGui::Checkbox("Bump: Line Extraction (high-pass) #2", &scene.bump_highpass2)) {
                compute_bump_from_raw2();
            }
            v = scene.mat.sss_strength;
            if (ImGui::SliderFloat("SSS", &v, 0.0f, 1.0f)) { scene.mat.sss_strength = v; rctx.needs_reset = true; }
            v = scene.mat.procedural_alpha;
            if (ImGui::SliderFloat("Proc Alpha", &v, 0.0f, 1.0f)) { scene.mat.procedural_alpha = v; rctx.needs_reset = true; }
            v = scene.mat.procedural_overlay;
            if (ImGui::SliderFloat("Proc Overlay", &v, 0.0f, 1.0f)) { scene.mat.procedural_overlay = v; rctx.needs_reset = true; }
        }

        if (ImGui::CollapsingHeader("Camera")) {
            ImGui::SliderFloat("Yaw", &ui.yaw, -3.14f, 3.14f);
            ImGui::SliderFloat("Pitch", &ui.pitch, -1.4f, 1.4f);
            ImGui::SliderFloat("Distance", &ui.dist, 1.5f, 15.0f);
        }

        ImGui::Separator();
        int target_ui = scene.interactive ? 1 : (scene.high_quality ? 32 : 6);
        ImGui::Text("Samples: %d/%d  Scale: %dx  HQ: %s",
            rctx.sample_count,
            target_ui,
            rctx.scale,
            scene.high_quality ? "ON" : "off");
        ImGui::Text("Mode: %s  Render: %d ms  FPS: %.1f",
            scene.interactive ? "Interactive" : "Idle",
            last_render_ms, io.Framerate);
        ImGui::Text("Tris: %zu", scene.mesh.tris.size());
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1),
            "ЛКМ-драг: вращение\nКолесо: зум");

        ImGui::End();

        // --- Плавающая кнопка Render HQ (левый нижний угол) ---
        {
            ImGui::SetNextWindowPos(ImVec2(20, H - 110), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.85f);
            ImGui::Begin("##render_btn_window", nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing);

            if (scene.high_quality) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.15f, 0.15f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.25f, 0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.10f, 0.10f, 1.0f));
                if (ImGui::Button("STOP RENDER", ImVec2(150, 52))) {
                    scene.high_quality = false;
                    rctx.needs_reset = true;
                }
                ImGui::PopStyleColor(3);
                ImGui::TextDisabled("HQ: %d / 32 samples", rctx.sample_count);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.55f, 0.20f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.70f, 0.30f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.10f, 0.45f, 0.15f, 1.0f));
                if (ImGui::Button("RENDER HQ", ImVec2(150, 52))) {
                    scene.high_quality = true;
                    rctx.needs_reset = true;
                    // Принудительно "отпускаем" пользователя, чтобы сразу войти в HQ
                    last_input = std::chrono::steady_clock::now() - std::chrono::seconds(1);
                }
                ImGui::PopStyleColor(3);
                ImGui::TextDisabled("Wireframe preview (click to trace)");
            }
            ImGui::End();
        }

        // --- Файловый диалог ---
        std::string fb_result = fb.draw();
        if (fb_result.rfind("SELECT:", 0) == 0) {
            std::string path = fb_result.substr(7);
            if (fb.callback_tag == "proc") {
                strncpy(ui.proc_path, path.c_str(), sizeof(ui.proc_path)-1);
                ui.proc_path[sizeof(ui.proc_path)-1] = 0;
                load_proc_tex(path);
                rebuild();
            } else if (fb.callback_tag == "proc2") {
                strncpy(ui.proc2_path, path.c_str(), sizeof(ui.proc2_path)-1);
                ui.proc2_path[sizeof(ui.proc2_path)-1] = 0;
                load_proc_tex2(path);
            } else if (fb.callback_tag == "obj") {
                strncpy(ui.obj_path, path.c_str(), sizeof(ui.obj_path)-1);
                ui.obj_path[sizeof(ui.obj_path)-1] = 0;
                scene.mesh = parse_obj(path);
                if (!scene.mesh.verts.empty()) {
                    scene.subdivision_level = 0;
                    rebuild();
                }
            } else if (fb.callback_tag == "uv") {
                strncpy(ui.uv_path, path.c_str(), sizeof(ui.uv_path)-1);
                ui.uv_path[sizeof(ui.uv_path)-1] = 0;
                load_uv_tex(path);
                rctx.needs_reset = true;
            }
        }

        ImGui::Render();
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, nullptr, nullptr);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), ren);
        SDL_RenderPresent(ren);
    }

    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
