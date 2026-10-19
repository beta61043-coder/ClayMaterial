#pragma once
#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "Math.h"
#include "Mesh.h"

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



Vec2 rotate_uv(Vec2 uv, float angle);
Vec3 sample_triplanar(Vec3 p, Vec3 n, const Image& tex, float inv_scale, Vec3 off3, const DisplaceParams& dp);
float triplanar_height(Vec3 p, Vec3 n, const Image& tex, float inv_scale, Vec3 off3, const DisplaceParams& dp);
Vec3 bump_normal_pair(const Scene& s, Vec3 p, Vec3 n, const Vec3& N_weights, const Image* t1, const Image* t2, const DisplaceParams& dp);
Vec3 shade_hit(const Scene& s, const BVH::Hit& hit, Vec3 view_dir, std::mt19937& gen);

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
void draw_line(std::vector<uint32_t>& buf, int W, int H, int x0, int y0, int x1, int y1, uint32_t color);
void draw_wireframe(std::vector<uint32_t>& buf, int W, int H, const Scene& scene);
