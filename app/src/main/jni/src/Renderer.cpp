#include "Renderer.h"
#include <cmath>

Vec2 rotate_uv(Vec2 uv, float deg) {
    if (std::abs(deg) < 0.0001f) return uv;
    float cu = uv.x - 0.5f, cv = uv.y - 0.5f;
    float ca = std::cos(deg*PI/180.f), sa = std::sin(deg*PI/180.f);
    return Vec2(std::fmod(cu*ca - cv*sa + 0.5f + 10.0f, 1.0f),
                std::fmod(cu*sa + cv*ca + 0.5f + 10.0f, 1.0f));
}


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



