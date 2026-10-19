#include "Renderer.h"

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

