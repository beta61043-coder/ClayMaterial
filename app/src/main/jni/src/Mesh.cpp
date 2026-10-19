#include "Mesh.h"
#include <fstream>
#include <sstream>
#include <algorithm>

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

