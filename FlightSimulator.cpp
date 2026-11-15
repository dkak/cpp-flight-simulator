// ============================================================================
//  C++ Flight Simulator  (v2)
//  Controls:  Mouse = pitch / heading   W / S = throttle / brake
//             Q / E = bank left / right  C = cockpit/chase view
//             R = reset to runway        ESC = quit
// ============================================================================
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

using glm::vec2; using glm::vec3; using glm::vec4; using glm::mat3; using glm::mat4;

// ----------------------------------------------------------------------------
//  Constants
// ----------------------------------------------------------------------------
constexpr float PI = 3.14159265f;
constexpr float GROUND_Y = -0.5f;     // flat ground level (runway area)
constexpr float PLANE_CLEARANCE = 0.20f;     // plane centre above ground when wheels touch
constexpr float PLANE_SCALE = 0.55f;
constexpr float MAX_SPEED = 15.0f;
constexpr float STALL_SPEED = 5.0f;
constexpr float TAKEOFF_SPEED = 6.0f;
constexpr float RUNWAY_HALF_W = 4.0f;
constexpr float RUNWAY_START_Z = 8.0f;
constexpr float RUNWAY_END_Z = -240.0f;
constexpr float WORLD_MIN_X = -600.0f, WORLD_MAX_X = 600.0f;
constexpr float WORLD_MIN_Z = -900.0f, WORLD_MAX_Z = 300.0f;
constexpr float FOG_DENSITY = 0.0026f;

const vec3 SUN_DIR = glm::normalize(vec3(0.6f, 0.5f, -0.3f));
const vec3 HORIZON_SRGB(0.72f, 0.84f, 0.95f);

// ----------------------------------------------------------------------------
//  Game state
// ----------------------------------------------------------------------------
struct Plane {
    vec3  pos{ 0.0f, GROUND_Y + PLANE_CLEARANCE, 3.0f };
    float yaw = -90.0f, pitch = 0.0f, roll = 0.0f;   // roll > 0 = right bank
    float speed = 0.0f, vSpeed = 0.0f;
    bool  onGround = true, stalled = false, crashed = false;
};

Plane plane;
int   winW = 1280, winH = 720;
bool  cockpitView = false;
float crashTimer = 0.0f;
float deltaTime = 0.0f, lastFrame = 0.0f;
int   throttleInput = 0;   // -1 brake, 0 none, +1 throttle
int   rollInput = 0;       // -1 left, +1 right
bool  firstMouse = true;
double lastX = 0.0, lastY = 0.0;

struct AABB { vec3 mn, mx; };
static const std::vector<AABB> obstacles = {
    { vec3(-22.0f, -0.5f, -40.0f), vec3(-15.0f, 4.0f, -28.0f) },   // hangar
    { vec3(8.0f, -0.5f, -21.0f), vec3(10.0f, 6.0f, -19.0f) },   // tower shaft
    { vec3(7.2f,  6.0f, -21.8f), vec3(10.8f, 7.6f, -18.2f) },   // control cab
};

// ----------------------------------------------------------------------------
//  Procedural terrain (value-noise fbm), flattened around the runway
// ----------------------------------------------------------------------------
static float hash2(int x, int z) {
    unsigned h = (unsigned)x * 374761393u + (unsigned)z * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (h & 0xFFFFFFu) / float(0xFFFFFF);
}
static float valueNoise(float x, float z) {
    int xi = (int)std::floor(x), zi = (int)std::floor(z);
    float fx = x - xi, fz = z - zi;
    float u = fx * fx * (3.0f - 2.0f * fx), v = fz * fz * (3.0f - 2.0f * fz);
    float a = hash2(xi, zi), b = hash2(xi + 1, zi), c = hash2(xi, zi + 1), d = hash2(xi + 1, zi + 1);
    return glm::mix(glm::mix(a, b, u), glm::mix(c, d, u), v);
}
static float fbm(float x, float z) {
    float sum = 0.0f, amp = 0.5f;
    for (int i = 0; i < 5; i++) { sum += amp * valueNoise(x, z); x *= 2.03f; z *= 2.03f; amp *= 0.5f; }
    return sum;
}
static float distanceToRunway(float x, float z) {
    float cz = glm::clamp(z, RUNWAY_END_Z - 40.0f, RUNWAY_START_Z + 40.0f);
    return std::sqrt(x * x + (z - cz) * (z - cz));
}
static float terrainHeight(float x, float z) {
    float n = fbm(x * 0.0035f + 10.0f, z * 0.0035f - 5.0f);
    float h = std::max(n - 0.38f, 0.0f) / 0.62f;
    float hills = h * h * 120.0f;
    float rolling = (fbm(x * 0.02f, z * 0.02f) - 0.5f) * 3.0f;
    float flat = glm::smoothstep(40.0f, 220.0f, distanceToRunway(x, z));
    return GROUND_Y + (hills + rolling) * flat;
}
static bool isOnRunway(const vec3& p) {
    return std::fabs(p.x) < RUNWAY_HALF_W + 1.0f && p.z < RUNWAY_START_Z && p.z > RUNWAY_END_Z;
}

// ----------------------------------------------------------------------------
//  Mesh building helpers.  Vertex = pos(3) normal(3) color(3).
//  A zero normal means "unlit / emissive".
// ----------------------------------------------------------------------------
using Verts = std::vector<float>;

struct Mesh { GLuint vao = 0, vbo = 0, ebo = 0; GLsizei count = 0; bool indexed = false; };

static void pushVert(Verts& v, const vec3& p, const vec3& n, const vec3& c) {
    v.insert(v.end(), { p.x, p.y, p.z, n.x, n.y, n.z, c.x, c.y, c.z });
}
static void addTri(Verts& v, vec3 a, vec3 b, vec3 c, vec3 col, bool unlit = false) {
    vec3 n(0.0f);
    if (!unlit) {
        vec3 cr = glm::cross(b - a, c - a);
        float len = glm::length(cr);
        n = len > 1e-8f ? cr / len : vec3(0, 1, 0);
    }
    pushVert(v, a, n, col); pushVert(v, b, n, col); pushVert(v, c, n, col);
}
static void addQuad(Verts& v, vec3 a, vec3 b, vec3 c, vec3 d, vec3 col, bool unlit = false) {
    addTri(v, a, b, c, col, unlit);
    addTri(v, a, c, d, col, unlit);
}
static void addBox(Verts& v, vec3 mn, vec3 mx, vec3 col, bool unlit = false) {
    vec3 p000(mn.x, mn.y, mn.z), p100(mx.x, mn.y, mn.z), p110(mx.x, mx.y, mn.z), p010(mn.x, mx.y, mn.z);
    vec3 p001(mn.x, mn.y, mx.z), p101(mx.x, mn.y, mx.z), p111(mx.x, mx.y, mx.z), p011(mn.x, mx.y, mx.z);
    addQuad(v, p001, p101, p111, p011, col, unlit);
    addQuad(v, p100, p000, p010, p110, col, unlit);
    addQuad(v, p000, p001, p011, p010, col, unlit);
    addQuad(v, p101, p100, p110, p111, col, unlit);
    addQuad(v, p010, p011, p111, p110, col, unlit);
    addQuad(v, p000, p100, p101, p001, col, unlit);
}
static void addEllipsoid(Verts& v, vec3 c, vec3 r, vec3 col, int rings = 5, int segs = 8, bool unlit = false) {
    auto local = [&](int ri, int si) {
        float phi = PI * ri / rings, th = 2.0f * PI * si / segs;
        return vec3(r.x * std::sin(phi) * std::cos(th), r.y * std::cos(phi), r.z * std::sin(phi) * std::sin(th));
        };
    auto emit = [&](vec3 l) {
        vec3 n = unlit ? vec3(0.0f) : glm::normalize(l / (r * r));
        pushVert(v, c + l, n, col);
        };
    for (int ri = 0; ri < rings; ri++)
        for (int si = 0; si < segs; si++) {
            vec3 a = local(ri, si), b = local(ri, si + 1), cc = local(ri + 1, si + 1), d = local(ri + 1, si);
            emit(a); emit(b); emit(cc);
            emit(a); emit(cc); emit(d);
        }
}
static void addCone(Verts& v, vec3 base, float radius, float height, int segs, vec3 col) {
    vec3 apex = base + vec3(0, height, 0);
    for (int i = 0; i < segs; i++) {
        float a0 = 2.0f * PI * i / segs, a1 = 2.0f * PI * (i + 1) / segs;
        vec3 p0 = base + vec3(radius * std::cos(a0), 0, radius * std::sin(a0));
        vec3 p1 = base + vec3(radius * std::cos(a1), 0, radius * std::sin(a1));
        addTri(v, apex, p1, p0, col);
    }
}
// Thin tapered slab (wings, stabilisers, fin). th = half-thickness vector.
static void addSlab(Verts& v, vec3 rLE, vec3 rTE, vec3 tLE, vec3 tTE, vec3 th, vec3 top, vec3 bottom, vec3 edge) {
    addQuad(v, rLE + th, tLE + th, tTE + th, rTE + th, top);
    addQuad(v, rLE - th, rTE - th, tTE - th, tLE - th, bottom);
    addQuad(v, rLE - th, tLE - th, tLE + th, rLE + th, edge);
    addQuad(v, rTE - th, rTE + th, tTE + th, tTE - th, edge);
    addQuad(v, tLE - th, tTE - th, tTE + th, tLE + th, edge);
}

static Mesh upload(const Verts& v, const std::vector<unsigned>* idx = nullptr) {
    Mesh m;
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STATIC_DRAW);
    const GLsizei stride = 9 * sizeof(float);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    if (idx) {
        glGenBuffers(1, &m.ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx->size() * sizeof(unsigned), idx->data(), GL_STATIC_DRAW);
        m.indexed = true;
        m.count = (GLsizei)idx->size();
    }
    else {
        m.count = (GLsizei)(v.size() / 9);
    }
    glBindVertexArray(0);
    return m;
}
static void drawMesh(const Mesh& m) {
    glBindVertexArray(m.vao);
    if (m.indexed) glDrawElements(GL_TRIANGLES, m.count, GL_UNSIGNED_INT, 0);
    else           glDrawArrays(GL_TRIANGLES, 0, m.count);
}
static void destroyMesh(Mesh& m) {
    glDeleteVertexArrays(1, &m.vao);
    glDeleteBuffers(1, &m.vbo);
    if (m.ebo) glDeleteBuffers(1, &m.ebo);
}

struct Rng {
    unsigned s;
    explicit Rng(unsigned seed) : s(seed) {}
    float next() { s = s * 1664525u + 1013904223u; return (s >> 8) / 16777216.0f; }
};

// ----------------------------------------------------------------------------
//  World meshes
// ----------------------------------------------------------------------------
static vec3 terrainColor(float x, float z, float h, float ny) {
    float alt = h - GROUND_Y;
    vec3 grass(0.22f, 0.47f, 0.13f), forest(0.11f, 0.31f, 0.10f), dry(0.46f, 0.43f, 0.22f);
    vec3 rock(0.43f, 0.40f, 0.37f), snow(0.95f, 0.96f, 1.0f);
    vec3 c = glm::mix(grass, forest, glm::smoothstep(4.0f, 25.0f, alt));
    float dryAmt = glm::smoothstep(0.55f, 0.75f, valueNoise(x * 0.01f + 40.0f, z * 0.01f)) * 0.6f;
    c = glm::mix(c, dry, dryAmt * (1.0f - glm::smoothstep(10.0f, 30.0f, alt)));
    float rockAmt = std::max(1.0f - glm::smoothstep(0.65f, 0.82f, ny), glm::smoothstep(30.0f, 45.0f, alt));
    c = glm::mix(c, rock, rockAmt);
    c = glm::mix(c, snow, glm::smoothstep(48.0f, 60.0f, alt) * glm::smoothstep(0.55f, 0.8f, ny));
    float variation = (valueNoise(x * 0.15f, z * 0.15f) - 0.5f) * 0.14f;
    return glm::clamp(c * (1.0f + variation), 0.0f, 1.0f);
}

static Mesh buildTerrain() {
    const int N = 200;
    const float cell = (WORLD_MAX_X - WORLD_MIN_X) / N;       // 6 units
    const int NZ = (int)((WORLD_MAX_Z - WORLD_MIN_Z) / cell);
    Verts v; v.reserve((N + 1) * (NZ + 1) * 9);
    const float e = 3.0f;
    for (int iz = 0; iz <= NZ; iz++)
        for (int ix = 0; ix <= N; ix++) {
            float x = WORLD_MIN_X + ix * cell, z = WORLD_MIN_Z + iz * cell;
            float h = terrainHeight(x, z);
            float dx = (terrainHeight(x + e, z) - terrainHeight(x - e, z)) / (2 * e);
            float dz = (terrainHeight(x, z + e) - terrainHeight(x, z - e)) / (2 * e);
            vec3 n = glm::normalize(vec3(-dx, 1.0f, -dz));
            pushVert(v, vec3(x, h, z), n, terrainColor(x, z, h, n.y));
        }
    std::vector<unsigned> idx; idx.reserve(N * NZ * 6);
    for (int iz = 0; iz < NZ; iz++)
        for (int ix = 0; ix < N; ix++) {
            unsigned i0 = iz * (N + 1) + ix, i1 = i0 + 1, i2 = i0 + (N + 1), i3 = i2 + 1;
            idx.insert(idx.end(), { i0, i2, i1, i1, i2, i3 });
        }
    return upload(v, &idx);
}

static Mesh buildRunway() {
    Verts v;
    vec3 asphalt(0.20f, 0.20f, 0.22f);
    float y = GROUND_Y + 0.02f;
    addQuad(v, vec3(-RUNWAY_HALF_W, y, RUNWAY_START_Z), vec3(RUNWAY_HALF_W, y, RUNWAY_START_Z),
        vec3(RUNWAY_HALF_W, y, RUNWAY_END_Z), vec3(-RUNWAY_HALF_W, y, RUNWAY_END_Z), asphalt);
    // Taxiway / apron leading to the hangar
    addQuad(v, vec3(-15.0f, y, -31.0f), vec3(-4.0f, y, -31.0f), vec3(-4.0f, y, -37.0f), vec3(-15.0f, y, -37.0f),
        asphalt * 1.15f);
    return upload(v);
}

static Mesh buildMarkings() {
    Verts v;
    vec3 white(0.92f, 0.92f, 0.90f);
    float y = GROUND_Y + 0.04f;
    auto rect = [&](float x0, float x1, float z0, float z1) {
        addQuad(v, vec3(x0, y, z0), vec3(x1, y, z0), vec3(x1, y, z1), vec3(x0, y, z1), white);
        };
    rect(-3.75f, -3.6f, RUNWAY_END_Z, RUNWAY_START_Z);                  // edge lines
    rect(3.6f, 3.75f, RUNWAY_END_Z, RUNWAY_START_Z);
    for (float z = 0.0f; z > RUNWAY_END_Z + 6.0f; z -= 12.0f) rect(-0.12f, 0.12f, z - 6.0f, z); // centre dashes
    for (int i = 0; i < 8; i++) {                                         // threshold bars
        float x = -3.2f + i * 0.8f;
        rect(x, x + 0.4f, 4.0f, 7.5f);
        rect(x, x + 0.4f, RUNWAY_END_Z + 1.0f, RUNWAY_END_Z + 4.5f);
    }
    for (float z : { -40.0f, -70.0f, -100.0f }) {                         // touchdown marks
        rect(-2.6f, -1.6f, z - 3.0f, z);
        rect(1.6f, 2.6f, z - 3.0f, z);
    }
    return upload(v);
}

static Mesh buildRunwayLights() {
    Verts v;
    for (float z = RUNWAY_START_Z; z >= RUNWAY_END_Z; z -= 12.0f)
        for (float s : { -1.0f, 1.0f }) {
            vec3 col(1.0f, 0.9f, 0.6f);
            if (z > -6.0f) col = vec3(0.2f, 1.0f, 0.3f);
            if (z < -228.0f) col = vec3(1.0f, 0.15f, 0.1f);
            addBox(v, vec3(s * 4.4f - 0.07f, GROUND_Y, z - 0.07f), vec3(s * 4.4f + 0.07f, GROUND_Y + 0.12f, z + 0.07f), col, true);
        }
    return upload(v);
}

static Mesh buildStructures() {
    Verts v;
    // Hangar
    vec3 wall(0.74f, 0.76f, 0.80f), roof(0.55f, 0.14f, 0.10f), dark(0.12f, 0.13f, 0.15f);
    addBox(v, vec3(-22.0f, GROUND_Y, -40.0f), vec3(-15.0f, 2.5f, -28.0f), wall);
    addTri(v, vec3(-22.0f, 2.5f, -28.0f), vec3(-15.0f, 2.5f, -28.0f), vec3(-18.5f, 4.0f, -28.0f), wall);
    addTri(v, vec3(-22.0f, 2.5f, -40.0f), vec3(-15.0f, 2.5f, -40.0f), vec3(-18.5f, 4.0f, -40.0f), wall);
    addQuad(v, vec3(-22.3f, 2.5f, -40.3f), vec3(-22.3f, 2.5f, -27.7f), vec3(-18.5f, 4.1f, -27.7f), vec3(-18.5f, 4.1f, -40.3f), roof);
    addQuad(v, vec3(-14.7f, 2.5f, -40.3f), vec3(-14.7f, 2.5f, -27.7f), vec3(-18.5f, 4.1f, -27.7f), vec3(-18.5f, 4.1f, -40.3f), roof);
    addQuad(v, vec3(-14.97f, GROUND_Y, -37.0f), vec3(-14.97f, GROUND_Y, -31.0f), vec3(-14.97f, 2.0f, -31.0f), vec3(-14.97f, 2.0f, -37.0f), dark);

    // Control tower
    vec3 concrete(0.85f, 0.85f, 0.88f), glass(0.18f, 0.38f, 0.52f), cap(0.3f, 0.3f, 0.32f);
    addBox(v, vec3(8.0f, GROUND_Y, -21.0f), vec3(10.0f, 6.0f, -19.0f), concrete);
    addBox(v, vec3(7.4f, 6.0f, -21.6f), vec3(10.6f, 7.4f, -18.4f), glass);
    addBox(v, vec3(7.2f, 7.4f, -21.8f), vec3(10.8f, 7.6f, -18.2f), cap);
    addBox(v, vec3(8.95f, 7.6f, -20.05f), vec3(9.05f, 9.0f, -19.95f), cap);
    return upload(v);
}

static Mesh buildBeacon() {
    Verts v;
    addEllipsoid(v, vec3(9.0f, 9.15f, -20.0f), vec3(0.2f), vec3(1.0f, 0.1f, 0.05f), 4, 8, true);
    return upload(v);
}

static Mesh buildFireball() {
    Verts v;
    addEllipsoid(v, vec3(0.0f), vec3(1.0f), vec3(1.0f, 0.55f, 0.1f), 6, 10, true);
    return upload(v);
}

static Mesh buildForest() {
    Verts v; Rng rng(1337);
    for (int i = 0; i < 6000; i++) {
        float x = -560.0f + rng.next() * 1120.0f, z = -860.0f + rng.next() * 1130.0f;
        float r1 = rng.next(), r2 = rng.next(), r3 = rng.next();
        if (distanceToRunway(x, z) < 30.0f) continue;
        if (valueNoise(x * 0.012f + 3.0f, z * 0.012f + 7.0f) < 0.45f) continue;   // clumped forests
        float h = terrainHeight(x, z);
        if (h - GROUND_Y > 38.0f) continue;
        float dx = (terrainHeight(x + 2, z) - terrainHeight(x - 2, z)) / 4.0f;
        float dz = (terrainHeight(x, z + 2) - terrainHeight(x, z - 2)) / 4.0f;
        if (1.0f / std::sqrt(1.0f + dx * dx + dz * dz) < 0.88f) continue;        // no trees on cliffs
        float H = 2.5f + r1 * 3.0f;
        vec3 base(x, h, z);
        vec3 leaf = r2 > 0.93f ? vec3(0.75f, 0.5f, 0.12f) : vec3(0.07f + 0.07f * r2, 0.26f + 0.16f * r3, 0.08f);
        float t = 0.05f * H;
        addBox(v, base + vec3(-t, 0, -t), base + vec3(t, 0.3f * H, t), vec3(0.30f, 0.20f, 0.10f));
        addCone(v, base + vec3(0, 0.2f * H, 0), 0.32f * H, 0.55f * H, 6, leaf);
        addCone(v, base + vec3(0, 0.5f * H, 0), 0.24f * H, 0.50f * H, 6, glm::clamp(leaf * 1.15f, 0.0f, 1.0f));
    }
    return upload(v);
}

static Mesh buildClouds() {
    Verts v; Rng rng(2024);
    for (int c = 0; c < 45; c++) {
        vec3 center(-520.0f + rng.next() * 1040.0f, 55.0f + rng.next() * 35.0f, -850.0f + rng.next() * 1100.0f);
        int puffs = 4 + (int)(rng.next() * 4.0f);
        for (int p = 0; p < puffs; p++) {
            vec3 off((rng.next() - 0.5f) * 30.0f, (rng.next() - 0.5f) * 3.0f, (rng.next() - 0.5f) * 14.0f);
            vec3 rad(7.0f + rng.next() * 8.0f, 2.5f + rng.next() * 2.5f, 5.0f + rng.next() * 5.0f);
            addEllipsoid(v, center + off, rad, vec3(0.97f, 0.98f, 1.0f), 4, 10);
        }
    }
    return upload(v);
}

// Aircraft in local space: nose toward -Z, up +Y, right +X, length 2, span 2.8
static Mesh buildAircraft() {
    Verts v;
    vec3 white(0.92f, 0.92f, 0.94f), belly(0.55f, 0.58f, 0.64f), red(0.80f, 0.14f, 0.10f), glass(0.18f, 0.34f, 0.50f);
    vec3 dark(0.14f, 0.14f, 0.16f);

    // Fuselage: 12-sided tapered tube
    struct Ring { float z, r, y; };
    const Ring rings[] = { { -0.95f, 0.06f, 0.0f },{ -0.80f, 0.17f, 0.0f },{ -0.30f, 0.23f, 0.0f },{ 0.35f, 0.17f, 0.02f },{ 0.95f, 0.05f, 0.08f } };
    const int S = 12;
    for (int i = 0; i < 4; i++)
        for (int s = 0; s < S; s++) {
            float a0 = 2.0f * PI * s / S, a1 = 2.0f * PI * (s + 1) / S, sy = std::sin(0.5f * (a0 + a1));
            auto pt = [&](const Ring& R, float a) { return vec3(R.r * std::cos(a), R.y + R.r * std::sin(a), R.z); };
            vec3 col = white;
            if (i == 0) col = red;
            else if (sy < -0.5f) col = belly;
            else if (std::fabs(sy) < 0.3f) col = red;                        // side stripe
            addQuad(v, pt(rings[i], a0), pt(rings[i], a1), pt(rings[i + 1], a1), pt(rings[i + 1], a0), col);
        }
    addEllipsoid(v, vec3(0.0f, 0.20f, -0.25f), vec3(0.13f, 0.11f, 0.30f), glass, 5, 10);   // canopy

    // Wings, stabilisers, fin
    for (float s : { -1.0f, 1.0f }) {
        addSlab(v, vec3(0, -0.10f, -0.40f), vec3(0, -0.10f, 0.22f), vec3(1.4f * s, -0.03f, -0.15f), vec3(1.4f * s, -0.03f, 0.15f),
            vec3(0, 0.025f, 0), white, belly, red);
        addSlab(v, vec3(0, 0.02f, 0.62f), vec3(0, 0.02f, 0.95f), vec3(0.55f * s, 0.03f, 0.80f), vec3(0.55f * s, 0.03f, 0.95f),
            vec3(0, 0.015f, 0), white, belly, red);
        // Landing gear
        addBox(v, vec3(s * 0.33f - 0.015f, -0.30f, -0.14f), vec3(s * 0.33f + 0.015f, -0.10f, -0.10f), dark);
        addEllipsoid(v, vec3(s * 0.33f, -0.31f, -0.12f), vec3(0.025f, 0.055f, 0.055f), vec3(0.05f), 4, 8);
    }
    addSlab(v, vec3(0, 0.15f, 0.55f), vec3(0, 0.12f, 0.97f), vec3(0, 0.55f, 0.86f), vec3(0, 0.55f, 0.97f),
        vec3(0.015f, 0, 0), red, red, red);
    addBox(v, vec3(-0.015f, -0.30f, -0.62f), vec3(0.015f, -0.12f, -0.58f), dark);
    addEllipsoid(v, vec3(0.0f, -0.31f, -0.60f), vec3(0.025f, 0.055f, 0.055f), vec3(0.05f), 4, 8);
    return upload(v);
}

static Mesh buildPropeller() {
    Verts v;
    vec3 c(0.10f, 0.10f, 0.11f);
    addBox(v, vec3(-0.03f, -0.45f, -0.01f), vec3(0.03f, 0.45f, 0.01f), c);
    addBox(v, vec3(-0.45f, -0.03f, -0.01f), vec3(0.45f, 0.03f, 0.01f), c);
    return upload(v);
}

static Mesh buildShadow() {
    Verts v;
    auto q = [&](float x0, float x1, float z0, float z1) {
        addQuad(v, vec3(x0, 0, z0), vec3(x1, 0, z0), vec3(x1, 0, z1), vec3(x0, 0, z1), vec3(0.0f), true);
        };
    q(-0.14f, 0.14f, -0.95f, 0.95f);                                   // fuselage
    q(-1.4f, -0.14f, -0.15f, 0.15f);  q(0.14f, 1.4f, -0.15f, 0.15f);   // wings
    q(-0.55f, -0.14f, 0.80f, 0.95f);  q(0.14f, 0.55f, 0.80f, 0.95f);   // tailplane
    return upload(v);
}

// ----------------------------------------------------------------------------
//  Shaders
// ----------------------------------------------------------------------------
static const char* MAIN_VS = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uViewProj;
uniform mat4 uModel;
uniform mat3 uNormalMat;
out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vColor;
void main() {
    vec4 w = uModel * vec4(aPos, 1.0);
    vWorldPos = w.xyz;
    vNormal = uNormalMat * aNormal;
    vColor = aColor;
    gl_Position = uViewProj * w;
}
)GLSL";

static const char* MAIN_FS = R"GLSL(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vColor;
out vec4 FragColor;
uniform vec3  uCamPos;
uniform vec3  uSunDir;
uniform vec3  uHorizon;
uniform float uFogDensity;
uniform float uGlow;
uniform float uAlpha;
uniform float uSpec;
void main() {
    vec3 albedo = pow(vColor, vec3(2.2));
    vec3 V = normalize(uCamPos - vWorldPos);
    float nl = length(vNormal);
    vec3 col;
    if (nl < 0.001) {
        col = albedo * uGlow;                       // emissive / unlit
    } else {
        vec3 N = vNormal / nl;
        if (dot(N, V) < 0.0) N = -N;                // two-sided lighting
        float diff = max(dot(N, uSunDir), 0.0);
        vec3 hemi = mix(vec3(0.18, 0.17, 0.14), vec3(0.28, 0.38, 0.52), N.y * 0.5 + 0.5);
        vec3 sunCol = vec3(1.25, 1.12, 0.92);
        col = albedo * (hemi + sunCol * diff);
        if (uSpec > 0.0) {
            vec3 H = normalize(uSunDir + V);
            col += sunCol * pow(max(dot(N, H), 0.0), 48.0) * uSpec * diff;
        }
    }
    float d = length(uCamPos - vWorldPos);
    float fog = clamp(1.0 - exp(-pow(d * uFogDensity, 1.6)), 0.0, 1.0);
    col = mix(col, pow(uHorizon, vec3(2.2)), fog);
    FragColor = vec4(pow(col, vec3(1.0 / 2.2)), uAlpha);
}
)GLSL";

static const char* SKY_VS = R"GLSL(
#version 330 core
out vec2 vNdc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vNdc = p * 2.0 - 1.0;
    gl_Position = vec4(vNdc, 1.0, 1.0);
}
)GLSL";

static const char* SKY_FS = R"GLSL(
#version 330 core
in vec2 vNdc;
out vec4 FragColor;
uniform mat4 uInvViewProj;
uniform vec3 uSunDir;
uniform vec3 uHorizon;
void main() {
    vec4 w = uInvViewProj * vec4(vNdc, 1.0, 1.0);
    vec3 dir = normalize(w.xyz / w.w);
    vec3 horizon = pow(uHorizon, vec3(2.2));
    vec3 zenith  = pow(vec3(0.16, 0.36, 0.78), vec3(2.2));
    vec3 col = mix(horizon, zenith, pow(clamp(dir.y, 0.0, 1.0), 0.5));
    float sd = max(dot(dir, uSunDir), 0.0);
    col += vec3(1.0, 0.85, 0.6) * (pow(sd, 8.0) * 0.25 + pow(sd, 200.0) * 0.6);
    col += vec3(1.0, 0.95, 0.85) * smoothstep(0.9997, 0.9999, sd) * 3.0;
    FragColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
)GLSL";

static const char* HUD_VS = R"GLSL(
#version 330 core
uniform vec4 uRect;   // x0, y0, x1, y1 in NDC
void main() {
    vec2 p = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    gl_Position = vec4(mix(uRect.xy, uRect.zw, p), 0.0, 1.0);
}
)GLSL";

static const char* HUD_FS = R"GLSL(
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;
void main() { FragColor = uColor; }
)GLSL";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(s, 1024, nullptr, log); std::cerr << "Shader compile error:\n" << log << std::endl; }
    return s;
}
static GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs), f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, 1024, nullptr, log); std::cerr << "Program link error:\n" << log << std::endl; }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ----------------------------------------------------------------------------
//  Flight model
// ----------------------------------------------------------------------------
static float approach(float v, float target, float maxDelta) {
    return v < target ? std::min(v + maxDelta, target) : std::max(v - maxDelta, target);
}
static vec3 forwardVec() {
    float cp = std::cos(glm::radians(plane.pitch));
    return glm::normalize(vec3(std::cos(glm::radians(plane.yaw)) * cp,
        std::sin(glm::radians(plane.pitch)),
        std::sin(glm::radians(plane.yaw)) * cp));
}
struct Basis { vec3 front, right, up, levelUp; };
static Basis planeBasis() {
    vec3 f = forwardVec();
    vec3 r0 = glm::normalize(glm::cross(f, vec3(0, 1, 0)));
    vec3 u0 = glm::cross(r0, f);
    mat4 R = glm::rotate(mat4(1.0f), glm::radians(plane.roll), f);
    vec3 u = glm::normalize(vec3(R * vec4(u0, 0.0f)));
    vec3 r = glm::normalize(glm::cross(f, u));
    return { f, r, u, u0 };
}

static void resetFlight() {
    plane = Plane();
    crashTimer = 0.0f;
}
static void crashPlane() {
    plane.crashed = true;
    plane.speed = 0.0f;
    crashTimer = 0.0f;
}

static void updatePlane(float dt) {
    Plane& p = plane;
    if (p.crashed) return;

    // Throttle / drag
    if (throttleInput > 0) { if (p.speed < MAX_SPEED) p.speed += 6.0f * dt; }
    else if (throttleInput < 0) { p.speed -= 6.0f * dt; }
    else { p.speed -= 1.5f * dt; }

    // Banking (only in the air) and bank-induced turning
    float rollTarget = p.onGround ? 0.0f : rollInput * 60.0f;
    p.roll += glm::clamp(rollTarget - p.roll, -90.0f * dt, 90.0f * dt);
    if (!p.onGround) p.yaw += std::sin(glm::radians(p.roll)) * 50.0f * (p.speed / MAX_SPEED) * dt;

    if (p.onGround) p.pitch = std::max(p.pitch, 0.0f);
    vec3 front = forwardVec();

    // Energy: climbing bleeds speed, diving gains it
    p.speed -= front.y * 4.0f * dt;
    p.speed = glm::clamp(p.speed, 0.0f, MAX_SPEED * 1.35f);

    // Lift / stall
    float liftFactor = glm::clamp(p.speed / STALL_SPEED, 0.0f, 1.0f);
    p.stalled = !p.onGround && p.speed < STALL_SPEED;
    if (p.stalled) {
        p.vSpeed -= 7.0f * (1.0f - liftFactor) * dt;
        p.pitch -= 22.0f * (1.0f - liftFactor) * dt;
    }
    else {
        p.vSpeed = approach(p.vSpeed, 0.0f, 10.0f * dt);
    }

    // Move
    vec3 moveDir = front;
    if (p.onGround&& p.speed < TAKEOFF_SPEED) moveDir.y = 0.0f;   // can't leave the ground too slowly
    bool wasOnGround = p.onGround;
    p.pos += moveDir * p.speed * dt;
    p.pos.y += p.vSpeed * dt;
    if (wasOnGround) p.speed = std::max(0.0f, p.speed - (isOnRunway(p.pos) ? 0.4f : 3.0f) * dt);

    p.pos.x = glm::clamp(p.pos.x, WORLD_MIN_X + 30.0f, WORLD_MAX_X - 30.0f);
    p.pos.z = glm::clamp(p.pos.z, WORLD_MIN_Z + 30.0f, WORLD_MAX_Z - 30.0f);
    p.pos.y = std::min(p.pos.y, 150.0f);

    // Terrain contact
    float groundY = terrainHeight(p.pos.x, p.pos.z) + PLANE_CLEARANCE;
    p.onGround = false;
    if (p.pos.y <= groundY) {
        float descent = -(moveDir.y * p.speed + p.vSpeed);
        bool hardLanding = descent > 3.5f;
        bool wingStrike = std::fabs(p.roll) > 35.0f;
        bool offRunwayFast = !isOnRunway(p.pos) && p.speed > 8.0f;
        bool hitWall = (groundY - p.pos.y) > 0.35f;
        if ((!wasOnGround && (hardLanding || wingStrike || offRunwayFast)) || hitWall) { crashPlane(); return; }
        p.pos.y = groundY;
        p.vSpeed = 0.0f;
        p.onGround = true;
    }

    // Buildings
    const float r = 0.6f;
    for (const AABB& b : obstacles)
        if (p.pos.x > b.mn.x - r && p.pos.x < b.mx.x + r &&
            p.pos.y > b.mn.y - r && p.pos.y < b.mx.y + r &&
            p.pos.z > b.mn.z - r && p.pos.z < b.mx.z + r) {
            crashPlane(); return;
        }
}

// ----------------------------------------------------------------------------
//  Input / callbacks
// ----------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow*, int width, int height) {
    winW = width; winH = height;
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow*, double xpos, double ypos) {
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    double dx = xpos - lastX, dy = lastY - ypos;
    lastX = xpos; lastY = ypos;
    if (plane.crashed) return;
    const float sensitivity = 0.1f;
    plane.yaw += (float)dx * sensitivity;
    plane.pitch += (float)dy * sensitivity;
    plane.pitch = glm::clamp(plane.pitch, -80.0f, 80.0f);
}

void processInput(GLFWwindow* window) {
    static bool prevC = false, prevR = false;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    bool c = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
    if (c && !prevC) cockpitView = !cockpitView;
    prevC = c;
    bool r = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
    if (r && !prevR) resetFlight();
    prevR = r;

    throttleInput = 0; rollInput = 0;
    if (plane.crashed) return;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) throttleInput = 1;
    else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) throttleInput = -1;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) rollInput = -1;   // bank left
    else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) rollInput = 1; // bank right
}

// ----------------------------------------------------------------------------
//  Main
// ----------------------------------------------------------------------------
int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(winW, winH, "C++ Flight Simulator", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;
    glEnable(GL_MULTISAMPLE);

    GLuint prog = makeProgram(MAIN_VS, MAIN_FS);
    GLuint skyProg = makeProgram(SKY_VS, SKY_FS);
    GLuint hudProg = makeProgram(HUD_VS, HUD_FS);
    GLuint emptyVAO; glGenVertexArrays(1, &emptyVAO);

    auto L = [&](const char* n) { return glGetUniformLocation(prog, n); };
    GLint uViewProj = L("uViewProj"), uModel = L("uModel"), uNormalMat = L("uNormalMat"), uCamPos = L("uCamPos");
    GLint uSunDir = L("uSunDir"), uHorizon = L("uHorizon"), uFogDensity = L("uFogDensity");
    GLint uGlow = L("uGlow"), uAlpha = L("uAlpha"), uSpec = L("uSpec");
    GLint skyInv = glGetUniformLocation(skyProg, "uInvViewProj"), skySun = glGetUniformLocation(skyProg, "uSunDir"),
        skyHor = glGetUniformLocation(skyProg, "uHorizon");
    GLint hudRect = glGetUniformLocation(hudProg, "uRect"), hudColor = glGetUniformLocation(hudProg, "uColor");

    Mesh terrain = buildTerrain(), runway = buildRunway(), markings = buildMarkings(), lights = buildRunwayLights();
    Mesh structures = buildStructures(), beacon = buildBeacon(), fireball = buildFireball();
    Mesh forest = buildForest(), clouds = buildClouds();
    Mesh aircraft = buildAircraft(), propeller = buildPropeller(), shadow = buildShadow();

    float propAngle = 0.0f, titleTimer = 0.0f;
    resetFlight();

    while (!glfwWindowShouldClose(window)) {
        float now = (float)glfwGetTime();
        deltaTime = std::min(now - lastFrame, 0.05f);
        lastFrame = now;

        processInput(window);
        updatePlane(deltaTime);
        if (plane.crashed) {
            crashTimer += deltaTime;
            if (crashTimer > 3.0f) resetFlight();
        }
        propAngle += (400.0f + plane.speed * 60.0f) * deltaTime;

        float terrainUnder = terrainHeight(plane.pos.x, plane.pos.z);
        float agl = std::max(0.0f, plane.pos.y - terrainUnder - PLANE_CLEARANCE);

        // ---- Window title telemetry (updated 10x / second)
        titleTimer += deltaTime;
        if (titleTimer > 0.1f) {
            titleTimer = 0.0f;
            std::ostringstream ss;
            if (plane.crashed) ss << "!!! CRASHED - resetting to runway !!!";
            else {
                if (plane.stalled) ss << "!!! STALL !!! | ";
                ss << "C++ Flight Simulator | " << (int)(plane.speed * 20) << " kts | AGL " << (int)(agl * 100)
                    << " ft | Pitch " << (int)plane.pitch << " | Roll " << (int)plane.roll
                    << " | [C] view  [R] reset";
            }
            glfwSetWindowTitle(window, ss.str().c_str());
        }

        // ---- Camera
        Basis b = planeBasis();
        vec3 eye, target, camUp;
        if (cockpitView) {
            eye = plane.pos + b.up * 0.12f + b.front * 0.15f;
            target = eye + b.front;
            camUp = b.up;
        }
        else {
            eye = plane.pos - b.front * 3.2f + b.levelUp * 0.9f;
            eye.y = std::max(eye.y, terrainHeight(eye.x, eye.z) + 0.4f);
            target = plane.pos + b.front * 2.0f;
            mat4 R = glm::rotate(mat4(1.0f), glm::radians(plane.roll * 0.4f), b.front);
            camUp = vec3(R * vec4(b.levelUp, 0.0f));
        }
        mat4 view = glm::lookAt(eye, target, camUp);
        mat4 proj = glm::perspective(glm::radians(55.0f), winW / (float)std::max(winH, 1), 0.15f, 1500.0f);
        mat4 viewProj = proj * view;

        // ---- Sky
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(skyProg);
        mat4 invVP = glm::inverse(proj * mat4(mat3(view)));
        glUniformMatrix4fv(skyInv, 1, GL_FALSE, glm::value_ptr(invVP));
        glUniform3fv(skySun, 1, glm::value_ptr(SUN_DIR));
        glUniform3fv(skyHor, 1, glm::value_ptr(HORIZON_SRGB));
        glBindVertexArray(emptyVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEnable(GL_DEPTH_TEST);

        // ---- World
        glUseProgram(prog);
        glUniformMatrix4fv(uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
        glUniform3fv(uCamPos, 1, glm::value_ptr(eye));
        glUniform3fv(uSunDir, 1, glm::value_ptr(SUN_DIR));
        glUniform3fv(uHorizon, 1, glm::value_ptr(HORIZON_SRGB));
        glUniform1f(uFogDensity, FOG_DENSITY);
        glUniform1f(uGlow, 1.0f); glUniform1f(uAlpha, 1.0f); glUniform1f(uSpec, 0.0f);

        auto setModel = [&](const mat4& m) {
            glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(m));
            mat3 nm = glm::transpose(glm::inverse(mat3(m)));
            glUniformMatrix3fv(uNormalMat, 1, GL_FALSE, glm::value_ptr(nm));
            };
        setModel(mat4(1.0f));

        drawMesh(terrain);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-1.0f, -1.0f); drawMesh(runway);
        glPolygonOffset(-2.0f, -2.0f); drawMesh(markings);
        glDisable(GL_POLYGON_OFFSET_FILL);
        drawMesh(structures);
        drawMesh(forest);
        drawMesh(clouds);
        drawMesh(lights);

        glUniform1f(uGlow, 0.35f + 0.65f * std::fabs(std::sin(now * 5.0f)));   // flashing beacon
        drawMesh(beacon);
        glUniform1f(uGlow, 1.0f);

        // ---- Aircraft (or fireball)
        if (plane.crashed) {
            float s = 0.5f + crashTimer * 2.5f;
            setModel(glm::translate(mat4(1.0f), plane.pos) * glm::scale(mat4(1.0f), vec3(s)));
            glUniform1f(uGlow, std::max(0.0f, 1.0f - crashTimer / 2.5f));
            drawMesh(fireball);
            glUniform1f(uGlow, 1.0f);
        }
        else if (!cockpitView) {
            mat4 rot(1.0f);
            rot[0] = vec4(b.right, 0.0f); rot[1] = vec4(b.up, 0.0f); rot[2] = vec4(-b.front, 0.0f);
            mat4 planeModel = glm::translate(mat4(1.0f), plane.pos) * rot * glm::scale(mat4(1.0f), vec3(PLANE_SCALE));
            glUniform1f(uSpec, 0.6f);
            setModel(planeModel);
            drawMesh(aircraft);
            setModel(planeModel * glm::translate(mat4(1.0f), vec3(0, 0, -1.0f)) *
                glm::rotate(mat4(1.0f), glm::radians(propAngle), vec3(0, 0, 1)));
            drawMesh(propeller);
            glUniform1f(uSpec, 0.0f);
        }

        // ---- Ground shadow (soft, alpha-blended)
        if (!plane.crashed && agl < 80.0f) {
            vec3 h = glm::normalize(vec3(b.front.x, 0.0f, b.front.z));
            mat4 rotS(1.0f);
            rotS[0] = vec4(glm::cross(h, vec3(0, 1, 0)), 0.0f); rotS[1] = vec4(0, 1, 0, 0); rotS[2] = vec4(-h, 0.0f);
            setModel(glm::translate(mat4(1.0f), vec3(plane.pos.x, terrainUnder + 0.06f, plane.pos.z)) * rotS *
                glm::scale(mat4(1.0f), vec3(PLANE_SCALE)));
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_FALSE);
            glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-4.0f, -4.0f);
            glUniform1f(uAlpha, 0.35f * (1.0f - agl / 80.0f));
            drawMesh(shadow);
            glUniform1f(uAlpha, 1.0f);
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);
        }

        // ---- HUD
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(hudProg);
        glBindVertexArray(emptyVAO);
        auto rect = [&](float x0, float y0, float x1, float y1, vec4 c) {
            glUniform4f(hudRect, x0, y0, x1, y1);
            glUniform4f(hudColor, c.r, c.g, c.b, c.a);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            };
        // Speed bar (bottom-left) with stall marker
        float sf = glm::clamp(plane.speed / MAX_SPEED, 0.0f, 1.0f);
        vec3 sc = plane.speed < STALL_SPEED ? vec3(0.95f, 0.25f, 0.2f) : vec3(0.3f, 0.9f, 0.4f);
        rect(-0.95f, -0.93f, -0.55f, -0.87f, vec4(0, 0, 0, 0.45f));
        rect(-0.95f, -0.93f, -0.95f + 0.4f * sf, -0.87f, vec4(sc, 0.85f));
        float mk = -0.95f + 0.4f * (STALL_SPEED / MAX_SPEED);
        rect(mk - 0.003f, -0.94f, mk + 0.003f, -0.86f, vec4(1, 1, 1, 0.9f));
        // Altitude bar (bottom-right)
        float af = glm::clamp(agl / 0.4f / 100.0f, 0.0f, 1.0f);   // full bar at 40 units AGL
        rect(0.55f, -0.93f, 0.95f, -0.87f, vec4(0, 0, 0, 0.45f));
        rect(0.55f, -0.93f, 0.55f + 0.4f * af, -0.87f, vec4(0.35f, 0.75f, 1.0f, 0.85f));
        // Centre marker
        rect(-0.012f, -0.002f, 0.012f, 0.002f, vec4(1, 1, 1, 0.7f));
        rect(-0.0012f, -0.02f, 0.0012f, 0.02f, vec4(1, 1, 1, 0.7f));
        // Warnings
        if (plane.stalled && std::sin(now * 14.0f) > 0.0f) rect(-0.35f, 0.82f, 0.35f, 0.90f, vec4(0.9f, 0.1f, 0.05f, 0.8f));
        if (plane.crashed) rect(-1.0f, -1.0f, 1.0f, 1.0f, vec4(0.8f, 0.05f, 0.02f, std::min(0.5f, crashTimer * 0.5f)));
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    for (Mesh* m : { &terrain, &runway, &markings, &lights, &structures, &beacon, &fireball, &forest, &clouds,
                     &aircraft, &propeller, &shadow })
        destroyMesh(*m);
    glDeleteVertexArrays(1, &emptyVAO);
    glDeleteProgram(prog); glDeleteProgram(skyProg); glDeleteProgram(hudProg);
    glfwTerminate();
    return 0;
}
