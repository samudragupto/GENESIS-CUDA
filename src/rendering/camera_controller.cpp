#include "camera_controller.h"
#include <cmath>
#include <cstring>
#include <algorithm>

void CameraController::init(float start_x, float start_y, float start_z,
                            int screen_width, int screen_height) {
    memset(keys, 0, sizeof(keys));

    camera.pos_x = start_x;
    camera.pos_y = start_y;
    camera.pos_z = start_z;
    camera.up_x = 0.0f;
    camera.up_y = 1.0f;
    camera.up_z = 0.0f;
    camera.fov = 60.0f;
    camera.near_plane = 0.1f;
    camera.far_plane = 5000.0f;
    camera.aspect = (float)screen_width / (float)screen_height;
    camera.yaw = -90.0f;
    camera.pitch = -30.0f;
    camera.speed = 50.0f;
    camera.sensitivity = 0.1f;

    move_speed = 50.0f;
    mouse_sensitivity = 0.1f;
    zoom_speed = 5.0f;

    last_mouse_x = (float)screen_width * 0.5f;
    last_mouse_y = (float)screen_height * 0.5f;
    mouse_captured = false;

    float yaw_rad = camera.yaw * 3.14159265f / 180.0f;
    float pitch_rad = camera.pitch * 3.14159265f / 180.0f;

    camera.look_x = camera.pos_x + cosf(yaw_rad) * cosf(pitch_rad);
    camera.look_y = camera.pos_y + sinf(pitch_rad);
    camera.look_z = camera.pos_z + sinf(yaw_rad) * cosf(pitch_rad);
}

void CameraController::processKeyDown(unsigned char key) {
    keys[key] = true;
}

void CameraController::processKeyUp(unsigned char key) {
    keys[key] = false;
}

void CameraController::processMouseMove(float x, float y) {
    if (!mouse_captured) return;

    float dx = x - last_mouse_x;
    float dy = last_mouse_y - y;

    last_mouse_x = x;
    last_mouse_y = y;

    camera.yaw += dx * mouse_sensitivity;
    camera.pitch += dy * mouse_sensitivity;

    if (camera.pitch > 89.0f) camera.pitch = 89.0f;
    if (camera.pitch < -89.0f) camera.pitch = -89.0f;
}

void CameraController::processMouseScroll(float delta) {
    camera.fov -= delta * zoom_speed;
    if (camera.fov < 10.0f) camera.fov = 10.0f;
    if (camera.fov > 120.0f) camera.fov = 120.0f;
}

void CameraController::processMouseButton(int button, bool pressed) {
    if (button == 1) {
        mouse_captured = pressed;
    }
}

void CameraController::update(float dt) {
    float yaw_rad = camera.yaw * 3.14159265f / 180.0f;
    float pitch_rad = camera.pitch * 3.14159265f / 180.0f;

    float front_x = cosf(yaw_rad) * cosf(pitch_rad);
    float front_y = sinf(pitch_rad);
    float front_z = sinf(yaw_rad) * cosf(pitch_rad);

    float right_x = sinf(yaw_rad);
    float right_y = 0.0f;
    float right_z = -cosf(yaw_rad);

    float speed = move_speed * dt;

    if (keys['w'] || keys['W']) {
        camera.pos_x += front_x * speed;
        camera.pos_y += front_y * speed;
        camera.pos_z += front_z * speed;
    }
    if (keys['s'] || keys['S']) {
        camera.pos_x -= front_x * speed;
        camera.pos_y -= front_y * speed;
        camera.pos_z -= front_z * speed;
    }
    if (keys['a'] || keys['A']) {
        camera.pos_x -= right_x * speed;
        camera.pos_z -= right_z * speed;
    }
    if (keys['d'] || keys['D']) {
        camera.pos_x += right_x * speed;
        camera.pos_z += right_z * speed;
    }
    if (keys['q'] || keys['Q']) {
        camera.pos_y += speed;
    }
    if (keys['e'] || keys['E']) {
        camera.pos_y -= speed;
    }

    if (keys['+'] || keys['=']) {
        move_speed *= 1.01f;
    }
    if (keys['-'] || keys['_']) {
        move_speed *= 0.99f;
        if (move_speed < 1.0f) move_speed = 1.0f;
    }

    camera.look_x = camera.pos_x + front_x;
    camera.look_y = camera.pos_y + front_y;
    camera.look_z = camera.pos_z + front_z;
}

Camera CameraController::getCamera() const {
    return camera;
}

void CameraController::lookAt(float x, float y, float z) {
    float dx = x - camera.pos_x;
    float dy = y - camera.pos_y;
    float dz = z - camera.pos_z;

    float dist_xz = sqrtf(dx * dx + dz * dz);
    camera.yaw = atan2f(dz, dx) * 180.0f / 3.14159265f;
    camera.pitch = atan2f(dy, dist_xz) * 180.0f / 3.14159265f;

    camera.look_x = x;
    camera.look_y = y;
    camera.look_z = z;
}

void CameraController::setPosition(float x, float y, float z) {
    camera.pos_x = x;
    camera.pos_y = y;
    camera.pos_z = z;
}

void CameraController::clampToBounds(float min_x, float min_y, float min_z,
                                     float max_x, float max_y, float max_z) {
    camera.pos_x = std::max(min_x, std::min(camera.pos_x, max_x));
    camera.pos_y = std::max(min_y, std::min(camera.pos_y, max_y));
    camera.pos_z = std::max(min_z, std::min(camera.pos_z, max_z));
}