#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include "render_common.cuh"

class CameraController {
public:
    Camera camera;

    float move_speed;
    float mouse_sensitivity;
    float zoom_speed;

    bool keys[256];
    float last_mouse_x;
    float last_mouse_y;
    bool mouse_captured;

    void init(float start_x, float start_y, float start_z,
              int screen_width, int screen_height);

    void processKeyDown(unsigned char key);
    void processKeyUp(unsigned char key);
    void processMouseMove(float x, float y);
    void processMouseScroll(float delta);
    void processMouseButton(int button, bool pressed);

    void update(float dt);

    Camera getCamera() const;

    void lookAt(float x, float y, float z);
    void setPosition(float x, float y, float z);

    void clampToBounds(float min_x, float min_y, float min_z,
                       float max_x, float max_y, float max_z);
};

#endif