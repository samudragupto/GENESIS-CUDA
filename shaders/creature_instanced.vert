#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

layout(location = 2) in vec3 iPosition;
layout(location = 3) in vec3 iColor;
layout(location = 4) in float iScale;
layout(location = 5) in float iRotation;

uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uSunDirection;

out vec3 vColor;
out vec3 vNormal;
out vec3 vWorldPos;
out float vNdotL;

void main() {
    float cosR = cos(iRotation);
    float sinR = sin(iRotation);

    mat3 rotation = mat3(
        cosR,  0.0, sinR,
        0.0,   1.0, 0.0,
        -sinR, 0.0, cosR
    );

    vec3 localPos = rotation * (aPosition * iScale);
    vec3 worldPos = localPos + iPosition;

    vec3 worldNormal = normalize(rotation * aNormal);

    vColor = iColor;
    vNormal = worldNormal;
    vWorldPos = worldPos;
    vNdotL = max(dot(worldNormal, normalize(uSunDirection)), 0.0);

    gl_Position = uProjection * uView * vec4(worldPos, 1.0);
}