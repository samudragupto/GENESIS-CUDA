#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in float aSize;

uniform mat4 uView;
uniform mat4 uProjection;
uniform float uPointScale;

out float vDepth;
out vec2 vScreenPos;

void main() {
    vec4 viewPos = uView * vec4(aPosition, 1.0);
    vDepth = -viewPos.z;

    gl_Position = uProjection * viewPos;

    float dist = length(viewPos.xyz);
    gl_PointSize = uPointScale * aSize / dist;

    vScreenPos = gl_Position.xy / gl_Position.w * 0.5 + 0.5;
}