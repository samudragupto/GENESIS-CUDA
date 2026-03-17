#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in float aBiomeId;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;
uniform vec3 uSunDirection;
uniform float uWaterLevel;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;
out float vHeight;
out float vBiomeId;
out float vShadowFactor;
out vec3 vViewDir;

void main() {
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(uNormalMatrix * aNormal);
    vTexCoord = aTexCoord;
    vHeight = aPosition.y;
    vBiomeId = aBiomeId;

    vec3 camPos = inverse(uView)[3].xyz;
    vViewDir = normalize(camPos - worldPos.xyz);

    float NdotL = max(dot(vNormal, uSunDirection), 0.0);
    vShadowFactor = NdotL;

    gl_Position = uProjection * uView * worldPos;
}