#version 450 core

in vec2 vUV;

uniform sampler2D uSceneTexture;
uniform sampler2D uHeatmapTexture;
uniform float uAlpha;
uniform int uEnabled;

out vec4 FragColor;

void main() {
    vec3 scene = texture(uSceneTexture, vUV).rgb;

    if (uEnabled == 0) {
        FragColor = vec4(scene, 1.0);
        return;
    }

    vec3 heatmap = texture(uHeatmapTexture, vUV).rgb;
    vec3 color = mix(scene, heatmap, uAlpha);

    FragColor = vec4(color, 1.0);
}