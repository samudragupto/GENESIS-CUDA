#version 450 core

in vec3 vColor;
in vec3 vNormal;
in vec3 vWorldPos;
in float vNdotL;

uniform vec3 uSunColor;
uniform vec3 uAmbientColor;
uniform float uSunIntensity;
uniform vec3 uCameraPos;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCameraPos - vWorldPos);

    vec3 ambient = vColor * uAmbientColor * 0.4;
    vec3 diffuse = vColor * vNdotL * uSunColor * uSunIntensity;

    float rim = 1.0 - max(dot(N, V), 0.0);
    rim = pow(rim, 3.0) * 0.3;
    vec3 rimColor = vColor * rim;

    vec3 color = ambient + diffuse + rimColor;

    float dist = length(vWorldPos - uCameraPos);
    float fog = 1.0 - exp(-dist * 0.0005);
    vec3 fogColor = uSunColor * 0.4 + vec3(0.3, 0.4, 0.5) * 0.6;
    color = mix(color, fogColor, clamp(fog, 0.0, 0.7));

    FragColor = vec4(color, 1.0);
}