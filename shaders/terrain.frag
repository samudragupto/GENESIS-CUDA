#version 450 core

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;
in float vHeight;
in float vBiomeId;
in float vShadowFactor;
in vec3 vViewDir;

uniform vec3 uSunDirection;
uniform vec3 uSunColor;
uniform vec3 uAmbientColor;
uniform float uSunIntensity;
uniform float uWaterLevel;
uniform float uTime;

out vec4 FragColor;

vec3 getBiomeColor(float biomeId, float height, float slope) {
    vec3 water = vec3(0.1, 0.3, 0.6);
    vec3 sand = vec3(0.76, 0.70, 0.50);
    vec3 grass = vec3(0.2, 0.5, 0.15);
    vec3 forest = vec3(0.1, 0.35, 0.08);
    vec3 rock = vec3(0.5, 0.45, 0.4);
    vec3 snow = vec3(0.9, 0.9, 0.95);
    vec3 desert = vec3(0.8, 0.65, 0.35);
    vec3 tundra = vec3(0.6, 0.65, 0.55);

    if (height < uWaterLevel) return water;
    if (height < uWaterLevel + 0.02) return sand;

    if (slope > 0.7) return rock;

    if (biomeId < 1.0) return grass;
    if (biomeId < 2.0) return forest;
    if (biomeId < 3.0) return desert;
    if (biomeId < 4.0) return tundra;
    if (biomeId < 5.0) return rock;
    if (biomeId < 6.0) return snow;

    return mix(grass, rock, clamp((height - 0.6) / 0.3, 0.0, 1.0));
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uSunDirection);
    vec3 V = normalize(vViewDir);
    vec3 H = normalize(L + V);

    float slope = 1.0 - N.y;

    vec3 baseColor = getBiomeColor(vBiomeId, vHeight, slope);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    vec3 diffuse = baseColor * NdotL * uSunColor * uSunIntensity;
    vec3 ambient = baseColor * uAmbientColor * 0.3;

    float specular = 0.0;
    if (vHeight < uWaterLevel + 0.01) {
        specular = pow(NdotH, 64.0) * 0.5;
    } else {
        specular = pow(NdotH, 16.0) * 0.1;
    }

    vec3 color = ambient + diffuse + vec3(specular) * uSunColor;

    float dist = length(vWorldPos);
    float fog = 1.0 - exp(-dist * 0.0003);
    vec3 fogColor = uSunColor * 0.5 + vec3(0.4, 0.5, 0.6) * 0.5;
    color = mix(color, fogColor, clamp(fog, 0.0, 0.8));

    if (vHeight < uWaterLevel) {
        float depth = (uWaterLevel - vHeight) * 10.0;
        float waterAlpha = clamp(depth, 0.3, 0.9);
        vec3 waterColor = vec3(0.05, 0.15, 0.4);
        color = mix(color, waterColor, waterAlpha);

        float wave = sin(vWorldPos.x * 2.0 + uTime * 1.5) *
                     cos(vWorldPos.z * 1.7 + uTime * 1.2) * 0.02;
        color += vec3(wave * 0.5);
    }

    FragColor = vec4(color, 1.0);
}