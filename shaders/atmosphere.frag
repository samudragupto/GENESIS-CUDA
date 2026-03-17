#version 450 core

in vec2 vUV;

uniform vec3 uSunDirection;
uniform vec3 uSunColor;
uniform float uSunIntensity;
uniform float uTimeOfDay;
uniform vec3 uCameraPos;
uniform mat4 uInvViewProj;

out vec4 FragColor;

vec3 skyColor(vec3 dir, vec3 sunDir) {
    float sunHeight = sunDir.y;
    float dayFactor = clamp(sunHeight * 2.0, 0.0, 1.0);
    float sunsetFactor = clamp(1.0 - abs(sunHeight) * 5.0, 0.0, 1.0);

    vec3 dayTop = vec3(0.1, 0.3, 0.8);
    vec3 dayBottom = vec3(0.5, 0.7, 1.0);
    vec3 nightTop = vec3(0.01, 0.01, 0.05);
    vec3 nightBottom = vec3(0.02, 0.02, 0.08);
    vec3 sunsetColor = vec3(1.0, 0.4, 0.1);

    float vFactor = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

    vec3 daySky = mix(dayBottom, dayTop, vFactor);
    vec3 nightSky = mix(nightBottom, nightTop, vFactor);

    vec3 sky = mix(nightSky, daySky, dayFactor);
    sky += sunsetColor * sunsetFactor * (1.0 - vFactor) * 0.5;

    float sunDot = max(dot(dir, sunDir), 0.0);
    float sunDisc = smoothstep(0.997, 0.999, sunDot);
    sky += uSunColor * sunDisc * 10.0 * dayFactor;

    float sunGlow = pow(max(sunDot, 0.0), 8.0);
    sky += uSunColor * sunGlow * 0.2 * dayFactor;

    return sky;
}

void main() {
    vec4 clipPos = vec4(vUV * 2.0 - 1.0, 1.0, 1.0);
    vec4 worldPos = uInvViewProj * clipPos;
    vec3 rayDir = normalize(worldPos.xyz / worldPos.w - uCameraPos);

    vec3 color = skyColor(rayDir, normalize(uSunDirection));

    FragColor = vec4(color, 1.0);
}