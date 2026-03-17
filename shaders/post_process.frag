#version 450 core

in vec2 vUV;

uniform sampler2D uColorTexture;
uniform sampler2D uBloomTexture;
uniform float uExposure;
uniform float uGamma;
uniform float uBloomIntensity;
uniform float uVignetteStrength;
uniform float uVignetteRadius;
uniform float uSaturation;
uniform int uFXAAEnabled;
uniform vec2 uTexelSize;

out vec4 FragColor;

vec3 acesToneMap(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 applySaturation(vec3 color, float sat) {
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luma), color, sat);
}

float fxaaLuma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 applyFXAA(sampler2D tex, vec2 uv, vec2 texel) {
    vec3 cc = texture(tex, uv).rgb;
    vec3 cn = texture(tex, uv + vec2(0.0, -texel.y)).rgb;
    vec3 cs = texture(tex, uv + vec2(0.0, texel.y)).rgb;
    vec3 ce = texture(tex, uv + vec2(texel.x, 0.0)).rgb;
    vec3 cw = texture(tex, uv + vec2(-texel.x, 0.0)).rgb;

    float lc = fxaaLuma(cc);
    float ln = fxaaLuma(cn);
    float ls = fxaaLuma(cs);
    float le = fxaaLuma(ce);
    float lw = fxaaLuma(cw);

    float lmin = min(min(min(ln, ls), min(le, lw)), lc);
    float lmax = max(max(max(ln, ls), max(le, lw)), lc);
    float lrange = lmax - lmin;

    if (lrange < max(0.0625, lmax * 0.125)) return cc;

    vec3 cnw = texture(tex, uv + vec2(-texel.x, -texel.y)).rgb;
    vec3 cne = texture(tex, uv + vec2(texel.x, -texel.y)).rgb;
    vec3 csw = texture(tex, uv + vec2(-texel.x, texel.y)).rgb;
    vec3 cse = texture(tex, uv + vec2(texel.x, texel.y)).rgb;

    return (cc * 4.0 + cn + cs + ce + cw + (cnw + cne + csw + cse) * 0.5) / 10.0;
}

void main() {
    vec3 color;

    if (uFXAAEnabled != 0) {
        color = applyFXAA(uColorTexture, vUV, uTexelSize);
    } else {
        color = texture(uColorTexture, vUV).rgb;
    }

    vec3 bloom = texture(uBloomTexture, vUV).rgb;
    color += bloom * uBloomIntensity;

    color *= uExposure;

    color = acesToneMap(color);

    color = applySaturation(color, uSaturation);

    color = pow(color, vec3(1.0 / uGamma));

    vec2 centered = vUV * 2.0 - 1.0;
    float dist = length(centered);
    float vignette = 1.0 - smoothstep(uVignetteRadius, 1.0, dist) * uVignetteStrength;
    color *= vignette;

    FragColor = vec4(color, 1.0);
}