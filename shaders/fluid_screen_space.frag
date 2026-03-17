#version 450 core

in float vDepth;
in vec2 vScreenPos;

uniform vec3 uWaterColor;
uniform float uWaterAlpha;
uniform vec3 uSunDirection;
uniform vec3 uSunColor;

out vec4 FragColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;

    float z = sqrt(1.0 - r2);

    vec3 normal = vec3(coord.x, coord.y, z);
    float NdotL = max(dot(normal, normalize(uSunDirection)), 0.0);

    vec3 color = uWaterColor * 0.3 + uWaterColor * NdotL * uSunColor * 0.7;

    float fresnel = pow(1.0 - z, 3.0);
    color += vec3(fresnel * 0.3) * uSunColor;

    float specular = pow(max(dot(reflect(-normalize(uSunDirection), normal),
                    vec3(0.0, 0.0, 1.0)), 0.0), 32.0);
    color += uSunColor * specular * 0.5;

    float alpha = uWaterAlpha * (1.0 - r2 * 0.3);

    FragColor = vec4(color, alpha);
}