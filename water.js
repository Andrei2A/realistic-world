/**
 * PHOTOREALISTIC WATER SHADER - Three.js
 * Senior Graphics Engineer Implementation
 *
 * Features:
 * - Gerstner Waves (GPU-based)
 * - Fresnel Effect
 * - Refraction with distortion
 * - Dual animated Normal Maps
 * - Planar Reflections
 * - Shore Foam
 * - Caustics
 */

import * as THREE from 'three';

// ============================================================
// WATER SHADER - VERTEX (Gerstner Waves)
// ============================================================
const waterVertexShader = `
uniform float uTime;
uniform vec4 uWave1; // direction.xy, steepness, wavelength
uniform vec4 uWave2;
uniform vec4 uWave3;
uniform vec4 uWave4;

varying vec3 vWorldPosition;
varying vec3 vWorldNormal;
varying vec4 vScreenPos;
varying vec2 vUv;

// Gerstner Wave Function - создает реалистичные волны с острыми гребнями
vec3 gerstnerWave(vec4 wave, vec3 p, inout vec3 tangent, inout vec3 binormal) {
    float steepness = wave.z;
    float wavelength = wave.w;

    float k = 2.0 * 3.14159 / wavelength;
    float c = sqrt(9.8 / k); // фазовая скорость
    vec2 d = normalize(wave.xy);
    float f = k * (dot(d, p.xz) - c * uTime);
    float a = steepness / k;

    // Накопление касательных для правильных нормалей
    tangent += vec3(
        -d.x * d.x * steepness * sin(f),
        d.x * steepness * cos(f),
        -d.x * d.y * steepness * sin(f)
    );
    binormal += vec3(
        -d.x * d.y * steepness * sin(f),
        d.y * steepness * cos(f),
        -d.y * d.y * steepness * sin(f)
    );

    return vec3(
        d.x * a * cos(f),
        a * sin(f),
        d.y * a * cos(f)
    );
}

void main() {
    vUv = uv;
    vec3 pos = position;

    vec3 tangent = vec3(1.0, 0.0, 0.0);
    vec3 binormal = vec3(0.0, 0.0, 1.0);

    // Применяем 4 волны Герстнера для сложной поверхности
    pos += gerstnerWave(uWave1, pos, tangent, binormal);
    pos += gerstnerWave(uWave2, pos, tangent, binormal);
    pos += gerstnerWave(uWave3, pos, tangent, binormal);
    pos += gerstnerWave(uWave4, pos, tangent, binormal);

    // Вычисляем нормаль из касательных
    vec3 normal = normalize(cross(binormal, tangent));

    vec4 worldPosition = modelMatrix * vec4(pos, 1.0);
    vWorldPosition = worldPosition.xyz;
    vWorldNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);

    vec4 clipPos = projectionMatrix * viewMatrix * worldPosition;
    vScreenPos = clipPos;

    gl_Position = clipPos;
}
`;

// ============================================================
// WATER SHADER - FRAGMENT (Fresnel, Refraction, Foam, etc.)
// ============================================================
const waterFragmentShader = `
precision highp float;

uniform float uTime;
uniform vec3 uSunDirection;
uniform vec3 uSunColor;
uniform vec3 uWaterColor;
uniform vec3 uWaterColorDeep;
uniform float uFresnelPower;
uniform float uRefractionStrength;
uniform float uFoamThreshold;
uniform float uFoamIntensity;

// Текстуры
uniform sampler2D uNormalMap1;
uniform sampler2D uNormalMap2;
uniform sampler2D uRefractionMap;
uniform sampler2D uReflectionMap;
uniform sampler2D uDepthMap;
uniform sampler2D uFoamTexture;

uniform vec2 uResolution;
uniform float uNear;
uniform float uFar;
uniform mat4 uProjectionMatrixInverse;
uniform mat4 uViewMatrixInverse;

varying vec3 vWorldPosition;
varying vec3 vWorldNormal;
varying vec4 vScreenPos;
varying vec2 vUv;

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

// Распаковка нормали из текстуры
vec3 unpackNormal(vec4 packednormal) {
    return packednormal.xyz * 2.0 - 1.0;
}

// Линейная глубина из буфера глубины
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * uNear * uFar) / (uFar + uNear - z * (uFar - uNear));
}

// Шум Перлина для пены
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// FBM для более сложного шума
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; i++) {
        value += amplitude * noise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// ============================================================
// MAIN FRAGMENT SHADER
// ============================================================

void main() {
    // Screen-space coordinates для текстур
    vec2 screenUV = vScreenPos.xy / vScreenPos.w * 0.5 + 0.5;

    // ========================================
    // 1. ANIMATED NORMAL MAPS (Dual Flow)
    // ========================================
    vec2 flowDir1 = vec2(0.03, 0.02);
    vec2 flowDir2 = vec2(-0.02, 0.03);

    // Две карты нормалей движутся в разных направлениях
    vec2 uv1 = vUv * 8.0 + uTime * flowDir1;
    vec2 uv2 = vUv * 12.0 + uTime * flowDir2;

    vec3 normal1 = unpackNormal(texture2D(uNormalMap1, uv1));
    vec3 normal2 = unpackNormal(texture2D(uNormalMap2, uv2));

    // Смешиваем нормали для мелкой ряби
    vec3 detailNormal = normalize(normal1 + normal2);

    // Комбинируем с геометрической нормалью волн
    vec3 finalNormal = normalize(vWorldNormal + detailNormal * 0.3);

    // ========================================
    // 2. FRESNEL EFFECT
    // ========================================
    vec3 viewDir = normalize(cameraPosition - vWorldPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, finalNormal), 0.0), uFresnelPower);
    fresnel = clamp(fresnel, 0.02, 0.98); // Минимум отражения даже сверху

    // ========================================
    // 3. REFRACTION (Преломление)
    // ========================================
    // Искажение UV на основе нормалей
    vec2 refractionOffset = finalNormal.xz * uRefractionStrength;
    vec2 refractionUV = screenUV + refractionOffset;
    refractionUV = clamp(refractionUV, 0.001, 0.999);

    vec3 refractionColor = texture2D(uRefractionMap, refractionUV).rgb;

    // Глубина воды для затухания цвета
    float sceneDepth = linearizeDepth(texture2D(uDepthMap, screenUV).r);
    float waterDepth = linearizeDepth(gl_FragCoord.z);
    float depthDifference = sceneDepth - waterDepth;

    // Затухание цвета с глубиной (Beer's Law)
    float depthFactor = 1.0 - exp(-depthDifference * 0.15);
    depthFactor = clamp(depthFactor, 0.0, 1.0);

    // Смешиваем refraction с цветом воды на глубине
    vec3 underwaterColor = mix(refractionColor, uWaterColorDeep, depthFactor * 0.8);

    // ========================================
    // 4. REFLECTION (Отражение)
    // ========================================
    vec2 reflectionOffset = finalNormal.xz * 0.05;
    vec2 reflectionUV = vec2(screenUV.x, 1.0 - screenUV.y) + reflectionOffset;
    reflectionUV = clamp(reflectionUV, 0.001, 0.999);

    vec3 reflectionColor = texture2D(uReflectionMap, reflectionUV).rgb;

    // ========================================
    // 5. SPECULAR HIGHLIGHTS (Блики солнца)
    // ========================================
    vec3 halfVector = normalize(viewDir + uSunDirection);
    float specular = pow(max(dot(finalNormal, halfVector), 0.0), 256.0);
    specular *= 2.0; // Интенсивность
    vec3 specularColor = uSunColor * specular;

    // ========================================
    // 6. SHORE FOAM (Пена у берега)
    // ========================================
    // Пена появляется где вода мелкая
    float foamLine = 1.0 - smoothstep(0.0, uFoamThreshold, depthDifference);

    // Анимированный шум для пены
    vec2 foamUV = vWorldPosition.xz * 0.5 + uTime * 0.1;
    float foamNoise = fbm(foamUV * 3.0);
    float foamPattern = fbm(foamUV * 8.0 + uTime * 0.5);

    // Волновая пена на гребнях
    float waveCrest = smoothstep(0.3, 0.5, vWorldPosition.y - position.y + 0.5);
    float foam = foamLine * foamNoise * uFoamIntensity;
    foam += waveCrest * foamPattern * 0.3;
    foam = clamp(foam, 0.0, 1.0);

    vec3 foamColor = vec3(0.95, 0.98, 1.0);

    // ========================================
    // 7. CAUSTICS (Каустика на дне)
    // ========================================
    vec2 causticUV = vWorldPosition.xz * 0.3;
    float caustic1 = noise(causticUV + uTime * 0.5);
    float caustic2 = noise(causticUV * 1.5 - uTime * 0.3);
    float caustics = (caustic1 * caustic2) * 2.0;
    caustics = pow(caustics, 2.0) * (1.0 - depthFactor);

    underwaterColor += caustics * uSunColor * 0.3;

    // ========================================
    // 8. FINAL COMPOSITION
    // ========================================
    // Смешиваем refraction и reflection по Френелю
    vec3 waterSurface = mix(underwaterColor, reflectionColor, fresnel);

    // Добавляем базовый цвет воды
    waterSurface = mix(waterSurface, uWaterColor, 0.1);

    // Добавляем блики
    waterSurface += specularColor;

    // Добавляем пену
    waterSurface = mix(waterSurface, foamColor, foam);

    // Subsurface scattering эффект (свет сквозь волну)
    float sss = pow(max(dot(viewDir, -uSunDirection), 0.0), 4.0);
    sss *= pow(max(dot(finalNormal, uSunDirection), 0.0), 0.5);
    waterSurface += uSunColor * sss * 0.2 * (1.0 - fresnel);

    // Прозрачность зависит от глубины и Френеля
    float alpha = mix(0.7, 0.95, fresnel);
    alpha = mix(alpha, 1.0, foam); // Пена непрозрачна

    gl_FragColor = vec4(waterSurface, alpha);
}
`;

// ============================================================
// WATER CLASS
// ============================================================
export class RealisticWater {
    constructor(renderer, scene, camera, options = {}) {
        this.renderer = renderer;
        this.scene = scene;
        this.camera = camera;

        // Настройки по умолчанию
        this.options = {
            width: options.width || 100,
            height: options.height || 100,
            segments: options.segments || 128,
            waterColor: options.waterColor || new THREE.Color(0x0077be),
            waterColorDeep: options.waterColorDeep || new THREE.Color(0x001e3d),
            sunDirection: options.sunDirection || new THREE.Vector3(0.5, 0.8, 0.3).normalize(),
            sunColor: options.sunColor || new THREE.Color(0xffffcc),
            ...options
        };

        this.clock = new THREE.Clock();

        this._initRenderTargets();
        this._initMaterial();
        this._initMesh();
        this._initReflectionCamera();
    }

    // ========================================
    // RENDER TARGETS для Reflection/Refraction
    // ========================================
    _initRenderTargets() {
        const pixelRatio = this.renderer.getPixelRatio();
        const width = window.innerWidth * pixelRatio;
        const height = window.innerHeight * pixelRatio;

        // Render target для refraction (что под водой)
        this.refractionRT = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat,
            encoding: THREE.sRGBEncoding
        });

        // Render target для reflection (отражение неба)
        this.reflectionRT = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat,
            encoding: THREE.sRGBEncoding
        });

        // Depth buffer для расчета глубины воды
        this.depthRT = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            format: THREE.RGBAFormat,
            type: THREE.FloatType
        });
    }

    // ========================================
    // MATERIAL с шейдерами
    // ========================================
    _initMaterial() {
        // Создаем процедурные Normal Maps
        const normalMap1 = this._createWaterNormalMap(256, 0.5);
        const normalMap2 = this._createWaterNormalMap(256, 0.3);

        this.uniforms = {
            uTime: { value: 0 },

            // Gerstner Waves параметры (direction.xy, steepness, wavelength)
            uWave1: { value: new THREE.Vector4(1.0, 0.0, 0.25, 20.0) },
            uWave2: { value: new THREE.Vector4(0.0, 1.0, 0.15, 15.0) },
            uWave3: { value: new THREE.Vector4(1.0, 1.0, 0.1, 10.0) },
            uWave4: { value: new THREE.Vector4(-0.5, 0.7, 0.08, 8.0) },

            // Освещение
            uSunDirection: { value: this.options.sunDirection },
            uSunColor: { value: this.options.sunColor },

            // Цвета воды
            uWaterColor: { value: this.options.waterColor },
            uWaterColorDeep: { value: this.options.waterColorDeep },

            // Эффекты
            uFresnelPower: { value: 3.0 },
            uRefractionStrength: { value: 0.02 },
            uFoamThreshold: { value: 1.5 },
            uFoamIntensity: { value: 0.8 },

            // Текстуры
            uNormalMap1: { value: normalMap1 },
            uNormalMap2: { value: normalMap2 },
            uRefractionMap: { value: this.refractionRT.texture },
            uReflectionMap: { value: this.reflectionRT.texture },
            uDepthMap: { value: this.depthRT.texture },
            uFoamTexture: { value: this._createFoamTexture() },

            // Camera
            uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
            uNear: { value: this.camera.near },
            uFar: { value: this.camera.far },
            uProjectionMatrixInverse: { value: new THREE.Matrix4() },
            uViewMatrixInverse: { value: new THREE.Matrix4() }
        };

        this.material = new THREE.ShaderMaterial({
            vertexShader: waterVertexShader,
            fragmentShader: waterFragmentShader,
            uniforms: this.uniforms,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });
    }

    // ========================================
    // MESH
    // ========================================
    _initMesh() {
        const geometry = new THREE.PlaneGeometry(
            this.options.width,
            this.options.height,
            this.options.segments,
            this.options.segments
        );
        geometry.rotateX(-Math.PI / 2);

        this.mesh = new THREE.Mesh(geometry, this.material);
        this.mesh.frustumCulled = false;
    }

    // ========================================
    // REFLECTION CAMERA (Planar Reflection)
    // ========================================
    _initReflectionCamera() {
        this.reflectionCamera = this.camera.clone();
        this.reflectionPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
        this.reflectionMatrix = new THREE.Matrix4();
    }

    // ========================================
    // PROCEDURAL NORMAL MAP
    // ========================================
    _createWaterNormalMap(size, scale) {
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');

        const imageData = ctx.createImageData(size, size);
        const data = imageData.data;

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const i = (y * size + x) * 4;

                // Генерируем высоты через несколько октав шума
                const nx = x / size;
                const ny = y / size;

                const h00 = this._noise2D(nx * 8, ny * 8) * scale;
                const h10 = this._noise2D((nx + 1/size) * 8, ny * 8) * scale;
                const h01 = this._noise2D(nx * 8, (ny + 1/size) * 8) * scale;

                // Вычисляем нормаль из градиента высот
                const dx = (h10 - h00) * 2.0;
                const dy = (h01 - h00) * 2.0;

                let nx2 = -dx;
                let ny2 = -dy;
                let nz = 1.0;

                // Нормализуем
                const len = Math.sqrt(nx2*nx2 + ny2*ny2 + nz*nz);
                nx2 /= len;
                ny2 /= len;
                nz /= len;

                // Упаковываем в [0, 255]
                data[i]     = (nx2 * 0.5 + 0.5) * 255;
                data[i + 1] = (ny2 * 0.5 + 0.5) * 255;
                data[i + 2] = (nz * 0.5 + 0.5) * 255;
                data[i + 3] = 255;
            }
        }

        ctx.putImageData(imageData, 0, 0);

        const texture = new THREE.CanvasTexture(canvas);
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        texture.magFilter = THREE.LinearFilter;

        return texture;
    }

    // ========================================
    // FOAM TEXTURE
    // ========================================
    _createFoamTexture() {
        const size = 256;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, size, size);

        // Рисуем случайные круги для пены
        for (let i = 0; i < 500; i++) {
            const x = Math.random() * size;
            const y = Math.random() * size;
            const r = Math.random() * 8 + 2;
            const alpha = Math.random() * 0.5 + 0.3;

            ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fill();
        }

        const texture = new THREE.CanvasTexture(canvas);
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;

        return texture;
    }

    // ========================================
    // SIMPLE 2D NOISE
    // ========================================
    _noise2D(x, y) {
        const n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
        return n - Math.floor(n);
    }

    // ========================================
    // UPDATE (вызывать каждый кадр)
    // ========================================
    update() {
        const delta = this.clock.getDelta();
        const elapsed = this.clock.getElapsedTime();

        // Обновляем время в шейдере
        this.uniforms.uTime.value = elapsed;

        // Обновляем матрицы камеры
        this.uniforms.uProjectionMatrixInverse.value.copy(this.camera.projectionMatrixInverse);
        this.uniforms.uViewMatrixInverse.value.copy(this.camera.matrixWorld);

        // Рендерим refraction и reflection
        this._renderRefraction();
        this._renderReflection();
    }

    // ========================================
    // RENDER REFRACTION
    // ========================================
    _renderRefraction() {
        this.mesh.visible = false;

        this.renderer.setRenderTarget(this.refractionRT);
        this.renderer.render(this.scene, this.camera);

        // Также рендерим depth
        this.renderer.setRenderTarget(this.depthRT);
        this.renderer.render(this.scene, this.camera);

        this.renderer.setRenderTarget(null);
        this.mesh.visible = true;
    }

    // ========================================
    // RENDER REFLECTION (Planar)
    // ========================================
    _renderReflection() {
        this.mesh.visible = false;

        // Позиция воды
        const waterY = this.mesh.position.y;

        // Отражаем камеру относительно плоскости воды
        this.reflectionCamera.copy(this.camera);
        this.reflectionCamera.position.y = -this.camera.position.y + 2 * waterY;
        this.reflectionCamera.up.set(0, -1, 0);
        this.reflectionCamera.lookAt(
            this.camera.position.x + this.camera.getWorldDirection(new THREE.Vector3()).x * 100,
            -this.camera.position.y + 2 * waterY,
            this.camera.position.z + this.camera.getWorldDirection(new THREE.Vector3()).z * 100
        );
        this.reflectionCamera.updateProjectionMatrix();

        // Clip plane чтобы не рендерить то что под водой
        const clipPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -waterY);
        this.renderer.clippingPlanes = [clipPlane];

        this.renderer.setRenderTarget(this.reflectionRT);
        this.renderer.render(this.scene, this.reflectionCamera);
        this.renderer.setRenderTarget(null);

        this.renderer.clippingPlanes = [];
        this.mesh.visible = true;
    }

    // ========================================
    // RESIZE HANDLER
    // ========================================
    resize(width, height) {
        const pixelRatio = this.renderer.getPixelRatio();

        this.refractionRT.setSize(width * pixelRatio, height * pixelRatio);
        this.reflectionRT.setSize(width * pixelRatio, height * pixelRatio);
        this.depthRT.setSize(width * pixelRatio, height * pixelRatio);

        this.uniforms.uResolution.value.set(width, height);
    }

    // ========================================
    // SETTERS для параметров
    // ========================================
    setWaveParams(waveIndex, direction, steepness, wavelength) {
        const waveName = `uWave${waveIndex}`;
        if (this.uniforms[waveName]) {
            this.uniforms[waveName].value.set(direction.x, direction.y, steepness, wavelength);
        }
    }

    setSunDirection(direction) {
        this.uniforms.uSunDirection.value.copy(direction).normalize();
    }

    setWaterColor(color, deepColor) {
        this.uniforms.uWaterColor.value.copy(color);
        if (deepColor) this.uniforms.uWaterColorDeep.value.copy(deepColor);
    }

    setFoamParams(threshold, intensity) {
        this.uniforms.uFoamThreshold.value = threshold;
        this.uniforms.uFoamIntensity.value = intensity;
    }

    // ========================================
    // DISPOSE
    // ========================================
    dispose() {
        this.refractionRT.dispose();
        this.reflectionRT.dispose();
        this.depthRT.dispose();
        this.material.dispose();
        this.mesh.geometry.dispose();
    }
}

// ============================================================
// SIMPLIFIED WATER (для интеграции в существующий проект)
// ============================================================
export function createSimpleRealisticWater(options = {}) {
    const {
        width = 30,
        height = 30,
        segments = 64,
        color = 0x0088aa,
        position = new THREE.Vector3(0, 0, 0)
    } = options;

    const geometry = new THREE.PlaneGeometry(width, height, segments, segments);
    geometry.rotateX(-Math.PI / 2);

    const material = new THREE.ShaderMaterial({
        uniforms: {
            uTime: { value: 0 },
            uWaterColor: { value: new THREE.Color(color) },
            uWaterColorDeep: { value: new THREE.Color(0x001830) },
            uSunDirection: { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() },
            uSunColor: { value: new THREE.Color(0xffffee) }
        },
        vertexShader: `
            uniform float uTime;
            varying vec3 vWorldPosition;
            varying vec3 vNormal;
            varying vec2 vUv;

            // Gerstner Wave
            vec3 gerstnerWave(vec2 dir, float steepness, float wavelength, vec3 p, inout vec3 tangent, inout vec3 binormal) {
                float k = 6.28318 / wavelength;
                float c = sqrt(9.8 / k);
                vec2 d = normalize(dir);
                float f = k * (dot(d, p.xz) - c * uTime);
                float a = steepness / k;

                tangent += vec3(-d.x * d.x * steepness * sin(f), d.x * steepness * cos(f), -d.x * d.y * steepness * sin(f));
                binormal += vec3(-d.x * d.y * steepness * sin(f), d.y * steepness * cos(f), -d.y * d.y * steepness * sin(f));

                return vec3(d.x * a * cos(f), a * sin(f), d.y * a * cos(f));
            }

            void main() {
                vUv = uv;
                vec3 pos = position;

                vec3 tangent = vec3(1, 0, 0);
                vec3 binormal = vec3(0, 0, 1);

                pos += gerstnerWave(vec2(1, 0), 0.15, 8.0, pos, tangent, binormal);
                pos += gerstnerWave(vec2(0, 1), 0.12, 6.0, pos, tangent, binormal);
                pos += gerstnerWave(vec2(1, 1), 0.08, 4.0, pos, tangent, binormal);
                pos += gerstnerWave(vec2(-0.5, 0.7), 0.05, 3.0, pos, tangent, binormal);

                vNormal = normalize(cross(binormal, tangent));
                vWorldPosition = (modelMatrix * vec4(pos, 1.0)).xyz;

                gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
            }
        `,
        fragmentShader: `
            uniform float uTime;
            uniform vec3 uWaterColor;
            uniform vec3 uWaterColorDeep;
            uniform vec3 uSunDirection;
            uniform vec3 uSunColor;

            varying vec3 vWorldPosition;
            varying vec3 vNormal;
            varying vec2 vUv;

            float noise(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec3 viewDir = normalize(cameraPosition - vWorldPosition);
                vec3 normal = normalize(vNormal);

                // Normal perturbation для мелкой ряби
                vec2 uv1 = vWorldPosition.xz * 0.5 + uTime * 0.02;
                vec2 uv2 = vWorldPosition.xz * 0.8 - uTime * 0.015;
                float ripple1 = noise(uv1 * 10.0);
                float ripple2 = noise(uv2 * 15.0);
                normal.xz += (ripple1 - 0.5 + ripple2 - 0.5) * 0.1;
                normal = normalize(normal);

                // Fresnel
                float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), 3.0);
                fresnel = clamp(fresnel, 0.05, 0.95);

                // Base color
                vec3 color = mix(uWaterColorDeep, uWaterColor, 0.5);

                // Sky reflection (simplified)
                vec3 reflectDir = reflect(-viewDir, normal);
                vec3 skyColor = mix(vec3(0.4, 0.6, 0.9), vec3(0.8, 0.9, 1.0), reflectDir.y * 0.5 + 0.5);

                color = mix(color, skyColor, fresnel);

                // Specular
                vec3 halfVec = normalize(viewDir + uSunDirection);
                float spec = pow(max(dot(normal, halfVec), 0.0), 128.0);
                color += uSunColor * spec * 1.5;

                // Sparkles
                float sparkle = noise(vWorldPosition.xz * 5.0 + uTime);
                sparkle = pow(sparkle, 20.0) * 2.0;
                color += vec3(sparkle);

                // Foam на гребнях
                float foam = smoothstep(0.2, 0.4, vWorldPosition.y - position.y + 0.3);
                foam *= noise(vWorldPosition.xz * 3.0 + uTime * 0.5);
                color = mix(color, vec3(1.0), foam * 0.5);

                // SSS эффект
                float sss = pow(max(dot(viewDir, -uSunDirection), 0.0), 3.0);
                color += vec3(0.0, 0.3, 0.2) * sss * 0.3;

                gl_FragColor = vec4(color, 0.9 - fresnel * 0.2);
            }
        `,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);

    // Функция обновления
    mesh.update = function(delta) {
        material.uniforms.uTime.value += delta || 0.016;
    };

    return mesh;
}

export default RealisticWater;
