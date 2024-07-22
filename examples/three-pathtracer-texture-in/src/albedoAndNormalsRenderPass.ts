import * as THREE from 'three';

export class AlbedoNormalPass {
    private albedoNormalMaterial: THREE.ShaderMaterial;
    private originalMaterials: Map<THREE.Object3D, THREE.Material | THREE.Material[]>;
    private albedoMaterials: Map<THREE.Object3D, THREE.Material | THREE.Material[]>;
    private samples: number;
    private useWorldSpaceNormals: boolean;

    constructor(samples = 4, useWorldSpaceNormals = true) {
        this.originalMaterials = new Map();
        this.albedoMaterials = new Map();
        this.samples = samples;
        this.useWorldSpaceNormals = useWorldSpaceNormals;

        this.albedoNormalMaterial = new THREE.RawShaderMaterial({
            vertexShader: `
        in vec3 position;
        in vec2 uv;
        in vec3 normal;

        out vec2 vUv;
        out vec3 vNormal;
        out vec3 vWorldPosition;

        uniform mat4 modelMatrix;
        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;
        uniform mat3 normalMatrix;

        void main() {
            vUv = uv;
            vNormal = normalMatrix * normal;
            vec4 worldPosition = modelMatrix * vec4(position, 1.0);
            vWorldPosition = worldPosition.xyz;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
            fragmentShader: `
        precision highp float;
        precision highp int;
        
        layout(location = 0) out vec4 gAlbedo;
        layout(location = 1) out vec4 gNormal;

        uniform vec3 color;
        uniform sampler2D map;
        uniform bool useMap;
        uniform vec2 resolution;
        uniform int samples;
        uniform bool useWorldSpaceNormals;

        in vec2 vUv;
        in vec3 vNormal;
        in vec3 vWorldPosition;

        vec3 getAlbedo(vec2 uv) {
            if (useMap) {
                return texture(map, uv).rgb;
            }
            return color;
        }

        void main() {
            vec2 pixelSize = 1.0 / resolution;
            vec3 albedoSum = vec3(0.0);
            vec3 normalSum = vec3(0.0);

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < samples; j++) {
                    vec2 offset = vec2(float(i), float(j)) / float(samples) - 0.5;
                    vec2 sampleUv = vUv + offset * pixelSize;
                    
                    // Sample albedo
                    vec3 albedo = getAlbedo(sampleUv);
                    albedoSum += albedo;

                    // Sample normal
                    vec3 normal;
                    if (useWorldSpaceNormals) {
                        vec3 dPdx = dFdx(vWorldPosition);
                        vec3 dPdy = dFdy(vWorldPosition);
                        normal = normalize(cross(dPdx, dPdy));
                    } else {
                        normal = normalize(vNormal);
                    }
                    normalSum += normal;
                }
            }

            // Average the samples
            float sampleCount = float(samples * samples);
            vec3 finalAlbedo = albedoSum / sampleCount;
            vec3 finalNormal = normalize(normalSum / sampleCount);

            // Clamp albedo to [0, 1] range
            finalAlbedo = clamp(finalAlbedo, 0.0, 1.0);

            // Output to G-Buffer
            gAlbedo = vec4(finalAlbedo, 1.0);
            gNormal = vec4(finalNormal, 1.0);  // Output normals directly in [-1, 1] range
        }
      `,
            uniforms: {
                color: { value: new THREE.Color(1, 1, 1) },
                map: { value: null },
                useMap: { value: false },
                resolution: { value: new THREE.Vector2() },
                samples: { value: this.samples },
                useWorldSpaceNormals: { value: this.useWorldSpaceNormals }
            },
            glslVersion: THREE.GLSL3
        });
    }

    render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, target: THREE.WebGLRenderTarget) {
        this.swapMaterials(scene);

        const oldRenderTarget = renderer.getRenderTarget();
        renderer.setRenderTarget(target);

        // Update resolution uniform
        const pixelRatio = renderer.getPixelRatio();
        const width = target.width * pixelRatio;
        const height = target.height * pixelRatio;
        this.albedoMaterials.forEach((material) => {
            if (material instanceof THREE.ShaderMaterial) {
                material.uniforms.resolution.value.set(width, height);
            }
        });

        renderer.render(scene, camera);
        renderer.setRenderTarget(oldRenderTarget);

        this.restoreMaterials();
    }

    private swapMaterials(object: THREE.Object3D) {
        if (object instanceof THREE.Mesh && object.material) {
            if (!this.originalMaterials.has(object))
                this.originalMaterials.set(object, object.material);

            if (this.albedoMaterials.has(object)) {
                object.material = this.albedoMaterials.get(object)!;
                return;
            }
            const material = object.material as THREE.MeshStandardMaterial;
            const newAlbedoMaterial = this.albedoNormalMaterial.clone();
            if (material.color) newAlbedoMaterial.uniforms.color.value.copy(material.color);
            newAlbedoMaterial.uniforms.map.value = material.map;
            newAlbedoMaterial.uniforms.useMap.value = !!material.map;
            this.albedoMaterials.set(object, newAlbedoMaterial);
            object.material = newAlbedoMaterial;
        }

        object.children.forEach(child => this.swapMaterials(child));
    }

    private restoreMaterials() {
        this.originalMaterials.forEach((material, object) => {
            if (object instanceof THREE.Mesh) {
                object.material = material;
            }
        });
        this.originalMaterials.clear();
    }

    setUseWorldSpaceNormals(value: boolean) {
        this.useWorldSpaceNormals = value;
        this.albedoMaterials.forEach((material) => {
            if (material instanceof THREE.ShaderMaterial) {
                material.uniforms.useWorldSpaceNormals.value = value;
            }
        });
    }
}