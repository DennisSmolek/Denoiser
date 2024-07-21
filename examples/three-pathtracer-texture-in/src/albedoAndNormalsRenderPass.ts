import * as THREE from 'three';

export class AlbedoNormalPass {
    private albedoNormalMaterial: THREE.ShaderMaterial;
    private originalMaterials: Map<THREE.Object3D, THREE.Material | THREE.Material[]>;
    private albedoMaterials: Map<THREE.Object3D, THREE.Material | THREE.Material[]>;
    private samples: number;

    constructor(samples = 4) {
        this.originalMaterials = new Map();
        this.albedoMaterials = new Map();
        this.samples = samples;

        this.albedoNormalMaterial = new THREE.RawShaderMaterial({
            vertexShader: `
        in vec3 position;
        in vec2 uv;
        in vec3 normal;

        out vec2 vUv;
        out vec3 vNormal;
        out vec4 vPosition;

        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;
        uniform mat3 normalMatrix;

        void main() {
          vUv = uv;
          vec3 transformedNormal = normalMatrix * normal;
          vNormal = normalize(transformedNormal);

          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vPosition = projectionMatrix * mvPosition;
          gl_Position = vPosition;
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

      in vec2 vUv;
      in vec3 vNormal;
      in vec4 vPosition;

      vec4 getAlbedo(vec2 uv) {
          if (useMap) {
              return texture(map, uv);
          }
          return vec4(color, 1.0);
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
                  albedoSum += getAlbedo(sampleUv).rgb;

                  // Sample normal
                  vec3 ddx = dFdx(vPosition.xyz);
                  vec3 ddy = dFdy(vPosition.xyz);
                  vec3 sampleNormal = normalize(cross(ddx, ddy));
                  normalSum += sampleNormal;
              }
          }

          // Average the samples
          float sampleCount = float(samples * samples);
          gAlbedo = vec4(albedoSum / sampleCount, 1.0);
          gNormal = vec4(normalize(normalSum) * 0.5 + 0.5, 1.0);
      }
    `,
            uniforms: {
                color: { value: new THREE.Color(1, 1, 1) },
                map: { value: null },
                useMap: { value: false },
                resolution: { value: new THREE.Vector2() },
                samples: { value: this.samples }
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
        if (object && (object as THREE.Mesh).material) {
            if (!this.originalMaterials.has(object))
                this.originalMaterials.set(object, (object as THREE.Mesh).material);

            if (this.albedoMaterials.has(object)) {
                (object as THREE.Mesh).material = this.albedoMaterials.get(object)!;
                return;
            }
            const material = (object as THREE.Mesh).material as THREE.MeshStandardMaterial;
            const newAlbedoMaterial = this.albedoNormalMaterial.clone();
            if (material.color) newAlbedoMaterial.uniforms.color.value.copy(material.color);
            newAlbedoMaterial.uniforms.map.value = material.map;
            newAlbedoMaterial.uniforms.useMap.value = !!material.map;
            this.albedoMaterials.set(object, newAlbedoMaterial);
            (object as THREE.Mesh).material = newAlbedoMaterial;
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
}