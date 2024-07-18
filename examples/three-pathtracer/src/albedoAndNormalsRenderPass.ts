import * as THREE from 'three';

export class AlbedoNormalPass {
    private albedoNormalMaterial: THREE.ShaderMaterial;
    private originalMaterials: Map<THREE.Object3D, THREE.Material | THREE.Material[]>;
    private albedoMaterials: Map<THREE.Object3D, THREE.Material | THREE.Material[]>;
    constructor() {
        this.originalMaterials = new Map();
        this.albedoMaterials = new Map();

        this.albedoNormalMaterial = new THREE.RawShaderMaterial({
            vertexShader: `
        in vec3 position;
        in vec2 uv;
        in vec3 normal;

        out vec2 vUv;
        out vec3 vNormal;

        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;
        uniform mat3 normalMatrix;

        
        void main() {
          vUv = uv;
          vec3 transformedNormal = normalMatrix * normal;
          vNormal = normalize(transformedNormal);

          vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
          gl_Position = projectionMatrix * mvPosition;
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
      in vec2 vUv;
      in vec3 vNormal;
  
      void main() {
        // Albedo output
        gAlbedo = vec4(color, 1.0);
        if (useMap) {
          gAlbedo = texture(map, vUv);
        }
  
        // Normal output
        gNormal = vec4(normalize(vNormal) * 0.5 + 0.5, 1.0);
      }
    `,
            uniforms: {
                color: { value: new THREE.Color(1, 1, 1) },
                map: { value: null },
                useMap: { value: false }
            },
            glslVersion: THREE.GLSL3
        });
    }

    render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, target: THREE.WebGLRenderTarget) {
        this.swapMaterials(scene);

        const oldRenderTarget = renderer.getRenderTarget();
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
        renderer.setRenderTarget(oldRenderTarget);

        this.restoreMaterials();
    }

    private swapMaterials(object: THREE.Object3D) {
        if (object && (object as THREE.Mesh).material) {
            // see if we've done this before
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