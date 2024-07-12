import * as THREE from 'three';

export class Renderer {
    private canvas: HTMLCanvasElement;
    private renderer: THREE.WebGLRenderer;
    private scene: THREE.Scene;
    private camera: THREE.OrthographicCamera;
    private textureLoader: THREE.TextureLoader;
    private fullscreenQuad: THREE.Mesh;

    private backendListeners = new Set<() => void>();
    private _ready = false;

    private renderTargetHolder?: THREE.WebGLRenderTarget;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.renderer = new THREE.WebGLRenderer({ canvas });
        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        this.textureLoader = new THREE.TextureLoader();

        this.fullscreenQuad = new THREE.Mesh(
            new THREE.PlaneGeometry(2, 2),
            new THREE.ShaderMaterial({
                vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = vec4(position.xy, 0.0, 1.0);
          }
        `,
                fragmentShader: `
          uniform sampler2D tDiffuse;
          varying vec2 vUv;
          void main() {
            gl_FragColor = texture2D(tDiffuse, vUv);
          }
        `,
                uniforms: {
                    tDiffuse: { value: null }
                }
            })
        );
        this.scene.add(this.fullscreenQuad);

        // Set initial size
        this.setSize(canvas.width, canvas.height);
        this.ready = true;
    }

    get ready() {
        return this._ready;
    }

    set ready(value: boolean) {
        this._ready = value;
        if (value) {
            for (const listener of this.backendListeners) {
                listener();
            }
        }
    }

    // backend listener when renderer is ready
    onReady(callback: () => void) {
        if (this.ready) callback();
        else this.backendListeners.add(callback);
        // return removal function
        return () => this.backendListeners.delete(callback);
    }

    setSize(width: number, height: number): void {
        this.canvas.width = width;
        this.canvas.height = height;
        this.renderer.setSize(width, height);
    }

    async loadTexture(url: string): Promise<THREE.Texture> {
        return new Promise((resolve, reject) => {
            this.textureLoader.load(url, resolve, undefined, reject);
        });
    }

    resetState(): void {
        const gl = this.renderer.getContext();
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindRenderbuffer(gl.RENDERBUFFER, null);
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.disable(gl.BLEND);
        gl.disable(gl.CULL_FACE);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.DITHER);
        gl.disable(gl.SCISSOR_TEST);
        gl.disable(gl.STENCIL_TEST);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        gl.useProgram(null);
    }

    renderToWebGLTexture(threeTexture: THREE.Texture): THREE.WebGLRenderTarget {
        this.resetState();

        const renderTarget = new THREE.WebGLRenderTarget(
            this.canvas.width,
            this.canvas.height,
            {
                format: THREE.RGBAFormat,
                type: THREE.UnsignedByteType
            }
        );

        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = threeTexture;
        this.renderer.setRenderTarget(renderTarget);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(null);

        this.resetState();

        this.renderTargetHolder = renderTarget;

        const textureProps = this.renderer.properties.get(renderTarget.texture);
        const outTexture = textureProps.__webglTexture

        return outTexture;
    }

    //merge and render
    mergeThenRender(webglTexture: WebGLTexture) {
        this.resetState();
        // load the old renderTarget
        const renderTarget = this.renderTargetHolder!;
        //put the webglTexture into the renderTarget
        const textureProps = this.renderer.properties.get(renderTarget.texture);
        textureProps.__webglTexture = webglTexture;


        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = renderTarget.texture;
        this.renderer.resetState();
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.scene, this.camera);
        this.renderer.resetState();
        this.resetState();

    }

    async demonstrateWorkflow(inputUrl: string): Promise<void> {
        // Step 1: Load ThreeJS texture from URL
        const threeTexture = await this.loadTexture(inputUrl);

        // Step 2: Render ThreeJS texture to WebGLTexture
        const webGLTexture = this.renderToWebGLTexture(threeTexture);

        // Step 3: Pass webGLTexture to other application (simulated)
        //const processedWebGLTexture = this.simulateOtherApplication(webGLTexture as WebGLTexture);

        // Step 4: Render processed WebGLTexture to canvas
        this.mergeThenRender(webGLTexture);
    }

    // Simulated other application processing
    /* private simulateOtherApplication(inputTexture: THREE.Texture): THREE.Texture {
         // In a real scenario, this would be replaced by actual processing in another application
         console.log("Processing texture in other application...");
         return inputTexture;
     }*/
}

export default Renderer;