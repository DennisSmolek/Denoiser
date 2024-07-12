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
    renderToWebGLTexture1(threeTexture: THREE.Texture): WebGLTexture {
        this.resetState();

        const renderTarget = new THREE.WebGLRenderTarget(
            this.canvas.width,
            this.canvas.height,
            {
                format: THREE.RGBAFormat,
                type: THREE.UnsignedByteType
            }
        );
        // Access the internal WebGL framebuffer
        const internalFrameBuffer = renderTarget.__webglFramebuffer; // This is how you might access it, but it's not recommended
        console.log('Rendertarget', renderTarget);
        const props = this.renderer.properties.get(renderTarget);
        console.log('render target properties', props);
        // can I use this props value to access the render target?

        console.log('new rt framebuffer:', internalFrameBuffer);

        const gl = this.renderer.getContext();

        // Create a framebuffer
        const framebuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

        // Create a texture to render to
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.canvas.width, this.canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        // Attach the texture to the framebuffer
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

        // Render the scene to the framebuffer
        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = threeTexture;
        this.renderer.setRenderTarget(null); // Ensure we're rendering to the framebuffer, not a Three.js RenderTarget
        this.renderer.render(this.scene, this.camera);

        // Clean up
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(framebuffer);

        this.resetState();

        return texture;
    }
    renderToRenderTarget(threeTexture: THREE.Texture): THREE.WebGLRenderTarget {
        this.resetState();

        const renderTarget = new THREE.WebGLRenderTarget(
            this.canvas.width,
            this.canvas.height,
            {
                format: THREE.RGBAFormat,
                type: THREE.UnsignedByteType
            }
        );
        const preTarget = this.renderer.properties.get(renderTarget).__webglFramebuffer;
        console.log('preTarget:', preTarget);

        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = threeTexture;
        this.renderer.setRenderTarget(renderTarget);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(null);

        this.resetState();

        console.log('Render Target.texture.texture:', renderTarget.texture);
        console.log('RenderTarget.__webglTexture:', (renderTarget.texture as any).__webglTexture);
        const props = this.renderer.properties.get(renderTarget);
        console.log('RenderTarget properties:', props);
        const postTarget = this.renderer.properties.get(renderTarget).__webglFramebuffer;
        console.log('postTarget:', postTarget);

        const textureProps = this.renderer.properties.get(renderTarget.texture);
        console.log('Texture properties:', textureProps);


        return renderTarget;
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
        const preTarget = this.renderer.properties.get(renderTarget).__webglFramebuffer;
        console.log('preTarget:', preTarget);

        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = threeTexture;
        this.renderer.setRenderTarget(renderTarget);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(null);

        this.resetState();

        this.renderTargetHolder = renderTarget;

        const textureProps = this.renderer.properties.get(renderTarget.texture);
        console.log('Texture properties:', textureProps);
        const outTexture = textureProps.__webglTexture

        return outTexture;
    }

    //merge and render
    mergeTextureAndTargetThenRender(webglTexture: WebGLTexture) {
        this.resetState();
        // load the old renderTarget
        const renderTarget = this.renderTargetHolder!;
        //put the webglTexture into the renderTarget
        const textureProps = this.renderer.properties.get(renderTarget.texture);
        textureProps.__webglTexture = webglTexture;


        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = renderTarget.texture;
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.scene, this.camera);

        this.resetState();

    }

    renderRenderTargetToCanvas(renderTarget: THREE.WebGLRenderTarget): void {
        this.resetState();

        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = renderTarget.texture;
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.scene, this.camera);

        this.resetState();
    }

    renderWebGLTextureToCanvasOld(webGLTexture: WebGLTexture): void {
        this.resetState();

        const gl = this.renderer.getContext();

        // Create a temporary Three.js texture from the WebGLTexture
        const tempTexture = new THREE.Texture();
        tempTexture.image = { width: this.canvas.width, height: this.canvas.height };
        tempTexture.format = THREE.RGBAFormat;
        tempTexture.type = THREE.UnsignedByteType;
        //(tempTexture as any).__webglTexture = webGLTexture;
        this.renderer.properties.get(tempTexture).__webglTexture = webGLTexture;
        tempTexture.needsUpdate = true;
        console.log('Temp texture props', this.renderer.properties.get(tempTexture));

        // Render the texture to the canvas
        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = tempTexture;
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.scene, this.camera);

        this.resetState();
    }

    renderWebGLTextureToCanvas3(webGLTexture: WebGLTexture): void {
        const gl = this.renderer.getContext();

        // Create a new THREE.Texture that wraps the WebGLTexture
        const texture = new THREE.Texture();
        texture.image = { width: this.canvas.width, height: this.canvas.height };
        texture.format = THREE.RGBAFormat;
        texture.type = THREE.UnsignedByteType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;

        // Manually update the texture
        // const oldTexture = (texture as any).__webglTexture;
        const props = this.renderer.properties.get(texture);
        props.__webglTexture = webGLTexture;
        props.__webglInit = true;
        texture.needsUpdate = true;

        console.log('Temp texture props', this.renderer.properties.get(texture));


        gl.bindTexture(gl.TEXTURE_2D, webGLTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

        // Update the shader material's texture
        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = texture;

        // Render
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.scene, this.camera);

        // Clean up
        //(texture as any).__webglTexture = oldTexture;
        texture.dispose();
    }

    renderWebGLTextureToCanvas(webGLTexture: WebGLTexture): void {
        const newTexture = new THREE.Texture();
        newTexture.image = webGLTexture;
        newTexture.needsUpdate = true;

        // Set texture parameters as needed
        newTexture.magFilter = THREE.LinearFilter;
        newTexture.wrapS = THREE.ClampToEdgeWrapping;
        newTexture.wrapT = THREE.ClampToEdgeWrapping;
        newTexture.minFilter = THREE.LinearFilter;

        // Update the shader material's texture
        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = newTexture;

        // Render
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.scene, this.camera);

    }

    async demonstrateWorkflow(inputUrl: string): Promise<void> {
        // Step 1: Load ThreeJS texture from URL
        const threeTexture = await this.loadTexture(inputUrl);

        // Step 2: Render ThreeJS texture to WebGLTexture
        const webGLTexture = this.renderToWebGLTexture(threeTexture);

        // Step 3: Pass webGLTexture to other application (simulated)
        //const processedWebGLTexture = this.simulateOtherApplication(webGLTexture);

        // Step 4: Render processed WebGLTexture to canvas
        //this.renderWebGLTextureToCanvas(processedWebGLTexture);
        this.mergeTextureAndTargetThenRender(webGLTexture);
    }

    // Simulated other application processing
    private simulateOtherApplication(inputTexture: THREE.Texture): THREE.Texture {
        // In a real scenario, this would be replaced by actual processing in another application
        console.log("Processing texture in other application...");
        return inputTexture;
    }
}

export default Renderer;

function getRTProps(renderer, rt: THREE.WebGLRenderTarget) {
    return renderer.properties.get(rt);
}
function RTHasFramebuffer(renderer, rt: THREE.WebGLRenderTarget): boolean {
    const props = getRTProps(renderer, rt);
    return props.__webglFramebuffer !== undefined;
}

function getRTFramebuffer(renderer: THREE.WebGLRenderer, rt: THREE.WebGLRenderTarget): WebGLFramebuffer {
    if (RTHasFramebuffer(renderer, rt)) {
        return getRTProps(renderer, rt).__webglFramebuffer;
    }
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    // if it doesn't have it, render a scene to it to get it
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
}