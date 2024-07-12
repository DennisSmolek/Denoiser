const debugging = false;
class WebGLStateManager {
    private gl: WebGLRenderingContext;
    private savedState: any = null;

    constructor(gl: WebGLRenderingContext) {
        this.gl = gl;
    }

    saveState() {
        const gl = this.gl;
        this.savedState = {
            program: gl.getParameter(gl.CURRENT_PROGRAM),
            activeTexture: gl.getParameter(gl.ACTIVE_TEXTURE),
            arrayBuffer: gl.getParameter(gl.ARRAY_BUFFER_BINDING),
            elementArrayBuffer: gl.getParameter(gl.ELEMENT_ARRAY_BUFFER_BINDING),
            framebuffer: gl.getParameter(gl.FRAMEBUFFER_BINDING),
            renderbuffer: gl.getParameter(gl.RENDERBUFFER_BINDING),
            texture: gl.getParameter(gl.TEXTURE_BINDING_2D),
            viewport: gl.getParameter(gl.VIEWPORT),
            blend: gl.isEnabled(gl.BLEND),
            depthTest: gl.isEnabled(gl.DEPTH_TEST),
            cullFace: gl.isEnabled(gl.CULL_FACE),
            scissorTest: gl.isEnabled(gl.SCISSOR_TEST),
            scissor: gl.getParameter(gl.SCISSOR_BOX),
            blendFunc: [
                gl.getParameter(gl.BLEND_SRC_RGB),
                gl.getParameter(gl.BLEND_DST_RGB),
                gl.getParameter(gl.BLEND_SRC_ALPHA),
                gl.getParameter(gl.BLEND_DST_ALPHA)
            ],
            blendEquation: [
                gl.getParameter(gl.BLEND_EQUATION_RGB),
                gl.getParameter(gl.BLEND_EQUATION_ALPHA)
            ],
            colorMask: gl.getParameter(gl.COLOR_WRITEMASK),
            clearColor: gl.getParameter(gl.COLOR_CLEAR_VALUE),
            pixelStoreParams: {
                unpackAlignment: gl.getParameter(gl.UNPACK_ALIGNMENT),
                unpackFlipY: gl.getParameter(gl.UNPACK_FLIP_Y_WEBGL),
                unpackPremultiplyAlpha: gl.getParameter(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL),
                unpackColorspaceConversion: gl.getParameter(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL)
            }
        };
    }

    restoreState() {
        if (!this.savedState) return;

        const gl = this.gl;
        const state = this.savedState;

        gl.useProgram(state.program);
        gl.activeTexture(state.activeTexture);
        gl.bindBuffer(gl.ARRAY_BUFFER, state.arrayBuffer);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state.elementArrayBuffer);
        gl.bindFramebuffer(gl.FRAMEBUFFER, state.framebuffer);
        gl.bindRenderbuffer(gl.RENDERBUFFER, state.renderbuffer);
        gl.bindTexture(gl.TEXTURE_2D, state.texture);
        gl.viewport(...state.viewport);
        gl.scissor(...state.scissor);

        state.blend ? gl.enable(gl.BLEND) : gl.disable(gl.BLEND);
        state.depthTest ? gl.enable(gl.DEPTH_TEST) : gl.disable(gl.DEPTH_TEST);
        state.cullFace ? gl.enable(gl.CULL_FACE) : gl.disable(gl.CULL_FACE);
        state.scissorTest ? gl.enable(gl.SCISSOR_TEST) : gl.disable(gl.SCISSOR_TEST);

        gl.blendFuncSeparate(...state.blendFunc);
        gl.blendEquationSeparate(...state.blendEquation);
        gl.colorMask(...state.colorMask);
        gl.clearColor(...state.clearColor);

        gl.pixelStorei(gl.UNPACK_ALIGNMENT, state.pixelStoreParams.unpackAlignment);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, state.pixelStoreParams.unpackFlipY);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, state.pixelStoreParams.unpackPremultiplyAlpha);
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, state.pixelStoreParams.unpackColorspaceConversion);
    }
}


export class WebGLRenderer {
    private canvas: HTMLCanvasElement;
    public gl: WebGL2RenderingContext;
    private program: WebGLProgram;
    private positionAttributeLocation: number;
    private texcoordAttributeLocation: number;
    private positionBuffer: WebGLBuffer;
    private texcoordBuffer: WebGLBuffer;
    private stateManager: WebGLStateManager;

    private _ready = false;
    private readyListeners = new Set<() => void>();

    constructor(canvas: HTMLCanvasElement) {
        if (!canvas) throw new Error('WebGL Renderer: Canvas element not found');
        this.canvas = canvas;
        const gl = this.canvas.getContext('webgl2');
        if (!gl) {
            throw new Error('WebGL2 not supported');
        }
        this.gl = gl;
        this.stateManager = new WebGLStateManager(this.gl);
        this.initShaders();
        this.setupContextLossHandling();
    }

    // Getters and Setters
    public get width(): number {
        return this.canvas.width;
    }

    public set width(width: number) {
        this.canvas.width = width;
    }

    public get height(): number {
        return this.canvas.height;
    }

    public set height(height: number) {
        this.canvas.height = height;
    }

    public get ready(): boolean {
        return this._ready;
    }

    public set ready(ready: boolean) {
        this._ready = ready;
        if (ready) {
            // biome-ignore lint/complexity/noForEach: <explanation>
            this.readyListeners.forEach(listener => listener.call(this));
        }
    }

    private initShaders(): void {
        const vertexShaderSource = `#version 300 es
            in vec4 a_position;
            in vec2 a_texcoord;
            out vec2 v_texcoord;
            void main() {
                gl_Position = a_position;
                v_texcoord = a_texcoord;
            }`;

        const fragmentShaderSource = `#version 300 es
            precision highp float;
            uniform sampler2D u_image;
            in vec2 v_texcoord;
            out vec4 outColor;
            void main() {
                outColor = texture(u_image, v_texcoord);
            }`;

        const vertexShader = this.compileShader(this.gl.VERTEX_SHADER, vertexShaderSource);
        const fragmentShader = this.compileShader(this.gl.FRAGMENT_SHADER, fragmentShaderSource);

        const program = this.gl.createProgram();
        if (!program) {
            throw new Error('Unable to create shader program');
        }
        this.program = program;

        this.gl.attachShader(this.program, vertexShader);
        this.gl.attachShader(this.program, fragmentShader);
        this.gl.linkProgram(this.program);

        if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
            throw new Error('Unable to initialize the shader program: ' + this.gl.getProgramInfoLog(this.program));
        }

        this.positionAttributeLocation = this.gl.getAttribLocation(this.program, "a_position");
        this.texcoordAttributeLocation = this.gl.getAttribLocation(this.program, "a_texcoord");

        const positionBuffer = this.gl.createBuffer();
        if (!positionBuffer) {
            throw new Error('Unable to create position buffer');
        }
        this.positionBuffer = positionBuffer;

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
            -1, -1,
            1, -1,
            -1, 1,
            -1, 1,
            1, -1,
            1, 1,
        ]), this.gl.STATIC_DRAW);

        const texcoordBuffer = this.gl.createBuffer();
        if (!texcoordBuffer) {
            throw new Error('Unable to create texcoord buffer');
        }
        this.texcoordBuffer = texcoordBuffer;

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texcoordBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array([
            0, 1,
            1, 1,
            0, 0,
            0, 0,
            1, 1,
            1, 0,
        ]), this.gl.STATIC_DRAW);
        this.ready = true;
    }

    private compileShader(type: number, source: string): WebGLShader {
        const shader = this.gl.createShader(type);
        if (!shader) {
            throw new Error('Unable to create shader');
        }

        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            throw new Error('An error occurred compiling the shaders: ' + this.gl.getShaderInfoLog(shader));
        }

        return shader;
    }

    public async loadTextureFromURL(url: string): Promise<WebGLTexture> {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => {
                const texture = this.gl.createTexture();
                if (!texture) {
                    reject(new Error('Unable to create texture'));
                    return;
                }
                this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, image);
                this.gl.generateMipmap(this.gl.TEXTURE_2D);
                resolve(texture);
            };
            image.onerror = reject;
            image.src = url;
        });
    }

    public drawTextureToCanvas(texture: WebGLTexture): void {
        console.log('Before drawing - WebGL state:');
        logEverything(this.gl);

        this.stateManager.saveState();

        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);
        this.gl.disable(this.gl.SCISSOR_TEST);

        this.gl.useProgram(this.program);

        this.gl.enableVertexAttribArray(this.positionAttributeLocation);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.vertexAttribPointer(this.positionAttributeLocation, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.enableVertexAttribArray(this.texcoordAttributeLocation);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texcoordBuffer);
        this.gl.vertexAttribPointer(this.texcoordAttributeLocation, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture);

        // Ensure consistent texture parameters
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);

        this.gl.clearColor(0, 0, 0, 1);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);

        // Unbind everything
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
        this.gl.bindTexture(this.gl.TEXTURE_2D, null);
        this.gl.useProgram(null);
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);

        // Disable vertex attributes

        this.gl.disableVertexAttribArray(this.positionAttributeLocation);
        this.gl.disableVertexAttribArray(this.texcoordAttributeLocation);

        //console.log('After drawing - WebGL state:');
        //logEverything(this.gl);

        this.stateManager.restoreState();

        // console.log('After state restoration - WebGL state:');
        //logEverything(this.gl);
    }

    private setupContextLossHandling(): void {
        this.canvas.addEventListener('webglcontextlost', (event) => {
            event.preventDefault();
            console.log('WebGL context lost. You should probably reload the page.');
        }, false);

        this.canvas.addEventListener('webglcontextrestored', () => {
            console.log('WebGL context restored. Reinitializing renderer...');
            this.initShaders();
        }, false);
    }

    public cleanup(): void {
        this.gl.deleteBuffer(this.positionBuffer);
        this.gl.deleteBuffer(this.texcoordBuffer);
        this.gl.deleteProgram(this.program);
        // Add any other cleanup necessary
    }

    public debugTransformTexture(inputTexture: WebGLTexture): WebGLTexture {
        const framebuffer = this.gl.createFramebuffer();
        if (!framebuffer) {
            throw new Error('Unable to create framebuffer');
        }
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);

        const outputTexture = this.gl.createTexture();
        if (!outputTexture) {
            throw new Error('Unable to create output texture');
        }
        this.gl.bindTexture(this.gl.TEXTURE_2D, outputTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.canvas.width, this.canvas.height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);

        this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, outputTexture, 0);

        if (this.gl.checkFramebufferStatus(this.gl.FRAMEBUFFER) !== this.gl.FRAMEBUFFER_COMPLETE) {
            throw new Error('Framebuffer is not complete');
        }

        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.useProgram(this.program);

        this.gl.enableVertexAttribArray(this.positionAttributeLocation);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.vertexAttribPointer(this.positionAttributeLocation, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.enableVertexAttribArray(this.texcoordAttributeLocation);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texcoordBuffer);
        this.gl.vertexAttribPointer(this.texcoordAttributeLocation, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, inputTexture);

        this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);

        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        this.gl.deleteFramebuffer(framebuffer);

        return outputTexture;
    }

    public onReady(callback: (this: WebGLRenderer) => void): () => void {
        if (this.ready) callback.call(this);
        else this.readyListeners.add(callback);

        // return removal
        return () => {
            this.readyListeners.delete(callback);
        };
    }
}


//* WebGL Logging Utils ===========================================
function logWebGLState(gl) {
    console.log("Current Program:", gl.getParameter(gl.CURRENT_PROGRAM));
    console.log("Active Texture:", gl.getParameter(gl.ACTIVE_TEXTURE));
    console.log("Viewport:", gl.getParameter(gl.VIEWPORT));
    console.log("Scissor Test:", gl.getParameter(gl.SCISSOR_TEST));
    console.log("Scissor Box:", gl.getParameter(gl.SCISSOR_BOX));
    console.log("Blend Enabled:", gl.getParameter(gl.BLEND));
    console.log("Depth Test Enabled:", gl.getParameter(gl.DEPTH_TEST));
    console.log("Blend src RGB:", gl.getParameter(gl.BLEND_SRC_RGB));
    console.log("Blend dst RGB:", gl.getParameter(gl.BLEND_DST_RGB));
    console.log("Blend equation RGB:", gl.getParameter(gl.BLEND_EQUATION_RGB));
}

function logProgram(gl) {
    console.log("Current Shader Program:", gl.getParameter(gl.CURRENT_PROGRAM));
}
function logFrameBuffers(gl) {
    console.log("Texture Binding:", gl.getParameter(gl.TEXTURE_BINDING_2D));
    console.log("Framebuffer Binding:", gl.getParameter(gl.FRAMEBUFFER_BINDING));
}
function logError(gl) {
    console.log("WebGL Error:", gl.getError());
}

function logViewport(gl) {
    console.log("Viewport:", gl.getParameter(gl.VIEWPORT));
    console.log("Scissor Test Enabled:", gl.getParameter(gl.SCISSOR_TEST));
    console.log("Scissor Box:", gl.getParameter(gl.SCISSOR_BOX));
}

function logUnpack(gl) {
    console.log("UNPACK_FLIP_Y_WEBGL:", gl.getParameter(gl.UNPACK_FLIP_Y_WEBGL));
    console.log(
        "UNPACK_PREMULTIPLY_ALPHA_WEBGL:",
        gl.getParameter(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL),
    );
    console.log(
        "UNPACK_COLORSPACE_CONVERSION_WEBGL:",
        gl.getParameter(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL),
    );
}

function logEverything(gl) {
    if (!debugging) return;
    logWebGLState(gl);
    logFrameBuffers(gl);
    logUnpack(gl);
    logError(gl);
}