export class WebGLStateManager {
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
        // @ts-ignore
        gl.viewport(...state.viewport);
        gl.scissor(state.scissor[0], state.scissor[1], state.scissor[2], state.scissor[3]);

        state.blend ? gl.enable(gl.BLEND) : gl.disable(gl.BLEND);
        state.depthTest ? gl.enable(gl.DEPTH_TEST) : gl.disable(gl.DEPTH_TEST);
        state.cullFace ? gl.enable(gl.CULL_FACE) : gl.disable(gl.CULL_FACE);
        state.scissorTest ? gl.enable(gl.SCISSOR_TEST) : gl.disable(gl.SCISSOR_TEST);

        gl.blendFuncSeparate(state.blendFunc[0], state.blendFunc[1], state.blendFunc[2], state.blendFunc[3]);
        // @ts-ignore
        gl.blendEquationSeparate(...state.blendEquation);
        // @ts-ignore
        gl.colorMask(...state.colorMask);
        // @ts-ignore
        gl.clearColor(...state.clearColor);

        gl.pixelStorei(gl.UNPACK_ALIGNMENT, state.pixelStoreParams.unpackAlignment);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, state.pixelStoreParams.unpackFlipY);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, state.pixelStoreParams.unpackPremultiplyAlpha);
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, state.pixelStoreParams.unpackColorspaceConversion);
    }
}

/* Cody code
    restoreWebGLState() {
        if (!this.gl || !this.webglProgram || !this.webglState) return;
        const gl = this.gl;
        const state = this.webglState;
        if (state !== null) {
            gl.bindVertexArray(state.VAO);
            gl.activeTexture(state.activeTexture);
            gl.bindTexture(gl.TEXTURE_2D, state.textureBinding);
            gl.useProgram(state.program);
            gl.bindRenderbuffer(gl.RENDERBUFFER, state.renderbufferBinding);
            gl.bindFramebuffer(gl.FRAMEBUFFER, state.framebufferBinding);
            //@ts-ignore
            gl.viewport(...state.viewport);

            if (state.scissorTest) gl.enable(gl.SCISSOR_TEST);
            else gl.disable(gl.SCISSOR_TEST);

            if (state.stencilTest) gl.enable(gl.STENCIL_TEST);
            else gl.disable(gl.STENCIL_TEST);

            if (state.depthTest) gl.enable(gl.DEPTH_TEST);
            else gl.disable(gl.DEPTH_TEST);

            if (state.blend) gl.enable(gl.BLEND);
            else gl.disable(gl.BLEND);

            if (state.cullFace) gl.enable(gl.CULL_FACE);
            else gl.disable(gl.CULL_FACE);
        }
        // restore the webgl state
        this.gl.useProgram(this.webglProgram);
        if (this.debugging) console.log('WebGL State Restored', this.webglProgram, this.gl, this.webglState);
    }
        */