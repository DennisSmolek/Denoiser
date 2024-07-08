//tf
import * as tf from '@tensorflow/tfjs';
import type { Tensor4D } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { Weights } from './weights';
import type { TensorMap } from './tza';
import { UNet } from './unet';
import { splitRGBA3D, concatenateAlpha3D } from './utils';
import '@tensorflow/tfjs-backend-webgpu';

type ImgInput = tf.PixelData | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;

type ListenerCalback = (data: any) => void;
type ModelInput = ImgInput | WebGLTexture | tf.GPUData | tf.Tensor3D;
type DenoiserProps = {
    filterType: 'rt' | 'rt_hdr' | 'rt_ldr' | 'rt_ldr_alb' | 'rt_ldr_alb_nrm' | 'rt_ldr_calb_cnrm';
    quality: 'fast' | 'high' | 'balanced';
    hdr: boolean;
    srgb: boolean;
    height: number;
    width: number;
    cleanAux: boolean;
    directionals: boolean;
    useColor: boolean;
    useAlbedo: boolean;
    useNormal: boolean;
};

export class Denoiser {
    // counter to how many times the model was built
    timesGenerated = 0;
    // webGL context
    gl?: WebGL2RenderingContext;
    //backend?: tf.MathBackendWebGL | WebGPUBackend | tf.MathBackendCPU;
    backend?: tf.KernelBackend
    backendLoaded = false;
    usingCustomBackend = false;

    // kept in its own object because changing these will require a rebuild
    props: DenoiserProps = {
        filterType: 'rt',
        quality: 'fast',
        hdr: false,
        srgb: false,
        height: 0,
        width: 0,
        cleanAux: false,
        directionals: false,
        useColor: true,
        useAlbedo: false,
        useNormal: false
    }

    // this allows us to bypass everything and pass a tensor directly to the model
    directInputTensor?: tf.Tensor3D;

    // configurable options that require rebuilding the model
    activeBackend: 'cpu' | 'webgl' | 'wasm' | 'webgpu' = 'webgpu';

    // how we want input and output to be handled
    inputMode: 'imgData' | 'webgl' | 'webgpu' | 'tensor' = 'imgData';
    outputMode: 'imgData' | 'webgl' | 'webgpu' | 'tensor' | 'float32' = 'imgData';

    //* Debug Props ---
    canvas?: HTMLCanvasElement;
    outputToCanvas = false;
    debugging = false;
    usePassThrough = false;

    //* Internal -----------------------------------

    // listeners for execution callbacks
    private listeners: Map<ListenerCalback, string> = new Map();
    // This holds listeners for when we save/restore state
    //private listenersRemaining = 0;

    // probably going to remove this
    private backendListeners: Set<ListenerCalback> = new Set();
    // Model Props ---
    private unet!: UNet;
    private inputTensors: Map<'color' | 'albedo' | 'normal', tf.Tensor3D> = new Map();
    private inputAlpha?: tf.Tensor3D;
    private oldOutputTensor?: tf.Tensor3D;

    //todo remove?
    private concatenatedImage!: tf.Tensor3D;

    // WebGL ---
    private webglProgram?: WebGLProgram;
    private webglState?: any;

    // holder for weights instance where we get tensorMaps
    private weights: Weights;
    private activeTensorMap!: TensorMap;
    private isDirty = true;

    constructor(preferedBackend = 'webgpu', canvas?: HTMLCanvasElement) {
        this.weights = Weights.getInstance();
        console.log('Denoiser initialized..');
        this.setupBackend(preferedBackend, canvas);
    }

    //* Getters and Setters ------------------------------
    get height() {
        return this.props.height;
    }
    set height(height: number) {
        this.isDirty = true;
        this.props.height = height;
    }
    get width() {
        return this.props.width;
    }
    set width(width: number) {
        this.isDirty = true;
        this.props.width = width;
    }
    set quality(quality: 'fast' | 'high' | 'balanced') {
        this.isDirty = true;
        this.props.quality = quality;
    }

    get backendReady() {
        return this.backendLoaded;
    }

    set backendReady(isReady: boolean) {
        this.backendLoaded = isReady;
        // biome-ignore lint/complexity/noForEach: <explanation>
        if (isReady) this.backendListeners.forEach((callback) => {
            callback(this.backend);
        });
    }

    //* Backend and Context ------------------------------

    // Take in potential contexts and set the backend
    //todo: should this be private?
    async setupBackend(prefered = 'webgpu', canvasOrDevice?: HTMLCanvasElement | GPUDevice) {
        // do the easy part first
        if (!canvasOrDevice) {
            console.log('Denoiser: No canvas provided, using default');
            await tf.setBackend(prefered);
            const backendName = tf.getBackend();
            this.backend = tf.engine().findBackend(backendName)
            // if webgl set the gl context
            //@ts-ignore
            this.gl = backendName === 'webgl' ? this.backend.gpgpu.gl : undefined;
            if (this.debugging) console.log(`Denoiser: Backend set to ${prefered}`);

            this.backendReady = true;
            return this.backend;
        }

        // We dont want to run mess with weird context for WASM or CPU backends
        if (prefered !== 'webgl' && prefered !== 'webgpu')
            throw new Error('Only webgl and webgpu are supported with custom contexts');

        let kernels: tf.KernelConfig[];
        //* Setup a webGPU backend with a custom device
        if (prefered === 'webgpu') {

        }

        //* Setup a webGL backend with a custom context
        // if multiple denoisers exist they can share contexts
        // register the backend if it doesn't exist
        if (tf.findBackend('denoiser-webgl') === null) {
            // we have to make sure all the kernels are registered
            kernels = tf.getKernelsForBackend('webgl');
            for (const kernelConfig of kernels) {
                const newKernelConfig = { ...kernelConfig, backendName: 'oidnflow-webgl' };
                tf.registerKernel(newKernelConfig);
            }
            const customBackend = new tf.MathBackendWebGL(canvasOrDevice as HTMLCanvasElement);
            tf.registerBackend('denoiser-webgl', () => customBackend);
        }
        await tf.setBackend('denoiser-webgl');

        //@ts-ignore
        this.gl = tf.engine().findBackend('denoiser-webgl').gpgpu.gl;
        console.log('%c Denoiser: Backend set to custom webgl', 'background: orange; color: white;');
        this.usingCustomBackend = true;
        this.backendReady = true;
        this.saveWebGLState();

        return this.backend;
    }
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
        console.log('WebGL State Restored', this.webglProgram, this.gl, this.webglState);
    }

    saveWebGLState() {
        if (!this.gl) return;
        const gl = this.gl;
        this.webglProgram = this.gl.getParameter(this.gl.CURRENT_PROGRAM);
        // state from Cody Bennett
        const state = {
            VAO: gl.getParameter(gl.VERTEX_ARRAY_BINDING),
            cullFace: gl.getParameter(gl.CULL_FACE),
            blend: gl.getParameter(gl.BLEND),
            depthTest: gl.getParameter(gl.DEPTH_TEST),
            stencilTest: gl.getParameter(gl.STENCIL_TEST),
            scissorTest: gl.getParameter(gl.SCISSOR_TEST),
            viewport: gl.getParameter(gl.VIEWPORT),
            framebufferBinding: gl.getParameter(gl.FRAMEBUFFER_BINDING),
            renderbufferBinding: gl.getParameter(gl.RENDERBUFFER_BINDING),
            program: gl.getParameter(gl.CURRENT_PROGRAM),
            activeTexture: gl.getParameter(gl.ACTIVE_TEXTURE),
            textureBinding: gl.getParameter(gl.TEXTURE_BINDING_2D),
        };
        this.webglState = state;
        console.log('WebGL State Saved', this.webglProgram, this.gl);
    }

    //* Build the unet using props ------------------------
    async build() {
        let startTime: number;
        let endTime: number;
        const { tensorMapLabel, size } = this.determineTensorMap();

        if (this.debugging) console.log(`Denoiser: starting Build with TensorMap: ${tensorMapLabel}...`);
        this.activeTensorMap = await this.weights.getCollection(tensorMapLabel);
        // calculate channels
        let channels = 0;
        if (this.props.useColor) channels += 3;
        if (this.props.useAlbedo) channels += 3;
        if (this.props.useNormal) channels += 3;

        if (this.debugging) startTime = performance.now();
        this.unet = new UNet({ weights: this.activeTensorMap, size, height: this.props.height, width: this.props.width, channels });

        if (this.usePassThrough) this.unet.debugBuild();
        else this.unet.build();

        if (this.debugging) {
            endTime = performance.now();
            console.log('Denoiser: Build Time:', endTime - startTime!);
        }
        this.timesGenerated++;
        this.isDirty = false;
    }

    // determine which tensormap to use
    private determineTensorMap() {
        // example rt_ldr_small | rt_ldr_calb_cnrm (with clean aux)
        let tensorMapLabel = this.props.filterType;
        tensorMapLabel += this.props.hdr ? '_hdr' : '_ldr';
        // if you use cleanAux you MUST provide both albedo and normal
        if (this.props.cleanAux) tensorMapLabel += '_calb_crnm';
        else {
            tensorMapLabel += this.props.useAlbedo ? '_alb' : '';
            tensorMapLabel += this.props.useNormal ? '_nrm' : '';
        }

        //* quality. 
        let size: 'small' | 'large' | 'default' = 'default';
        // small and large only exist in specific cases
        const hasSmall = ['rt_hdr', 'rt_ldr', 'rt_hdr_alb', 'rt_ldr_alb', 'rt_hdr_alb_nrm', 'rt_ldr_alb_nrm', 'rt_hdr_calb_cnrm', 'rt_ldr_calb_cnrm'];
        const hasLarge = ['rt_alb', 'rt_nrm', 'rt_hdr_calb_cnrm', 'rt_ldr_calb_cnrm'];

        if (this.props.quality === 'fast' && hasSmall.includes(tensorMapLabel)) size = 'small';
        else if (this.props.quality === 'high' && hasLarge.includes(tensorMapLabel)) size = 'large';

        if (size !== 'default') tensorMapLabel += `_${size}`;

        return { tensorMapLabel, size };
    }

    //* Execute the denoiser ------------------------------
    // execute with image data inputs
    async execute(colorInput?: ModelInput, albedoInput?: ModelInput, normalInput?: ModelInput) {
        console.log('colorInput:', colorInput, 'albedoInput:', albedoInput, 'normalInput:', normalInput);
        if (colorInput || albedoInput || normalInput) switch (this.inputMode) {
            case 'webgl':
                //todo: implement webgl input
                break;
            case 'webgpu':
                //todo: implement webgpu input
                break;
            case 'tensor':
                if (colorInput) this.setInputTensor('color', colorInput as tf.Tensor3D);
                if (albedoInput) this.setInputTensor('albedo', albedoInput as tf.Tensor3D);
                if (normalInput) this.setInputTensor('normal', normalInput as tf.Tensor3D);
                break;
            default:
                if (colorInput) this.setImage('color', colorInput as ImgInput);
                if (albedoInput) this.setImage('albedo', albedoInput as ImgInput);
                if (normalInput) this.setImage('normal', normalInput as ImgInput);
        }

        return this.executeModel();
    }

    // actually execute the model with the set inputs of this class
    private async executeModel() {
        let startTime: number;
        let endTime: number;
        if (this.debugging) console.log('%c Denoiser: Denoising...', 'background: blue; color: white;');
        // if we need to rebuild the model
        if (this.isDirty) await this.build();

        if (this.debugging) startTime = performance.now();

        // process and send the input to the model
        const inputTensor = await this.handleModelInput();
        this.unet.inputTensor = inputTensor;

        // execute the model
        const result = await this.unet.execute();

        // this helps us debug input/output by making the model a pure pass through
        if (this.usePassThrough) {
            // check if its totally equal
            const isEqual = tf.equal(inputTensor, result);
            console.log('Make sure model is a pass trough:');
            isEqual.all().print();
        }
        // process the output
        const output = await this.handleModelOutput(result);

        if (this.debugging) {
            //console.log('Output Tensor')
            //  output.print(); 
            endTime = performance.now();
            console.log(`Denoiser: Execution Time: ${endTime - startTime!}ms`);
        }
        return this.handleReturn(output);
    }

    // take the set input tensors and concatenate them. ready things for model input
    private async handleModelInput(): Promise<Tensor4D> {
        // if we have direct input bypass everything
        if (this.directInputTensor) return this.directInputTensor.expandDims(0) as Tensor4D;

        // TODO: if we ever want to denoise just normal or albedo this will have to change as it requires color
        const color = this.inputTensors.get('color');
        const albedo = this.inputTensors.get('albedo');
        const normal = this.inputTensors.get('normal');

        //if (!color && !albedo && !normal) throw new Error('Denoiser must have an input set before execution.');
        if (!color) throw new Error('Denoiser must have an input set before execution.');

        // Concatenate images along the channel dimension
        // if the concatenated image is not the same as the color image dispose of it
        //todo: this can all probably be wrapped in a tf.tidy
        if (albedo || normal && this.concatenatedImage !== color) this.concatenatedImage.dispose();
        if (albedo) {
            if (normal) this.concatenatedImage = tf.concat([color, albedo, normal], -1);
            else this.concatenatedImage = tf.concat([color, albedo], -1);
        } else if (normal) {
            this.concatenatedImage = tf.concat([color, normal], -1);
        } else this.concatenatedImage = color;
        // Reshape for batch size
        return this.concatenatedImage.expandDims(0) as Tensor4D; // Now shape is [1, height, width, 9]
    }

    // Take the model output and ready it to be returned
    private async handleModelOutput(result: tf.Tensor4D): Promise<tf.Tensor3D> {
        const outputImage = tf.tidy(() => {
            //TODO this is unnecessarily verbose
            let output = result.squeeze() as tf.Tensor3D;
            // With float32 we had strange 1.0000001 values this limits to expected outputs
            output = tf.clipByValue(output, 0, 1);
            // flip the image vertically
            // output = tf.reverse(output, [0]);
            // check the datatype and shape of the outputImage
            // if (this.debugging) console.log('Output Image shape:', output.shape, 'dtype:', output.dtype);
            if (this.inputAlpha) output = concatenateAlpha3D(output, this.inputAlpha);
            return output;
        });
        // if there is an old output tensor dispose of it
        if (this.oldOutputTensor) this.oldOutputTensor.dispose();
        this.oldOutputTensor = outputImage;
        return outputImage;
    }

    // handle returns and callbacks
    private handleReturn(outputTensor: tf.Tensor3D) {
        // if we are dumping to canvas
        if (this.outputToCanvas && this.canvas)
            tf.browser.toPixels(outputTensor, this.canvas);

        // handle listeners
        this.listeners.forEach((returnType, callback) => {
            this.handleCallback(outputTensor, returnType, callback);
        });

        // output for direct execution
        // only fire if we have no listeners
        if (this.listeners.size === 0)
            return this.handleCallback(outputTensor, this.outputMode);
        return false;
    }

    private async handleCallback(outputTensor: tf.Tensor3D, returnType: string, callback?: ListenerCalback) {
        let toReturn: tf.Tensor3D | ImageData | WebGLTexture | GPUBuffer;
        let data: tf.GPUData;
        const startTime = performance.now();
        switch (returnType) {
            case 'tensor':
                toReturn = outputTensor;
                break;
            case 'webgl':
                data = outputTensor.dataToGPU();
                if (!data.texture) throw new Error('Denoiser: Could not convert to webGL texture');
                toReturn = data.texture;
                break;
            case 'webgpu':
                data = outputTensor.dataToGPU();
                if (!data.buffer) throw new Error('Denoiser: Could not convert to webGPU buffer');
                toReturn = data.buffer;
                break;
            case 'float32': {
                const float32Data = outputTensor.dataSync() as Float32Array;
                toReturn = float32Data;
                break;
            }
            default: {
                // reflip for standard imgData
                // const flip = tf.reverse(outputTensor, [0]);
                const pixelData = await tf.browser.toPixels(outputTensor);
                // flip.dispose();
                toReturn = new ImageData(pixelData, this.props.width, this.props.height);
            }
        }
        if (this.debugging) console.log(`Denoiser: Callback Prep duration: ${performance.now() - startTime}ms`);

        if (callback) callback(toReturn!);
        return toReturn!;
    }

    //* Set the Inputs -----------------------------------

    // set the input tensor
    setInputTensor(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor3D) {
        // console.log('Setting Image Tensor:', name, tensor.shape);
        // destroy existing tensor
        const existingTensor = this.inputTensors.get(name);
        if (existingTensor) existingTensor.dispose();

        this.inputTensors.set(name, tensor);
    }

    //* Data Input ---
    setData(name: 'color' | 'albedo' | 'normal', data: Float32Array | Uint8Array, height: number, width: number, channels = 4) {
        // check if data is a float32 and reject otherwise
        if (data.constructor !== Float32Array)
            throw new Error('Invalid Input data type. Must be a Float32Array');
        /*
        const pixelLength = noAlpha ? 3 : 4;
        // check the length of the data
         Im removing this check for now because it might be a unnecessary performance hit
        if (data.length / pixelLength !== width * height) {
            throw new Error('Invalid data length. Length must be equal to count * 4.');
        }*/
        const baseTensor = tf.tensor3d(data, [height, width, channels])
        if (channels === 4) {
            // split the alpha channel from the rgb data (NOTE: destroys the baseTensor)
            const { rgb, alpha } = splitRGBA3D(baseTensor);
            this.setInputTensor(name, rgb);
            this.inputAlpha = alpha;
            // we only care about the alpha of the color input
            if (name === 'color') this.inputAlpha = alpha;
        } else this.setInputTensor(name, baseTensor);


    }

    //* Image Input ---
    //set the image and tensor, normalize (potentailly) flipY if needed
    setImage(name: 'color' | 'albedo' | 'normal', imgData: ImgInput, flipY = false) {
        let finalData = imgData;
        // if input is color lets take the height and width and set it on this
        if (name === 'color' && !this.height && !this.width) {
            // if the input is an html image use the natural height and width
            if (imgData instanceof HTMLImageElement) {
                this.height = imgData.naturalHeight;
                this.width = imgData.naturalWidth;
                // check if the image is css scaled, if so correct the data
                if (hasSizeMissmatch(imgData)) {
                    console.log('Image is css scaled, getting correct image data');
                    finalData = getCorrectImageData(imgData);
                }


            } else if (imgData.height && imgData.width) {
                this.height = imgData.height;
                this.width = imgData.width;
            }
            console.log('Setting height and width from image:', this.height, this.width);
        }

        this.setInputTensor(name, tf.tidy(() => {
            let tensor: tf.Tensor3D;
            if (name === 'normal') {
                tensor = this.createImageTensor(finalData, false);
            } else tensor = this.createImageTensor(finalData);
            if (!flipY) return tensor;
            return tf.reverse(tensor, [0]);
        }));
    }

    // creates the image tensor
    createImageTensor(input: ImgInput, normalize = true): tf.Tensor3D {
        return tf.tidy(() => {
            console.log('Image input');
            console.dir(input);
            const imgTensor = tf.browser.fromPixels(input);
            console.log('image tensor:', imgTensor.shape, imgTensor.dtype);
            if (!normalize) return imgTensor as tf.Tensor3D;
            // normalize the tensor
            //const normalized = imgTensor.toFloat().div(tf.scalar(255));
            // normalize the tensor to OIDN expect range of -1 to 1
            const normalized = imgTensor.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));
            return normalized as tf.Tensor3D;
        });
    }

    setCanvas(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.outputToCanvas = true;
    }

    //* Listeners ---------------------------------------
    // Add a listener to the denoiser with a return function to stop listening
    onExecute(listener: ListenerCalback, responseType: string) {
        this.listeners.set(listener, responseType);
        return () => this.listeners.delete(listener);
    }

    onBackendReady(listener: ListenerCalback) {
        // if the backend is already ready fire the listener
        if (this.backendReady) listener(this.backend);
        else this.backendListeners.add(listener);
        return () => this.backendListeners.delete(listener);
    }
}

//* Utils ----------------------------------------------

// take a css scaled image and use a canvas to get the actual size
function getCorrectImageData(img: HTMLImageElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

// check if the height/width dont math
function hasSizeMissmatch(img: HTMLImageElement) {
    if (!img.naturalHeight || !img.naturalWidth) return true;
    return img.height !== img.naturalHeight || img.width !== img.naturalWidth;
}


/*
// get the alpha channel from a float32 array
function getAlphaChannel(imageData: Float32Array, count: number): Float32Array {
    const alphaData = new Float32Array(count);
    let j = 0; // Index for alphaData array

    for (let i = 3; i < imageData.length; i += 4) {
        alphaData[j++] = imageData[i];
    }

    return alphaData;
}

//* This functionality should be moved to tensorflow
// this is actually more useful
// take a float32array and return two float32arrays, one without the alpha and one just the alpha
function splitAlphaChannel(imageData: Float32Array, count: number): [Float32Array, Float32Array] {
    const rgbData = new Float32Array(count * 3);
    const alphaData = new Float32Array(count);
    let j = 0; // Index for rgbData array
    let k = 0; // Index for alphaData array

    for (let i = 0; i < imageData.length; i += 4) {
        rgbData[j++] = imageData[i];
        rgbData[j++] = imageData[i + 1];
        rgbData[j++] = imageData[i + 2];
        alphaData[k++] = imageData[i + 3];
    }

    return [rgbData, alphaData];
}

function addAlphaChannel(imageData: Float32Array, itemCount: number, alphaData?: Float32Array): Float32Array {
    const startTime = performance.now();
    const output = new Float32Array(itemCount * 4); // Create a new array with space for the alpha channel
    let j = 0; // Index for output array

    for (let i = 0; i < imageData.length; i += 3) {
        output[j++] = imageData[i];     // R
        output[j++] = imageData[i + 1]; // G
        output[j++] = imageData[i + 2]; // B
        // If we have saved alpha data, use it; otherwise, default to 1 (fully opaque)
        output[j++] = alphaData ? alphaData[i / 3] : 1;
    }
    console.log(`Denoiser: Alpha Channel Addition duration: ${performance.now() - startTime}ms`);

    return output;
}
    */