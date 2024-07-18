//tf
import * as tf from '@tensorflow/tfjs';
import type { Tensor4D } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { Weights } from './weights';
import type { TensorMap } from './tza';
import { UNet } from './unet';
import { splitRGBA3D, concatenateAlpha3D } from './utils';
import { GPUTensorTiler } from './improvedTileHandling';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { WebGLStateManager } from './webglStateManager';

//* Types ----------------------------------------------
type ImgInput = tf.PixelData | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;

type ListenerCalback = (data: any) => void;
type ModelInput = null | ImgInput | WebGLTexture | tf.GPUData | tf.Tensor3D;
type DenoiserProps = {
    filterType: 'rt' | 'rt_hdr' | 'rt_ldr' | 'rt_ldr_alb' | 'rt_ldr_alb_nrm' | 'rt_ldr_calb_cnrm';
    quality: 'fast' | 'high' | 'balanced';
    hdr: boolean;
    srgb: boolean;
    height: number;
    width: number;
    cleanAux: boolean;
    dirtyAux: boolean;
    directionals: boolean;
    useColor: boolean;
    useAlbedo: boolean;
    useNormal: boolean;
};
type InputOptions = {
    flipY?: boolean;
    colorspace?: 'srgb' | 'linear';
    height?: number;
    width?: number;
    channels?: 3 | 4;
}

export class Denoiser {
    // counter to how many times the model was built
    timesGenerated = 0;
    private _gl?: WebGL2RenderingContext;
    device?: GPUDevice;
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
        dirtyAux: false,
        directionals: false,
        useColor: true,
        useAlbedo: false,
        useNormal: false
    }

    // this allows us to bypass everything and pass a tensor directly to the model
    directInputTensor?: tf.Tensor3D;

    //* Cofig props ---
    // how we want input and output to be handled
    inputMode: 'imgData' | 'webgl' | 'webgpu' | 'tensor' = 'imgData';
    outputMode: 'imgData' | 'webgl' | 'webgpu' | 'tensor' | 'float32' = 'imgData';
    flipOutputY = false;

    //* Debug Props ---
    canvas?: HTMLCanvasElement;
    outputToCanvas = false;
    debugging = false;
    usePassThrough = false;

    //* Tiling ---
    private _useTiling = false;
    private _tilingUserBlocked = false
    tileStride = 16;
    tileSize = 256;

    //* Internal -----------------------------------

    // listeners for execution callbacks
    private listeners: Map<ListenerCalback, string> = new Map();
    private backendListeners: Set<ListenerCalback> = new Set();
    // Model Props ---
    private unet!: UNet;
    private tiler?: GPUTensorTiler;
    private inputTensors: Map<'color' | 'albedo' | 'normal', tf.Tensor3D> = new Map();
    private inputAlpha?: tf.Tensor3D;
    private oldOutputTensor?: tf.Tensor3D;

    // WebGL ---
    private webglStateManager?: WebGLStateManager;

    // holder for weights instance where we get tensorMaps
    private weights: Weights;
    private activeTensorMap!: TensorMap;
    private isDirty = true;

    constructor(preferedBackend = 'webgl', canvasOrDevice?: HTMLCanvasElement | GPUDevice) {
        this.weights = Weights.getInstance();
        console.log('Running new tiler')
        console.log('%c Denoiser initialized..', 'background: #d66b00; color: white;');
        this.setupBackend(preferedBackend, canvasOrDevice);
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

    get quality() {
        return this.props.quality;
    }

    set quality(quality: 'fast' | 'high' | 'balanced') {
        this.isDirty = true;
        this.props.quality = quality;
    }

    get hdr() {
        return this.props.hdr;
    }

    set hdr(hdr: boolean) {
        this.isDirty = true;
        this.props.hdr = hdr;
    }
    get dirtyAux() {
        return this.props.dirtyAux;
    }
    set dirtyAux(dirtyAux: boolean) {
        this.isDirty = true;
        if (dirtyAux && this.props.cleanAux) this.props.cleanAux = false;
        this.props.dirtyAux = dirtyAux;
    }
    // Backend---

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

    get gl(): WebGL2RenderingContext | undefined {
        return this._gl;
    }
    set gl(gl: WebGL2RenderingContext) {
        if (!gl) return;
        this._gl = gl;
        this.webglStateManager = new WebGLStateManager(gl);
    }

    get useTiling() {
        return this._useTiling;
    }
    set useTiling(useTiling: boolean) {
        this._tilingUserBlocked = !useTiling;
        this._useTiling = useTiling;
    }

    // Weights --

    set weightsPath(path: string) {
        this.weights.path = path;
    }
    get weightsPath() {
        return this.weights.path!;
    }
    set weightsUrl(url: string) {
        this.weights.url = url;
    }
    get weightsUrl() {
        return this.weights.url!;
    }

    //* Backend and Context ------------------------------

    // Take in potential contexts and set the backend
    //todo: should this be private?
    async setupBackend(prefered = 'webgpu', canvasOrDevice?: HTMLCanvasElement | GPUDevice) {
        // do the easy part first
        if (!canvasOrDevice) {
            if (this.debugging) console.log('Denoiser: No canvas provided, using default');
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
            this.device = canvasOrDevice as GPUDevice;
            kernels = tf.getKernelsForBackend('webgpu');
            kernels.forEach(kernelConfig => {
                const newKernelConfig = { ...kernelConfig, backendName: 'denoiser-webgpu' };
                tf.registerKernel(newKernelConfig);
            });
            tf.registerBackend('denoiser-webgpu', async () => {
                return new WebGPUBackend(this.device!);
            });
            await tf.setBackend('denoiser-webgpu');
            console.log('%c Denoiser: Backend set to custom WebGPU', 'background: teal; color: white;');
            this.usingCustomBackend = true;
            this.backendReady = true;


            return this.backend;
        }

        //* Setup a webGL backend with a custom context
        // if multiple denoisers exist they can share contexts
        // register the backend if it doesn't exist
        if (tf.findBackend('denoiser-webgl') === null) {
            // we have to make sure all the kernels are registered
            kernels = tf.getKernelsForBackend('webgl');
            for (const kernelConfig of kernels) {
                const newKernelConfig = { ...kernelConfig, backendName: 'denoiser-webgl' };
                tf.registerKernel(newKernelConfig);
            }
            const customBackend = new tf.MathBackendWebGL(canvasOrDevice as HTMLCanvasElement);
            tf.registerBackend('denoiser-webgl', () => customBackend);
            console.log('registered kernels for custom webgl backend', kernels);
        }
        await tf.setBackend('denoiser-webgl');
        await tf.ready();

        //@ts-ignore
        this.gl = tf.engine().findBackend('denoiser-webgl').gpgpu.gl;
        console.log('%c Denoiser: Backend set to custom webgl', 'background: orange; color: white;');
        this.usingCustomBackend = true;
        this.backendReady = true;
        this.webglStateManager?.saveState();

        return this.backend;
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

        // calculate dimensions
        const height = this.useTiling ? this.tileSize : this.props.height;
        const width = this.useTiling ? this.tileSize : this.props.width;
        if (this.debugging) startTime = performance.now();
        this.unet = new UNet({ weights: this.activeTensorMap, size, height, width, channels });

        const model = (this.usePassThrough) ? await this.unet.debugBuild() : await this.unet.build();
        if (!model) throw new Error('UNet Model failed to build');

        //* Tiling
        if (this.useTiling) {
            this.tiler = new GPUTensorTiler(model, this.tileSize || 256)
            console.log('%c Denoiser: Using Tiling Output', 'background: green; color: white;');
        }

        if (this.debugging) {
            endTime = performance.now();
            if (this.debugging) console.log('Denoiser: Build Time:', endTime - startTime!);
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
        if (this.props.cleanAux) tensorMapLabel += '_calb_cnrm';
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
        if (this.webglStateManager) this.webglStateManager.restoreState();
        if (colorInput || albedoInput || normalInput) switch (this.inputMode) {
            case 'webgl':
                if (!this.height || !this.width) throw new Error('Denoiser: Height and Width must be set when executing with webGL input.');
                if (colorInput) this.setInputTexture('color', colorInput as WebGLTexture, this.height, this.width);
                if (albedoInput) this.setInputTexture('albedo', albedoInput as WebGLTexture, this.height, this.width);
                if (normalInput) this.setInputTexture('normal', normalInput as WebGLTexture, this.height, this.width);
                break;

            case 'webgpu':
                if (!this.height || !this.width) throw new Error('Denoiser: Height and Width must be set when executing with webGPU input.');
                if (colorInput) this.setInputBuffer('color', colorInput as GPUBuffer, this.height, this.width);
                if (albedoInput) this.setInputBuffer('albedo', albedoInput as GPUBuffer, this.height, this.width);
                if (normalInput) this.setInputBuffer('normal', normalInput as GPUBuffer, this.height, this.width);
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
        if (this.debugging) console.log('%c Denoiser: Denoising...', 'background: blue; color: white;');
        if (this.debugging) startTime = performance.now();

        // process and send the input to the model
        const inputTensor = await this.handleModelInput();

        // if we need to rebuild the model
        if (this.isDirty) await this.build();

        // Execute model with tiling or standard
        const result = this.useTiling ? await this.tiler!.processLargeTensor(inputTensor)
            : await this.unet.execute(inputTensor);

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
            if (this.debugging) console.log(`Denoiser: Execution Time: ${performance.now() - startTime!}ms`);
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

        // if we have extra images we need to use tiling
        if (!this._tilingUserBlocked && (albedo || normal)) this.useTiling = true;

        // if we have both aux images and not blocked by the user, set cleanAux to true
        if (albedo && normal && !this.props.dirtyAux) this.props.cleanAux = true;

        // concat the images
        return tf.tidy(() => {
            let concatenatedImage: tf.Tensor3D;
            if (albedo) {
                if (normal) concatenatedImage = tf.concat([color, albedo, normal], -1);
                else concatenatedImage = tf.concat([color, albedo], -1);
            } else concatenatedImage = color;

            // Reshape for batch size
            return concatenatedImage.expandDims(0) as Tensor4D; // Now shape is [1, height, width, 9]
        });
    }

    // Take the model output and ready it to be returned
    private async handleModelOutput(result: tf.Tensor4D): Promise<tf.Tensor3D> {
        const outputImage = tf.tidy(() => {
            let output = result.squeeze() as tf.Tensor3D;
            // With float32 we had strange 1.0000001 values this limits to expected outputs
            if (!this.hdr) output = tf.clipByValue(output, 0, 1);
            // flip the image vertically
            if (this.flipOutputY) output = tf.reverse(output, [0]);
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
    private async handleReturn(outputTensor: tf.Tensor3D) {
        // if we are dumping to canvas
        if (this.outputToCanvas && this.canvas)
            tf.browser.toPixels(outputTensor, this.canvas);

        // handle listeners
        this.listeners.forEach((returnType, callback) => {
            this.handleCallback(outputTensor, returnType, callback);
        });

        // output for direct execution
        // only fire if we have no listeners
        let toReturn: any;
        if (this.listeners.size === 0)
            toReturn = await this.handleCallback(outputTensor, this.outputMode);

        if (this.gl) this.webglStateManager?.saveState();
        return toReturn;
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
                // WebGL too must have the alpha channel
                if (!this.inputAlpha) data = concatenateAlpha3D(outputTensor).dataToGPU();
                else data = outputTensor.dataToGPU({ customTexShape: [this.height, this.width] });
                if (!data.texture) throw new Error('Denoiser: Could not convert to webGL texture');
                toReturn = data.texture;
                // todo: this might be dangerous
                data.tensorRef.dispose();
                break;

            case 'webgpu':
                // webGPU MUST have the alpha channel
                // if we didn't add it back (because of data input) we need to now with blanks
                if (!this.inputAlpha) data = concatenateAlpha3D(outputTensor).dataToGPU();
                else data = outputTensor.dataToGPU();
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
        // if (this.debugging) console.log(`Denoiser: Callback Prep duration: ${performance.now() - startTime}ms`);

        if (callback) callback(toReturn!);
        return toReturn!;
    }

    //* Set the Inputs -----------------------------------

    // set the input tensor. Rarely accessed directly but used by all internals
    setInputTensor(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor3D) {
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
    // TODO change this to setInputImage
    //set the image and tensor, normalize (potentailly) flipY if needed
    setImage(name: 'color' | 'albedo' | 'normal', imgData: ImgInput, flipY = false) {
        let finalData = imgData;
        // if input is color lets take the height and width and set it on this
        if (imgData instanceof HTMLImageElement && hasSizeMissmatch(imgData)) {
            // check if the image is css scaled, if so correct the data
            if (this.debugging) console.log('Image is css scaled, getting correct image data');
            finalData = getCorrectImageData(imgData);
        }
        if (name === 'color') {

            let inHeight = 0;
            let inWidth = 0;

            // if the input is an html image use the natural height and width
            if (imgData instanceof HTMLImageElement) {
                inHeight = imgData.naturalHeight;
                inWidth = imgData.naturalWidth;
            } else if (imgData.height && imgData.width) {
                inHeight = imgData.height;
                inWidth = imgData.width;
            }
            if (inHeight || inWidth) {
                if (inHeight && inHeight !== this.height) this.height = inHeight;
                if (inWidth && inWidth !== this.width) this.width = inWidth;

                if (this.debugging && (inHeight !== this.height || inWidth !== this.width)) {
                    console.warn('Denoiser: Image size does not match denoiser size, resizing may occur.');
                }
            }
        } else if (name === 'albedo') this.props.useAlbedo = true;
        else if (name === 'normal') this.props.useNormal = true;


        this.setInputTensor(name, tf.tidy(() => {
            let tensor: tf.Tensor3D;
            if (name === 'normal') {
                tensor = this.createImageTensor(finalData, true);
            } else tensor = this.createImageTensor(finalData);
            if (!flipY) return tensor;
            return tf.reverse(tensor, [0]);
        }));
    }

    // creates the image tensor
    createImageTensor(input: ImgInput, isNormalMap = false): tf.Tensor3D {
        return tf.tidy(() => {
            const imgTensor = tf.browser.fromPixels(input);
            // standard normalization
            if (!isNormalMap) return imgTensor.toFloat().div(tf.scalar(255)) as tf.Tensor3D;

            // normalize the tensor to OIDN expect range of -1 to 1
            const normalized = imgTensor.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));
            return normalized as tf.Tensor3D;
        });
    }

    // for a webGPU buffer create and set it
    setInputBuffer(name: 'color' | 'albedo' | 'normal', buffer: GPUBuffer, height: number, width: number, channels = 4) {
        const baseTensor = tf.tensor({ buffer: buffer }, [height, width, channels], 'float32') as tf.Tensor3D;

        // if the channels is 4 we need to strip the alpha channel
        if (channels === 4) {
            // split the alpha channel from the rgb data (NOTE: destroys the baseTensor)
            const { rgb, alpha } = splitRGBA3D(baseTensor);
            this.setInputTensor(name, rgb);
            this.inputAlpha = alpha;
            // we only care about the alpha of the color input
            if (name === 'color') this.inputAlpha = alpha;
        } else this.setInputTensor(name, baseTensor);
    }

    // for a webGL texture create and set it
    setInputTexture(name: 'color' | 'albedo' | 'normal', texture: WebGLTexture, height?: number, width?: number, options: InputOptions = {}) {
        //options
        const { flipY, colorspace, channels } = options;
        // if passed, overwrite the height and width of the class
        if (name === 'color' && (height !== this.height || width !== this.width)) {
            if (height) this.height = height;
            if (width) this.width = width;
        } else if (name === 'albedo') this.props.useAlbedo = true;
        else if (name === 'normal') this.props.useNormal = true;
        const baseTensor = tf.tidy(() => {
            let toReturn = tf.tensor({ texture, height: this.height, width: this.width, channels: 'RGBA' }, [this.height, this.width, channels || 4], 'float32') as tf.Tensor3D;
            //if flipping
            if (flipY) toReturn = tf.reverse(toReturn, [0]);

            if (colorspace === 'linear') {
                console.log('Colorspace converted from linear to srgb')
                toReturn = tensorLinearToSRGB(toReturn);
            }
            // leaveing seperate incase we want to add handing
            return toReturn;
        });

        // if the channels is 4 we need to strip the alpha channel
        if (!channels) {
            // split the alpha channel from the rgb data (NOTE: destroys the baseTensor)
            const { rgb, alpha } = splitRGBA3D(baseTensor);
            this.setInputTensor(name, rgb);
            // we only care about the alpha of the color input
            if (name === 'color') this.inputAlpha = alpha;
            else alpha.dispose();
            // get rid of the baseTensor
            //todo this might be done in splitRGBA3D
            baseTensor.dispose();
        } else this.setInputTensor(name, baseTensor);
    }

    // this is mostly an internal debug thing but is super helpful
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

// convert a tensor with linear color encoding to sRGB
function tensorLinearToSRGB(tensor: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
        const gamma = tf.scalar(2.2);
        const sRGB = tensor.pow(gamma);
        return sRGB;
    }) as tf.Tensor3D;
}
