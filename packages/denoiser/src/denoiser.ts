//tf
import * as tf from '@tensorflow/tfjs';
import { Weights } from './weights';
import { UNet } from './unet';
import { splitRGBA3D, concatenateAlpha3D, getCorrectImageData, hasSizeMissmatch, tensorLinearToSRGB } from './utils';
import { GPUTensorTiler } from './tiler';
import { WebGLStateManager } from './webglStateManager';
import { setupBackend, determineTensorMap, handleModelInput, handleModelOutput, handleCallback, adjustSize, handleInputTensors } from './denoiserUtils';

// TODO: These shouldnt have to be imported
import type { TensorMap, DenoiserProps, ImgInput, InputOptions, ListenerCalback, ModelInput } from 'types/types';


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
    _useTiling = false;
    _tilingUserBlocked = false
    tileStride = 16;
    tileSize = 256;

    //* Internal -----------------------------------

    // listeners for execution callbacks
    private listeners: Map<ListenerCalback, string> = new Map();
    private backendListeners: Set<ListenerCalback> = new Set();
    // Model Props ---
    private unet!: UNet;
    private tiler?: GPUTensorTiler;
    inputTensors: Map<'color' | 'albedo' | 'normal', tf.Tensor3D> = new Map();
    inputAlpha?: tf.Tensor3D;
    oldOutputTensor?: tf.Tensor3D;

    // WebGL ---
    public webglStateManager?: WebGLStateManager;

    // holder for weights instance where we get tensorMaps
    private weights: Weights;
    private activeTensorMap!: TensorMap;
    private isDirty = true;

    constructor(preferedBackend = 'webgl', canvasOrDevice?: HTMLCanvasElement | GPUDevice) {
        this.weights = Weights.getInstance();
        console.log('%c Denoiser initialized..', 'background: #d66b00; color: white;');
        setupBackend(this, preferedBackend, canvasOrDevice);
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
        if (isReady) this.backendListeners.forEach((callback) => callback(this.backend));
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

    //* Build the unet using props ------------------------
    async build() {
        let startTime: number;
        let endTime: number;
        const { tensorMapLabel, size } = determineTensorMap(this.props);

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
        const inputTensor = await handleModelInput(this);

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
        const output = await handleModelOutput(this, result);

        if (this.debugging) {
            //console.log('Output Tensor')
            //  output.print(); 
            if (this.debugging) console.log(`Denoiser: Execution Time: ${performance.now() - startTime!}ms`);
        }
        return this.handleReturn(output);
    }

    // handle returns and callbacks
    private async handleReturn(outputTensor: tf.Tensor3D) {
        // if we are dumping to canvas
        if (this.outputToCanvas && this.canvas)
            tf.browser.toPixels(outputTensor, this.canvas);

        // handle listeners
        this.listeners.forEach((returnType, callback) => {
            handleCallback(this, outputTensor, returnType, callback);
        });

        // output for direct execution, only fire if we have no listeners
        let toReturn: unknown;
        if (this.listeners.size === 0)
            toReturn = await handleCallback(this, outputTensor, this.outputMode);

        if (this.gl) this.webglStateManager?.saveState();
        return toReturn;
    }

    //* Set the Inputs -----------------------------------
    // set the input tensor. Rarely accessed directly but used by all internals
    setInputTensor(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor3D) {
        // destroy existing tensor
        this.inputTensors.get(name)?.dispose();
        this.inputTensors.set(name, tensor);
    }

    //* Image Input ---
    setImage(name: 'color' | 'albedo' | 'normal', imgData: ImgInput, flipY = false) {
        console.warn('setImage is deprecated, use setInputImage instead');
        this.setInputImage(name, imgData, flipY);
    }

    //set the image and tensor, normalize (potentailly) flipY if needed
    setInputImage(name: 'color' | 'albedo' | 'normal', imgData: ImgInput, flipY = false) {
        let finalData = imgData;
        // if input is color lets take the height and width and set it on this
        if (imgData instanceof HTMLImageElement && hasSizeMissmatch(imgData)) {
            // check if the image is css scaled, if so correct the data
            if (this.debugging) console.log('Image is css scaled, getting correct image data');
            finalData = getCorrectImageData(imgData);
        }
        if (name === 'color') {
            let inHeight = 0, inWidth = 0;
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

                if (this.debugging && (inHeight !== this.height || inWidth !== this.width))
                    console.warn('Denoiser: Image size does not match denoiser size, resizing may occur.');
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
    setInputBuffer(name: 'color' | 'albedo' | 'normal', buffer: GPUBuffer, options: InputOptions = {}) {
        if (name === 'color') adjustSize(this, options);
        const baseTensor = tf.tensor({ buffer: buffer }, [this.height, this.width, options.channels || 4], 'float32') as tf.Tensor3D;
        handleInputTensors(this, name, baseTensor, options);
    }

    // for a webGL texture create and set it
    setInputTexture(name: 'color' | 'albedo' | 'normal', texture: WebGLTexture, options: InputOptions = {}) {
        if (name === 'color') adjustSize(this, options);
        const baseTensor = tf.tensor({ texture, height: this.height, width: this.width, channels: 'RGBA' }, [this.height, this.width, options.channels || 4], 'float32') as tf.Tensor3D;
        handleInputTensors(this, name, baseTensor, options);
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
