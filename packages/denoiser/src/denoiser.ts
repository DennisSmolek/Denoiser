//tf
import * as tf from '@tensorflow/tfjs';
import { Weights } from './weights';
import { UNet } from './unet';
import { formatTime, getCorrectImageData, hasSizeMissmatch } from './utils';
import { GPUTensorTiler } from './tiler';
import { WebGLStateManager } from './webglStateManager';
import { setupBackend, determineTensorMap, handleModelInput, handleModelOutput, handleCallback, adjustSize, handleInputTensors, warmstart, profileModel } from './denoiserUtils';
//import { logMemoryUsage } from './utils';
// TODO: These shouldnt have to be imported
import type { TensorMap, DenoiserProps, ImgInput, InputOptions, ListenerCalback, ModelInput } from 'types/types';


export class Denoiser {
    // counter to how many times the model was built
    timesGenerated = 0;
    // these probably can be deleted
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
        batchSize: 4,
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

    //* Tiling ---
    _useTiling = false;
    _tilingUserBlocked = false
    tileStride = 16;
    tileSize = 256;

    //* Internal -----------------------------------
    // listeners for execution callbacks
    private listeners: Map<ListenerCalback, string> = new Map();
    private backendListeners: Set<ListenerCalback> = new Set();
    private progressListeners: Set<(progress: number) => void> = new Set();

    private aborted = false;
    // Model Props ---
    private unet!: UNet;
    private tiler?: GPUTensorTiler;
    inputTensors: Map<'color' | 'albedo' | 'normal', tf.Tensor3D> = new Map();
    inputAlpha?: tf.Tensor3D;
    oldOutputTensor?: tf.Tensor3D;

    // WebGL ---
    public webglStateManager?: WebGLStateManager;
    private _gl?: WebGL2RenderingContext;
    device?: GPUDevice;
    warmstart = true;

    // holder for weights instance where we get tensorMaps
    private weights: Weights;
    private activeTensorMap!: TensorMap;
    private isDirty = true;

    //* Debugging -----------------------------------
    canvas?: HTMLCanvasElement;
    outputToCanvas = false;
    debugging = false;
    profiling = false;
    usePassThrough = false;

    stats: { [key: string]: number | string } = {};
    timers: { [key: string]: number } = {
        buildStart: 0,
        buildEnd: 0,
        executionStart: 0,
        executionEnd: 0,
        tilingStart: 0,
        tilingEnd: 0

    };

    constructor(preferedBackend = 'webgl', canvasOrDevice?: HTMLCanvasElement | GPUDevice) {
        this.weights = Weights.getInstance();
        tf.enableProdMode();
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

    get srgb() {
        return this.props.srgb;
    }
    set srgb(srgb: boolean) {
        this.isDirty = true;
        this.props.srgb = srgb;
        if (this.tiler) this.tiler.srgb = srgb;
    }

    get batchSize() {
        return this.props.batchSize;
    }

    set batchSize(batchSize: number) {
        this.isDirty = true;
        this.props.batchSize = batchSize;
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
        if (isReady) this.backendListeners.forEach((callback) => callback(this.backend));
    }

    get gl(): WebGL2RenderingContext | undefined {
        return this._gl;
    }
    set gl(gl: WebGL2RenderingContext) {
        if (!gl) return;
        this._gl = gl;
        this.webglStateManager = new WebGLStateManager(gl);
        this.webglStateManager.captureCurrentState();
    }

    get useTiling() {
        return this._useTiling;
    }
    set useTiling(useTiling: boolean) {
        this._tilingUserBlocked = !useTiling;
        this._useTiling = useTiling;
    }

    set debuggTF(debug: boolean) {
        if (debug) tf.enableDebugMode();
        else tf.enableProdMode();
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

        // if we already have a UNet instance, dispose it
        if (this.unet) this.unet.dispose();
        if (this.tiler) this.tiler.dispose();

        this.startTimer('build');
        this.unet = new UNet({ weights: this.activeTensorMap, size, height, width, channels });

        const model = (this.usePassThrough) ? await this.unet.debugBuild() : await this.unet.build();
        if (!model) throw new Error('UNet Model failed to build');
        this.stopTimer('build');

        // Warmstart the model
        if (this.warmstart) {
            this.startTimer('warmstart');
            warmstart(model, height, width, channels);
            this.stopTimer('warmstart');
        }

        //* Tiling
        if (this.useTiling) this.tiler = new GPUTensorTiler(model, { tileSize: this.tileSize, srgb: this.props.srgb, batchSize: this.props.batchSize });


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
                if (colorInput) await this.setInputTexture('color', colorInput as WebGLTexture);
                if (albedoInput) await this.setInputTexture('albedo', albedoInput as WebGLTexture);
                if (normalInput) await this.setInputTexture('normal', normalInput as WebGLTexture);
                break;

            case 'webgpu':
                if (!this.height || !this.width) throw new Error('Denoiser: Height and Width must be set when executing with webGPU input.');
                if (colorInput) await this.setInputBuffer('color', colorInput as GPUBuffer);
                if (albedoInput) await this.setInputBuffer('albedo', albedoInput as GPUBuffer);
                if (normalInput) await this.setInputBuffer('normal', normalInput as GPUBuffer);
                break;

            case 'tensor':
                if (colorInput) this.setInputTensor('color', colorInput as tf.Tensor3D);
                if (albedoInput) this.setInputTensor('albedo', albedoInput as tf.Tensor3D);
                if (normalInput) this.setInputTensor('normal', normalInput as tf.Tensor3D);
                break;

            default:
                if (colorInput) this.setInputImage('color', colorInput as ImgInput);
                if (albedoInput) this.setInputImage('albedo', albedoInput as ImgInput);
                if (normalInput) this.setInputImage('normal', normalInput as ImgInput);
        }

        return this.executeModel();
    }

    // actually execute the model with the set inputs of this class
    private async executeModel() {
        if (this.debugging) console.log('%c Denoiser: Denoising...', 'background: blue; color: white;');
        // reset an aborted flag
        this.aborted = false;
        this.startTimer('execution');
        // process and send the input to the model
        const inputTensor = await handleModelInput(this);

        // if we need to rebuild the model
        if (this.isDirty) await this.build();

        // If profiling
        if (this.profiling) {
            const testTensor = GPUTensorTiler.generateSampleInput(this.tileSize, this.tileSize);
            await profileModel(this.unet.model, testTensor);
        }

        // Execute model with tiling or standard
        this.startTimer('tiling');
        const result = this.useTiling ? await this.tiler!.processLargeTensor(inputTensor, (progress) => this.handleProgress(progress))
            : await this.unet.execute(inputTensor);
        this.stopTimer('tiling');

        // if we abort
        if (this.aborted) return this.handleAbort();

        // process the output
        const output = await handleModelOutput(this, result);
        return this.handleReturn(output);
    }

    // handle returns and callbacks
    private async handleReturn(outputTensor: tf.Tensor3D) {
        // if we are dumping to canvas
        if (this.outputToCanvas && this.canvas)
            tf.browser.toPixels(outputTensor, this.canvas);

        // handle listeners
        this.listeners.forEach((returnType, callback) =>
            handleCallback(this, outputTensor, returnType, callback));

        // output for direct execution, only fire if we have no listeners
        let toReturn: unknown;
        if (this.listeners.size === 0)
            toReturn = await handleCallback(this, outputTensor, this.outputMode);

        if (this.gl) this.webglStateManager?.saveState();
        this.stopTimer('execution');
        this.logStats();
        return toReturn;
    }

    private async handleAbort() {
        if (this.gl) this.webglStateManager?.saveState();
        this.stopTimer('execution');
    }

    //* Set the Inputs -----------------------------------
    // set the input tensor. Rarely accessed directly but used by all internals
    setInputTensor(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor3D) {
        if (!tensor) throw new Error('Denoiser: No tensor provided');
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
        if (!imgData) throw new Error('Denoiser: No image data provided');
        let finalData = imgData;
        // if input is color lets take the height and width and set it on this
        if (imgData instanceof HTMLImageElement && hasSizeMissmatch(imgData)) {
            // check if the image is css scaled, if so correct the data
            if (this.debugging) console.log('Image is css scaled, getting correct image data');
            finalData = getCorrectImageData(imgData);
        }
        if (name === 'color') {
            this.props.useColor = true;
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
            return tf.tidy(() => {
                const imageTensor = this.createImageTensor(finalData, name === 'normal');
                if (!flipY) return imageTensor;
                return tf.reverse(imageTensor, [0]);
            });
        }));
    }

    // creates the image tensor
    createImageTensor(input: ImgInput, isNormalMap = false): tf.Tensor3D {
        return tf.tidy(() => {
            const imgTensor = tf.browser.fromPixels(input);
            // standard normalization
            if (!isNormalMap) return imgTensor.toFloat().div(tf.scalar(255)) as tf.Tensor3D;

            // normalize the tensor to OIDN expect range of -1 to 1
            return imgTensor.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));
        });
    }

    // for a webGPU buffer create and set it
    async setInputBuffer(name: 'color' | 'albedo' | 'normal', buffer: GPUBuffer, options: InputOptions = {}) {
        if (!buffer) throw new Error('Denoiser: No buffer provided');
        if (name === 'color') adjustSize(this, options);
        const baseTensor = tf.tensor({ buffer: buffer }, [this.height, this.width, options.channels || 4], 'float32') as tf.Tensor3D;
        await handleInputTensors(this, name, baseTensor, options);
    }

    // for a webGL texture create and set it
    async setInputTexture(name: 'color' | 'albedo' | 'normal', texture: WebGLTexture, options: InputOptions = {}) {
        if (!texture) throw new Error('Denoiser: No texture provided');
        if (name === 'color') adjustSize(this, options);
        const baseTensor = tf.tensor({ texture, height: this.height, width: this.width, channels: 'RGBA' }, [this.height, this.width, options.channels || 4], 'float32') as tf.Tensor3D;
        await handleInputTensors(this, name, baseTensor, options);
    }

    //* Data Input ---
    setInputData(name: 'color' | 'albedo' | 'normal', data: Float32Array | Uint8Array, options: InputOptions = {}) {
        if (!data) throw new Error('Denoiser: No data provided');
        // check if data is a float32 and reject otherwise
        if (data.constructor !== Float32Array)
            throw new Error('Invalid Input data type. Must be a Float32Array');

        const baseTensor = tf.tensor3d(data, [this.height, this.width, options.channels || 4])
        handleInputTensors(this, name, baseTensor, options);
    }

    // this is mostly an internal debug thing but is super helpful
    setCanvas(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.outputToCanvas = true;
    }
    // clear the input tensors and mark dirty
    resetInputs() {
        console.log('%c Denoiser: RESETTING INPUTS', 'background: red; color: white');
        this.inputTensors.forEach((tensor) => tensor.dispose());
        this.inputTensors.clear();
        if (this.inputAlpha) {
            this.inputAlpha.dispose();
            this.inputAlpha = undefined;
        }
        if (this.directInputTensor) this.directInputTensor.dispose();
        this.props.useColor = false;
        this.props.useAlbedo = false;
        this.props.useNormal = false;
        this.isDirty = true;
    }
    // cancel current execution
    abort() {
        this.aborted = true;
        if (this.debugging) console.log('%c Denoiser: ABORTING', 'background: red; color: white');

        if (this.tiler) this.tiler.abort();
    }

    dispose() {
        this.unet.dispose();
        if (this.tiler) this.tiler.dispose();
        this.inputTensors.forEach((tensor) => tensor.dispose());
        if (this.inputAlpha) this.inputAlpha.dispose();
        if (this.directInputTensor) this.directInputTensor.dispose();
    }

    //* Listeners ---------------------------------------

    private handleProgress(progress: number) {
        this.progressListeners.forEach((listener) => listener(progress));
    }

    // add a listener for progress of the tiler
    onProgress(listener: (progress: number) => void) {
        this.progressListeners.add(listener);
        return () => this.progressListeners.delete(listener);
    }


    // Add a listener to the denoiser with a return function to stop listening
    onExecute(listener: ListenerCalback, responseType = this.outputMode) {
        this.listeners.set(listener, responseType);
        return () => this.listeners.delete(listener);
    }

    onBackendReady(listener: ListenerCalback) {
        // if the backend is already ready fire the listener
        if (this.backendReady) listener(this.backend);
        else this.backendListeners.add(listener);
        return () => this.backendListeners.delete(listener);
    }
    //* Debuging ----------------------------------------
    startTimer(name: string) {
        if (!this.debugging) return;
        this.timers[`${name}In`] = performance.now();
    }
    stopTimer(name: string) {
        if (!this.debugging) return;
        this.timers[`${name}Out`] = performance.now();
        this.stats[name] = this.timers[`${name}Out`] - this.timers[`${name}In`];
    }
    logStats() {
        if (!this.debugging) return;
        console.log('%c Denoiser Stats:', 'background: #73BFB8; color: white');
        // parse the times and give nicer formats
        const formattedStats = Object.entries(this.stats).reduce((acc, [key, value]) => {
            acc[key] = (typeof value === "string") ? value : formatTime(value as number);
            return acc;
        }, {} as { [key: string]: string });
        console.table(formattedStats);
    }
}
