//tf
import * as tf from '@tensorflow/tfjs';
import type { Tensor4D } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { Weights } from './weights';
import type { TensorMap } from './tza';
import { UNet } from './unet';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

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
    gl?: WebGLRenderingContext;
    backend?: tf.MathBackendWebGL | WebGPUBackend | tf.MathBackendCPU;

    // kept in its own object because changing these will require a rebuild
    props: DenoiserProps = {
        filterType: 'rt',
        quality: 'fast',
        hdr: false,
        srgb: false,
        height: 512,
        width: 512,
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

    private unet!: UNet;
    private inputTensors: Map<'color' | 'albedo' | 'normal', tf.Tensor3D> = new Map();
    private inputAlpha?: Float32Array;

    //todo remove?
    private concatenatedImage!: tf.Tensor3D;

    // holder for weights instance where we get tensorMaps
    private weights: Weights;
    private activeTensorMap!: TensorMap;
    private isDirty = false;

    constructor(preferedBackend = 'webgpu', canvas?: HTMLCanvasElement) {
        this.weights = Weights.getInstance();
        console.log('Denoiser initialized..');
        this.setupBackend(preferedBackend, canvas);
    }

    //* Getters and Setters ------------------------------
    set height(height: number) {
        this.isDirty = true;
        this.props.height = height;
    }
    set width(width: number) {
        this.isDirty = true;
        this.props.width = width;
    }
    set quality(quality: 'fast' | 'high' | 'balanced') {
        this.isDirty = true;
        this.props.quality = quality;
    }

    // Take in potential contexts and set the backend
    //todo: should this be private?
    async setupBackend(prefered = 'webgpu', canvas?: HTMLCanvasElement) {
        // do the easy part first
        if (!canvas) {
            await tf.setBackend(prefered);
            //@ts-ignore
            this.backend = tf.getBackend();
            console.log(`Denoiser: Backend set to ${prefered}`);
            return this.backend;
        }

        if (prefered === 'webGPU')
            throw new Error('We havent setup custom webGPU Contexts yet');

        if (prefered !== 'webgl')
            throw new Error('Only webgl and webgpu are supported with custom contexts');
        //* Setup a webGL backend with a custom context
        // if multiple denoisers exist they can share contexts
        // register the backend if it doesn't exist
        if (tf.findBackend('oidnflow-webgl') === null) {
            // we have to make sure all the kernels are registered
            const kernels = tf.getKernelsForBackend('webgl');
            for (const kernelConfig of kernels) {
                const newKernelConfig = { ...kernelConfig, backendName: 'oidnflow-webgl' };
                tf.registerKernel(newKernelConfig);
            }
            const customBackend = new tf.MathBackendWebGL(canvas);
            tf.registerBackend('oidnflow-webgl', () => customBackend);
        }
        await tf.setBackend('oidnflow-webgl');
        //@ts-ignore
        const backend = tf.getBackend();
        //@ts-ignore
        const testGl = tf.engine().findBackend(backend).gpgpu;
        console.log('testGl:', testGl);
        console.log('%c Denoiser: Backend set to custom webgl', 'background: orange; color: white;');
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
        if (colorInput || albedoInput || normalInput) switch (this.inputMode) {
            case 'webgl':
                //todo: implement webgl input
                break;
            case 'webgpu':
                //todo: implement webgpu input
                break;
            case 'tensor':
                if (colorInput) this.setImageTensor('color', colorInput as tf.Tensor3D);
                if (albedoInput) this.setImageTensor('albedo', albedoInput as tf.Tensor3D);
                if (normalInput) this.setImageTensor('normal', normalInput as tf.Tensor3D);
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

        //        if (!color && !albedo && !normal) throw new Error('Denoiser must have an input set before execution.');
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
            // check the datatype and shape of the outputImage
            if (this.debugging) console.log('Output Image shape:', output.shape, 'dtype:', output.dtype);
            return output;
        });
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
        return this.handleCallback(outputTensor, this.outputMode);
    }

    private async handleCallback(outputTensor: tf.Tensor3D, returnType: string, callback?: ListenerCalback) {
        let toReturn: tf.Tensor3D | ImageData | WebGLTexture | GPUBuffer;
        let data: tf.GPUData;
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
            case 'float32':
                toReturn = addAlphaChannel(outputTensor.dataSync() as Float32Array, outputTensor.shape[0] * outputTensor.shape[1], this.inputAlpha);
                break;
            default: {
                const pixelData = await tf.browser.toPixels(outputTensor);
                toReturn = new ImageData(pixelData, this.props.width, this.props.height);
            }
        }
        if (callback) callback(toReturn!);
        return toReturn!;
    }

    //* Set the Inputs -----------------------------------

    // set the input tensor
    setInputTensor(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor3D) {
        console.log('Setting Image Tensor:', name, tensor.shape);
        // destroy existing tensor
        const existingTensor = this.inputTensors.get(name);
        if (existingTensor) existingTensor.dispose();

        this.inputTensors.set(name, tensor);
    }

    //* Data Input ---
    setData(name: 'color' | 'albedo' | 'normal', data: Float32Array, height: number, width: number, noAlpha = false) {
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
        if (!noAlpha) {
            const [rgbData, alphaData] = splitAlphaChannel(data, width * height);
            this.setInputTensor(name, tf.tensor3d(rgbData, [height, width, 3]));
            // we only care about the alpha of the color input
            if (name === 'color') this.inputAlpha = alphaData;
        } else this.setInputTensor(name, tf.tensor3d(data, [height, width, 3]));
    }

    //* Image Input ---
    //set the image and tensor
    setImage(name: 'color' | 'albedo' | 'normal', imgData: ImgInput) {
        let tensor: tf.Tensor3D;
        if (name === 'normal') {
            // TODO determine if this is the best way to handle normal maps
            tensor = this.createImageTensor(imgData, false);
        } else tensor = this.createImageTensor(imgData);
        this.setInputTensor(name, tensor);
    }

    // creates the image tensor
    createImageTensor(input: ImgInput, normalize = true): tf.Tensor3D {
        return tf.tidy(() => {
            const imgTensor = tf.browser.fromPixels(input);
            if (!normalize) return imgTensor as tf.Tensor3D;
            // normalize the tensor
            const normalized = imgTensor.toFloat().div(tf.scalar(255));
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
}

//* Utils ----------------------------------------------
// get the alpha channel from a float32 array
function getAlphaChannel(imageData: Float32Array, count: number): Float32Array {
    const alphaData = new Float32Array(count);
    let j = 0; // Index for alphaData array

    for (let i = 3; i < imageData.length; i += 4) {
        alphaData[j++] = imageData[i];
    }

    return alphaData;
}

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
    const output = new Float32Array(itemCount * 4); // Create a new array with space for the alpha channel
    let j = 0; // Index for output array

    for (let i = 0; i < imageData.length; i += 3) {
        output[j++] = imageData[i];     // R
        output[j++] = imageData[i + 1]; // G
        output[j++] = imageData[i + 2]; // B
        // If we have saved alpha data, use it; otherwise, default to 1 (fully opaque)
        output[j++] = alphaData ? alphaData[i / 3] : 1;
    }

    return output;
}