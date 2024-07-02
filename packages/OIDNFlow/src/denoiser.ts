//tf
import * as tf from '@tensorflow/tfjs';
import type { Tensor4D } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { Weights } from './weights';
import type { TensorMap } from './tza';
import { UNet } from './unet';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

type ImgInput = tf.PixelData | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;

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
    // core weights will be pre-loaded but replaceable
    weights: Weights;
    // webGL context
    gl?: WebGLRenderingContext;
    backend?: tf.MathBackendWebGL | WebGPUBackend | tf.MathBackendCPU;

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

    // configurable options that require rebuilding the model
    activeBackend: 'cpu' | 'webgl' | 'wasm' | 'webgpu' = 'webgpu';

    //* changable options
    //if we output inplace on the same input
    inplace = false;
    // used when debugging
    canvas?: HTMLCanvasElement;
    outputToCanvas = false;

    debugging = false;

    // holders
    // TODO set actual options
    //tfBackend: tf.MathBackendWebGL;

    private unet!: UNet;
    private inputTensors: Map<'color' | 'albedo' | 'normal', tf.Tensor3D> = new Map();

    private concatenatedImage!: tf.Tensor3D;

    // this allows us to bypass everything and pass a tensor directly to the model
    directInputTensor?: tf.Tensor3D;
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
            if (prefered === 'webgl') {
                //@ts-ignore
                this.backend = tf.getBackend() as tf.MathBackendWebGL;
                console.log('backend:', this.backend);
                this.gl = this.backend.gpgpu.gl;
                return this.backend
            }
        }

        if (prefered === 'webGPU')
            throw new Error('We havent setup custom webGPU Contexts yet');

        if (prefered !== 'webgl')
            throw new Error('Only webgl and webgpu are supported with custom contexts');

        // register the backend if it doesn't exist
        if (tf.findBackend('oidnflow-webgl') === null) {
            const customBackend = new tf.MathBackendWebGL(canvas);
            tf.registerBackend('oidnflow-webgl', () => customBackend);
        }

        //tensorflow does this to restore to the default backend but I dont know if we need it
        const savedBackend = tf.getBackend();
        await tf.setBackend('oidnflow-webgl');
        //@ts-ignore
        const backend = tf.getBackend();
        backend;
        //@ts-ignore
        const testGl = tf.engine().findBackend(backend).gpgpu;
        console.log('testGl:', testGl);
        console.log('%c Denoiser: Backend set to custom webgl', 'background: orange; color: white;');
        return this.backend;
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

    async execute() {
        console.log('%c Denoiser: Denoising...', 'background: blue; color: white;');

        const startTime = performance.now();
        if (this.isDirty) await this.build();
        // send the input to the model
        const inputTensor = await this.handleInput();
        this.unet.inputTensor = inputTensor;

        const result = await this.unet.execute();
        //const result = tf.clone(inputTensor);
        if (this.debugging) {
            const isEqual = tf.equal(inputTensor, result);
            // check if its totally equal
            console.log('Make sure model is a pass trough:');
            isEqual.all().print();
        }

        const output = await this.handleOutput(result);
        //console.log('Output Tensor')
        //  output.print();
        const endTime = performance.now();
        console.log(`Denoiser: Execution Time: ${endTime - startTime}ms`);
        return output;
    }

    private async handleInput(): Promise<Tensor4D> {
        // TODO: if we ever want to denoise just normal or albedo this will have to change as it requires color
        const color = this.inputTensors.get('color');
        const albedo = this.inputTensors.get('albedo');
        const normal = this.inputTensors.get('normal');

        // make sure we have inputs if not throw an error
        if (!color) throw new Error('Denoiser must support color pass at the moment :(');

        // Step 2: Concatenate images along the channel dimension
        // if the concatenated image is not the same as the color image dispose of it
        //todo: this can all probably be wrapped in a tf.tidy
        if (albedo || normal && this.concatenatedImage !== color) this.concatenatedImage.dispose();
        if (albedo) {
            if (normal) this.concatenatedImage = tf.concat([color, albedo, normal], -1);
            else this.concatenatedImage = tf.concat([color, albedo], -1);
        } else if (normal) {
            this.concatenatedImage = tf.concat([color, normal], -1);
        } else this.concatenatedImage = color;
        // Step 3: Reshape for batch size
        return this.concatenatedImage.expandDims(0) as Tensor4D; // Now shape is [1, height, width, 9]
    }

    private async handleOutput(result: tf.Tensor4D) {
        // Now that we've ensured output is not an array, we can safely call squeeze()
        let outputImage = result.squeeze() as tf.Tensor3D;
        // outputImage.print();
        // stops infinite values but might be whiting out things
        outputImage = tf.clipByValue(outputImage, 0, 1);
        // check the datatype and shape of the outputImage
        console.log('Output Image shape:', outputImage.shape, 'dtype:', outputImage.dtype);
        // normalize the output

        /* This normalization block is kinda dirty and comes from the input data not being normalized first
        lets try normalizing the input and see how it flows through.
        // Check if the maximum value in the tensor is greater than 1 to decide on normalization
        outputImage = outputImage.toFloat(); // Ensure the tensor is in float32 format
        tf.tidy(() => {
            const maxValue = outputImage.max().arraySync();
            // Asserting the type of maxValue to be number
            if (typeof maxValue === 'number' && maxValue > 1) {
                outputImage = outputImage.div(tf.scalar(255));
            }
        });
        */
        if (this.outputToCanvas && this.canvas) {
            tf.browser.toPixels(outputImage, this.canvas);
        }
        return outputImage;
    }

    // raw debug for input testing. should draw the image EXACTLY
    rawCanvasDraw() {
        const color = this.inputTensors.get('color');
        if (!color || !this.canvas) return;
        tf.browser.toPixels(color, this.canvas);
    }

    async build() {
        console.log('Denoiser starting Build...');
        const { tensorMapLabel, size } = this.determineTensorMap();
        console.log('Denoiser: Using tensor map:', tensorMapLabel, 'size:', size)
        this.activeTensorMap = await this.weights.getCollection(tensorMapLabel);
        // calculate channels
        let channels = 0;
        if (this.props.useColor) channels += 3;
        if (this.props.useAlbedo) channels += 3;
        if (this.props.useNormal) channels += 3;

        this.unet = new UNet({ weights: this.activeTensorMap, size, height: this.props.height, width: this.props.width, channels });

        const startTime = performance.now();
        if (this.debugging) this.unet.debugBuild();
        else this.unet.build();
        const endTime = performance.now();
        console.log('Denoiser: Build Time:', endTime - startTime);

        this.timesGenerated++;
        this.isDirty = false;
    }

    //set the image and tensor
    setImage(name: 'color' | 'albedo' | 'normal', imgData: ImgInput) {
        const tensor = this.createImageTensor(imgData);
        this.setImageTensor(name, tensor);
    }

    // creates the image tensor
    createImageTensor(input: ImgInput): tf.Tensor3D {
        const imgTensor = tf.browser.fromPixels(input);
        // normalize the tensor
        const normalized = imgTensor.toFloat().div(tf.scalar(255));
        return normalized as tf.Tensor3D;
        //return imgTensor;
    }

    // set the input tensor on the unet
    setImageTensor(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor3D) {
        this.inputTensors.set(name, tensor);
    }

    setCanvas(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.outputToCanvas = true;
    }
}