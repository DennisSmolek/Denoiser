//tf
import * as tf from '@tensorflow/tfjs';
import { Weights } from './weights';
import type { TensorMap } from './tza';
import { UNet } from './unet';

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

    // changable options
    //if we output inplace on the same input
    inplace = false;
    // used when debugging
    canvas?: HTMLCanvasElement;
    outputToCanvas = false;

    // holders
    // TODO set actual options
    //tfBackend: tf.MathBackendWebGL;

    private unet!: UNet;
    private inputTensors: Map<'color' | 'albedo' | 'normal', tf.Tensor3D> = new Map();
    private activeTensorMap!: TensorMap;
    private isDirty = false;

    constructor() {
        this.weights = Weights.getInstance();
        console.log('Denoiser initialized');
    }

    //* Getters and Setters ------------------------------
    set height(height: number) {
        if (this.timesGenerated > 0) {
            this.isDirty = true;
            console.log('Model is dirty');
        }
        this.props.height = height;
    }
    set width(width: number) {
        if (this.timesGenerated > 0) {
            this.isDirty = true;
            console.log('Model is dirty');
        }
        this.props.width = width;
    }
    set quality(quality: 'fast' | 'high' | 'balanced') {
        if (this.timesGenerated > 0) {
            this.isDirty = true;
            console.log('Model is dirty');
        }
        this.props.quality = quality;
    }

    // determine which tensormap to use
    private determineTensorMap() {
        // example rt_ldr_small | rt_ldr_calb_cnrm (with clean aux)
        let size: 'small' | 'large' | 'default' = 'default';
        let tensorMapLabel = this.props.filterType;
        tensorMapLabel += this.props.hdr ? '_hdr' : '_ldr';
        // if you use cleanAux you MUST provide both albedo and normal
        if (this.props.cleanAux) tensorMapLabel += '_calb_crnm';
        else {
            tensorMapLabel += this.props.useAlbedo ? '_alb' : '';
            tensorMapLabel += this.props.useNormal ? '_nrm' : '';
        }

        //* quality. 
        // small and large only exist in specific cases
        const hasSmall = ['rt_hdr', 'rt_ldr', 'rt_hdr_alb', 'rt_ldr_alb', 'rt_hdr_alb_nrm', 'rt_ldr_alb_nrm', 'rt_hdr_calb_cnrm', 'rt_ldr_calb_cnrm'];
        const hasLarge = ['rt_alb', 'rt_nrm', 'rt_hdr_calb_cnrm', 'rt_ldr_calb_cnrm'];

        if (this.props.quality === 'fast' && hasSmall.includes(tensorMapLabel)) size = 'small';
        else if (this.props.quality === 'high' && hasLarge.includes(tensorMapLabel)) size = 'large';

        if (size !== 'default') tensorMapLabel += `_${size}`;


        return { tensorMapLabel, size };
    }

    async execute() {
        console.log('Denoising...');
        if (this.isDirty) await this.build();
        const result = await this.unet.execute();
        if (this.outputToCanvas || this.canvas) {
            tf.browser.toPixels(result, this.canvas);
        }
        return result;
    }

    async build() {
        console.log('Building model...');
        const { tensorMapLabel, size } = this.determineTensorMap();
        console.log('Denoiser: Using tensor map:', tensorMapLabel)
        console.log('Denoiser: Using size:', size)
        this.activeTensorMap = await this.weights.getCollection(tensorMapLabel);
        // calculate channels
        let channels = 0;
        if (this.props.useColor) channels += 3;
        if (this.props.useAlbedo) channels += 3;
        if (this.props.useNormal) channels += 3;
        this.unet = new UNet({ weights: this.activeTensorMap, size, height: this.props.height, width: this.props.width, channels });
        this.unet.build();
        // if we already have input tensors set them
        for (const [name, tensor] of this.inputTensors) {
            this.unet.setImage(name, tensor);
        }
        this.timesGenerated++;
    }

    //set the image and tensor
    setImage(name: 'color' | 'albedo' | 'normal', imgData: ImgInput) {
        const tensor = this.createImageTensor(imgData);
        this.setImageTensor(name, tensor);
    }

    // creates the image tensor
    createImageTensor(input: ImgInput): tf.Tensor3D {
        const imgTensor = tf.browser.fromPixels(input);
        return imgTensor;
    }

    // set the input tensor on the unet
    setImageTensor(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor3D) {
        this.inputTensors.set(name, tensor);
        this.unet.setImage(name, tensor);
    }

    setCanvas(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.outputToCanvas = true;
    }
}