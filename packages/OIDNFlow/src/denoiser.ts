//tf
import * as tf from '@tensorflow/tfjs';
import { Weights } from './weights';
import { loadTestTZA } from './tza';

export class Denoiser {
    // counter to how many times the model was built
    timesGenerated = 0;
    // core weights will be pre-loaded but replaceable
    weights: Weights;

    // configurable options that require rebuilding the model
    activeBackend: 'cpu' | 'webgl' | 'wasm' | 'webgpu' = 'webgpu';

    // changable options
    //if we output inplace on the same input
    inplace = false;

    // holders
    // TODO set actual options
    tfBackend: tf.MathBackendWebGL | null = null;

    constructor() {
        this.weights = Weights.getInstance();
        console.log('Denoiser initialized');

    }

    async execute() {
        console.log('Denoising...');
        const testTensorMap = await this.weights.getCollection();
        for (const tensor of testTensorMap) {
            console.log(tensor[0]);
            tensor[1].print();
        }


    }

    // Other methods...
}