//tf
import * as tf from '@tensorflow/tfjs';
import { Weights } from './weights';

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

    constructor(inputString?: string) {
        this.weights = Weights.getInstance();
        if (inputString) {
            this.baseString = inputString;
        }
    }

    execute() {
        console.log(`Denoising...${this.baseString}`);
    }

    // Other methods...
}