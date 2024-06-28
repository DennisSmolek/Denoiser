// a UNet is the actual TF model
import * as tf from '@tensorflow/tfjs';
import type { Weights } from './weights';

// this has to be destroyed and is immutable so props wont be exposed
export class UNet {
    weights: Weights;
    filterType: 'rt' | 'rtLightmap' = 'rt';
    quality: 'high' | 'balanced' | 'fast' = 'fast';
    height: number;
    width: number;
    cleanAux = false;
    hdr = false;
    srgb = false;
    directionals = false;

    // not sure on this one
    inputScale = 1.0;
    numRuns = 1;

    //* Inputs
    color: tf.Tensor2D;
    albedo: tf.Tensor2D;
    normal: tf.Tensor2D;

    //TODO make type for the inputs required for the constructor
    constructor({ weights }) {
        this.weights = weights
    }

    async build() {
        //TODO build the model
    }

    async commit() {
        //TODO commit the model
    }
    setImage(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor2D) {
        // only allow the known names
        if (!['color', 'albedo', 'normal'].includes(name)) {
            throw new Error(`UNet: Unknown input name: ${name}`);
        }
        // destroy the old inputs
        if (this[name]) this[name].dispose();

        this[name] = tensor;
    }

    //destroy the model
    destroy() {
        //TODO destroy the model
    }
}