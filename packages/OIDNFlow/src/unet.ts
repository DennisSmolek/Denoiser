// a UNet is the actual TF model
import * as tf from '@tensorflow/tfjs';
import type { Tensor4D, SymbolicTensor } from '@tensorflow/tfjs';
import type { TensorMap } from './tza';

export type UnetProps = {
    weights: TensorMap,
    size: 'small' | 'large' | 'xl' | 'default',
    height: number,
    width: number,
    channels: number
};

// this has to be destroyed and is immutable so props wont be exposed
export class UNet {
    height = 0;
    width = 0;
    inChannels = 0;
    // changes params of the uNet
    size: 'small' | 'large' | 'xl' | 'default' = 'default';

    //* internal
    private _inputTensor!: Tensor4D;
    private model!: tf.LayersModel;
    weights: TensorMap;

    constructor({ weights, size, height, width, channels }: UnetProps) {
        this.weights = weights;
        this.size = size;
        this.height = height;
        this.width = width;
        this.inChannels = channels;
    }

    //* Getter and Setters ------------------------------
    set inputTensor(inputTensor: Tensor4D) {
        // cleanup old one
        if (this._inputTensor) this._inputTensor.dispose();
        this._inputTensor = inputTensor;
    }

    async execute() {
        // because I want to see the weights data
        //console.log('input weights:', this.weights);

        if (!this._inputTensor) throw new Error('Input tensor not set!');
        if (!this.model) throw new Error('Model not built yet!');

        // Feed to the network
        const output = this.model.predict(this._inputTensor);

        // Ensure output is a single Tensor before calling Tensor-specific methods
        if (Array.isArray(output))
            throw new Error('Expected model to return a single tensor, but it returned an array.');

        return output as tf.Tensor4D;
    }

    async debugBuild() {
        console.log('%cBuilding Passthrough UNet...', 'color: #EFA00B')
        // Build a passthrough model that returns the input as the output
        const input = tf.input({ shape: [this.height, this.width, this.inChannels] });
        const output = input;
        this.model = tf.model({ inputs: input, outputs: output });

        // because this should be a 0 process pass through lets check equality
        this.model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        return this.model;
    }

    async build() {
        // debug all stats
        console.log(`%cBuilding Standard UNet, %cHeight:${this.height} | Width:${this.width} | Channels:${this.inChannels} | Size:${this.size}`, 'color: #57A773', 'color: white; background-color: #57A773; padding: 2px; border-radius: 4px; font-weight: bold;');

        // input layer ---------------------------------
        const input = tf.input({ shape: [this.height, this.width, this.inChannels] });

        //* Encoder
        // Block 1 -------------------------------------
        let x = this.convLayer('enc_conv0', input);
        x = this.convLayer('enc_conv1', x);
        const pool1 = this.poolLayer(x);
        x = pool1;

        // Block 2 -------------------------------------
        x = this.convLayer('enc_conv2', pool1);
        const pool2 = this.poolLayer(x);

        // Block 3 -------------------------------------
        x = this.convLayer('enc_conv3', pool2);
        const pool3 = this.poolLayer(x);

        // Block 4 -------------------------------------
        x = this.convLayer('enc_conv4', pool3);
        const pool4 = this.poolLayer(x);

        //* Bottleneck ----------------------------------
        x = this.convLayer('enc_conv5a', pool4);
        x = this.convLayer('enc_conv5b', x);

        //* Decoder 
        // Block 4 -------------------------------------
        x = this.upsampleLayer(x);
        x = this.concatenateLayer(x, pool3);
        x = this.convLayer('dec_conv4a', x);
        x = this.convLayer('dec_conv4b', x);

        // Block 3 -------------------------------------
        x = this.upsampleLayer(x);
        x = this.concatenateLayer(x, pool2);
        x = this.convLayer('dec_conv3a', x);
        x = this.convLayer('dec_conv3b', x);

        // Block 2 -------------------------------------
        x = this.upsampleLayer(x);
        x = this.concatenateLayer(x, pool1);
        x = this.convLayer('dec_conv2a', x);
        x = this.convLayer('dec_conv2b', x);

        // Block 1 -------------------------------------
        x = this.upsampleLayer(x);
        x = this.concatenateLayer(x, input);
        x = this.convLayer('dec_conv1a', x);
        x = this.convLayer('dec_conv1b', x);

        // Output -------------------------------------
        x = this.convLayer('dec_conv0', x) as tf.SymbolicTensor;

        // Create the model
        this.model = tf.model({ inputs: input, outputs: x });

        // Compile the model
        this.model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

        // log the model summary
        //this.model.summary();
        // Return the model
        return this.model;
    }

    // generate the conv layers and add weights
    private convLayer(name: string, source: SymbolicTensor): SymbolicTensor {
        const weights = this.weights.get(`${name}.weight`);
        const biases = this.weights.get(`${name}.bias`);
        if (!weights || !biases)
            throw new Error(`UNet: Missing weights or biases for layer: ${name}`);

        const layer = tf.layers.conv2d({
            filters: weights.shape[3]!,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu',
            trainable: false,
            weights: [weights, biases]
        });

        return layer.apply(source) as SymbolicTensor;
    }

    private poolLayer(source: SymbolicTensor): SymbolicTensor {
        return tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(source) as SymbolicTensor;
    }

    private upsampleLayer(source: SymbolicTensor): SymbolicTensor {
        return tf.layers.upSampling2d({ size: [2, 2] }).apply(source) as SymbolicTensor;
    }

    private concatenateLayer(source1: SymbolicTensor, source2: SymbolicTensor): SymbolicTensor {
        return tf.layers.concatenate().apply([source1, source2]) as SymbolicTensor;
    }

    //destroy the model
    destroy() {
        if (this.model) this.model.dispose();
        //dispose of the inputs
        if (this._inputTensor) this._inputTensor.dispose();
    }
}


/* Save this for handling alignment/padding
function calculatePadding(size: number, alignment: number): [number, number] {
const remainder = size % alignment;
if (remainder === 0) {
  return [0, 0]; // No padding needed
} else {
  const padding = alignment - remainder;
  // Split padding evenly on both sides
  const pad1 = Math.floor(padding / 2);
  const pad2 = padding - pad1;
  return [pad1, pad2];
}
}

function padTensor(tensor: tf.Tensor, alignment: number): tf.Tensor {
const [height, width] = tensor.shape.slice(1); // Assuming NHWC format
const [padTop, padBottom] = calculatePadding(height, alignment);
const [padLeft, padRight] = calculatePadding(width, alignment);
 
// Apply padding, assuming NHWC format. Adjust if using a different format.
const paddedTensor = tf.pad(tensor, [[0, 0], [padTop, padBottom], [padLeft, padRight], [0, 0]]);
return paddedTensor;
}

// Example usage
const alignment = 16;
const inputTensor = tf.tensor4d([...], [1, 256, 256, 3]); // Fill in tensor data
const paddedTensor = padTensor(inputTensor, alignment);

// encoder channels
// Encoder channels
        let ec1 = 32;
        let ec2 = 48;
        let ec3 = 64;
        let ec4 = 80;
        let ec5 = 96;
        let dc4 = 112;
        let dc3 = 96;
        let dc2a = 64;
        let dc2b = 64;
        let dc1a = 64;
        let dc1b = 32;
        if (this.size === 'small') {
            ec1 = 32;
            ec2 = 32;
            ec3 = 32;
            ec4 = 32;
            ec5 = 32;
            dc4 = 64;
            dc3 = 64;
            dc2a = 64;
            dc2b = 32;
            dc1a = 32;
            dc1b = 32;
        };
*/