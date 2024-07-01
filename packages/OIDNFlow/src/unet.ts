// a UNet is the actual TF model
import * as tf from '@tensorflow/tfjs';
import type { Tensor4D } from '@tensorflow/tfjs';
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
    private outChannels = 3;
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

    async buildPassthrough() {
        console.log('%cBuilding Passthrough UNet...', 'color: #EFA00B')
        // Build a passthrough model that returns the input as the output
        const input = tf.input({ shape: [this.height, this.width, this.inChannels] });
        const output = input;
        this.model = tf.model({ inputs: input, outputs: output });

        // because this should be a 0 process pass through lets check equality
        this.model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        return this.model;
    }

    // build a debug model that takes input does a simple downsize and upscale
    async debugBuild() {
        console.log('%cBuilding DEBUGING UNet...', 'color: yellow');
        const input = tf.input({ shape: [this.height, this.width, this.inChannels] });

        // Downsize
        const downsize = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(input) as tf.SymbolicTensor;

        // Upscale
        const upscale = tf.layers.upSampling2d({ size: [2, 2] }).apply(downsize) as tf.SymbolicTensor;

        // Output
        const output = tf.layers.conv2d({
            filters: this.outChannels,
            kernelSize: 1,
            padding: 'same',
            activation: 'linear',
        }).apply(upscale) as tf.SymbolicTensor;

        // Create the model
        this.model = tf.model({ inputs: input, outputs: output });

        // Compile the model
        this.model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        // Log the model summary
        this.model.summary();

        // Return the model
        return this.model;
    }

    async build() {
        // debug all stats
        console.log('%cBuilding Standard UNet...', 'color: #57A773');
        console.log('Height:', this.height);
        console.log('Width:', this.width);
        console.log('Channels:', this.inChannels);
        console.log('Size:', this.size);
        console.log('Weights:', this.weights);


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

        // input layer ---------------------------------
        const input = tf.input({ shape: [this.height, this.width, this.inChannels] });

        //* Encoder
        // Block 1 -------------------------------------
        let x = this.convLayer('enc_conv0', this.inChannels, ec1, true).apply(input);
        x = this.convLayer('enc_conv1', ec1, ec1).apply(x);
        const pool1 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(x) as tf.SymbolicTensor;

        // Block 2 -------------------------------------
        x = this.convLayer('enc_conv2', ec1, ec2).apply(pool1);
        const pool2 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(x) as tf.SymbolicTensor;

        // Block 3 -------------------------------------
        x = this.convLayer('enc_conv3', ec2, ec3).apply(pool2);
        const pool3 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(x) as tf.SymbolicTensor;

        // Block 4 -------------------------------------
        x = this.convLayer('enc_conv4', ec3, ec4).apply(pool3);
        const pool4 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(x);

        //* Bottleneck ----------------------------------
        x = this.convLayer('enc_conv5a', ec4, ec5).apply(pool4);
        x = this.convLayer('enc_conv5b', ec5, ec5).apply(x);

        //* Decoder 
        // Block 4 -------------------------------------
        x = tf.layers.upSampling2d({ size: [2, 2] }).apply(x) as tf.SymbolicTensor;
        x = tf.layers.concatenate().apply([x, pool3]);
        x = this.convLayer('dec_conv4a', ec5 + ec3, dc4).apply(x) as tf.SymbolicTensor;
        x = this.convLayer('dec_conv4b', dc4, dc4).apply(x);

        // Block 3 -------------------------------------
        x = tf.layers.upSampling2d({ size: [2, 2] }).apply(x) as tf.SymbolicTensor;
        x = tf.layers.concatenate().apply([x, pool2]);
        x = this.convLayer('dec_conv3a', dc4 + ec2, dc3).apply(x) as tf.SymbolicTensor;
        x = this.convLayer('dec_conv3b', dc3, dc3).apply(x);

        // Block 2 -------------------------------------
        x = tf.layers.upSampling2d({ size: [2, 2] }).apply(x) as tf.SymbolicTensor;
        x = tf.layers.concatenate().apply([x, pool1]);
        x = this.convLayer('dec_conv2a', dc3 + ec1, dc2a).apply(x) as tf.SymbolicTensor;
        x = this.convLayer('dec_conv2b', dc2a, dc2b).apply(x);

        // Block 1 -------------------------------------
        x = tf.layers.upSampling2d({ size: [2, 2] }).apply(x) as tf.SymbolicTensor;
        x = tf.layers.concatenate().apply([x, input]);
        x = this.convLayer('dec_conv1a', dc2b + this.inChannels, dc1a).apply(x) as tf.SymbolicTensor;
        x = this.convLayer('dec_conv1b', dc1a, dc1b).apply(x);

        // Output -------------------------------------
        x = this.convLayer('dec_conv0', dc1b, this.outChannels).apply(x) as tf.SymbolicTensor;

        // Create the model
        this.model = tf.model({ inputs: input, outputs: x });

        // Compile the model
        this.model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

        // log the model summary
        this.model.summary();
        // Return the model
        return this.model;
    }
    // generate the conv layers and add weights
    private convLayer(name: string, in_channels: number, out_channels: number, isFirstLayer = false): tf.layers.Layer {
        // Define the layer without specifying the inputShape in the initial configuration.
        // The input shape is only needed if this is the first layer in the model.
        const layer = tf.layers.conv2d({
            filters: out_channels,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu',
        });

        // Assuming `this.weights` is a map where keys are layer names and values are objects
        // with `weight` and `bias` properties, each being a tf.Tensor.
        const weights = this.weights.get(`${name}.weight`);
        const biases = this.weights.get(`${name}.bias`);


        if (!weights || !biases) {
            throw new Error(`UNet: Missing weights or biases for layer: ${name}`);
        }

        // Manually build the layer with the expected input shape if it's not already built.
        // This is crucial if you're setting weights before connecting the layer to an input.
        // Note: Adjust `[null, null, in_channels]` based on your actual input shape requirements.
        // Adjust here for the first layer
        if (isFirstLayer) {
            // For the first layer, explicitly set the input shape including the channels
            layer.build([null, null, null, in_channels]); // Adjusted for the input shape
        } else if (!layer.built) {
            layer.build([null, null, null, in_channels]);
        }
        // Set the weights and biases.
        layer.setWeights([weights, biases]);

        return layer;
    }


    async execute() {
        // because I want to see the weights data
        //console.log('input weights:', this.weights);

        if (!this._inputTensor) throw new Error('Input tensor not set!');
        if (!this.model) throw new Error('Model not built yet!');
        // Step 4: Feed to the network
        const output = this.model.predict(this._inputTensor);

        // Ensure output is a single Tensor before calling Tensor-specific methods
        if (Array.isArray(output)) {
            throw new Error('Expected model to return a single tensor, but it returned an array.');
        }

        return output as tf.Tensor4D;
    }


    //destroy the model
    destroy() {
        if (this.model) this.model.dispose();
        //dispose of the inputs
        if (this.inputTensor) this.inputTensor.dispose();
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
*/