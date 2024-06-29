// a UNet is the actual TF model
import * as tf from '@tensorflow/tfjs';
import type { ConvLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/convolutional';
import { Weights } from './weights';
import { TensorMap } from './tza';

// this has to be destroyed and is immutable so props wont be exposed
export class UNet {
    weights: TensorMap;
    filterType: 'rt' | 'rtlightmap' = 'rt';
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

    // model parameters
    outChannels = 3;
    size: 'small' | 'large' | 'xl' | 'default' = 'default';

    //* Inputs
    color: tf.Tensor3D;
    albedo: tf.Tensor3D;
    normal: tf.Tensor3D;

    //* internal
    private model: tf.LayersModel;

    //TODO make type for the inputs required for the constructor
    constructor({ weights }: { weights: TensorMap }) {
        this.weights = weights
    }

    //* Getters n Setters
    get inChannels() {
        let channels = 0;
        if (this.color) channels += 3;
        if (this.albedo) channels += 3;
        if (this.normal) channels += 3;
        return channels;
    }

    async build() {
        //TODO build the model
        const ic = this.inChannels;
        const oc = this.outChannels;

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
        // put base convolutions in a map so we can wrap them later
        const convs = new Map();

        convs.set('enc_conv0', Conv(ic, ec1));
        convs.set('enc_conv1', Conv(ec1, ec1));
        convs.set('enc_conv2', Conv(ec1, ec2));
        convs.set('enc_conv3', Conv(ec2, ec3));
        convs.set('enc_conv4', Conv(ec3, ec4));
        convs.set('enc_conv5a', Conv(ec4, ec5));
        convs.set('enc_conv5b', Conv(ec5, ec5));
        convs.set('dec_conv4a', Conv(ec5 + ec3, dc4));
        convs.set('dec_conv4b', Conv(dc4, dc4));
        convs.set('dec_conv3a', Conv(dc4 + ec2, dc3));
        convs.set('dec_conv3b', Conv(dc3, dc3));
        convs.set('dec_conv2a', Conv(dc3 + ec1, dc2a));
        convs.set('dec_conv2b', Conv(dc2a, dc2b));
        convs.set('dec_conv1a', Conv(dc2b + ic, dc1a));
        convs.set('dec_conv1b', Conv(dc1a, dc1b));
        convs.set('dec_conv0', Conv(dc1b, oc));

        const model = tf.sequential();

        // input layer ---------------------------------
        const input = tf.layers.inputLayer({ inputShape: [this.height, this.width, ic] });
        model.add(input);

        //* Encoder
        // Block 1 -------------------------------------
        model.add(this.convLayer('enc_conv0', ic, ec1));
        model.add(this.convLayer('enc_conv1', ec1, ec1));
        const pool1 = model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Block 2 -------------------------------------
        model.add(this.convLayer('enc_conv2', ec1, ec2));
        const pool2 = model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Block 3 -------------------------------------
        model.add(this.convLayer('enc_conv3', ec2, ec3));
        const pool3 = model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Block 4 -------------------------------------
        model.add(this.convLayer('enc_conv4', ec3, ec4));
        const pool4 = model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Bottleneck ----------------------------------
        model.add(this.convLayer('enc_conv5a', ec4, ec5));
        model.add(this.convLayer('enc_conv5b', ec5, ec5));

        //* Decoder 
        // Block 4 -------------------------------------
        model.add(tf.layers.upSampling2d({ size: [2, 2] }));



    }
    // generate the conv layers and add weights
    convLayer(name: string, in_channels: number, out_channels: number): tf.layers.Layer {
        const layer = tf.layers.conv2d({
            filters: out_channels,
            kernelSize: 3,
            padding: 'same',
            activation: 'relu',
            inputShape: [null, null, in_channels]
        });
        this.setWeightsAndBiases(name, layer);
        return layer;
    }

    // todo: determine if this can be merged with convLayer
    setWeightsAndBiases(name: string, layer: tf.layers.Layer) {
        // get the weights and biases from the weights map
        const weights = this.weights.get(`${this.filterType}_${name}_weights`);
        const biases = this.weights.get(`${this.filterType}_${name}_biases`);

        if (!weights || !biases) throw new Error(`UNet: Missing weights or biases for layer: ${name}`);

        // set the weights and biases
        layer.setWeights([weights, biases]);
    }

    async commit() {
        //TODO commit the model
    }

    async execute() {
        //TODO execute the model
        // Assuming `model` is your TensorFlow.js model
        // and `rgbImage`, `albedoImage`, `normalsImage` are your input images as tf.Tensor3D

        // Step 2: Concatenate images along the channel dimension
        let concatenatedImage: tf.Tensor3D;
        if (this.albedo) {
            if (this.normal) concatenatedImage = tf.concat([this.color, this.albedo, this.normal], -1);
            else concatenatedImage = tf.concat([this.color, this.albedo], -1);
        } else if (this.normal) {
            concatenatedImage = tf.concat([this.color, this.normal], -1);
        } else concatenatedImage = this.color;


        // Step 3: Reshape for batch size
        const inputTensor = concatenatedImage.expandDims(0); // Now shape is [1, height, width, 9]

        // Step 4: Feed to the network
        const output = this.model.predict(inputTensor);
    }
    setImage(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor3D) {
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


// Network layers
function Conv(in_channels: number, out_channels: number): tf.layers.Layer {
    return tf.layers.conv2d({
        filters: out_channels,
        kernelSize: 3,
        padding: 'same',
        activation: 'linear',
        inputShape: [null, null, in_channels]
    });
}

function convParams(in_channels: number, out_channels: number): ConvLayerArgs {
    return {
        filters: out_channels,
        kernelSize: 3,
        padding: "same",
        activation: 'relu',
        //TODO: this may not be needed in tf.js
        inputShape: [null, null, in_channels]
    };
}

function relu(x: tf.Tensor): tf.Tensor {
    return tf.relu(x);
}

function pool(x: tf.Tensor): tf.Tensor {
    return tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(x) as tf.Tensor;
}

function upsample(x: tf.Tensor): tf.Tensor {
    return tf.layers.upSampling2d({ size: [2, 2] }).apply(x) as tf.Tensor;
}

function concat(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.concat([a, b], -1);
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
const inputTensor = tf.tensor4d([...], [1, /* height */, /* width */, /* channels */]); // Fill in tensor data
//const paddedTensor = padTensor(inputTensor, alignment);
