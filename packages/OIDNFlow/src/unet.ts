// a UNet is the actual TF model
import * as tf from '@tensorflow/tfjs';
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
    weights: TensorMap;
    height = 0;
    width = 0;

    // model parameters
    inChannels = 0;
    private outChannels = 3;
    size: 'small' | 'large' | 'xl' | 'default' = 'default';

    //* Inputs
    color?: tf.Tensor3D;
    albedo?: tf.Tensor3D;
    normal?: tf.Tensor3D;

    private concatenatedImage!: tf.Tensor3D;
    private inputTensor!: tf.Tensor3D;
    // this allows us to bypass everything and pass a tensor directly to the model
    directInputTensor?: tf.Tensor3D;

    //* internal
    private model!: tf.LayersModel;

    constructor({ weights, size, height, width, channels }: UnetProps) {
        this.weights = weights;
        this.size = size;
        this.height = height;
        this.width = width;
        this.inChannels = channels;
    }

    async build() {
        // debug all stats
        console.log('%cBuilding UNet...', 'color: yellow');
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
        const weightsTransposed = weights.transpose([2, 3, 1, 0]);

        // Set the weights and biases.
        layer.setWeights([weightsTransposed, biases]);

        return layer;
    }

    // todo: determine if this can be merged with convLayer
    private setWeightsAndBiases(name: string, layer: tf.layers.Layer) {
        // get the weights and biases from the weights map
        console.log('Weights:', this.weights)
        const weights = this.weights.get(`${name}.weight`);
        const biases = this.weights.get(`${name}.bias`);

        if (!weights || !biases) throw new Error(`UNet: Missing weights or biases for layer: ${name}`);

        // set the weights and biases
        layer.setWeights([weights, biases]);
    }

    async execute() {
        // TODO: if we ever want to denoise just normal or albedo this will have to change as it requires color
        // make sure we have inputs if not throw an error
        if (!this.color) throw new Error('UNet: Missing input image');

        // Step 2: Concatenate images along the channel dimension
        // if the concatenated image is not the same as the color image dispose of it
        if (this.albedo || this.normal && this.concatenatedImage !== this.color) this.concatenatedImage.dispose();
        if (this.albedo) {
            if (this.normal) this.concatenatedImage = tf.concat([this.color, this.albedo, this.normal], -1);
            else this.concatenatedImage = tf.concat([this.color, this.albedo], -1);
        } else if (this.normal) {
            this.concatenatedImage = tf.concat([this.color, this.normal], -1);
        } else this.concatenatedImage = this.color;
        // Step 3: Reshape for batch size
        // destroy the old input tensor
        if (this.inputTensor) this.inputTensor.dispose();
        this.inputTensor = this.concatenatedImage.expandDims(0); // Now shape is [1, height, width, 9]

        // Step 4: Feed to the network
        const output = this.model.predict(this.directInputTensor || this.inputTensor);

        // Ensure output is a single Tensor before calling Tensor-specific methods
        if (Array.isArray(output)) {
            throw new Error('Expected model to return a single tensor, but it returned an array.');
        }

        // Now that we've ensured output is not an array, we can safely call squeeze()
        let outputImage = output.squeeze() as tf.Tensor3D;
        outputImage = tf.clipByValue(outputImage, 0, 1);
        return outputImage;
    }
    setImage(name: 'color' | 'albedo' | 'normal', tensor: tf.Tensor3D) {
        // only allow the known names
        if (!['color', 'albedo', 'normal'].includes(name))
            throw new Error(`UNet: Unknown input name: ${name}`);

        //destroy the old inputs
        // biome-ignore lint/style/noNonNullAssertion: <explanation>
        if (this[name]) this[name]!.dispose();

        this[name] = tensor;
    }

    //destroy the model
    destroy() {
        if (this.model) this.model.dispose();
        //dispose of the inputs
        if (this.color) this.color.dispose();
        if (this.albedo) this.albedo.dispose();
        if (this.normal) this.normal.dispose();

        // extra tensors
        if (this.concatenatedImage) this.concatenatedImage.dispose();
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