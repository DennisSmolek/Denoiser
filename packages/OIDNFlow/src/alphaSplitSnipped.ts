import * as tf from '@tensorflow/tfjs';

// Function to split RGBA tensor into RGB and A tensors for Tensor3D
function splitRGBA3D(inputTensor: tf.Tensor3D, skipDispose = false): { rgb: tf.Tensor3D, alpha: tf.Tensor3D } {
    // Assuming inputTensor shape is [height, width, 4] where the last dimension is RGBA
    const rgb = tf.slice(inputTensor, [0, 0, 0], [-1, -1, 3]); // Take the first 3 channels (RGB)
    const alpha = tf.slice(inputTensor, [0, 0, 3], [-1, -1, 1]); // Take the 4th channel (Alpha)
    if (!skipDispose) inputTensor.dispose(); // Dispose the input tensor if not skipped
    return { rgb, alpha };
}

// Function to concatenate the alpha channel back to the RGB tensor for Tensor3D
function concatenateAlpha3D(rgbTensor: tf.Tensor3D, alphaTensor: tf.Tensor3D): tf.Tensor3D {
    return tf.concat([rgbTensor, alphaTensor], -1); // Concatenate along the channel dimension
}

// Example usage for Tensor3D
const height = 256; // Example height
const width = 256; // Example width
const rgbaData = new Float32Array(height * width * 4); // Example RGBA data
const inputTensor3D = tf.tensor3d(rgbaData, [height, width, 4]); // Create a 3D tensor

const { rgb, alpha } = splitRGBA3D(inputTensor3D); // Split the tensor into RGB and Alpha

// After processing the RGB tensor with your model, concatenate the alpha channel back
// const outputRGB = model.predict(rgb); // Example model prediction on RGB tensor
// const outputRGBA = concatenateAlpha3D(outputRGB, alpha); // Add the alpha channel back