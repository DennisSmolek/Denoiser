import * as tf from '@tensorflow/tfjs';

// Function to split RGBA tensor into RGB and A tensors for Tensor3D
export function splitRGBA3D(inputTensor: tf.Tensor3D, disposeInputs = true): { rgb: tf.Tensor3D, alpha: tf.Tensor3D } {
    // Assuming inputTensor shape is [height, width, 4] where the last dimension is RGBA
    const rgb = tf.slice(inputTensor, [0, 0, 0], [-1, -1, 3]); // Take the first 3 channels (RGB)
    const alpha = tf.slice(inputTensor, [0, 0, 3], [-1, -1, 1]); // Take the 4th channel (Alpha)
    if (disposeInputs) inputTensor.dispose(); // Dispose the input tensor if not skipped
    return { rgb, alpha };
}

// Function to concatenate the alpha channel back to the RGB tensor for Tensor3D
export function concatenateAlpha3D(rgbTensor: tf.Tensor3D, alphaTensor: tf.Tensor3D, disposeInputs = false): tf.Tensor3D {
    const concatenatedTensor = tf.concat([rgbTensor, alphaTensor], -1); // Concatenate along the channel dimension
    if (disposeInputs) {
        rgbTensor.dispose();
        alphaTensor.dispose();
    }
    return concatenatedTensor;
}