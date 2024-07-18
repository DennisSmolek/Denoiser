import * as tf from '@tensorflow/tfjs';

// Function to split RGBA tensor into RGB and A tensors for Tensor3D
export async function splitRGBA3D(inputTensor: tf.Tensor3D, disposeInputs = true): { rgb: tf.Tensor3D, alpha: tf.Tensor3D } {
    // Assuming inputTensor shape is [height, width, 4] where the last dimension is RGBA
    const rgb = tf.slice(inputTensor, [0, 0, 0], [-1, -1, 3]); // Take the first 3 channels (RGB)
    const alpha = tf.slice(inputTensor, [0, 0, 3], [-1, -1, 1]); // Take the 4th channel (Alpha)
    if (disposeInputs) inputTensor.dispose(); // Dispose the input tensor if not skipped
    return { rgb, alpha };
}

// Function to concatenate the alpha channel back to the RGB tensor for Tensor3D
export function concatenateAlpha3D(rgbTensor: tf.Tensor3D, alphaTensor?: tf.Tensor3D, disposeInputs = false): tf.Tensor3D {
    //if there in no alpha tensor, create one with all 1s
    if (!alphaTensor) {
        alphaTensor = tf.tidy(() => {
            return tf.ones(rgbTensor.shape.slice(0, 2).concat([1]) as [number, number, number], 'float32');
        });
    }


    const concatenatedTensor = tf.concat([rgbTensor, alphaTensor], -1); // Concatenate along the channel dimension
    if (disposeInputs) {
        rgbTensor.dispose();
        alphaTensor.dispose();
    }
    return concatenatedTensor;
}

//* Utils ----------------------------------------------

// take a css scaled image and use a canvas to get the actual size
export function getCorrectImageData(img: HTMLImageElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get canvas context');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

// check if the height/width dont math
export function hasSizeMissmatch(img: HTMLImageElement) {
    if (!img.naturalHeight || !img.naturalWidth) return true;
    return img.height !== img.naturalHeight || img.width !== img.naturalWidth;
}

// convert a tensor with linear color encoding to sRGB
export function tensorLinearToSRGB(tensor: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
        const gamma = tf.scalar(2.2);
        const sRGB = tensor.pow(gamma);
        return sRGB;
    }) as tf.Tensor3D;
}

// output how many tensors are in memory
export function memoryStats(preString = '') {
    console.log(`${preString} Tensors in memory:`, tf.memory().numTensors);
}