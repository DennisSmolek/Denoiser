// Converted from the C++ version in OIDN

import * as tf from '@tensorflow/tfjs';

// Assuming TensorMap is a Map where keys are string and values are tf.Tensor
type TensorMap = Map<string, tf.Tensor>;

function parseTZA(buffer: ArrayBuffer): TensorMap {
    const tensorMap = new Map<string, tf.Tensor>();

    let view = new DataView(buffer);
    let offset = 0;

    // Parse the magic value
    const magic = view.getUint16(offset, true);
    offset += 2;
    if (magic !== 0x41D7) {
        throw new Error('Invalid or corrupted weights blob');
    }

    // Parse the version
    const majorVersion = view.getUint8(offset++);
    const minorVersion = view.getUint8(offset++);
    if (majorVersion !== 2) {
        throw new Error('Unsupported weights blob version');
    }

    // Parse the table offset and jump to the table
    const tableOffset = view.getBigUint64(offset, true);
    offset = Number(tableOffset);

    // Parse the number of tensors
    const numTensors = view.getUint32(offset, true);
    offset += 4;

    for (let i = 0; i < numTensors; ++i) {
        // Parse the name
        const nameLen = view.getUint16(offset, true);
        offset += 2;
        const name = new TextDecoder().decode(buffer.slice(offset, offset + nameLen));
        offset += nameLen;

        // Parse the number of dimensions
        const ndims = view.getUint8(offset++);

        // Parse the shape of the tensor
        const shape: number[] = [];
        for (let j = 0; j < ndims; ++j) {
            shape.push(view.getUint32(offset, true));
            offset += 4;
        }

        // Parse the layout of the tensor (not directly used in TensorFlow.js)
        const layout = new TextDecoder().decode(buffer.slice(offset, offset + ndims));
        offset += ndims;

        // Parse the data type of the tensor
        const dataType = new TextDecoder().decode(buffer.slice(offset, offset + 1));
        offset += 1;

        // Parse the offset to the tensor data
        const tensorOffset = Number(view.getBigUint64(offset, true));
        offset += 8;

        // Depending on the data type, create the tensor
        let tensor: tf.Tensor;
        switch (dataType) {
            case 'f':
                tensor = tf.tensor(new Float32Array(buffer.slice(tensorOffset, tensorOffset + shape.reduce((a, b) => a * b) * 4)), shape);
                break;
            case 'h':
                // TensorFlow.js does not directly support Float16, so we might need to convert it to Float32
                console.warn('Float16 is not directly supported, converting to Float32');
                tensor = tf.tensor(new Uint16Array(buffer.slice(tensorOffset, tensorOffset + shape.reduce((a, b) => a * b) * 2)), shape).cast('float32');
                break;
            default:
                throw new Error('Invalid tensor data type');
        }

        // Add the tensor to the map
        tensorMap.set(name, tensor);
    }

    return tensorMap;
}