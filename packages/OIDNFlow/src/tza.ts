// Converted from the C++ version in OIDN

import * as tf from '@tensorflow/tfjs';


// Assuming TensorMap is a Map where keys are string and values are tf.Tensor
export type TensorMap = Map<string, tf.Tensor>;


export async function loadTestTZA(filename = 'rt_ldr_alb_nrm_small.tza'): Promise<TensorMap> {
    console.log('file we are loading:', filename);
    const tza = await loadDefaultTZAFile(filename);
    //debugTZAFile(tza);
    return parseTZA(tza);
}

// Load a TZA file
export async function loadTZAFile(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok)
        throw new Error(`Failed to load TZA file from ${url}`);

    console.log('File loaded:', url);
    const buffer = await response.arrayBuffer();
    return buffer;
}

// load a TZA file from the default tzas in the library
export async function loadDefaultTZAFile(fileName: string): Promise<ArrayBuffer> {
    const url = getTZAFilePath(fileName);
    console.log('Loading default file:', url)
    return loadTZAFile(url);
}
function getTZAFilePath(fileName: string): string {
    if (!fileName) throw new Error('No filename provided in path getting');
    // Assuming the library's package name is 'my-library' and files are in 'tza' folder
    const relativePath = `./tzas/${fileName}`;
    console.log('relative path:', relativePath);
    return new URL(relativePath, import.meta.url).href;
}

export function debugTZAFile(buffer: ArrayBuffer): void {
    const dataView = new DataView(buffer);
    const magic = dataView.getUint16(0, true); // Assuming little-endian
    const majorVersion = dataView.getUint8(2);
    const minorVersion = dataView.getUint8(3);
    // Use getBigUint64 for the table offset
    const tableOffsetBigInt = dataView.getBigUint64(4, true); // Assuming little-endian
    // Convert BigInt to number for the byteOffset parameter in DataView
    const tableOffset = Number(tableOffsetBigInt);
    if (tableOffset >= buffer.byteLength) {
        console.error("Table offset is out of bounds.");
        return;
    }
    const tableSizeDataView = new DataView(buffer, tableOffset);
    const tableSize = tableSizeDataView.getUint32(0, true); // Assuming little-endian

    console.log(`Magic Value: ${magic.toString(16)}`);
    console.log(`Version: ${majorVersion}.${minorVersion}`);
    console.log(`Table Offset: ${tableOffset}`);
    console.log(`Table Size: ${tableSize}`);
}

export function parseTZA(buffer: ArrayBuffer): TensorMap {
    const tensorMap = new Map<string, tf.Tensor>();

    const view = new DataView(buffer);
    // log out the data view stats
    console.log('byteLength:', view.byteLength);
    console.log('byteOffset:', view.byteOffset);
    console.log('buffer:', view.buffer);

    let offset = 0;

    // Parse the magic value
    const magic = view.getUint16(offset, true);
    offset += 2;
    if (magic !== 0x41D7) {
        throw new Error('Invalid or corrupted weights blob');
    }
    // Parse the version
    const majorVersion = view.getUint8(offset++);
    //const minorVersion = view.getUint8(offset++);
    // increment offset by 1 while ignoring the minor version
    offset++;
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
        // const layout = new TextDecoder().decode(buffer.slice(offset, offset + ndims));
        offset += ndims;

        // Parse the data type of the tensor
        const dataType = new TextDecoder().decode(buffer.slice(offset, offset + 1));
        offset += 1;

        // Parse the offset to the tensor data
        const tensorOffset = Number(view.getBigUint64(offset, true));
        offset += 8;

        // Depending on the data type, create the tensor

        //console.log('dataType', dataType)
        //console.log('Creating tensor with dataType:', dataType);
        //console.log('Expected shape:', shape);
        const numElements = shape.reduce((a, b) => a * b);
        //console.log('Expected number of elements:', numElements);

        let slicedBuffer: Float32Array | Uint16Array;
        if (dataType === 'f') {
            slicedBuffer = new Float32Array(buffer.slice(tensorOffset, tensorOffset + numElements * 4));
        } else if (dataType === 'h') {
            // Assuming conversion to Float32 is needed
            const uint16Array = new Uint16Array(buffer.slice(tensorOffset, tensorOffset + numElements * 2));
            // Convert Uint16Array to Float32Array if necessary
            slicedBuffer = new Float32Array(uint16Array.length);
            for (let i = 0; i < uint16Array.length; i++) {
                slicedBuffer[i] = uint16Array[i]; // Simple conversion, adjust as needed
            }
        } else {
            throw new Error('Invalid tensor data type');
        }

        //console.log('Sliced buffer length:', slicedBuffer.length);
        if (slicedBuffer.length !== numElements) {
            console.error('Mismatch between expected number of elements and sliced buffer length');
            // Adjust calculations or handle error
        }

        const tensor = tf.tensor(slicedBuffer, shape);

        // Add the tensor to the map
        tensorMap.set(name, tensor);
    }

    return tensorMap;
}