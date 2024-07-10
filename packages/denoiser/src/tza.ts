import * as tf from '@tensorflow/tfjs';

// Assuming TensorMap is a Map where keys are string and values are tf.Tensor
export type TensorMap = Map<string, tf.Tensor>;

// Original Load a TZA file
/*
export async function loadTZAFile(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load TZA file from ${url}`);
    return await response.arrayBuffer();
}
    */
export async function loadTZAFile(url: string): Promise<ArrayBuffer> {
    // Use the HEAD method to get headers without downloading the file
    const headResponse = await fetch(url, { method: 'HEAD' });
    if (!headResponse.ok) throw new Error(`Failed to fetch TZA file headers from ${url}`);

    // Attempt to extract the filename from the Content-Disposition header
    const contentDisposition = headResponse.headers.get('Content-Disposition');
    let filename: string | null | undefined = null;
    if (contentDisposition) {
        const matches = contentDisposition.match(/filename="?(.+?)"?$/);
        if (matches?.[1]) {
            filename = matches[1];
        }
    }

    // If filename is not found in Content-Disposition, parse it from the URL
    if (!filename) {
        const urlObj = new URL(url);
        filename = urlObj.pathname.split('/').pop();
    }

    // Check if the filename ends with '.tza'
    if (!filename || !filename.endsWith('.tza')) {
        throw new Error(`The file at ${url} does not appear to be a TZA file based on its extension.`);
    }

    // If the file extension is correct, proceed to download the file
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load TZA file from ${url}`);
    return await response.arrayBuffer();
}

// load a TZA file from the default tzas in the library
export async function loadDefaultTZAFile(fileName: string, subDirectory?: string): Promise<ArrayBuffer> {
    //@ts-ignore
    //const viteDevMode = import.meta.env.DEV;
    // removing this hack for now
    const viteDevMode = false;

    const url = viteDevMode ? getdevTZAFilePath(fileName) : getTZAFilePath(fileName, subDirectory);
    return loadTZAFile(url);
}

// this only works in dev environments

function getdevTZAFilePath(fileName: string): string {
    if (!fileName) throw new Error('No filename provided in path getting');
    // Assuming the library's package name is 'my-library' and files are in 'tza' folder
    const relativePath = `./tzas/${fileName}`;
    console.log('relative path:', relativePath);
    return new URL(relativePath, import.meta.url).href;
}


// take an optional file path from the root or get the file from the root of the site

function getTZAFilePath(fileName: string, subDirectory?: string): string {
    if (!fileName) throw new Error('No filename provided in path getting');
    const relativePath = `/${subDirectory ? `${subDirectory}/` : 'tzas/'}${fileName}`;
    // console.log('root path:', relativePath);
    return new URL(relativePath, import.meta.url).href;
}


export function parseTZA(buffer: ArrayBuffer): TensorMap {
    const tensorMap = new Map<string, tf.Tensor>();
    const view = new DataView(buffer);
    // log out the data view stats
    //console.log('byteLength:', view.byteLength);
    // console.log('byteOffset:', view.byteOffset);
    // console.log('buffer:', view.buffer);

    let offset = 0;

    // Parse the magic value
    const magic = view.getUint16(offset, true);
    offset += 2;
    if (magic !== 0x41D7) throw new Error('Invalid or corrupted weights blob');

    // Parse the version
    const majorVersion = view.getUint8(offset++);
    //const minorVersion = view.getUint8(offset++);
    // increment offset by 1 while ignoring the minor version
    offset++;
    if (majorVersion !== 2) throw new Error('Unsupported weights blob version');

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
        //console.log(`Weight shape for ${name}:`, shape)

        // Parse the layout of the tensor (not directly used in TensorFlow.js)
        // const layout = new TextDecoder().decode(buffer.slice(offset, offset + ndims));
        offset += ndims;

        // Parse the data type of the tensor
        const dataType = new TextDecoder().decode(buffer.slice(offset, offset + 1));
        offset += 1;

        // Parse the offset to the tensor data
        const tensorOffset = Number(view.getBigUint64(offset, true));
        offset += 8;

        const numElements = shape.reduce((a, b) => a * b);

        let slicedBuffer: Float32Array | Uint16Array;
        if (dataType === 'f') {
            slicedBuffer = new Float32Array(buffer.slice(tensorOffset, tensorOffset + numElements * 4));
        } else if (dataType === 'h') {
            // Assuming conversion to Float32 is needed
            slicedBuffer = float16ToFloat32(buffer.slice(tensorOffset, tensorOffset + numElements * 2));
        } else throw new Error('Invalid tensor data type');

        //console.log('Sliced buffer length:', slicedBuffer.length);
        if (slicedBuffer.length !== numElements)
            throw new Error('Mismatch between expected number of elements and sliced buffer length');


        const tensor = tf.tidy(() => {
            let out = tf.tensor(slicedBuffer, shape);
            // transpose the weight from OIHW to HWIO
            if (shape.length === 4) out = out.transpose([2, 3, 1, 0]);
            return out;
        });
        // Add the tensor to the map
        tensorMap.set(name, tensor);
    }

    return tensorMap;
}

//* Debug Utils ----------------------------
export async function loadTestTZA(filename = 'rt_ldr_alb_nrm_small.tza'): Promise<TensorMap> {
    const tza = await loadDefaultTZAFile(filename);
    return parseTZA(tza);
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

//* Utils ----------------------------

function float16ToFloat32(buffer: ArrayBuffer): Float32Array {
    const length = buffer.byteLength / 2; // Each Float16 takes 2 bytes
    const input = new Uint16Array(buffer);
    const output = new Float32Array(length);

    for (let i = 0; i < length; i++) {
        const value = input[i];
        const sign = (value & 0x8000) >> 15;
        const exponent = (value & 0x7C00) >> 10;
        const fraction = value & 0x03FF;

        if (exponent === 0) {
            if (fraction === 0) {
                output[i] = sign === 0 ? 0.0 : -0.0;
            } else {
                // Subnormal numbers
                const normalizedFraction = fraction / 0x400;
                const m = Math.pow(2, 1 - 15); // 2^(1-15) for normalization
                output[i] = (sign === 0 ? 1 : -1) * normalizedFraction * m;
            }
        } else if (exponent === 0x1F) {
            // Infinity and NaN
            if (fraction === 0) {
                output[i] = sign === 0 ? Infinity : -Infinity;
            } else {
                output[i] = NaN;
            }
        } else {
            // Normalized numbers
            const normalizedExponent = exponent - 15 + 127; // Adjust exponent from Float16 to Float32
            const normalizedFraction = fraction / 0x400;
            output[i] = (sign === 0 ? 1 : -1) * (1 + normalizedFraction) * Math.pow(2, normalizedExponent - 127);
        }
    }

    return output;
}