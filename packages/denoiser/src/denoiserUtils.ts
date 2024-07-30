// denoiserUtils.ts
import * as tf from '@tensorflow/tfjs';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import type { Denoiser } from './denoiser';
import type { DenoiserProps, InputOptions } from 'types/types';
import { concatenateAlpha3D, formatBytes, splitRGBA3D } from './utils';

export async function setupBackend(denoiser: Denoiser, prefered = 'webgl', canvasOrDevice?: HTMLCanvasElement | GPUDevice) {
    // do the easy part first
    if (!canvasOrDevice) {
        if (denoiser.debugging) console.log('Denoiser: No canvas provided, using default');
        await tf.setBackend(prefered);
        const backendName = tf.getBackend();
        denoiser.backend = tf.engine().findBackend(backendName)
        // if webgl set the gl context
        //@ts-ignore
        denoiser.gl = backendName === 'webgl' ? denoiser.backend.gpgpu.gl : undefined;
        if (denoiser.debugging) console.log(`Denoiser: Backend set to ${prefered}`);
        await tf.ready();
        denoiser.backendReady = true;
        return denoiser.backend;
    }

    // We dont want to run mess with weird context for WASM or CPU backends
    if (prefered !== 'webgl' && prefered !== 'webgpu')
        throw new Error('Only webgl and webgpu are supported with custom contexts');

    let kernels: tf.KernelConfig[];
    //* Setup a webGPU backend with a custom device
    if (prefered === 'webgpu') {
        denoiser.device = canvasOrDevice as GPUDevice;
        kernels = tf.getKernelsForBackend('webgpu');
        // biome-ignore lint/complexity/noForEach: <explanation>
        kernels.forEach(kernelConfig => {
            const newKernelConfig = { ...kernelConfig, backendName: 'denoiser-webgpu' };
            tf.registerKernel(newKernelConfig);
        });
        tf.registerBackend('denoiser-webgpu', async () => {
            return new WebGPUBackend(denoiser.device!);
        });
        await tf.setBackend('denoiser-webgpu');
        console.log('%c Denoiser: Backend set to custom WebGPU', 'background: teal; color: white;');
        denoiser.usingCustomBackend = true;
        denoiser.backendReady = true;

        return denoiser.backend;
    }

    //* Setup a webGL backend with a custom context
    // to get here the canvas has to be passed and the rendering context is actually already on the canvas
    denoiser.gl = (canvasOrDevice as HTMLCanvasElement).getContext('webgl2')!;

    // if multiple denoisers exist they can share contexts
    // register the backend if it doesn't exist
    if (tf.findBackend('denoiser-webgl') === null) {
        // we have to make sure all the kernels are registered
        kernels = tf.getKernelsForBackend('webgl');
        for (const kernelConfig of kernels) {
            const newKernelConfig = { ...kernelConfig, backendName: 'denoiser-webgl' };
            tf.registerKernel(newKernelConfig);
        }
        const customBackend = new tf.MathBackendWebGL(canvasOrDevice as HTMLCanvasElement);
        tf.registerBackend('denoiser-webgl', () => customBackend);
    }
    await tf.setBackend('denoiser-webgl');
    await tf.ready();

    // see what happens if we force textures to be f16
    tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
    //tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);
    // attempting speed increases
    tf.env().set('WEBGL_EXP_CONV', true);
    tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true);
    console.log('%c Denoiser: Backend set to custom webgl', 'background: orange; color: white;');
    denoiser.usingCustomBackend = true;
    denoiser.backendReady = true;
    denoiser.webglStateManager?.saveState();

    return denoiser.backend;
}

// Resolves the tensor map label and size based on the denoiser props
export function determineTensorMap(props: DenoiserProps) {
    // example rt_ldr_small | rt_ldr_calb_cnrm (with clean aux)
    let tensorMapLabel = props.filterType;
    tensorMapLabel += props.hdr ? '_hdr' : '_ldr';
    // if you use cleanAux you MUST provide both albedo and normal
    if (props.useAlbedo && props.useNormal && props.cleanAux) tensorMapLabel += '_calb_cnrm';
    else {
        tensorMapLabel += props.useAlbedo ? '_alb' : '';
        tensorMapLabel += props.useNormal ? '_nrm' : '';
    }

    //* quality. 
    let size: 'small' | 'large' | 'default' = 'default';
    // small and large only exist in specific cases
    const hasSmall = ['rt_hdr', 'rt_ldr', 'rt_hdr_alb', 'rt_ldr_alb', 'rt_hdr_alb_nrm', 'rt_ldr_alb_nrm', 'rt_hdr_calb_cnrm', 'rt_ldr_calb_cnrm'];
    const hasLarge = ['rt_alb', 'rt_nrm', 'rt_hdr_calb_cnrm', 'rt_ldr_calb_cnrm'];

    if (props.quality === 'fast' && hasSmall.includes(tensorMapLabel)) size = 'small';
    else if (props.quality === 'high' && hasLarge.includes(tensorMapLabel)) size = 'large';

    if (size !== 'default') tensorMapLabel += `_${size}`;

    return { tensorMapLabel, size };

}

//adjust the size based on the input tensor size
export function adjustSize(denoiser: Denoiser, options: InputOptions) {
    const { height, width } = options;
    if (!height || !width) return;
    if (height !== denoiser.height || width !== denoiser.width) {
        console.log('Values before adjustment:', denoiser.width, denoiser.height, width, height);
        if (height) denoiser.height = height;
        if (width) denoiser.width = width;
        if (denoiser.debugging) console.log('Denoiser: Adjusted size to', denoiser.width, denoiser.height);
    }
}

// warmstart the model if using webGL
export function warmstart(model: tf.LayersModel, height: number, width: number, channels: number) {
    const backendName = tf.getBackend();
    // Warmstart for webGL only
    if (backendName !== 'webgl' && backendName !== 'denoiser-webgl') return;
    console.log('Warming up the model for WebGL');
    tf.env().set('ENGINE_COMPILE_ONLY', true);
    (model.predict(tf.randomNormal([1, height, width, channels])) as tf.Tensor4D).dataSync();
    (tf.backend() as tf.MathBackendWebGL).checkCompileCompletion();
    (tf.backend() as tf.MathBackendWebGL).getUniformLocations();
    tf.env().set('ENGINE_COMPILE_ONLY', false);
}

// prepare and process input tensors
export async function handleInputTensors(denoiser: Denoiser, name: 'color' | 'albedo' | 'normal', inputTensor: tf.Tensor3D, options: InputOptions) {
    if (name === 'color') denoiser.props.useColor = true;
    else if (name === 'albedo') denoiser.props.useAlbedo = true;
    else if (name === 'normal') denoiser.props.useNormal = true;
    //options
    const { flipY, channels } = options;
    const baseTensor = tf.tidy(() => {
        let toReturn = inputTensor;
        if (flipY) toReturn = tf.reverse(toReturn, [0]);
        // if we are a normal we need to normalize the tensor into the range [-1, 1]
        if (name === 'normal') toReturn = toReturn.sub(0.5).mul(2);
        // destroy the original input tensor
        if (toReturn !== inputTensor) inputTensor.dispose();
        // leaveing seperate incase we want to add handing
        return toReturn;
    });
    // if the channels is 4 we need to strip the alpha channel
    if (!channels) {
        // split the alpha channel from the rgb data (NOTE: destroys the baseTensor)
        const { rgb, alpha } = await splitRGBA3D(baseTensor, true);
        denoiser.setInputTensor(name, rgb);
        // we only care about the alpha of the color input
        if (name === 'color') denoiser.inputAlpha = alpha;
        else alpha.dispose();
    } else denoiser.setInputTensor(name, baseTensor);
}

export async function handleModelInput(denoiser: Denoiser): Promise<tf.Tensor4D> {
    // destroy the old model output to save some memory
    if (denoiser.oldOutputTensor) denoiser.oldOutputTensor.dispose();
    // take the set input tensors and concatenate them. ready things for model input
    // if we have direct input bypass everything
    if (denoiser.directInputTensor) return denoiser.directInputTensor.expandDims(0) as tf.Tensor4D;
    denoiser.startTimer('modelInput');
    // TODO: if we ever want to denoise just normal or albedo this will have to change as it requires color
    const color = denoiser.inputTensors.get('color');
    const albedo = denoiser.inputTensors.get('albedo');
    const normal = denoiser.inputTensors.get('normal');

    //if (!color && !albedo && !normal) throw new Error('Denoiser must have an input set before execution.');
    if (!color) throw new Error('Denoiser must have an input set before execution.');

    // if we have extra images we need to use tiling, also if the height/width is not a divisible by 16
    if ((!denoiser._tilingUserBlocked && (albedo || normal)) || (denoiser.height % 16 !== 0 || denoiser.width % 16 !== 0)) denoiser.useTiling = true;

    // if we have both aux images and not blocked by the user, set cleanAux to true
    if (albedo && normal && !denoiser.props.dirtyAux) denoiser.props.cleanAux = true;

    // concat the images
    const toReturn = tf.tidy(() => {
        let concatenatedImage: tf.Tensor3D;
        if (albedo) {
            if (normal) concatenatedImage = tf.concat([color, albedo, normal], -1);
            else concatenatedImage = tf.concat([color, albedo], -1);
        } else concatenatedImage = color;

        // Reshape for batch size
        return concatenatedImage.expandDims(0) as tf.Tensor4D; // Now shape is [1, height, width, 9]
    });
    // dispose of the input tensors
    color.dispose();
    if (albedo) albedo.dispose();
    if (normal) normal.dispose();

    denoiser.stopTimer('modelInput');
    return toReturn;
}
// Take the model output and ready it to be returned
export async function handleModelOutput(denoiser: Denoiser, result: tf.Tensor4D): Promise<tf.Tensor3D> {
    denoiser.startTimer('modelOutput');
    const outputImage = tf.tidy(() => {
        let output = result.squeeze() as tf.Tensor3D;
        // With float32 we had strange 1.0000001 values this limits to expected outputs
        if (!denoiser.hdr) output = tf.clipByValue(output, 0, 1);
        // flip the image vertically
        if (denoiser.flipOutputY) output = tf.reverse(output, [0]);
        // if (denoiser.debugging) console.log('Output Image shape:', output.shape, 'dtype:', output.dtype);
        if (denoiser.inputAlpha) output = concatenateAlpha3D(output, denoiser.inputAlpha, true);
        //dispose if input
        result.dispose();
        return output;
    });
    // if there is an old output tensor dispose of it
    if (denoiser.oldOutputTensor) denoiser.oldOutputTensor.dispose();
    denoiser.oldOutputTensor = outputImage;
    denoiser.stopTimer('modelOutput');
    return outputImage;
}

export async function handleCallback(denoiser: Denoiser, outputTensor: tf.Tensor3D, returnType: string, callback?: Function | undefined) {
    let toReturn: tf.Tensor3D | ImageData | WebGLTexture | GPUBuffer;
    let data: tf.GPUData;
    //const startTime = performance.now();
    switch (returnType) {
        case 'webgl':
        case 'webgpu':
            // WebGL and webGPU both require the alpha channel
            if (!denoiser.inputAlpha) data = concatenateAlpha3D(outputTensor).dataToGPU({ customTexShape: [denoiser.height, denoiser.width] });
            else data = outputTensor.dataToGPU({ customTexShape: [denoiser.height, denoiser.width] });

            toReturn = returnType === 'webgl' ? data.texture! : data.buffer!;
            if (!toReturn) throw new Error('Denoiser: Could not convert to GPU Accessible Tensor');

            // Dispose the tensor reference
            data.tensorRef.dispose();
            break;

        case 'tensor':
            toReturn = outputTensor;
            break;

        case 'float32': {
            const float32Data = outputTensor.dataSync() as Float32Array;
            toReturn = float32Data;
            break;
        }
        default: {
            const pixelData = await tf.browser.toPixels(outputTensor);
            toReturn = new ImageData(pixelData, denoiser.props.width, denoiser.props.height);
        }
    }
    if (callback) callback(toReturn!);
    return toReturn!;
}


export async function profileModel(model: tf.LayersModel, inputTensor: tf.Tensor4D) {
    // Warm-up run
    const warmupResult = await model.predict(inputTensor);
    tf.dispose(warmupResult);

    // Actual profiling
    const profileInfo = await tf.profile(() => {
        return model.predict(inputTensor);
    });

    console.log('Profile information:', profileInfo);

    // Log specific metrics
    // console.log('Total time (ms):', profileInfo.totalTimeMs);
    //console.log('Kernel execution time (ms):', profileInfo.kernelMs);

    // Log individual kernel executions
    profileInfo.kernels.forEach((kernel, index) => {
        console.log(`Kernel ${index}:`, kernel.name, kernel.kernelTimeMs, 'ms', 'Bytes Added:', formatBytes(kernel.bytesAdded));
    });

    // Memory usage
    const memoryInfo = tf.memory();
    console.log('Memory usage:', memoryInfo);

    // Don't forget to dispose of tensors
    tf.dispose(inputTensor);
}