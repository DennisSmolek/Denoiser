

// Super basic webGPU setup half pulled from various ai and my dumb ideas
export class WebGPURenderer {
    device?: GPUDevice;
    canvas: HTMLCanvasElement;
    isReady = false;
    readyListeners: Set<(renderer: WebGPURenderer) => void> = new Set();
    constructor(canvas: HTMLCanvasElement) {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported on this browser.");
        }
        this.canvas = canvas;
        this.init();

    }
    async init() {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("No appropriate GPUAdapter found.");
        }
        this.device = await adapter.requestDevice();
        this.isReady = true;
        this.readyListeners.forEach((callback) => callback(this));
    }

    renderBuffer(buffer: GPUBuffer) {
        renderToCanvas(this.device!, this.canvas, buffer);
    }
    renderTestImage(imageSource = "./noisey.jpg") {
        renderToCanvas(this.device!, this.canvas, null, imageSource);
    }

    getImageBuffer(imageSource: string) {
        return processToBuffer(this.device!, null, imageSource);
    }

    onReady(callback: (renderer: WebGPURenderer) => void) {
        if (this.isReady) {
            callback(this);
        } else {
            this.readyListeners.add(callback);
        }
        // return a cleanup function
        return () => this.readyListeners.delete(callback);
    }
}

// Shader code
const shaderCode = `
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texCoord: vec2f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0)
    );

    var texCoord = array<vec2f, 6>(
        vec2f(0.0, 1.0),
        vec2f(1.0, 1.0),
        vec2f(0.0, 0.0),
        vec2f(0.0, 0.0),
        vec2f(1.0, 1.0),
        vec2f(1.0, 0.0)
    );

    var output: VertexOutput;
    output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
    output.texCoord = texCoord[vertexIndex];
    return output;
}

@group(0) @binding(0) var<storage, read> inputBuffer: array<f32>;

@fragment
fn fragmentMain(@location(0) texCoord: vec2f) -> @location(0) vec4f {
    let x = u32(texCoord.x * 1280.0);
    let y = u32(texCoord.y * 720.0);
    let index = (y * 1280u + x) * 4u;
    
    let r = inputBuffer[index];
    let g = inputBuffer[index + 1u];
    let b = inputBuffer[index + 2u];
    let a = inputBuffer[index + 3u];
    
    return vec4f(r, g, b, a);
}
`;

async function initWebGPU(device: GPUDevice, outputCanvas: HTMLCanvasElement) {
    const context = outputCanvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    if (context) context.configure({
        device: device,
        format: canvasFormat,
    });

    const shaderModule = device.createShaderModule({
        code: shaderCode
    });

    const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{
                format: canvasFormat
            }]
        },
        primitive: {
            topology: 'triangle-list',
        }
    });

    return { device, context, pipeline };
}

async function createGPUBufferFromImage(device: GPUDevice, imageSrc: string) {
    // Load the image
    const img = await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = imageSrc;
        img.crossOrigin = "anonymous";  // Enable loading from other domains
    });

    // Create a canvas to draw the image
    const offscreenCanvas = new OffscreenCanvas(1280, 720);
    const ctx = offscreenCanvas.getContext('2d');

    // Draw the image, scaling it to fit 1280x720
    ctx.drawImage(img, 0, 0, 1280, 720);

    // Get the image data
    const imageData = ctx.getImageData(0, 0, 1280, 720);

    // Create a Float32Array to hold the normalized pixel data
    const floatArray = new Float32Array(1280 * 720 * 4);

    // Normalize the pixel data from 0-255 to 0-1
    for (let i = 0; i < imageData.data.length; i++) {
        floatArray[i] = imageData.data[i] / 255;
    }

    // Create and populate the buffer
    const buffer = device.createBuffer({
        size: floatArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });

    new Float32Array(buffer.getMappedRange()).set(floatArray);
    buffer.unmap();

    return buffer;
}

// Updated main function to include image loading option
export async function renderToCanvas(device: GPUDevice, canvas: HTMLCanvasElement, inputBuffer?: GPUBuffer, imageSource = null) {
    const { context, pipeline } = await initWebGPU(device, canvas);

    let gpuBuffer: GPUBuffer;
    if (imageSource) {
        gpuBuffer = await createGPUBufferFromImage(device, imageSource);
    } else if (inputBuffer) {
        gpuBuffer = inputBuffer;
    } else throw new Error("Either inputBuffer or imageSource must be provided");

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: gpuBuffer }
        }],
    });

    function render() {
        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6);  // 6 vertices for 2 triangles
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
    }

    render();
}
async function processToBuffer(device: GPUDevice, inputBuffer: GPUBuffer | null, imageSource: string | null): Promise<GPUBuffer> {
    // Initialize WebGPU (we don't actually need a canvas for this operation)
    const { pipeline } = await initWebGPU(device, document.createElement('canvas'));

    // Determine the input: either use the provided buffer or create one from the image
    let sourceBuffer: GPUBuffer;
    if (imageSource) {
        sourceBuffer = await createGPUBufferFromImage(device, imageSource);
    } else if (inputBuffer) {
        sourceBuffer = inputBuffer;
    } else {
        throw new Error("Either inputBuffer or imageSource must be provided");
    }

    // Create an output buffer
    const outputBuffer = device.createBuffer({
        size: 1280 * 720 * 4 * 4, // 1280x720 pixels, 4 channels (RGBA), 4 bytes per float
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'processBuffer'
    });

    // Create a compute pipeline for processing
    const computeShaderModule = device.createShaderModule({
        code: `
            @group(0) @binding(0) var<storage, read> inputBuffer: array<f32>;
            @group(0) @binding(1) var<storage, read_write> outputBuffer: array<f32>;

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                let y = global_id.y;
                if (x >= 1280u || y >= 720u) {
                    return;
                }
                let index = (y * 1280u + x) * 4u;
                outputBuffer[index] = inputBuffer[index];
                outputBuffer[index + 1u] = inputBuffer[index + 1u];
                outputBuffer[index + 2u] = inputBuffer[index + 2u];
                outputBuffer[index + 3u] = inputBuffer[index + 3u];
            }
        `
    });

    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: computeShaderModule,
            entryPoint: 'main',
        },
    });

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: sourceBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
        ],
    });

    // Create a command encoder and pass
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(1280 / 16), Math.ceil(720 / 16));
    computePass.end();

    // Submit the command buffer
    device.queue.submit([commandEncoder.finish()]);

    // Return the output buffer
    return outputBuffer;
}