![title-card-resized](https://github.com/DennisSmolek/Denoiser/assets/1397052/ffe87fd5-00e6-464e-b8a2-ba80402b9d2f)

## AI Denoising that runs in the browser.

#### Based on the Open Image Denoiser and powered by Tensorflow.js



Because of Tensorflow’s broad support, Denoiser works in almost any javascript environment but runs fastest where it can run WebGL, and most advanced on WebGPU.


### Basic Javascript Example
```ts
import { Denoiser } from "denoiser";
//* Elements
const noisey = document.getElementById("noisey-img");
const outputCanvas = document.getElementById("output-canvas");
const button = document.getElementById("execute-button");

const denoiser = new Denoiser();
// set the canvas for quick denoising output
denoiser.setCanvas(outputCanvas);

// function to denoise the image when clicked
async function doDenoise() {
	await denoiser.execute(noisey);
}

//* add a click litener to the button
button.addEventListener("click", doDenoise);
```
[See This Example Live](example.com/link)

## Overview

[Open Image Denoiser (OIDN)](https://github.com/RenderKit/oidn) is a Native Neural Network Denoiser designed specifically for use with pathtracing to get the highest quality results from the lowest number of samples. 

Denoiser uses the same model structure (UNet) and OIDN’s pre-trained weights, with tensorflow.js to offer high quality denoising in the browser.

We take a variety of different inputs (Images, ImgData, Canvas Data, WebGLTextures, GPUBuffers, ArrayBuffers, etc) and convert them into tensorflow tensors.

We then pre-process and bundle the inputs to ready them for execution.

Then we build a custom model at runtime based on the inputs and configuration, run the model, and ready the results.

If the size of the images/data is too large we automatically tile/batch the input to reduce GPU memory load.

With the results from the model, Denoiser can immediately render them to a canvas, return a new image, output a texture, a WebGPU Buffer, etc. based on your choices.


Using Tensorflow, we keep as much of these operations as possible on the GPU making things incredibly fast.

Denoiser has a ton of options and configurations for different needs and platforms, I’ll do my best to document them.

### Install:
```sh
yarn add denoiser
```
#### Including the weights

I tried many things to do this automatically, but my options were to make the library size huge with the weights bundled, or have the user load the weights their own.

I have bundled the weights in the `node_module` and you can make a [script to copy them]() in your build process.

Or you can just copy the weights you will use [from here]() and put them into a folder named `tzas`;

By default, `Denoiser` will look for these in the root path, so `https://yourapp.com/tzas`

If you are using Vite, this just means create a `tzas` folder in your `public` folder and put whatever weights you might use. 
*(all of the examples use this method)*

#### If your path isn't on the root or you want to load weights from a URL you can do that too:
```ts
denoiser.weightPath = 'path/from/root/tzas';
//or override completely with a URL
denoiser.weightUrl = 'https://whereyourweightsare.com/tzas';
```
### Getting Set Up
There are a [ton of options](#denoiser) and inputs that control how things flow through the denoiser. You probably need none of it.

1. [Install](#install) and put weights in the root folder.
2. Import and create the denoiser:
```ts
import { Denoiser } from 'denoiser';
const denoiser = new Denoiser();
```
**DONE**

Seriously, most other things are taken care of automatically.
The only optional adjustment you have to set is the quality.

Default `quality` is `'fast'` which will work for 90% of what you do. You can also set `'balanced'` which ranges from 10-200% slower, with not that much difference in quality.

As of 7/24 we don't support `'high'` as OIDN lies about this anyway. Even if you set it, unless all the other props are set it actually just runs `'balanced'`. I'll start supporting it [When I add other UNets](#Unets)

The other possible option you could use is `hdr`. This changes how the model loads and is mostly used in textures/buffers if you know what you are doing. If you're using hdr data, be sure to set this. *(note: if you are using images as input, you absolutely don't need hdr )*

#### Standard Options example
```ts
denoiser.quality = 'balanced';
denoiser.hdr = true;
// these can be set too to help with WebGL/WebGPU modes
denoiser.height = 720;
denoiser.width = 1280;
```
Checkout setting up with [WebGL](#webgl) and [WebGPU](#webgpu) for more advanced users.

Full list of [Denoiser Props](#denoiser)

***

### Getting Data In
There are two ways of getting data into the denoiser. Setting a input **BEFORE** execution with a explicit set function,
 or setting the input **DURING** execution using `inputMode` to indicate how you want the inputs handled.

#### execute(*color?, albedo?, normal?*)
`Denoiser.execute()` is the most common input and output method combining the explicit input/output handlers into one action.

Using `inputMode` and `outputMode` you can inform the denoiser what kind of data you will send and expect when you run `execute()`

*By default, `inputMode` and `outputMode` are set to `imgData`*;

```ts
const noisey = document.getElementById("noisey-img");
const albedo = document.getElementById("albedo-img");

denoiser.outputMode = 'webgpu';
const outputBuffer = await denoiser.execute(color, albedo);
// send the buffer to the renderer directly
renderer.renderBuffer(outputBuffer);
```
#### inputMode/outputMode
```ts
inputMode: 'imgData' | 'webgl' | 'webgpu' | 'tensor';
outputMode: 'imgData' | 'webgl' | 'webgpu' | 'tensor' | 'float32';
```
---

#### ImageData *(default)*

The default way to get data in is using `ImgData` from a canvas or even just passing a HTML `Image` object directly.

```ts
const noisey = document.getElementById("noisey-img");

// using the execute function
const output = await denoiser.execute(noisey);

// using setImage
denoiser.setImage('color', noisey);
const output = denoiser.execute();

```

#### setImage(name, data)
name: `'color' | 'albedo' | 'normal'`

data: `PixelData|ImageData|HTMLImageElement|HTMLCanvasElement| HTMLVideoElement|ImageBitmap`

`color` and `albedo` inputs are regular (non-normalized) RGB(a) inputs. (sRGB Colorspace)

`normal` is linear and as a saved image assumed to be in 0-255. Note: When normalized (OIDN expects) so we transform these to [-1, 1]

*NOTE: Tensorflow automatically strips all alpha channels from image data. When we return it all alpha will be set as 100%. If you need alpha output consider a different method of input*

---
#### WebGL Texture
##### *Make sure to read: [Running in a shared WebGL Context]()*
*(Note: WebGL is being weird at the moment)*
#### SetInputTexture(`name, texture, height, width, channels?`)
If the denoiser is running in the same context you can pass a `WebGLTexture` directly as input without needing to sync the CPU/GPU.

Textures are expected to already be normalize [0, 1] for `color`, `albedo` and [-1, 1] for `normal`

Be sure to set the height and width of the texture, we can't determine this from the data alone. You can set these before directly on the denoiser.

We assume the data will have an alpha channel, and will actually parse this data back onto the texture when returning it.(it will NOT be denoised) If your texture doesn't have alpha set the channels to 3.

```ts
// ** somewhere inside a loop **
const texture = frameBuffer.texture;
denoiser.setInputTexture('color', texture, renderer.height, renderer.width);

// If you want to use texture with execute
denoiser.inputMode = 'webgl' // only has to be set anytime before and only once.
const { colorTexture, albedoTexture, normalTexture} = renderer.getMRTOutput();
const outputTexture = await denoiser.execute(colorTexture, albedoTexture, normalTexture);
// render the results
renderer.renderTexture(outputTexture);
```
---
#### WebGPU Buffer
##### *Make sure to read: [Running WebGPU](#webgpu)*
WebGPU is the easiest advanced setup to get data in/out of the Neural Network without causing a CPU/GPU Sync.

Buffers are expected to already be normalize [0, 1] for `color`, `albedo` and [-1, 1] for `normal`

Be sure to set the height and width as we can't determine that from data. You can set these before directly on the denoiser.

> [!WARNING]
>Also a note about `usage:` I got errors until I made sure the `GPUBufferUsage.COPY_DST` was set. I'm not sure what Tensorflow is doing with the buffer that it needs to write to it but it will throw errors without this set.



```ts
// setup code
const denoiser = new Denoiser('webgpu', renderer.device);
denoiser.height = 720;
denoiser.width = 1280;

/* Somewhere deep in your render code */
// Create an output buffer
const outputBuffer = device.createBuffer({
    size: 1280 * 720 * 4 * 4, // 1280x720 pixels, 4 channels (RGBA), 4 bytes per float
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'processBuffer'
});
/* Do Fancy WebGPU Stuff */
// all done on the GPU, send it to denoise. (note, no need to set height/width if pre-set)
denoiser.setInputBuffer('color', outputBuffer);

//* Different ex With execute -----------------------------

denoiser.inputMode = 'webgpu'; // only needs to be set once
const inBuffer = renderer.getLatestBuffer();
denoiser.execute(inBuffer);
```


---
#### Data Buffer
##### setData(`name, Float32Array | Uint8Array, height, width, channels?`)
You probably don't actually need this.
A case where you would is if using HDR loaded on the CPU like with threejs's [RGBELoader](https://github.com/mrdoob/three.js/blob/master/examples/jsm/loaders/RGBELoader.js) and then get the data out as a `Float32Array` or if using some sort of camera input. 
```ts
    const rawData = system.getRawOutput();
    denoiser.setData('color', rawData, system.height, system.width)
    setData(name: 'color' | 'albedo' | 'normal', data: Float32Array | Uint8Array, height: number, width: number, channels = 4) {

```
---

### Getting Data Out
Just like with input there are two main methods for getting data out of the denoiser. You can use the output of the `execute()` function with `outputMode` set to whatever you like, or you can create an `executionListener` that will fire on every execution and output whatever way you set, regardless of `outputMode`

There is also the super easy `setCanvas()` which is fine for many cases.

#### setCanvas(`HTMLCanvasElement`)
This dumps the outputTensor directly to a canvas on every execution. This is faster than pulling the data and drawing it yourself, as Tensorflow draws the canvas directly with the tensor data.

I use this often for debugging as it's guaranteed to draw exactly what the `outputTensor` holds. Very useful for testing renderers. 

It also runs regardless of `outputMode` 
```ts
const noisey = document.getElementById("noisey-img");
const outputCanvas = document.getElementById("output-canvas");

const denoiser = new Denoiser();
// set the canvas for quick denoising output
denoiser.setCanvas(outputCanvas);

// we dont care about the output data, just be sure noisey is loaded. might wrap this in a onLoaded to be 100%
denoiser.execute(noisey);

//To deactivate it, you have to set:
denoiser.outputToCanvas = false;
```
---
#### execute()
Execute is the primary way to get data in/out of the denoiser, but not the only way.
If you are only calling the denoiser once and you are handling input/output in the same place execute is the way to go.

It's particualarly useful for things like timers or setting/restoring state after the denoiser runs that need to bracket the denoise step.

```ts
//* Basic Example ===
//load
const noisy = renderer.getCanvasInfo();
//denoise
const imgData = await denoiser.execute(noisy);
//draw
renderer.drawToCanvas(imgData);

//* Advanced Bracketing Example ===
denoiser.outputMode = 'webgl';
denoiser.height = 720;
denoiser.width = 1280;
const startTime = performance.now();
// load
const colorTexture = renderer.getColorTexture();
// advanced thing, don't worry about it
densoiser.restoreWebGLState();
// denoise
const outputTexture = await denoiser.execute(colorTexture);
denoiser.saveWebGLState();
//draw
renderer.drawTexture(outputTexture);
statsOutput('renderDenoised', startTime, performance.now());
```
---

#### onExecute(`callback(output), outputMode`)
Attaching a listener to the denoiser lets you decouple input, execution, and output in a much greater/more flexible way.

You can add as many listeners as you want that each have their own `outputMode`s meaning you could send one to a compute shader and another to a renderer so you can compare results for example.

Adding a listener returns a return function which will remove the listener.

NOTE: Using listeners disables the output when you call `execute()` to avoid costly outputHandling

#### Decoupled Handling Example:
```ts
const device = renderer.getDevice();
const denoiser = new Denoiser('webgpu', device);
// attach execution listener now
denoiser.onExecute((outputBuffer) => {
    //draw
    renderer.renderBuffer(outputBuffer);
}, 'webgpu');


async function whenSomethingHappens(url: string) {
    const buffer = await renderer.makeBuffer(url);
    // set the input on the denoiser
    denoiser.setInputBuffer('color', buffer, renderer.height, renderer.width);
}

function doDenoise() {
    denoiser.execute();
}

loadButton.addEventHandler('pointerDown', (() => whenSomethingHappens(input.value));
denoiseButton.addEventHandler('pointerDown', doDenoise);
```

---

### Other UNets
Why did I change the name and this isn't just an OIDN for the web?

Now that I have a UNet operating and am much more comfortable with tensorflow there are [other more advanced/modern UNets](https://github.com/cszn/SCUNet) that I think could be used in conjunction with the OIDN UNet.

OIDN is specifically designed for pathtracers and 3D renderers with MRT outputs of albedo and normal data. The noise is almost always uniformly black which means noise due to sensor data, compression, or other sources is often replicated. 

There are also many methods to potentially speed up the denoising/handling that would be very different that the core OIDN Unets. Therefore, I didn't feel right calling it OIDNFlow, OIWDN, etc as it would be fundementally different.

Once I have other UNet's/Models in the works I will add the "Large" UNet that is required for "high" quality.




Problem is, as of right now it has almost no mobile support and limited desktop support.

Also, *(although untested)* it was reported that for tensorflow, webGL is actually still slightly faster to execute.


SUper advanced future stuff:


(I haven't exposed the option to override with a single `tensorMap`)