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



### Install:
```sh
yarn add denoiser
```



building and outputting the tzas
```ts
// Example webpack.config.js snippet
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
    // Other webpack config
    plugins: [
        new CopyPlugin({
            patterns: [
                { from: 'node_modules/denoiser/tzas', to: 'tzas' },
            ],
        }),
    ],
};

```
(why not OIDN Denoiser?)