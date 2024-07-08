import { Denoiser } from "denoiser";
import "./index.css";

const denoiser = new Denoiser();
denoiser.debugging = true;

// Helper to know when the denoiser is ready
denoiser.onBackendReady(() => {
	console.log("Denoiser created!", denoiser);
});

// Get the image for input
const noisey = document.getElementById("noisey-img") as HTMLImageElement;
// get the canvas for output
const outputCanvas = document.getElementById("output") as HTMLCanvasElement;

// set the canvas for quick denoising output
denoiser.setCanvas(outputCanvas);
// set the image to denoise
//denoiser.setImage("color", noisey);

// function to denoise the image
async function doDenoise() {
	console.dir("noisey", noisey);
	const startTime = performance.now();
	await denoiser.execute(noisey);
	const endTime = performance.now();
	console.log("Denoising took", endTime - startTime, "ms");
}

// add a click litener to the button
const button = document.getElementById("execute-button") as HTMLButtonElement;
button.addEventListener("click", doDenoise);

let device: GPUDevice;

async function webgpu() {
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) return false;
	device = await adapter.requestDevice();
	console.log("Device Setup", device);
}
webgpu();
