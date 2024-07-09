import { Denoiser } from "denoiser";
import "./index.css";
import { WebGPURenderer } from "./webGPUDemo";

let denoiser: Denoiser;

//* WebGPU ===========================================
// get the canvas for output
const outputCanvas = document.getElementById("output") as HTMLCanvasElement;
// const rawOutputCanvas = document.getElementById("rawOutput") as HTMLCanvasElement;
// create the renderer
const renderer = new WebGPURenderer(outputCanvas);
renderer.onReady(() => {
	console.log("Renderer is ready");
	//renderer.renderTestImage();
	denoiser = new Denoiser("webgpu", renderer.device);
	denoiser.onBackendReady(() => setupDenoising());
	//denoiser.setCanvas(rawOutputCanvas);
});

//* Denoising ===========================================
// Get the three inputs
const noisey = document.getElementById("noisey") as HTMLImageElement;
const albedo = document.getElementById("albedo") as HTMLImageElement;
const normal = document.getElementById("normal") as HTMLImageElement;

// because we have to wait a little longer for the webGPU backend lets make a handy function
function setupDenoising() {
	denoiser.debugging = true; // uncomment if you want detailed logs
	// use tiling with webGPU and larger images. (May become standard)
	denoiser.useTiling = true;
	// activate the button
	button.disabled = false;

	// add an execution listener
	denoiser.onExecute((outputBuffer) => {
		console.log("Output buffer", outputBuffer);
		renderer.renderBuffer(outputBuffer);

		// this tells the system we want a webGPU buffer
	}, "webgpu");
}

// function to denoise the image when clicked
async function doDenoise() {
	const startTime = performance.now();

	// different levels of execution
	await denoiser.execute(noisey, albedo, normal);
	//await denoiser.execute(noisey, albedo);
	//await denoiser.execute(noisey);

	updateTimeDisplay(startTime);
}

//* add a click litener to the button
const button = document.getElementById("execute-button") as HTMLButtonElement;
button.addEventListener("click", doDenoise);

//* Utilities for the Demo ===========================================

//* visibility toggles
// get the button to toggle denose visibility
const denoisedButton = document.getElementById(
	"denoised-button",
) as HTMLButtonElement;
const originalButton = document.getElementById(
	"original-button",
) as HTMLButtonElement;
// get the sources holder
const sources = document.getElementById("sources") as HTMLDivElement;

// attach the click listeners
denoisedButton.addEventListener("click", () => setVisible());
originalButton.addEventListener("click", () => setVisible("original"));

// function to toggle visibility of source images
function setVisible(className?: string) {
	if (className) sources.className = className;
	else sources.className = "";
}

//* Show the time it took to denoise
function updateTimeDisplay(startTime: number) {
	// get the time span
	const timeOutput = document.getElementById("time-output") as HTMLSpanElement;
	const stats = document.getElementById("stats") as HTMLDivElement;
	const endTime = performance.now();
	const duration = Math.round((endTime - startTime) * 100) / 100;
	timeOutput.innerText = `${duration}ms`;
	// activate the denoised button
	denoisedButton.disabled = false;
	// show the stats
	stats.style.display = "block";
}

//* toggle the quality
// get the toggle input
const qualityToggle = document.getElementById(
	"quality-toggle",
) as HTMLInputElement;
qualityToggle.addEventListener("change", toggleQuality);

// get the parent so we can add a class to it
const qualityHolder = document.getElementById("quality") as HTMLDivElement;
function toggleQuality() {
	const quality = qualityToggle.checked ? "balanced" : "fast";
	denoiser.quality = quality;
	console.log("Quality set to", quality);
	qualityHolder.className = quality;
}
