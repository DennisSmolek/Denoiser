import { Denoiser } from "denoiser";
import "./index.css";
import { loadImageAndRender } from "./webGLDemo";

//* WebGL ===========================================
// get the canvas for output
const outputCanvas = document.getElementById("output") as HTMLCanvasElement;
const WEBGL_ATTRIBUTES = {
	alpha: true,
	antialias: false,
	premultipliedAlpha: false,
	preserveDrawingBuffer: false,
	depth: false,
	stencil: false,
	failIfMajorPerformanceCaveat: true,
	powerPreference: "low-power",
};
const context = outputCanvas.getContext(
	"webgl2",
	WEBGL_ATTRIBUTES,
) as WebGL2RenderingContext;
console.log("first draw");
if (context) loadImageAndRender("./noisey.jpg", context);

//* Denoising ===========================================

setTimeout(() => {
	const denoiser = new Denoiser("webgl", outputCanvas);
	console.log("second draw");
	if (context) loadImageAndRender("./noisey.jpg", context);
	denoiser.debugging = true; // uncomment if you want detailed logs
	setTimeout(() => {
		console.log("third draw");
		if (context) loadImageAndRender("./normal.jpg", context);
	}, 5000);
}, 5000);
// Get the three inputs
const noisey = document.getElementById("noisey") as HTMLImageElement;
const albedo = document.getElementById("albedo") as HTMLImageElement;
const normal = document.getElementById("normal") as HTMLImageElement;

// function to denoise the image when clicked
async function doDenoise() {
	const startTime = performance.now();
	await denoiser.execute(noisey, albedo, normal);
	updateTimeDisplay(startTime);
}

//* add a click litener to the button
const button = document.getElementById("execute-button") as HTMLButtonElement;
button.addEventListener("click", doDenoise);

//* Utilities for the Demo ------------------------------

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
