import { Denoiser } from "denoiser";
import "./index.css";

//* Denoising ===========================================

const denoiser = new Denoiser();
//denoiser.usePassThrough = true;
denoiser.debugging = true;
// denoiser.debugging = true; // uncomment if you want detailed logs

// Helper to know when the denoiser is ready, not needed
denoiser.onBackendReady(() => {
	console.log("Denoiser created!", denoiser);
});

//* Elements
// Get the image for input
const noisey = document.getElementById("noisey-img") as HTMLImageElement;
// get the canvas for output
const outputCanvas = document.getElementById("output") as HTMLCanvasElement;
//---

// set the canvas for quick denoising output
denoiser.setCanvas(outputCanvas);
// set the image to denoise
//denoiser.setImage("color", noisey);

// function to denoise the image when clicked
async function doDenoise() {
	const startTime = performance.now();
	await denoiser.execute(noisey);
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
