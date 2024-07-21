import Dropzone from "dropzone";

import { Denoiser } from "denoiser";
import "./index.css";
import "dropzone/dist/dropzone.css";

//* Denoising ===========================================

const denoiser = new Denoiser("webgl");
denoiser.debugging = true; // uncomment if you want detailed logs
denoiser.useTiling = true; // uncomment if you want to use tiling
// get the canvas for output
const outputCanvas = document.getElementById("output") as HTMLCanvasElement;
denoiser.setCanvas(outputCanvas);
// set the image to denoise
//denoiser.setImage("color", noisey);

// function to denoise the image when clicked
async function doDenoise() {
	const startTime = performance.now();
	await denoiser.execute();
	updateTimeDisplay(startTime);
}

//* add a click litener to the button
const button = document.getElementById("execute-button") as HTMLButtonElement;
button.addEventListener("click", doDenoise);

//* Dropzone ===========================================
const dropzone = new Dropzone("#dropzone", {
	autoProcessQueue: false, // Prevent automatic upload
	url: "/file/post", // Dummy URL, required but won't be used
});

async function handleFile(file: File) {
	// if not an image we can draw to a canvas reject it
	if (!file.type.startsWith("image")) {
		console.error("File is not an image");
		return;
	}
	// clear the original canvas
	const ctx = outputCanvas.getContext("2d");
	if (ctx) ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

	// draw the preview and get the new data
	const newImgData = await drawPreview(file);
	denoiser.setImage("color", newImgData);

	// enable the denoiser button
	button.disabled = false;
}

dropzone.on("addedfile", (file) => {
	console.log("File added", file);
	handleFile(file);
});

//* Canvas Drawing ===========================================
const previewCanvas = document.getElementById("preview") as HTMLCanvasElement;

async function drawPreview(file: File) {
	return new Promise<HTMLImageElement>((resolve, reject) => {
		const ctx = previewCanvas.getContext("2d");
		if (!ctx) return reject();

		const img = new Image();
		img.onload = () => {
			previewCanvas.width = img.width;
			previewCanvas.height = img.height;
			ctx.drawImage(img, 0, 0);
			resolve(img);
		};
		img.onerror = reject;
		img.src = URL.createObjectURL(file);
	});
}

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
