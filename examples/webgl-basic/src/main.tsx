import { Denoiser } from "denoiser";
import "./index.css";
import { WebGLRenderer } from "./webGLDemo";

let denoiser: Denoiser;

//* WebGL ===========================================
// get the canvas for output
const outputCanvas = document.getElementById("output") as HTMLCanvasElement;
// debugging canvas, you wont need this but it helps if you break stuff
export const rawOutputCanvas = document.getElementById("rawOutput");
// create the renderer
const renderer = new WebGLRenderer(outputCanvas!);
renderer.onReady(() => {
	denoiser = new Denoiser("webgl", outputCanvas);
	denoiser.onBackendReady(() => setupDenoising());
});

//* Denoising ===========================================
// Get the three inputs
export const noisey = document.getElementById("noisey") as HTMLImageElement;
const albedo = document.getElementById("albedo") as HTMLImageElement;
const normal = document.getElementById("normal") as HTMLImageElement;

// because we have to wait a little longer for the webGPU backend lets make a handy function
function setupDenoising() {
	//denoiser.debugging = true; // uncomment if you want detailed logs
	// if something isn't looking right, this will dump the outputBuffer to the canvas
	//denoiser.setCanvas(rawOutputCanvas);
	//denoiser.usePassThrough = true; // bypass denoising model

	// use tiling with webGPU and larger images. (May become standard)
	denoiser.useTiling = true;
	// activate the button
	button.disabled = false;

	// add an execution listener
	denoiser.onExecute((outputTexture) => {
		renderer.drawTextureToCanvas(outputTexture);
		// this tells the system we want a webGL texture
	}, "webgl");
}

// function to denoise the image when clicked
async function doDenoise() {
	const startTime = performance.now();

	// render a image to texture
	const noiseyTexture = await renderer.loadTextureFromURL("./noisey.jpg");
	await denoiser.setInputTexture("color", noiseyTexture, {
		height: 720,
		width: 1280,
	});
	// pass null as we set the buffer manually
	denoiser.execute(null, albedo, normal);

	// different levels of execution
	//await denoiser.execute(noisey, albedo, normal);
	//await denoiser.execute(noisey, albedo);
	//await denoiser.execute(noisey);
	//denoiser.execute();

	updateTimeDisplay(startTime);
}

//* add a click litener to the button
const button = document.getElementById("execute-button") as HTMLButtonElement;
button.addEventListener("click", doDenoise);

//* WebGL Logging Utils ===========================================
function logWebGLState(gl) {
	console.log("Current Program:", gl.getParameter(gl.CURRENT_PROGRAM));
	console.log("Active Texture:", gl.getParameter(gl.ACTIVE_TEXTURE));
	console.log("Viewport:", gl.getParameter(gl.VIEWPORT));
	console.log("Scissor Test:", gl.getParameter(gl.SCISSOR_TEST));
	console.log("Scissor Box:", gl.getParameter(gl.SCISSOR_BOX));
	console.log("Blend Enabled:", gl.getParameter(gl.BLEND));
	console.log("Depth Test Enabled:", gl.getParameter(gl.DEPTH_TEST));
	console.log("Blend src RGB:", gl.getParameter(gl.BLEND_SRC_RGB));
	console.log("Blend dst RGB:", gl.getParameter(gl.BLEND_DST_RGB));
	console.log("Blend equation RGB:", gl.getParameter(gl.BLEND_EQUATION_RGB));
}

export function logProgram(gl) {
	console.log("Current Shader Program:", gl.getParameter(gl.CURRENT_PROGRAM));
}
function logFrameBuffers(gl) {
	console.log("Texture Binding:", gl.getParameter(gl.TEXTURE_BINDING_2D));
	console.log("Framebuffer Binding:", gl.getParameter(gl.FRAMEBUFFER_BINDING));
}
function logError(gl) {
	console.log("WebGL Error:", gl.getError());
}

export function logViewport(gl) {
	console.log("Viewport:", gl.getParameter(gl.VIEWPORT));
	console.log("Scissor Test Enabled:", gl.getParameter(gl.SCISSOR_TEST));
	console.log("Scissor Box:", gl.getParameter(gl.SCISSOR_BOX));
}

function logUnpack(gl) {
	console.log("UNPACK_FLIP_Y_WEBGL:", gl.getParameter(gl.UNPACK_FLIP_Y_WEBGL));
	console.log(
		"UNPACK_PREMULTIPLY_ALPHA_WEBGL:",
		gl.getParameter(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL),
	);
	console.log(
		"UNPACK_COLORSPACE_CONVERSION_WEBGL:",
		gl.getParameter(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL),
	);
}

export function logEverything(gl) {
	logWebGLState(gl);
	logFrameBuffers(gl);
	logUnpack(gl);
	logError(gl);
}

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

function webGLTextureToFloat32Array(
	gl: WebGLRenderingContext,
	texture: WebGLTexture,
	width: number,
	height: number,
): Float32Array {
	// Create a framebuffer and attach the texture
	const framebuffer = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D,
		texture,
		0,
	);

	// Check if the framebuffer is complete
	if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
		throw new Error("Unable to read texture data");
	}

	// Read the pixel data
	const pixels = new Float32Array(width * height * 4);
	gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, pixels);

	// Clean up
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	gl.deleteFramebuffer(framebuffer);

	return pixels;
}

export function compareWebGLTextures(
	gl: WebGLRenderingContext,
	texture1: WebGLTexture,
	texture2: WebGLTexture,
	width: number,
	height: number,
	epsilon: number = 1e-6,
): {
	areEqual: boolean;
	differences: { index: number; value1: number; value2: number }[];
} {
	// Convert textures to Float32Arrays
	const data1 = webGLTextureToFloat32Array(gl, texture1, width, height);
	const data2 = webGLTextureToFloat32Array(gl, texture2, width, height);

	// Compare the arrays
	const differences: { index: number; value1: number; value2: number }[] = [];
	let areEqual = true;

	for (let i = 0; i < data1.length; i++) {
		if (Math.abs(data1[i] - data2[i]) > epsilon) {
			areEqual = false;
			differences.push({ index: i, value1: data1[i], value2: data2[i] });
		}
	}

	return { areEqual, differences };
}
