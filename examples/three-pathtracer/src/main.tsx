import "./index.css";
import { Renderer } from "./threePathtracerDemo";

//* WebGL ===========================================
// get the canvas for output
const outputCanvas = document.getElementById("output") as HTMLCanvasElement;
// debugging canvas, you wont need this but it helps if you break stuff
export const rawOutputCanvas = document.getElementById("rawOutput");
// create the renderer
const renderer = new Renderer(outputCanvas!);
renderer.onReady(() => {
	//denoiser = new Denoiser("webgl", outputCanvas);
	//denoiser.onBackendReady(() => setupDenoising());
});

// Update the stats on the page
renderer.onStats((stats: { samples: number; isDenoised: boolean }) => {
	// samples
	document.getElementById("samples")!.textContent = Math.floor(
		stats!.samples,
	).toString();

	// denoised
	const denoisedLabel = document.getElementById("denoised")!;
	if (stats.isDenoised) {
		denoisedLabel.style.display = "block";
	} else {
		denoisedLabel.style.display = "none";
	}
});

//* Utilities for the Demo ===========================================

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
	renderer.denoiser.quality = quality;
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
