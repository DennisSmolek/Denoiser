import { useRef } from "react";
import { useThree } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import { useControls, button } from "leva";

import * as THREE from "three";
import "./App.css";
// for base testing import oidn
import { Denoiser } from "denoiser";
import { useFrame } from "@react-three/fiber";
import { OrbitControls, useFBO } from "@react-three/drei";

function App() {
	const hasMounted = useRef(false);
	const planeRef = useRef<THREE.Mesh>(null);
	const planeUint8Ref = useRef<THREE.Mesh>(null);
	const denoiser = useRef<Denoiser | null>(null);
	const fullTexture = useRef<THREE.Texture | null>(null);
	//const fancyTextureSet = useRef(false);
	const imgElement = useRef<HTMLImageElement>();
	const denoiserStart = useRef(performance.now());
	const tfActive = useRef(false);

	const { gl } = useThree();

	const renderTarget = useFBO(512, 512, { type: THREE.UnsignedByteType });

	// load the sample textures
	const [voltronTexture, , voltron2] = useTexture([
		"voltron.png",
		"pirates.png",
		"voltron2.png",
	]);

	function initializeDenoiserPrivateContext() {
		// create a canvas element
		const canvas = document.createElement("canvas");
		canvas.width = 512;
		canvas.height = 512;
		// get the webGL context
		const gl = getWebGLRenderingContext(canvas);
		console.log("context for private canvas:", gl);
		return initializeDenoiser(true, canvas);
	}

	function initializeDenoiser(shared = false, canvas = gl.domElement) {
		// stop the denoiser from double mounting
		if (hasMounted.current) return;
		hasMounted.current = true;

		//temp to work out three issues
		tfActive.current = true;
		gl.resetState();
		console.log("Three blocked");
		console.log("Shared value:", shared);
		// create the denoiser
		const dn = shared ? new Denoiser("webgl", canvas) : new Denoiser("webgl");
		//const dn = new Denoiser("webgl");
		denoiser.current = dn;
		denoiser.current.debugging = true;

		// create the float32 texture
		if (!fullTexture.current) {
			// create a new texture with 32 bit float data
			fullTexture.current = new THREE.DataTexture(
				new Float32Array(512 * 512 * 4),
				512,
				512,
				THREE.RGBAFormat,
				THREE.FloatType,
			);
			fullTexture.current.flipY = true;
			fullTexture.current.colorSpace = THREE.NoColorSpace;
			fullTexture.current.repeat.set(1, 1);
			fullTexture.current.wrapS = THREE.ClampToEdgeWrapping;
			fullTexture.current.wrapT = THREE.ClampToEdgeWrapping;
			// set the texture to the plane
			if (planeRef.current) {
				//@ts-ignore
				planeRef.current.material.map = fullTexture.current;
				console.log("Inital Float32 Texture set on the plane");
			}
		}

		//* ImageData testing
		dn.onExecute((data) => {
			console.log(
				"%cImageData Denoising complete",
				"background-color: #00FFC5; color: black;",
			);
			console.log(
				"Time to denoise (ImgData):",
				performance.now() - denoiserStart.current,
			);
			console.log("Image data Buffer from the denoiser", data.data);
			// update the texture data
			gl.resetState();
			tfActive.current = false;
			console.log("Three restored in ImgData");
			const u8Texture = (
				planeUint8Ref.current!.material as THREE.MeshBasicMaterial
			).map;
			if (!u8Texture) throw new Error("No texture");
			u8Texture.image = data;
			u8Texture.colorSpace = THREE.NoColorSpace;
			u8Texture.needsUpdate = true;
			console.log("updated ImgData Texture", u8Texture);
			// because three is acting weird
			setTimeout(() => {
				// set the image plane to the voltron texture
				const newMat = new THREE.MeshBasicMaterial({ map: voltron2 });

				planeUint8Ref.current!.material = newMat;

				console.log(
					"Plane set back to voltron texture",
					(planeUint8Ref.current!.material as THREE.MeshBasicMaterial).map,
				);

				setTimeout(() => {
					console.log("trying to set it to the texture again");

					// set the texture to the plane
					(planeUint8Ref.current!.material as THREE.MeshBasicMaterial)
						.map!.image = data;
					(planeUint8Ref.current!.material as THREE.MeshBasicMaterial)
						.map!.needsUpdate = true;
					console.log(
						"Plane set to the denoised texture",
						(planeUint8Ref.current!.material as THREE.MeshBasicMaterial).map,
					);
				}, 5000);
			}, 10000);
		}, "imgData");

		//* Float32 testing
		/*
		dn.onExecute((data) => {
			console.log(
				`%cFloat32 Denoising complete, time: ${performance.now() - denoiserStart.current}`,
				"background-color: #0075A2; color: white;",
			);
			console.log("Float32 Buffer from the denoiser", data);
			// update the texture data
			updateTextureData({ data, height: 512, width: 512 });
			//console.log("updated Float32 Texture", fullTexture.current);
		}, "float32");
		*/

		dn.onBackendReady(() => {
			tfActive.current = false;
			console.log("Three restored");
		});
	}
	/*
	const updateTextureData = (newTexture: {
		data: Float32Array;
		width: number;
		height: number;
	}) => {
		if (!fullTexture.current) return;
		//quick test to see if it's double encoding
		fullTexture.current.image.data = newTexture.data;

		//fullTexture.current.image.data = applySRGBEncoding(newTexture.data);
		fullTexture.current.needsUpdate = true;
		(planeRef.current!.material as THREE.MeshBasicMaterial).needsUpdate = true;
		console.log(
			"Time to update texture:",
			performance.now() - denoiserStart.current,
		);
		//console.log("updated Float32 Texture", fullTexture.current);
	};
*/
	const snapshotTestImage = () => {
		if (!denoiser.current) return;

		// block three from rendering
		tfActive.current = true;
		console.log("Three blocked");

		// set a timeout to restore three
		setTimeout(() => {
			tfActive.current = false;
			console.log("Three restored");
		}, 5000);

		const imageElement = new Image();
		imageElement.src = "pirates.png";

		const handleLoad = () => {
			imgElement.current = imageElement;
			setTimeout(() => {
				const dn = denoiser.current;
				if (!dn) return;
				const { height, width } = imageElement;
				dn.height = height;
				dn.width = width;
				//console.log(`Image loaded with height: ${height} and width: ${width}`);
				// pass the image to the denoiser
				dn.setImage("color", imageElement);
				dn.execute();
			}, 1);
		};

		// Add event listener
		imageElement.addEventListener("load", handleLoad);
	};

	const snapShotRenderTarget = () => {
		denoiserStart.current = performance.now();
		let pixelBuffer:
			| Uint8Array
			| Uint8ClampedArray
			| Uint16Array
			| Float32Array = readRenderTargetData(renderTarget);

		if (pixelBuffer.constructor.name === "Uint8Array") {
			// convert to Uint8ClampedArray
			pixelBuffer = new Uint8ClampedArray(pixelBuffer.buffer);
			// create a new image data object
			const imageData = new ImageData(
				pixelBuffer,
				renderTarget.width,
				renderTarget.height,
			);
			// set the denoiser image data
			denoiser.current!.setImage("color", imageData, true);
		}
		console.log(
			"Time to read render target:",
			performance.now() - denoiserStart.current,
		);
		// read pixels from the render target to the buffer
		//console.log("RenderTarget texture:", renderTarget.texture);
		//console.log("Render target coverted to buffer:", pixelBuffer);

		// execute the model
		denoiser.current!.execute();
		/*
		// send to the denoiser
		denoiser.current!.setData(
			"color",
			pixelBuffer,
			renderTarget.width,
			renderTarget.height,
		);
		});
		*/
	};
	const readRenderTargetData = (renderTarget: THREE.WebGLRenderTarget) => {
		const texture = renderTarget.texture;
		const { width, height } = renderTarget.texture.image;
		//console.log("Texture height and width:", height, width);
		let pixelBuffer: Uint8Array | Uint16Array | Float32Array;
		if (texture.type === THREE.FloatType) {
			pixelBuffer = new Float32Array(width * height * 4);
		} else if (texture.type === THREE.HalfFloatType) {
			pixelBuffer = new Uint16Array(width * height * 4);
		} else {
			pixelBuffer = new Uint8Array(width * height * 4);
		}

		// Read pixels from the render target to the buffer
		gl.readRenderTargetPixels(renderTarget, 0, 0, width, height, pixelBuffer);
		return pixelBuffer;
	};

	// loop
	useFrame((state) => {
		const { gl, scene, camera } = state;
		// see if this clears the webGL issue
		if (tfActive.current) return;
		gl.resetState();

		gl.setRenderTarget(renderTarget);
		gl.render(scene, camera);
		//gl.resetState();

		// reset to default render target
		gl.setRenderTarget(null);
		gl.render(scene, camera);
		gl.resetState();
	}, 1);

	useControls({
		snapShotRenderTargetOld: button(snapShotRenderTarget),
		snapshotTestImage: button(snapshotTestImage),
		initializeDenoiser: button(() => initializeDenoiser()),
		initializeDenoiserThreeJS: button(() => initializeDenoiser(true)),
		initializeDenoiserPrivateContext: button(initializeDenoiserPrivateContext),
	});
	return (
		<>
			<OrbitControls />
			<mesh ref={planeRef} rotation={[0, 1, 0]}>
				<planeGeometry args={[10, 10]} />
				<meshBasicMaterial map={voltronTexture} />
			</mesh>
			<mesh ref={planeUint8Ref} position={[6, 0, 0]} rotation={[0, -1, 0]}>
				<planeGeometry args={[10, 10]} />
				<meshBasicMaterial map={voltronTexture} />
			</mesh>

			<mesh position={[3, -6, 1]}>
				<torusKnotGeometry args={[1, 0.4, 100, 16]} />
				<meshStandardMaterial color="#C47AC0" metalness={1.0} />
			</mesh>

			<directionalLight position={[0, 10, 0]} intensity={10} />
		</>
	);
}

export default App;

export function applySRGBEncoding(imageData: Float32Array): Float32Array {
	const encodedData = new Float32Array(imageData.length);
	for (let i = 0; i < imageData.length; i += 4) {
		const r = imageData[i];
		const g = imageData[i + 1];
		const b = imageData[i + 2];
		const a = imageData[i + 3];

		encodedData[i] = Math.pow(r, 2.2);
		encodedData[i + 1] = Math.pow(g, 2.2);
		encodedData[i + 2] = Math.pow(b, 2.2);
		encodedData[i + 3] = a;
	}
	return encodedData;
}

function getWebGLRenderingContext(canvas: HTMLCanvasElement) {
	const WEBGL_ATTRIBUTES = {
		alpha: false,
		antialias: false,
		premultipliedAlpha: true,
		preserveDrawingBuffer: false,
		depth: false,
		stencil: false,
		failIfMajorPerformanceCaveat: true,
		powerPreference: "low-power",
	};
	return canvas.getContext("webgl2", WEBGL_ATTRIBUTES);
}
