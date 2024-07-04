import { useState, useEffect, useRef } from "react";
import { useThree } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import { useControls, button } from "leva";

import * as THREE from "three";
import "./App.css";
// for base testing import oidn
import { Denoiser } from "OIDNFlow";
import { useFrame } from "@react-three/fiber";
import { OrbitControls, useFBO } from "@react-three/drei";

function App() {
	const hasMounted = useRef(false);
	const planeRef = useRef<THREE.Mesh>(null);
	const denoiser = useRef<Denoiser | null>(null);
	const fullTexture = useRef<THREE.Texture | null>(null);
	const { gl } = useThree();

	const renderTarget = useFBO(512, 512, { type: THREE.FloatType });

	// load the sample textures
	const [voltronTexture, piratesTexture] = useTexture([
		"voltron.png",
		"pirates.png",
	]);

	// Initialize the denoiser
	useEffect(() => {
		if (hasMounted.current) return;
		hasMounted.current = true;
		console.log("pirates", piratesTexture);
		denoiser.current = new Denoiser("webgl", gl.domElement);
		denoiser.current.debugging = true;
		//denoiser.current.outputMode = "webgl";
		denoiser.current.outputMode = "float32";

		// get the height and width of the noisey image
	}, []);

	const updateTextureData = (newTexture: ImageData) => {
		if (!fullTexture.current) {
			// create a new texture with 32 bit float data
			fullTexture.current = new THREE.DataTexture(
				newTexture.data,
				newTexture.width,
				newTexture.height,
				THREE.RGBAFormat,
				THREE.FloatType,
			);
			fullTexture.current.repeat.set(1, 1);
			fullTexture.current.wrapS = THREE.ClampToEdgeWrapping;
			fullTexture.current.wrapT = THREE.ClampToEdgeWrapping;
		} else fullTexture.current.image.data = newTexture.data;
		fullTexture.current.needsUpdate = true;
		console.log("updated Texture", fullTexture.current);
	};

	const snapShotRenderTarget = () => {
		const pixelBuffer = readRenderTargetData(renderTarget);
		// read pixels from the render target to the buffer
		console.log("RenderTarget texture:", renderTarget.texture);
		console.log("Render target coverted to buffer:", pixelBuffer);

		// this will pass the data directly to the texture update
		updateTextureData({
			data: pixelBuffer,
			height: renderTarget.height,
			width: renderTarget.width,
		});
		// just because im lazy set the texture to the plane
		// make a totally new material
		const newMat = new THREE.MeshBasicMaterial({ map: fullTexture.current });

		// set it on the plane
		planeRef.current!.material = newMat;
		console.log(
			"%cPlane material updated",
			"background-color: violet; color: white;",
		);
	};
	const readRenderTargetData = (renderTarget: THREE.WebGLRenderTarget) => {
		const texture = renderTarget.texture;
		const { width, height } = renderTarget.texture.image;
		console.log("Texture Type:", texture.type);
		console.log("Texture height and width:", height, width);
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
		gl.resetState();

		gl.setRenderTarget(renderTarget);
		gl.render(scene, camera);
		gl.resetState();

		// reset to default render target
		gl.setRenderTarget(null);
		//gl.resetState();
	});

	// when the image loads get the data and setup the denoiser
	useEffect(() => {
		if (!denoiser.current) return;
		const imageElement = new Image();
		imageElement.src = "voltron.png";

		const handleLoad = () => {
			console.log("Image loaded");
			setTimeout(() => {
				const dn = denoiser.current;
				if (!dn) return;
				const { height, width } = imageElement;
				dn.height = height;
				dn.width = width;
				console.log(`Image loaded with height: ${height} and width: ${width}`);
				// pass the image to the denoiser
				dn.setImage("color", imageElement);

				// build the model

				dn.execute().then((data) => {
					console.log("Denoising complete");
					console.log("Buffer from the denoiser", data);
					console.log("denoiser height and width", dn.height, dn.width);
					//updateTextureData({ data, height: dn.height, width: dn.width });
				});
			}, 1000);
		};

		// Add event listener
		imageElement.addEventListener("load", handleLoad);

		// Remove event listener on cleanup
		return () => {
			imageElement.removeEventListener("load", handleLoad);
		};
	}, []);

	const values = useControls({
		updateRenderTarget: button(snapShotRenderTarget),
	});
	return (
		<>
			<OrbitControls />
			<mesh ref={planeRef}>
				<planeGeometry args={[10, 10]} />
				<meshBasicMaterial map={voltronTexture} />
			</mesh>

			<mesh position={[0, 0, 1]}>
				<torusKnotGeometry args={[1, 0.4, 100, 16]} />
				<meshStandardMaterial color="#C47AC0" metalness={1.0} />
			</mesh>

			<directionalLight position={[0, 10, 0]} intensity={10} />
		</>
	);
}

export default App;
