import { useState, useEffect, useRef } from "react";
import { useThree } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";

import * as THREE from "three";
import "./App.css";
// for base testing import oidn
import { Denoiser } from "OIDNFlow";

function App() {
	const hasMounted = useRef(false);
	const noiseyImage = useRef<HTMLImageElement>(null);
	const cleanedCanvas = useRef<HTMLCanvasElement>(null);

	const denoiser = useRef<Denoiser | null>(null);

	const { gl } = useThree();

	// load the sample textures
	const [voltronTexture, piratesTexture] = useTexture([
		"voltron.png",
		"pirates.png",
	]);

	// Initialize the denoiser
	useEffect(() => {
		if (hasMounted.current) return;
		hasMounted.current = true;
		console.log("gl", gl);
		denoiser.current = new Denoiser("webgl", gl.domElement);
		//denoiser.current.debugging = true;

		// get the height and width of the noisey image
	}, []);

	// when the image loads get the data and setup the denoiser
	useEffect(() => {
		const imageElement = noiseyImage.current;

		if (!imageElement || !denoiser.current) return;

		const handleLoad = () => {
			setTimeout(() => {
				const dn = denoiser.current;
				const canvas = cleanedCanvas.current;
				if (!dn || !canvas) return;
				const { height, width } = imageElement;
				dn.height = height;
				dn.width = width;
				console.log(`Image loaded with height: ${height} and width: ${width}`);
				// pass the image to the denoiser
				dn.setImage("color", imageElement);

				// build the model
				dn.execute().then(() => {
					console.log("Denoising complete");
				});
			}, 5000);
		};

		// Add event listener
		imageElement.addEventListener("load", handleLoad);

		// Remove event listener on cleanup
		return () => {
			imageElement.removeEventListener("load", handleLoad);
		};
	}, []);
	return (
		<>
			<mesh>
				<planeGeometry args={[1, 1]} />
				<meshBasicMaterial color={"red"} />
			</mesh>
		</>
	);
}

export default App;
