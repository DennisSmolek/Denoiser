import { useState, useEffect, useRef } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
// for base testing import oidn
import { Denoiser } from "denoiser";

function App() {
	const [count, setCount] = useState(0);
	const hasMounted = useRef(false);
	const noiseyImage = useRef<HTMLImageElement>(null);
	const cleanedCanvas = useRef<HTMLCanvasElement>(null);

	const denoiser = useRef<Denoiser | null>(null);
	// Initialize the denoiser
	useEffect(() => {
		if (hasMounted.current) return;
		hasMounted.current = true;
		denoiser.current = new Denoiser();
		denoiser.current.debugging = true;
		//	denoiser.current.usePassThrough = true;

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
				dn.setCanvas(canvas);
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
			<div>
				<img src="./noisey.jpg" ref={noiseyImage} alt="Noisey" />
				<canvas ref={cleanedCanvas} id="output" width="512" height="512" />
			</div>
		</>
	);
}

export default App;
