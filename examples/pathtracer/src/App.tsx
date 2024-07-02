import { useState, useEffect, useRef } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
// for base testing import oidn
import { Denoiser } from "OIDNFlow";

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
				dn.setCanvas(canvas);
				dn.setImage("color", imageElement);

				// build the model
				dn.execute().then(() => {
					console.log("Denoising complete");
					console.log("second pass delay 15 seconds...");
					setTimeout(() => {
						// clear the canvas
						console.log("clear");
						const ctx = canvas.getContext("2d");
						if (!ctx) return;
						ctx.clearRect(0, 0, width, height);
						//draw the canvas red to show it was cleared
						ctx.fillStyle = "red";
						ctx.fillRect(0, 0, width, height);
						setTimeout(() => {
							// set the canvas to green to show we are in a new draw phase
							ctx.fillStyle = "green";
							ctx.fillRect(0, 0, width, height);
							dn.execute().then(() => {
								console.log("Second Denoising complete");
								// go for a third time
								console.log("third pass delay 10 seconds...");
								setTimeout(() => {
									// clear the canvas
									console.log("clear");
									const ctx = canvas.getContext("2d");
									if (!ctx) return;
									ctx.clearRect(0, 0, width, height);
									//draw the canvas red to show it was cleared
									ctx.fillStyle = "red";
									ctx.fillRect(0, 0, width, height);
									setTimeout(() => {
										// set the canvas to green to show we are in a new draw phase
										ctx.fillStyle = "green";
										ctx.fillRect(0, 0, width, height);
										dn.execute().then(() => {
											console.log("Third Denoising complete");
										});
									}, 5000);
								}, 5000);
							});
						}, 5000);
					}, 10000);
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
