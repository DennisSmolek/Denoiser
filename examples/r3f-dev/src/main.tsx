import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { Canvas } from "@react-three/fiber";

ReactDOM.createRoot(document.getElementById("root")!).render(
	<React.StrictMode>
		<OuterApp />
	</React.StrictMode>,
);

function OuterApp() {
	return (
		<Canvas
			id="r3fCanvas"
			//orthographic
			//camera={{ position: [0, 0, 10], zoom: 1, near: 0.1, far: 1000 }}
		>
			<App />
		</Canvas>
	);
}
