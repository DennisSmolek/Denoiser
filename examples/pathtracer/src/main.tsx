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
		<Canvas>
			<App />
		</Canvas>
	);
}
