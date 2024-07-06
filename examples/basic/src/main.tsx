import { Denoiser } from "OIDNFlow";

const denoiser = new Denoiser();
// make pure hdr work
denoiser.props.hdr = true;

console.log("Denoiser created!", denoiser);
