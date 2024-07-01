/* Weights from C++ rtfilter
models.hdr           = {blobs::weights::rt_hdr,
                        blobs::weights::rt_hdr_small};
models.hdr_alb       = {blobs::weights::rt_hdr_alb,
                        blobs::weights::rt_hdr_alb_small};
models.hdr_alb_nrm   = {blobs::weights::rt_hdr_alb_nrm,
                        blobs::weights::rt_hdr_alb_nrm_small};
models.hdr_calb_cnrm = {blobs::weights::rt_hdr_calb_cnrm,
                        blobs::weights::rt_hdr_calb_cnrm_small,
                        blobs::weights::rt_hdr_calb_cnrm_large};
models.ldr           = {blobs::weights::rt_ldr,
                        blobs::weights::rt_ldr_small};
models.ldr_alb       = {blobs::weights::rt_ldr_alb,
                        blobs::weights::rt_ldr_alb_small};
models.ldr_alb_nrm   = {blobs::weights::rt_ldr_alb_nrm,
                        blobs::weights::rt_ldr_alb_nrm_small};
models.ldr_calb_cnrm = {blobs::weights::rt_ldr_calb_cnrm,
                        blobs::weights::rt_ldr_calb_cnrm_small};
models.alb           = {blobs::weights::rt_alb,
                        nullptr,
                        blobs::weights::rt_alb_large};
models.nrm           = {blobs::weights::rt_nrm,
                        nullptr,
                        blobs::weights::rt_nrm_large};
                        */

import { type TensorMap, loadDefaultTZAFile, loadTZAFile, parseTZA } from "./tza";

/* The "models" are just collections of weights based on these main parameters:
If there is a beauty pass (you can denoise just albedo and normal)
If there is a normal pass
If there is an albedo pass
If those passes are already clean (cleanAux)
If the image is HDR
If the image is sRGB (based on HDR and if its a normal)
If the image has directionals
The quality of the denoising

The models I expect most are ldr and ldr_calb_cnrm which is a plain beauty pass 
or a beauty pass with CLEAN albedo and normal.
Becasuse these are not actual tensorflow models I wont call them that, internally they
were already typed as TensorMaps
*/
type Collections = Map<string, TensorMap>;
export class Weights {
    private static instance: Weights;
    private collections: Collections;

    private constructor() {
        //console.log('Weights initialized...');
        this.collections = new Map();
    }

    public static getInstance(): Weights {
        if (!Weights.instance) Weights.instance = new Weights();
        return Weights.instance;
    }

    async getCollection(collection = 'rt_ldr', path?: string): Promise<TensorMap> {
        //if we already have the collection return it
        if (this.collections.has(collection)) return this.collections.get(collection)!;
        // else load it remote
        let buffer: ArrayBuffer;
        if (path) buffer = await loadTZAFile(path);
        else buffer = await loadDefaultTZAFile(`${collection}.tza`);

        const tensorMap = parseTZA(buffer);
        this.collections.set(collection, tensorMap);
        return tensorMap;
    }

    // util
    has(collection: string) {
        return this.collections.has(collection);
    }
}