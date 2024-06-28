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

export class Weights {
    private static instance: Weights;
    private weights: Map<string, Float32Array>;

    private constructor() {
        console.log('weights initialized');
        this.weights = new Map();
    }

    public static getInstance(): Weights {
        if (!Weights.instance) {
            Weights.instance = new Weights();
        }
        return Weights.instance;
    }

    add(weight: Float32Array, name: string) {
        this.weights.set(name, weight);
    }

    // Other methods...
}
