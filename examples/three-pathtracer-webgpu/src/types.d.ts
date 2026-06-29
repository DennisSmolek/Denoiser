/// <reference types="@webgpu/types" />

// three r185's "./webgpu" entry doesn't expose a types condition; re-export the
// core three types and treat the WebGPU-only additions as untyped for this glue example.
declare module 'three/webgpu' {
  export * from 'three';
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const WebGPURenderer: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const PostProcessing: any;
}

// The webgpu-pathtracer branch ships JS source without type declarations.
declare module 'three-gpu-pathtracer/webgpu' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const WebGPUPathTracer: any;
}
