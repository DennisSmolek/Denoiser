import * as THREE from 'three';
import { WebGLPathTracer } from 'three-gpu-pathtracer';
import { generateRadialFloorTexture } from './utils/generateRadialFloorTexture.js';
import { LDrawLoader } from 'three/examples/jsm/loaders/LDrawLoader.js';
import { LDrawUtils } from 'three/examples/jsm/utils/LDrawUtils.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import { Pane } from 'tweakpane';
import * as EssentialsPlugin from '@tweakpane/plugin-essentials';

import { Denoiser } from "denoiser";
import { AlbedoNormalPass } from './albedoAndNormalsRenderPass.js';

type statsObject = { samples: number, isDenoised: boolean };
export class Renderer {
    private renderer: THREE.WebGLRenderer;
    pathtracer!: WebGLPathTracer;
    denoiser!: Denoiser;
    pane!: Pane;
    private fpsGraph!: any;

    params = {
        envmapUrl: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/modern_buildings_2_2k.hdr',
        outputBase: 0,
        model: 'tokyo_Packed',
        samples: { min: 1, max: 6 },
        denoiseAt: 6,
        pfade: 500,
        dfade: 500,
        useCanvasInput: false,
        showBackdrop: false,
        useAux: true,
        denoiserEnabled: true,
        paused: false,
        pathtracerEnabled: true,
        useFast: true,
        sampleCount: 0,
        outSpace: 2,
        ptOutSpace: 2,
        denoiserInSpace: 0,
        toneMapping: THREE.ACESFilmicToneMapping,
        tonemappingExposure: 1

    };
    // stats for output
    stats = {
        samples: 0,
        isDenoised: false,
        denoising: false,
        inputHandleTime: 0,
        executionTime: 0,
        renderTime: 0,
    }

    private scene = new THREE.Scene();
    private flatScene = new THREE.Scene();

    private camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    private flatCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // RenderTargets
    private colorRenderTarget!: THREE.WebGLRenderTarget;
    private albedoNormalRenderTarget!: THREE.WebGLRenderTarget;
    private denoiserRenderTarget = new THREE.WebGLRenderTarget(1, 1);
    private conversionRenderTarget!: THREE.WebGLRenderTarget;
    private denoiserTargetStaged = false;

    // Passes
    private albedoNormalPass = new AlbedoNormalPass();

    // Output Textures
    private outputTextures = new Map<string, THREE.Texture>();

    //* Data
    private models = new Map<string, THREE.Object3D>();

    // status for denoiser
    private denoising = false;
    private denoiseBlocked = false;


    // holders for timers
    private timers = {
        inputStartTime: 0,
        executionStartTime: 0,
        renderStartTime: 0,
        pfadeStart: 0,
        dfadeStart: 0,
    }

    // Scene objects
    private fullscreenQuad?: THREE.Mesh;
    public controls: OrbitControls;
    private modelHolder = new THREE.Object3D();
    private backdrop: THREE.Mesh | undefined;

    private statsListeners = new Set<(arg0: statsObject) => void>();
    private backendListeners = new Set<(arg: Renderer) => void>();
    private _ready = false;

    constructor(canvas: HTMLCanvasElement) {
        this.renderer = new THREE.WebGLRenderer({ canvas, });
        //this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        // Set initial size
        //this.setSize(canvas.width, canvas.height);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.setupPane();
        this.setupRenderTargets();
        this.setupMainScene();
        this.setupFlatScene();
        this.setupPathtracer();
        this.setupDenoiser();
        this.setupProgressListeners();
        //bind the mouse so when the domElemnt is clicked we can hide the denoiser
        this.renderer.domElement.addEventListener('mousedown', () => this.hideDenoisedOutput());


    }
    //* Getters n Setters ----------------------------
    get gl() {
        return this.renderer.getContext() as WebGL2RenderingContext;
    }
    get ready() {
        return this._ready;
    }
    set ready(value: boolean) {
        this._ready = value;
        if (value) for (const listener of this.backendListeners) listener(this);
    }

    //* Initializers ----------------------------
    private setupRenderTargets() {
        // Set up your render targets
        this.colorRenderTarget = new THREE.WebGLRenderTarget(
            this.renderer.domElement.width,
            this.renderer.domElement.height,
        );
        this.albedoNormalRenderTarget = new THREE.WebGLRenderTarget(
            this.renderer.domElement.width,
            this.renderer.domElement.height,
            { count: 2 }
        );
        this.albedoNormalRenderTarget.textures[0].name = 'albedo';
        this.albedoNormalRenderTarget.textures[1].name = 'normal';
        // for converteing pathtracer output to a usable format
        this.conversionRenderTarget = new THREE.WebGLRenderTarget(
            this.renderer.domElement.width,
            this.renderer.domElement.height,
            {
                format: THREE.RGBAFormat,
                type: THREE.UnsignedByteType
            }
        );
        //@ts-ignore
        this.conversionRenderTarget.colorspace = THREE.SRGBColorSpace;
    }

    //* Scene Setup ----------------------------
    private setupMainScene() {
        this.scene.background = new THREE.Color('#A7ABDD');
        this.scene.add(this.modelHolder);
        this.modelHolder.position.set(-2, 0.13, 0);

        // Floor ---
        const floorTex = generateRadialFloorTexture(2048);
        const floorPlane = new THREE.Mesh(
            new THREE.PlaneGeometry(),
            new THREE.MeshStandardMaterial({
                map: floorTex,
                transparent: true,
                color: 0x111111,
                roughness: 0.1,
                metalness: 0.0,
                side: THREE.DoubleSide,
            })
        );
        floorPlane.scale.setScalar(15);
        floorPlane.rotation.x = - Math.PI / 2;
        this.scene.add(floorPlane);

        // Test sphere to test current state
        const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(1, 32, 32),
            new THREE.MeshStandardMaterial({
                color: 0x00ff00,
                roughness: 0.5,
                metalness: 0.0,
            })
        );

        sphere.position.set(0, 1, 0);
        //this.scene.add(sphere);

        // Backdrop wall to check constant noise
        this.backdrop = new THREE.Mesh(
            new THREE.PlaneGeometry(30, 10),
            new THREE.MeshStandardMaterial({
                color: '#724CF9',
                roughness: 0.5,
                metalness: 0.5,
            })
        );

        this.backdrop.position.set(0, 5, -5);
        this.backdrop.visible = this.params.showBackdrop;
        this.scene.add(this.backdrop);

        //const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        //this.scene.add(ambientLight);
        // this.scene.add(new THREE.DirectionalLight(0xffffff, 0.5));
        this.scene.add(this.camera);
        this.camera.position.set(-3.589, 1.17, 5.64);
        this.controls.target.set(-0.01, 2.13, -0.65)
        this.camera.aspect = this.renderer.domElement.width / this.renderer.domElement.height;
        this.camera.updateProjectionMatrix();

        // document.body.appendChild(this.statsWidget.dom);
        this.updateEnvMap();

        // load the default model
        this.setModel('tokyo_Packed');
    }

    // this outputs the final images to the screen
    private setupFlatScene() {
        this.fullscreenQuad = new THREE.Mesh(
            new THREE.PlaneGeometry(2, 2),
            new THREE.ShaderMaterial({
                vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = vec4(position.xy, 0.0, 1.0);
          }
        `,
                fragmentShader: `
            uniform sampler2D color;
            uniform sampler2D albedo;
            uniform sampler2D normal;
            uniform sampler2D pathtraced;
            uniform sampler2D denoised;
            uniform int base;
            uniform float pBlend;
            uniform float dBlend;
            uniform bool canvasIn;
            uniform bool readingPT;
            uniform int outputSpace;
            uniform int ptOutputSpace;
            uniform int denoiserInSpace;


            varying vec2 vUv;

            vec4 LinearToSRGB(vec4 value) {
                return vec4(pow(value.rgb, vec3(1.0 / 2.2)), value.a);
            }
            vec4 SRGBToLinear(vec4 value) {
                return vec4(pow(value.rgb, vec3(2.2)), value.a);
            }

          void main() {
            
            vec4 finalOut;
            vec4 tBase = texture2D(color, vUv);
            vec4 pOut = mix(tBase, texture2D(pathtraced, vUv), pBlend);
            vec4 finalDenoised = texture2D(denoised, vUv);
            if(canvasIn) {
                finalDenoised = SRGBToLinear(finalDenoised);
            }
                switch(denoiserInSpace) {
                    case 1:
                        finalDenoised = SRGBToLinear(finalDenoised);
                        break;
                    case 2:
                        finalDenoised = LinearToSRGB(finalDenoised);
                        break;
                    default:
                        break;
                }

            vec4 dOut = mix(pOut, finalDenoised , dBlend);
            
            switch (base) {
                case 1:
                    finalOut = tBase;
                    break;
                case 2:
                    finalOut = texture2D(albedo, vUv);
                    break;
                case 3:
                    finalOut = texture2D(normal, vUv);
                    break;
                case 4:
                    finalOut = texture2D(pathtraced, vUv);
                    break;
                case 5:
                    finalOut = finalDenoised;
                    break;
               default:
                    finalOut = dOut; 
                    break;
            }
            if(readingPT) {
                vec4 ptOut = texture2D(pathtraced, vUv);
                switch(ptOutputSpace) {
                    case 1:
                        ptOut = SRGBToLinear(ptOut);
                        break;
                    case 2:
                        ptOut = LinearToSRGB(ptOut);
                        break;
                    default:
                        break;
                }
            } else {
                switch(outputSpace) {
                    case 1:
                        finalOut = SRGBToLinear(finalOut);
                        break;
                    case 2:
                        finalOut = LinearToSRGB(finalOut);
                        break;
                    default:
                        
                        break;
                }
            }
            gl_FragColor = finalOut;
            
          }
        `,
                uniforms: {
                    color: { value: null },
                    albedo: { value: null },
                    normal: { value: null },
                    pathtraced: { value: null },
                    denoised: { value: null },
                    base: { value: 0.0 },
                    pBlend: { value: 0.0 },
                    dBlend: { value: 0.0 },
                    canvasIn: { value: false },
                    readingPT: { value: false },
                    outputSpace: { value: 2 },
                    ptOutputSpace: { value: 2 },
                    denoiserInSpace: { value: 0 },
                }
            })
        );
        this.flatScene.add(this.fullscreenQuad);
    }
    // move me later
    setSpaceValue(space: string, value: number) {
        //@ts-ignore
        this.fullscreenQuad!.material.uniforms[space].value = value;
    }
    // Setup controls
    setupPane() {
        this.pane = new Pane({
            container: this.renderer.domElement.parentElement!,
        });
        this.pane.registerPlugin(EssentialsPlugin);
        // fps
        this.fpsGraph = this.pane.addBlade({
            view: 'fpsgraph',
            label: 'fps',
            rows: 2
        });
        this.pane.addBinding(this.params, 'sampleCount', { label: 'Samples', readonly: true });


        const options = this.pane.addFolder({ title: 'Options', expanded: false, });
        const envOptions = Object.entries(envMaps).map(([key, value]) => ({ text: key, value: value }));
        options.addBinding(this.params, 'envmapUrl', {
            label: 'Environment Map',
            view: 'list',
            options: envOptions
        }).on('change', () => this.updateEnvMap());
        // models
        options.addBinding(this.params, 'model', {
            label: 'Model',
            view: 'list',
            options: [
                { text: 'Tokyo', value: 'tokyo_Packed' },
                { text: 'Bonsai', value: 'bonsai_Packed' },
                { text: 'Bonsai-Cherry', value: 'bonsai-cherry_Packed' },
            ]
        }).on('change', () => this.setModel(this.params.model));


        options.addBinding(this.params, 'showBackdrop', { label: 'Show Backdrop' }).on('change', () => {
            if (this.backdrop) this.backdrop.visible = this.params.showBackdrop;
            // we also have to update the scene
            this.sceneUpdated();
        });
        // Renderer Options
        const rendererFolder = options.addFolder({ title: 'Renderer', expanded: false, });
        // dropdowns for colorspaces
        rendererFolder.addBinding(this.params, 'outSpace', { label: 'Output Space', view: 'select', options: [{ text: 'Default', value: 0 }, { text: 'sRGB to linear', value: 1 }, { text: 'linear to srgb', value: 2 }] })
            .on('change', () => this.setSpaceValue('outputSpace', this.params.outSpace));

        rendererFolder.addBinding(this.params, 'ptOutSpace', { label: 'PT Output Space', view: 'select', options: [{ text: 'Default', value: 0 }, { text: 'sRGB to linear', value: 1 }, { text: 'linear to srgb', value: 2 }] })
            .on('change', () => this.setSpaceValue('ptOutputSpace', this.params.ptOutSpace));
        rendererFolder.addBinding(this.params, 'denoiserInSpace', { label: 'Denoiser Input Space', view: 'select', options: [{ text: 'Default', value: 0 }, { text: 'sRGB to linear', value: 1 }, { text: 'linear to srgb', value: 2 }] })
            .on('change', () => this.setSpaceValue('denoiserInSpace', this.params.denoiserInSpace));
        // renderer tonemapping
        rendererFolder.addBlade({ view: 'separator' });
        rendererFolder.addBinding(this.params, 'toneMapping', {
            label: 'Tone Mapping',
            view: 'select',
            options: [
                { text: 'No Tone Mapping', value: THREE.NoToneMapping },
                { text: 'Linear Tone Mapping', value: THREE.LinearToneMapping },
                { text: 'Reinhard Tone Mapping', value: THREE.ReinhardToneMapping },
                { text: 'Cineon Tone Mapping', value: THREE.CineonToneMapping },
                { text: 'ACES Filmic Tone Mapping', value: THREE.ACESFilmicToneMapping },
                { text: 'AgX Tone Mapping', value: THREE.AgXToneMapping },
                { text: 'Neutral Tone Mapping', value: THREE.NeutralToneMapping },
                { text: 'Custom Tone Mapping', value: THREE.CustomToneMapping },
            ],
        }).on('change', () => {
            console.log('Tonemapping before', this.renderer.toneMapping);
            this.renderer.toneMapping = this.params.toneMapping;
            console.log('Tonemapping after', this.renderer.toneMapping);
        });
        rendererFolder.addBinding(this.params, 'tonemappingExposure', {
            label: 'Tonemapping Exposure',
            view: 'slider',
            min: -5,
            max: 5,
            step: 0.1
        }).on('change', () => {
            this.renderer.toneMappingExposure = this.params.tonemappingExposure;
        });

        // Pathtracer options ----------------------------
        const ptFolder = options.addFolder({ title: 'Pathtracer', expanded: false, });
        ptFolder.addBinding(this.params, 'pathtracerEnabled', { label: 'Enable Pathtracer' })
        ptFolder.addBinding(this.params, 'samples', { label: 'Samples', min: 1, max: 100, step: 1 })

        // Denoiser options ----------------------------
        const dnFolder = options.addFolder({ title: 'Denoiser', expanded: false, });
        dnFolder.addBinding(this.params, 'denoiserEnabled', { label: 'Enable Denoiser' });
        dnFolder.addBinding(this.params, 'useFast', { label: 'Use Fast Quality' })
            .on('change', () => {
                this.denoiser.quality = this.params.useFast ? 'fast' : 'balanced';
                //this.denoiser.resetInputs();
                this.pathtracer.reset();
                this.hideDenoisedOutput();
            });

        dnFolder.addBinding(this.params, 'denoiseAt', { label: 'Denoise At', min: 1, max: 100, step: 1 });
        dnFolder.addBinding(this.params, 'useAux', { label: 'Use Aux Textures' })
            .on('change', () => {
                this.denoiser.resetInputs();
                this.hideDenoisedOutput();
                this.pathtracer.reset();
            });

        dnFolder.addBinding(this.params, 'useCanvasInput', { label: 'Use Canvas Input' })
            .on('change', () => {
                this.denoiser.resetInputs();
                this.denoiser.flipOutputY = this.params.useCanvasInput;
                //@ts-ignore
                this.fullscreenQuad!.material.uniforms.canvasIn.value = this.params.useCanvasInput;
                this.pathtracer.reset();
                this.hideDenoisedOutput();
            });

        // options.addBinding(this, 'paused', { label: 'Pause' });
        options.addBinding(this.params, 'outputBase', {
            label: 'Final Output:',
            view: 'select',
            options: [
                { text: 'Beauty', value: 0 },
                { text: 'color', value: 1 },
                { text: 'Albedo', value: 2 },
                { text: 'Normals', value: 3 },
                { text: 'Pathtraced', value: 4 },
                { text: 'Denoised', value: 5 },
            ],
        }).on('change', () => this.updateBaseOutput(this.params.outputBase));
        const abortButton = this.pane.addButton({
            title: 'Abort Denoising'
        });
        abortButton.on('click', () => {
            this.denoiser.abort();
            this.denoising = false;
        });
    }

    setupProgressListeners() {
        // wrapper for the span itself
        const inProgressSpan = document.getElementById('denoising');
        const progressOut = document.getElementById('progress');
        this.denoiser.onProgress((progress: number) => {
            if (0 < progress && progress < 1) {
                inProgressSpan!.style.display = 'flex';
                progressOut!.style.display = 'block';
            }
            else {
                inProgressSpan!.style.display = 'none';
                progressOut!.style.display = 'none';
            }
            progressOut!.innerHTML = `${progress * 100}%`;
            console.log('Tiling Progress:', progress);
        });

    }
    //* Pathtracer Methods ----------------------------

    // Setup the pathtracer
    setupPathtracer() {
        this.pathtracer = new WebGLPathTracer(this.renderer);
        this.pathtracer.setScene(this.scene, this.camera);

        // disable default fade and protections
        this.pathtracer.renderToCanvas = false;
        this.pathtracer.renderDelay = 100;
        this.pathtracer.fadeDuration = 0;
        this.pathtracer.minSamples = 0;
        this.pathtracer.renderScale = 1;
        this.pathtracer.multipleImportanceSampling = true;

        // update the camera when the controls change
        this.controls.addEventListener('change', () => {
            this.pathtracer.updateCamera();
            this.denoiseBlocked = false;
            // console.log('Camera position', this.camera.position);
            //console.log('Camera target', this.controls.target);
        });
    }

    // render the pathtracer and fire various events
    renderPathtracer() {
        this.pathtracer.renderSample();
        this.outputTextures.set('pathtraced', this.pathtracer.target.texture);
        this.stats.samples = Math.floor(this.pathtracer.samples);
        this.params.sampleCount = this.stats.samples;
        // if the samples is 1, the pathtracer has just started
        if (this.stats.samples === 1) {
            this.timers.pfadeStart = Date.now();
            // set the denoiser to 0
            this.timers.dfadeStart = 0;
        }
        if (!this.params.useCanvasInput) this.statsUpdated();
    }
    // take the input texture, apply tonemapping and color conversion
    simulateOutput() {
        //this.resetState();
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.readingPT = { value: true };
        this.renderer.setRenderTarget(this.conversionRenderTarget);
        this.renderer.render(this.flatScene, this.flatCamera);
        this.renderer.setRenderTarget(null);
        // reset the full screen quad
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.readingPT = { value: false };
        //this.resetState();
        this.outputTextures.set('pathtracedSRGB', this.conversionRenderTarget.texture);
    }

    //* Denoiser Methods ----------------------------
    setupDenoiser() {
        // Initialization
        this.denoiser = new Denoiser("webgl", this.renderer.domElement);
        this.denoiser.outputMode = 'webgl';
        this.denoiser.inputMode = 'webgl';
        this.denoiser.debugging = true;
        //this.denoiser.hdr = true;
        //this.denoiser.usePassThrough = true;
        this.denoiser.height = this.renderer.domElement.height;
        this.denoiser.width = this.renderer.domElement.width;
        //this.denoiser.flipOutputY = true;

        // for debugging dump the denoiser to the rawOutput canvas
        //this.denoiser.setCanvas(document.getElementById("rawOutput") as HTMLCanvasElement);

        this.denoiser.onBackendReady(() => {
            // start the renderer and let other systems know
            this.ready = true;
            this.animate();
        });

        // on execution
        this.denoiser.onExecute((denoisedWebGLTexture: WebGLTexture) => {
            // pull the original color texture
            const denoisedTexture = this.denoiserRenderTarget.texture;
            // flipY
            denoisedTexture.flipY = true;

            // insert the webGL into this texture
            const denoisedTextureProps = this.renderer.properties.get(denoisedTexture);
            denoisedTextureProps.__webglTexture = denoisedWebGLTexture;

            this.outputTextures.set('denoised', denoisedTexture);
            this.stats.isDenoised = true;
            this.statsUpdated();
            this.timers.dfadeStart = Date.now();
            this.denoising = false;
            this.stats.executionTime = performance.now() - this.timers.executionStartTime;
            this.stats.denoising = false;
            console.log('Denoising Stats')
            console.table(this.stats);
            const denoisedSpan = document.getElementById('denoised');
            denoisedSpan!.style.display = 'block';
        }, 'webgl')
    }

    async runDenoiser() {
        if (this.denoising || this.denoiseBlocked || !this.params.denoiserEnabled) return;
        // we dont want to run the denoiser if we arent on a beauty output
        if (this.params.outputBase !== 0) return;
        // Take the pathtracer output, tonemap and convert it 
        this.simulateOutput();
        this.resetState();
        // block future denoising attempts
        // todo: do we need two blockers?
        this.denoising = true;
        this.denoiseBlocked = true;
        console.log('%c Denoise Step Started...', 'color: purple');
        this.timers.inputStartTime = performance.now();

        // get the webgl texture out of this
        const colorTexture = this.getWebGLTexture(this.outputTextures.get('pathtracedSRGB')!, true);
        const albedoTexture = this.getWebGLTexture(this.outputTextures.get('albedo')!, true);
        const normalTexture = this.getWebGLTexture(this.outputTextures.get('normal')!, true);

        // run like normal with textures feeding in/out
        if (this.params.useCanvasInput) {
            // can we pass the canvas directly?
            console.log('Setting input image directly from canvas')
            this.denoiser.setInputImage('color', this.renderer.domElement!);
            // flip the inputs to match the canvas input
            if (this.params.useAux) {
                await this.denoiser.setInputTexture('albedo', albedoTexture, { flipY: true });
                await this.denoiser.setInputTexture('normal', normalTexture, { flipY: true });
            }
        } else {
            await this.denoiser.setInputTexture('color', colorTexture);
            if (this.params.useAux) {
                await this.denoiser.setInputTexture('albedo', albedoTexture);
                await this.denoiser.setInputTexture('normal', normalTexture);
            }
        }
        this.stats.inputHandleTime = performance.now() - this.timers.inputStartTime;
        console.log('Denoiser Input time', Math.round(this.stats.inputHandleTime));

        const denoisedSpan = document.getElementById('denoised');
        denoisedSpan!.style.display = 'none';
        //todo: with input mode set I should be able to pass these directly
        this.timers.executionStartTime = performance.now();
        this.denoiser.execute();
    }

    //* Renderer managment ----------------------------

    // backend listener when renderer is ready
    onReady(callback: () => void) {
        if (this.ready) callback();
        else this.backendListeners.add(callback);
        // return removal function
        return () => this.backendListeners.delete(callback);
    }

    setSize(width: number, height: number): void {
        this.renderer.domElement.width = width;
        this.renderer.domElement.height = height;
        this.renderer.setSize(width, height);
    }

    resetState(): void {
        const gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindRenderbuffer(gl.RENDERBUFFER, null);
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.disable(gl.BLEND);
        gl.disable(gl.CULL_FACE);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.DITHER);
        gl.disable(gl.SCISSOR_TEST);
        gl.disable(gl.STENCIL_TEST);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        gl.useProgram(null);
    }

    //* For when you change things in the control panel
    updateEnvMap() {
        new RGBELoader().load(this.params.envmapUrl, (texture: THREE.Texture) => {
            texture.mapping = THREE.EquirectangularReflectionMapping;
            this.scene.background = texture;
            this.scene.environment = texture;
            this.pathtracer.updateEnvironment();
            this.hideDenoisedOutput();
        });
    }

    updateBaseOutput(base = 0) {
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.base.value = base;
    }

    // update the uniforms of the output materials
    updateTextures() {
        for (const [name, texture] of this.outputTextures) {
            const material = this.fullscreenQuad!.material as THREE.ShaderMaterial;
            material.uniforms[name] = { value: texture };
        }
        // update the blends of textures
        let pfade = Math.min(1, Math.max(0, (Date.now() - this.timers.pfadeStart) / this.params.pfade));
        let dfade = Math.min(1, Math.max(0, (Date.now() - this.timers.dfadeStart) / this.params.dfade));
        if (this.stats.samples < this.params.samples.min) pfade = 0;
        if (this.timers.dfadeStart === 0) dfade = 0;
        //force disable
        if (!this.params.pathtracerEnabled) pfade = 0;
        if (!this.params.denoiserEnabled) dfade = 0;
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.pBlend.value = pfade;
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.dBlend.value = dfade;
    }

    // when the mouse goes down initially we need to hide the denoiser right away
    hideDenoisedOutput() {
        this.pathtracer.reset();
        this.timers.dfadeStart = 0;
        this.denoiseBlocked = false;
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.dBlend.value = 0;
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.pBlend.value = 0;
    }

    // update the pathtracer scene when things in the scene chage
    sceneUpdated() {
        this.pathtracer.setScene(this.scene, this.camera);
        this.hideDenoisedOutput();
    }

    statsUpdated() {
        this.statsListeners.forEach(listener => listener(this.stats));
    }

    onStats(callback: (arg0: { samples: number, isDenoised: boolean }) => void) {
        this.statsListeners.add(callback);
        return () => this.statsListeners.delete(callback);
    }

    //* Model Handling ----------------------------
    // set the model after loading it
    async setModel(modelName: string) {
        // get the model, load if needed
        const model = await this.loadModel(modelName);
        // remove old model
        this.modelHolder.clear();
        // add new model
        this.modelHolder.add(model);
        this.sceneUpdated();
        return true;
    }

    // load the model, store it in the cache return a callback
    async loadModel(modelName: string): Promise<THREE.Object3D> {
        if (this.models.has(modelName)) return this.models.get(modelName)!;

        const loader = new LDrawLoader();
        loader.smoothNormals = true;
        let model = await loader.loadAsync(`./models/${modelName}.mpd`);
        // Remove the lines and reduce the roughness
        const toRemove: THREE.LineSegments[] = [];
        model.traverse(c => {
            if (isLineSegments(c)) toRemove.push(c);
            if (isMesh(c)) (c.material as THREE.MeshStandardMaterial).roughness *= 0.25;
        });
        for (const c of toRemove) c.parent?.remove(c);

        // merge the object
        model = LDrawUtils.mergeObject(model);
        model.scale.setScalar(0.01);

        // model needs to be flipped
        model.rotation.x = Math.PI;

        this.models.set(modelName, model);
        return model;
    }

    private animate() {
        this.fpsGraph.begin();
        if (!this.params.paused) {
            this.controls.update();
            this.draw();
        }
        this.fpsGraph.end();

        requestAnimationFrame(() => this.animate());
    }

    private draw() {
        if (this.params.paused) return;
        this.renderer.resetState();
        const originalEnvMap = this.scene.environment;
        const originalBackground = this.scene.background;

        // to setup the denoiser correct with three we need to run the texture at least once
        //TODO Three actually has staging on textures, this can be removed once done #16
        if (!this.denoiserTargetStaged) {
            this.renderer.setRenderTarget(this.denoiserRenderTarget);
            this.renderer.render(this.scene, this.camera);
            this.denoiserTargetStaged = true;
        }

        //* Standard base pass
        this.renderer.setRenderTarget(this.colorRenderTarget);
        this.renderer.render(this.scene, this.camera);
        this.outputTextures.set('color', this.colorRenderTarget.texture);

        //* Albedos and normals pass
        if (this.params.useAux) {
            this.scene.environment = null;
            this.scene.background = null;
            this.albedoNormalPass.render(this.renderer, this.scene, this.camera, this.albedoNormalRenderTarget);
            this.outputTextures.set('albedo', this.albedoNormalRenderTarget.textures[0]);
            this.outputTextures.set('normal', this.albedoNormalRenderTarget.textures[1]);
        }

        //* Pathtracer Pass
        this.scene.environment = originalEnvMap;
        this.scene.background = originalBackground;
        this.renderer.setRenderTarget(null);
        if (this.params.pathtracerEnabled && this.pathtracer.samples < this.params.samples.max) {
            this.renderPathtracer();
        }

        //* Final output
        this.updateTextures();
        this.renderer.render(this.flatScene, this.flatCamera);
        this.renderer.resetState();
        //* Denoiser Pass
        if (Math.round(this.pathtracer.samples) === this.params.denoiseAt) {
            this.runDenoiser();
        }
    }

    //* Utilities that require things in the class ===========================================
    // Get the webGLTexture out of a renderTarget
    getWebGLTexture(input: THREE.WebGLRenderTarget | THREE.Texture, asTexture = false): WebGLTexture {
        const baseTexture = asTexture ? input : (input as THREE.WebGL3DRenderTarget).texture;
        const textureProps = this.renderer.properties.get(baseTexture);
        return textureProps.__webglTexture;
    }
}

export default Renderer;

//* Utils ===========================================

function isLineSegments(object: THREE.Object3D): object is THREE.LineSegments {
    return (object as THREE.LineSegments).isLineSegments === true;
}

function isMesh(object: THREE.Object3D): object is THREE.Mesh {
    return (object as THREE.Mesh).isMesh === true;
}



const envMaps = {
    'Royal Esplanade': 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/equirectangular/royal_esplanade_1k.hdr',
    'Moonless Golf': 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/equirectangular/moonless_golf_1k.hdr',
    'Overpass': 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/equirectangular/pedestrian_overpass_1k.hdr',
    'Venice Sunset': 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/equirectangular/venice_sunset_1k.hdr',
    'Small Studio': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/studio_small_05_1k.hdr',
    'Pfalzer Forest': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/phalzer_forest_01_1k.hdr',
    'Leadenhall Market': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/leadenhall_market_1k.hdr',
    'Kloppenheim': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/kloppenheim_05_1k.hdr',
    'Hilly Terrain': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/hilly_terrain_01_1k.hdr',
    'Circus Arena': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/circus_arena_1k.hdr',
    'Chinese Garden': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/chinese_garden_1k.hdr',
    'Autoshop': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/autoshop_01_1k.hdr',

    'Measuring Lab': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/vintage_measuring_lab_2k.hdr',
    'Whale Skeleton': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/whale_skeleton_2k.hdr',
    'Hall of Mammals': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/hall_of_mammals_2k.hdr',

    'Drachenfels Cellar': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/drachenfels_cellar_2k.hdr',
    'Adams Place Bridge': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/adams_place_bridge_2k.hdr',
    'Sepulchral Chapel Rotunda': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/sepulchral_chapel_rotunda_2k.hdr',
    'Peppermint Powerplant': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/peppermint_powerplant_2k.hdr',
    'Noon Grass': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/noon_grass_2k.hdr',
    'Narrow Moonlit Road': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/narrow_moonlit_road_2k.hdr',
    'St Peters Square Night': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/st_peters_square_night_2k.hdr',
    'Brown Photostudio 01': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/brown_photostudio_01_2k.hdr',
    'Rainforest Trail': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/rainforest_trail_2k.hdr',
    'Brown Photostudio 07': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/brown_photostudio_07_2k.hdr',
    'Brown Photostudio 06': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/brown_photostudio_06_2k.hdr',
    'Dancing Hall': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/dancing_hall_2k.hdr',
    'Aristea Wreck Puresky': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/aristea_wreck_puresky_2k.hdr',
    'Modern Buildings 2': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/modern_buildings_2_2k.hdr',
    'Thatch Chapel': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/thatch_chapel_2k.hdr',
    'Vestibule': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/vestibule_2k.hdr',
    'Blocky Photo Studio': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/blocky_photo_studio_1k.hdr',
    'Christmas Photo Studio 07': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/christmas_photo_studio_07_2k.hdr',
    'Aerodynamics Workshop': 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/master/hdri/aerodynamics_workshop_1k.hdr',

};

// Old Methods
/*
renderToWebGLTexture(threeTexture: THREE.Texture): THREE.WebGLRenderTarget {
        this.resetState();

        const renderTarget = new THREE.WebGLRenderTarget(
            this.renderer.domElement.width,
            this.renderer.domElement.height,
            {
                format: THREE.RGBAFormat,
                type: THREE.UnsignedByteType
            }
        );

        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = threeTexture;
        this.renderer.setRenderTarget(renderTarget);
        this.renderer.render(this.scene, this.camera);
        this.renderer.setRenderTarget(null);

        this.resetState();

        this.renderTargetHolder = renderTarget;

        const textureProps = this.renderer.properties.get(renderTarget.texture);
        const outTexture = textureProps.__webglTexture

        return outTexture;
    }

    //merge and render
    mergeThenRender(webglTexture: WebGLTexture) {
        this.resetState();
        // load the old renderTarget
        const renderTarget = this.renderTargetHolder!;
        //put the webglTexture into the renderTarget
        const textureProps = this.renderer.properties.get(renderTarget.texture);
        textureProps.__webglTexture = webglTexture;


        (this.fullscreenQuad.material as THREE.ShaderMaterial).uniforms.tDiffuse.value = renderTarget.texture;
        this.renderer.resetState();
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.scene, this.camera);
        this.renderer.resetState();
        this.resetState();

    }
        */