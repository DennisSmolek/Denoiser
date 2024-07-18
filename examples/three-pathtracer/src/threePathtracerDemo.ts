import * as THREE from 'three';
import { WebGLPathTracer } from 'three-gpu-pathtracer';
// pathtracer demo imports
import { generateRadialFloorTexture } from './utils/generateRadialFloorTexture.js';
import { LDrawLoader } from 'three/examples/jsm/loaders/LDrawLoader.js';
import { LDrawUtils } from 'three/examples/jsm/utils/LDrawUtils.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import { Pane } from 'tweakpane';

import { Denoiser } from "denoiser";
import { AlbedoNormalPass } from './albedoAndNormalsRenderPass.js';

type statsObject = { samples: number, isDenoised: boolean };
export class Renderer {
    private canvas: HTMLCanvasElement;
    private renderer: THREE.WebGLRenderer;
    private scene = new THREE.Scene();
    private flatScene = new THREE.Scene();

    private camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    private flatCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // RenderTargets
    private colorRenderTarget!: THREE.WebGLRenderTarget;
    private albedoNormalRenderTarget!: THREE.WebGLRenderTarget;
    private denoiserRenderTarget = new THREE.WebGLRenderTarget(1, 1);
    private denoiserTargetStaged = false;
    // Passes
    private albedoNormalPass = new AlbedoNormalPass();

    // Output Textures
    private outputTextures = new Map<string, THREE.Texture>();
    // Data
    private models = new Map<string, THREE.Object3D>();
    public params = {
        envmapName: 'Aristea Wreck Puresky',
        outputBase: 0,
        minSamples: 1,
        maxSamples: 6,
        denoiseAt: 5,
        pfade: 500,
        dfade: 500
    };
    // stats for output
    stats = {
        samples: 0,
        isDenoised: false,
    }
    pane!: Pane;

    private fullscreenQuad?: THREE.Mesh;

    public pathtracer!: WebGLPathTracer;
    public denoiser!: Denoiser;

    public statsWidget = new Stats();
    public controls: OrbitControls;


    paused = false;
    pathtracerEnabled = true;
    doAlbedoAndNormals = true;

    modelHolder = new THREE.Object3D();

    // holders for fade timers
    private pfadeStart = 0;
    private dfadeStart = 0;

    // status for denoiser
    private denoising = false;
    private denoiseBlocked = false;

    private backendListeners = new Set<() => void>();
    private _ready = false;

    private statsListeners = new Set<(arg0: statsObject) => void>();

    public gl: WebGL2RenderingContext;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;

        this.renderer = new THREE.WebGLRenderer({ canvas });
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.gl = this.renderer.getContext() as WebGL2RenderingContext;
        // Set initial size
        this.setSize(canvas.width, canvas.height);

        this.setupPane();
        this.setupRenderTargets();
        this.setupMainScene();
        this.setupFlatScene();
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.setupPathtracer();
        // setup denoiser
        this.setupDenoiser();
    }
    //* Getters n Setters ----------------------------

    get ready() {
        return this._ready;
    }

    set ready(value: boolean) {
        this._ready = value;
        if (value) {
            for (const listener of this.backendListeners) {
                listener();
            }
        }
    }
    //* Initializers ----------------------------
    private setupRenderTargets() {
        // Set up your render targets
        this.colorRenderTarget = new THREE.WebGLRenderTarget(
            this.canvas.width,
            this.canvas.height,
        );

        this.albedoNormalRenderTarget = new THREE.WebGLRenderTarget(
            this.canvas.width,
            this.canvas.height,
            {
                count: 2,
                type: THREE.UnsignedByteType  // Changed from FloatType
            }
        );

        this.albedoNormalRenderTarget.textures[0].name = 'albedo';
        this.albedoNormalRenderTarget.textures[1].name = 'normal';
    }

    //* Scene Setup ----------------------------
    private setupMainScene() {
        this.scene.background = new THREE.Color('#A7ABDD');
        this.scene.add(this.modelHolder);

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

        // sphere to test current state
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
        const backdrop = new THREE.Mesh(
            new THREE.PlaneGeometry(30, 10),
            new THREE.MeshStandardMaterial({
                color: '#724CF9',
                roughness: 0.5,
                metalness: 0.5,
            })
        );

        backdrop.position.set(0, 5, -5);
        //backdrop.rotation.x = Math.PI / 2;
        this.scene.add(backdrop);


        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        this.scene.add(new THREE.DirectionalLight(0xffffff, 0.5));
        this.scene.add(this.camera);
        this.camera.position.set(0, 8, 15);
        this.camera.lookAt(sphere.position);
        this.camera.aspect = this.canvas.width / this.canvas.height;
        this.camera.updateProjectionMatrix();

        document.body.appendChild(this.statsWidget.dom);
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
          varying vec2 vUv;

            

          void main() {
            
            vec4 finalOut;
            vec4 tBase = texture2D(color, vUv);
            vec4 pOut = mix(tBase, texture2D(pathtraced, vUv), pBlend);
            vec4 dOut = mix(pOut, texture2D(denoised, vUv), dBlend);
           
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
                    finalOut = texture2D(denoised, vUv);
                    break;
                default:
                    finalOut = dOut; 
                    break;
            }
             gl_FragColor = LinearTosRGB(finalOut);
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
                }
            })
        );
        this.flatScene.add(this.fullscreenQuad);

    }
    // Setup controls
    setupPane() {
        this.pane = new Pane({
            title: 'Settings',
            expanded: false,
            container: this.canvas.parentElement!,
        });
        const options = Object.entries(envMaps).map(([key, value]) => ({ text: key, value: value }));
        this.pane.addBinding(this.params, 'envmapName', {
            label: 'Environment Map',
            view: 'list',
            options: options
        }).on('change', () => this.updateEnvMap());
        this.pane.addBinding(this, 'paused', { label: 'Pause' });
        this.pane.addBinding(this.params, 'outputBase', {
            label: 'Output Pass',
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
        });
    }

    // render the pathtracer and fire various events
    renderPathtracer() {
        this.pathtracer.renderSample();
        this.outputTextures.set('pathtraced', this.pathtracer.target.texture);
        this.stats.samples = Math.floor(this.pathtracer.samples);
        // if the samples is 1, the pathtracer has just started
        if (this.stats.samples === 1) {
            this.pfadeStart = Date.now();
            // set the denoiser to 0
            this.dfadeStart = 0;
        }
        this.statsUpdated();
    }

    //* Denoiser Methods ----------------------------
    setupDenoiser() {
        // Initialization
        this.denoiser = new Denoiser("webgl", this.canvas);
        this.denoiser.outputMode = 'webgl';
        this.denoiser.inputMode = 'webgl';
        this.denoiser.debugging = true;
        this.denoiser.hdr = true;
        //this.denoiser.usePassThrough = true;

        // for debugging dump the denoiser to the rawOutput canvas
        //const rawOutputCanvas = document.getElementById("rawOutput") as HTMLCanvasElement;
        //this.denoiser.setCanvas(rawOutputCanvas);


        this.denoiser.onBackendReady(() => {
            // start the renderer and let other systems know
            this.ready = true;
            this.animate();
        });


        // on execution
        this.denoiser.onExecute(denoisedWebGLTexture => {

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
            this.dfadeStart = Date.now();
            this.denoising = false;
        }, 'webgl')
    }

    async runDenoiser() {
        if (this.denoising || this.denoiseBlocked) return;
        this.resetState();
        // block future denoising attempts
        this.denoising = true;
        this.denoiseBlocked = true;
        console.log('Running denoiser');

        // get the webgl texture out of this
        const colorTexture = this.getWebGLTexture(this.outputTextures.get('pathtraced')!, true);
        const albedoTexture = this.getWebGLTexture(this.outputTextures.get('albedo')!, true);
        const normalTexture = this.getWebGLTexture(this.outputTextures.get('normal')!, true);

        this.denoiser.setInputTexture('color', colorTexture, this.canvas.height, this.canvas.width, { colorspace: 'linear' });
        this.denoiser.setInputTexture('albedo', albedoTexture, this.canvas.height, this.canvas.width);
        this.denoiser.setInputTexture('normal', normalTexture, this.canvas.height, this.canvas.width);
        //todo: with input mode set I should be able to pass these directly
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
        this.canvas.width = width;
        this.canvas.height = height;
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
        new RGBELoader().load(envMaps[this.params.envmapName as keyof typeof envMaps], (texture) => {
            texture.mapping = THREE.EquirectangularReflectionMapping;
            this.scene.background = texture;
            this.scene.environment = texture;
            this.pathtracer.updateEnvironment();
        });
    }

    updateBaseOutput(base = 0) {
        console.log('setting output to', base);
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.base.value = base;
    }

    // update the uniforms of the output materials
    updateTextures() {
        for (const [name, texture] of this.outputTextures) {
            const material = this.fullscreenQuad!.material as THREE.ShaderMaterial;
            material.uniforms[name] = { value: texture };
        }
        // update the blends of textures

        let pfade = Math.min(1, Math.max(0, (Date.now() - this.pfadeStart) / this.params.pfade));
        let dfade = Math.min(1, Math.max(0, (Date.now() - this.dfadeStart) / this.params.dfade));
        if (this.stats.samples < this.params.minSamples) {

            pfade = 0;
        }

        if (this.dfadeStart === 0) {
            dfade = 0;
        }

        //console.log('pfade', pfade, 'dfade', dfade, 'dfadestart', this.dfadeStart);
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.pBlend.value = pfade;
        (this.fullscreenQuad!.material as THREE.ShaderMaterial).uniforms.dBlend.value = dfade;
    }




    // update the pathtracer scene when things in the scene chage
    sceneUpdated() {
        this.pathtracer.setScene(this.scene, this.camera);
    }

    statsUpdated() {
        this.statsListeners.forEach(listener => listener(this.stats));
    }

    onStats(callback: (arg0: { samples: number, isDenoised: boolean }) => void) {
        this.statsListeners.add(callback);
        return () => this.statsListeners.delete(callback);
    }


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
        if (this.models.has(modelName)) {
            return this.models.get(modelName)!;
        }

        const loader = new LDrawLoader();
        //await loader.preloadMaterials('https://raw.githubusercontent.com/gkjohnson/ldraw-parts-library/master/colors/ldcfgalt.ldr');
        loader.smoothNormals = true;
        let model = await loader.loadAsync(`./models/${modelName}.mpd`);
        model.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                //
            }
        });
        // merge the object
        model = LDrawUtils.mergeObject(model);
        model.scale.setScalar(0.01);

        // model needs to be flipped
        model.rotation.x = Math.PI;


        this.models.set(modelName, model);
        return model;
    }




    private animate() {

        this.statsWidget.begin();
        if (!this.paused) this.controls.update();
        this.draw();
        this.statsWidget.end();

        requestAnimationFrame(() => this.animate());
    }

    private draw() {
        if (this.paused) return;
        this.renderer.resetState();
        const originalEnvMap = this.scene.environment;
        const originalBackground = this.scene.background;

        // to setup the denoiser correct with three we need to run the texture at least once
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
        if (this.doAlbedoAndNormals) {
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
        if (this.pathtracerEnabled && this.pathtracer.samples < this.params.maxSamples) {
            this.renderPathtracer();
        }


        //* Final output
        this.updateTextures();
        this.renderer.render(this.flatScene, this.flatCamera);
        this.renderer.resetState();
        //* Denoiser Pass
        if (this.stats.samples === this.params.denoiseAt) {
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
            this.canvas.width,
            this.canvas.height,
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