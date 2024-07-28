import * as tf from '@tensorflow/tfjs';
import { formatBytes, isMobile } from './utils';

type TilerOptions = {
    tileSize?: number;
    overlap?: number;
    batchSize?: number;
    debugging?: boolean;
    srgb?: boolean;
};

export class GPUTensorTiler {
    private model: tf.LayersModel;
    private tileSize: [number, number];
    private overlap: number;
    private batchSize: number;
    //private blendingMask: tf.Tensor2D | null = null;
    private blendingMasks: Map<string, tf.Tensor2D> = new Map();
    private debugging: boolean;
    private aborting = false;
    public srgb: boolean;

    //* Debugging
    timers: { [key: string]: number } = {};
    stats: { [key: string]: number } = {};

    constructor(model: tf.LayersModel, options?: TilerOptions) {

        console.log('%c Denoiser: Using Tiling Output', 'background: green; color: white;');
        this.model = model;
        const tileSize = options?.tileSize || 256;
        this.tileSize = [tileSize, tileSize];
        this.overlap = options?.overlap || 32;
        this.batchSize = options?.batchSize || this.calculateBatchSize();;
        this.debugging = options?.debugging || false;
        this.srgb = options?.srgb || false;
    }

    private calculateBatchSize(): number {
        // Implement logic to determine batch size based on available GPU memory
        // This is a placeholder implementation
        if (isMobile()) return 1;
        return 4;
    }

    private getTileDims(index: number, total: number, size: number, stride: number): [number, number] {
        const start = index * stride;
        const end = Math.min(start + size, total);
        return [start, end - start];
    }

    // TODO move to central location
    private logMemoryUsage(stage: string, bgColor = 'orange') {
        if (this.debugging) {
            const memoryInfo = tf.memory() as any;
            console.log(`%cMemory usage at ${stage}:`, `background: ${bgColor}; color: white;`);
            console.table({
                numBytes: formatBytes(memoryInfo.numBytes),
                numBytesInGPU: formatBytes(memoryInfo.numBytesInGPU),
                numBytesInGPUAllocated: formatBytes(memoryInfo.unreliable ? 0 : memoryInfo.numBytesInGPUAllocated),
                numBytesInGPUFree: formatBytes(memoryInfo.unreliable ? 0 : memoryInfo.numBytesInGPUFree),
                numDataBuffers: memoryInfo.numDataBuffers,
                numTensors: memoryInfo.numTensors
            });
        }
    }

    //* Core method, takes input and processes it
    async processLargeTensor(rawInput: tf.Tensor4D, statusCallback?: (progress: number) => void): Promise<tf.Tensor4D> {
        this.logMemoryUsage('Start of processLargeTensor');
        this.startTimer('processLargeTensor');
        if (this.aborting) this.aborting = false;

        const [batchSize, height, width, channels] = rawInput.shape;
        const [tileHeight, tileWidth] = this.tileSize;
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        const totalTiles = tilesY * tilesX;
        const batches = Math.ceil(totalTiles / this.batchSize);

        const processedTiles: tf.Tensor4D[] = [];
        this.startTimer('allbatches');

        // If we are set as SRGB that means the input is in SRGB and we need to convert it to linear
        const input = this.srgb ? this.sRGBToLinear(rawInput) : rawInput;

        for (let batch = 0; batch < batches; batch++) {
            // If aborting, return the original input (keeps ts happy)
            if (this.aborting) {
                if (statusCallback) statusCallback(0);
                return rawInput.clone();
            }

            this.startTimer(`processLargeTensorBatch${batch + 1}`);
            const batchTiles = tf.tidy(() => {
                const start = batch * this.batchSize;
                const end = Math.min(start + this.batchSize, totalTiles);

                return Array.from({ length: end - start }, (_, i) => {

                    const index = start + i;
                    const y = Math.floor(index / tilesX);
                    const x = index % tilesX;

                    const [startY, curHeight] = this.getTileDims(y, height, tileHeight, strideY);
                    const [startX, curWidth] = this.getTileDims(x, width, tileWidth, strideX);

                    const tile = input.slice([0, startY, startX, 0], [1, curHeight, curWidth, channels]);
                    const paddedTile = tile.pad([
                        [0, 0],
                        [0, tileHeight - curHeight],
                        [0, tileWidth - curWidth],
                        [0, 0]
                    ]) as tf.Tensor4D;

                    return paddedTile;

                    // Add colored border for diagnostic purposes
                    //return this.addColoredBorder(paddedTile, 2, [1, 0, 0]); // 2-pixel wide red border

                });
            });

            this.logMemoryUsage(`After slicing batch ${batch + 1}/${batches}`);

            this.startTimer(`predictBatch${batch + 1}`);
            const batchInput = tf.concat(batchTiles, 0);
            const batchOutput = this.model.predict(batchInput) as tf.Tensor4D;
            this.stopTimer(`predictBatch${batch + 1}`);
            this.logMemoryUsage(`After prediction of batch ${batch + 1}/${batches}`, '#1098F7');

            // Split the batch output back into individual tiles
            for (let i = 0; i < batchOutput.shape[0]; i++) {
                const processedTile = batchOutput.slice([i, 0, 0, 0], [1, tileHeight, tileWidth, 3]);
                processedTiles.push(tf.keep(processedTile));
            }

            batchOutput.dispose();
            batchInput.dispose();
            batchTiles.forEach(tile => tile.dispose());

            this.logMemoryUsage(`After processing batch ${batch + 1}/${batches}`);
            this.stopTimer(`processLargeTensorBatch${batch + 1}`);

            // Call the status callback if provided
            if (statusCallback) {
                const progress = (batch + 1) / batches;
                statusCallback(progress);
            }

            // Allow the UI to update and potentially free up GPU memory
            await tf.nextFrame();
        }

        this.stopTimer('allbatches');
        this.stopTimer('processLargeTensor');

        let result = this.reassembleTilesWithBlending(processedTiles, [batchSize, height, width, 3]);
        if (this.srgb) result = this.linearToSRGB(result);

        processedTiles.forEach(tile => tf.dispose(tile));
        this.logMemoryUsage('End of processLargeTensor');
        this.logResults();

        return result;
    }

    private reassembleTilesWithBlending(tiles: tf.Tensor4D[], outputShape: [number, number, number, number]): tf.Tensor4D {
        this.logMemoryUsage('Start of reassembleTilesWithBlending');
        this.startTimer('reassembleTilesWithBlending');
        const [batchSize, height, width, channels] = outputShape;
        const [tileHeight, tileWidth] = this.tileSize;
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        let reassembled = tf.zeros(outputShape);
        let weights = tf.zeros([batchSize, height, width, 1]);

        let tileIndex = 0;
        for (let y = 0; y < tilesY; y++) {
            for (let x = 0; x < tilesX; x++) {
                this.startTimer(`reassembleTile${tileIndex + 1}`);
                // Use the cached position-aware blending mask
                const [startY, curHeight] = this.getTileDims(y, height, tileHeight, strideY);
                const [startX, curWidth] = this.getTileDims(x, width, tileWidth, strideX);

                const tileMask = this.getBlendingMask(curHeight, curWidth, y, x, tilesY, tilesX).expandDims(0).expandDims(-1);


                const [paddedWeightedTile, paddedTileWeights] = tf.tidy(() => {

                    const tile = tiles[tileIndex];
                    const slicedTile = tile.slice([0, 0, 0, 0], [1, curHeight, curWidth, channels]);


                    this.logMemoryUsage(`After slicing & masking tile ${tileIndex + 1}/${tiles.length}`);
                    const weightedTile = slicedTile.mul(tileMask);
                    const paddedWeightedTile = tf.pad(weightedTile, [
                        [0, 0],
                        [startY, height - startY - curHeight],
                        [startX, width - startX - curWidth],
                        [0, 0]
                    ]);
                    this.logMemoryUsage(`After weighting & padding tile ${tileIndex + 1}/${tiles.length}`);

                    const paddedTileWeights = tf.pad(tileMask, [
                        [0, 0],
                        [startY, height - startY - curHeight],
                        [startX, width - startX - curWidth],
                        [0, 0]
                    ]);
                    this.logMemoryUsage(`After padding weights of tile ${tileIndex + 1}/${tiles.length}`);
                    return [paddedWeightedTile, paddedTileWeights];
                });

                const newTile = reassembled.add(paddedWeightedTile);
                reassembled.dispose();
                reassembled = newTile;
                const newWeights = weights.add(paddedTileWeights);
                weights.dispose();
                weights = newWeights;

                paddedWeightedTile.dispose();
                paddedTileWeights.dispose();
                this.stopTimer(`reassembleTile${tileIndex + 1}`);
                this.logMemoryUsage(`After adding tile ${tileIndex + 1}/${tiles.length}`, '#1098F7');
                tileIndex++;
            }
        }

        const result = tf.tidy(() => {
            return reassembled.div(weights.add(1e-8));
        });

        reassembled.dispose();
        weights.dispose();
        this.stopTimer('reassembleTilesWithBlending');
        this.logMemoryUsage('End of reassembleTilesWithBlending');

        return result as tf.Tensor4D;
    }

    private getBlendingMask(height: number, width: number, tileY: number, tileX: number, tilesY: number, tilesX: number): tf.Tensor2D {
        const key = `${height},${width},${tileY},${tileX},${tilesY},${tilesX}`;

        if (!this.blendingMasks.has(key)) {
            const mask = this.createImprovedBlendingMask(height, width, tileY, tileX, tilesY, tilesX);
            this.blendingMasks.set(key, mask);
        }

        return this.blendingMasks.get(key)!;
    }

    // Blending mask is not position aware
    private createImprovedBlendingMask(height: number, width: number, tileY: number, tileX: number, tilesY: number, tilesX: number): tf.Tensor2D {
        return tf.tidy(() => {
            const mask = tf.buffer([height, width]);

            for (let i = 0; i < height; i++) {
                for (let j = 0; j < width; j++) {
                    let yWeight = 1;
                    let xWeight = 1;

                    if (tileY > 0) {
                        yWeight = Math.min(yWeight, this.sigmoidBlend(i / this.overlap));
                    }
                    if (tileY < tilesY - 1) {
                        yWeight = Math.min(yWeight, this.sigmoidBlend((height - 1 - i) / this.overlap));
                    }
                    if (tileX > 0) {
                        xWeight = Math.min(xWeight, this.sigmoidBlend(j / this.overlap));
                    }
                    if (tileX < tilesX - 1) {
                        xWeight = Math.min(xWeight, this.sigmoidBlend((width - 1 - j) / this.overlap));
                    }

                    const smoothWeight = Math.min(yWeight, xWeight);
                    mask.set(smoothWeight, i, j);
                }
            }

            return mask.toTensor();
        }) as tf.Tensor2D;
    }

    private sigmoidBlend(x: number): number {
        return 1 / (1 + Math.exp(-12 * (x - 0.5)));
    }

    startTimer(name: string) {
        if (!this.debugging) return;
        this.timers[`${name}In`] = performance.now();
    }
    stopTimer(name: string) {
        if (!this.debugging) return;
        this.timers[`${name}Out`] = performance.now();
        this.stats[name] = this.timers[`${name}Out`] - this.timers[`${name}In`];
    }

    logResults() {
        if (!this.debugging) return;
        console.log('Tiler Results :');
        console.table(this.stats);
    }

    dispose() {
        for (const mask of this.blendingMasks.values()) {
            mask.dispose();
        }
        this.blendingMasks.clear();
    }

    abort() {
        if (this.debugging) console.log('%cTiler aborted', 'color: white; background-color: red');
        this.aborting = true;

    }

    //* Colorspace handing ----------------------
    // TODO move to central location
    sRGBToLinear(inputTensor: tf.Tensor): tf.Tensor4D {

        const toReturn = tf.tidy(() => {
            return tf.where(tf.greater(inputTensor, 0.04045),
                tf.pow(tf.div(tf.add(inputTensor, 0.055), 1.055), 2.4),
                tf.div(inputTensor, 12.92)
            );
        }) as tf.Tensor4D;
        inputTensor.dispose();
        return toReturn;
    }
    // TODO move to central location
    linearToSRGB(inputTensor: tf.Tensor): tf.Tensor4D {
        const toReturn = tf.tidy(() => {
            return tf.where(tf.greater(inputTensor, 0.0031308),
                tf.sub(tf.mul(tf.pow(inputTensor, 1 / 2.4), 1.055), 0.055),
                tf.mul(inputTensor, 12.92)
            );
        }) as tf.Tensor4D;
        inputTensor.dispose();
        return toReturn;
    }

    //* Debug Utils ==============================
    addColoredBorder(tile: tf.Tensor4D, borderWidth: number, color: [number, number, number]): tf.Tensor4D {
        return tf.tidy(() => {
            const [_, height, width, channels] = tile.shape;
            const border = tf.ones([1, height, width, channels]);
            const coloredBorder = border.mul(tf.tensor(color));

            const withBorder = tf.where(
                tf.logicalOr(
                    tf.logicalOr(
                        tf.less(tf.range(0, height).reshape([1, height, 1, 1]), borderWidth),
                        tf.greater(tf.range(0, height).reshape([1, height, 1, 1]), height - borderWidth - 1)
                    ),
                    tf.logicalOr(
                        tf.less(tf.range(0, width).reshape([1, 1, width, 1]), borderWidth),
                        tf.greater(tf.range(0, width).reshape([1, 1, width, 1]), width - borderWidth - 1)
                    )
                ),
                coloredBorder,
                tile
            );

            return withBorder;
        }) as tf.Tensor4D;
    }

    async visualizeBlendRegions(output: tf.Tensor4D): Promise<tf.Tensor4D> {
        return tf.tidy(() => {
            const [_, height, width] = output.shape;
            const blendMask = tf.buffer([height, width, 1]);

            const tilesY = Math.ceil(height / (this.tileSize[0] - this.overlap));
            const tilesX = Math.ceil(width / (this.tileSize[1] - this.overlap));

            for (let y = 0; y < tilesY; y++) {
                for (let x = 0; x < tilesX; x++) {
                    const [startY, curHeight] = this.getTileDims(y, height, this.tileSize[0], this.tileSize[0] - this.overlap);
                    const [startX, curWidth] = this.getTileDims(x, width, this.tileSize[1], this.tileSize[1] - this.overlap);

                    const mask = this.getBlendingMask(curHeight, curWidth, y, x, tilesY, tilesX);
                    const maskData = mask.arraySync();

                    for (let i = 0; i < curHeight; i++) {
                        for (let j = 0; j < curWidth; j++) {
                            if (maskData[i][j] < 1) {
                                blendMask.set(1, startY + i, startX + j, 0);
                            }
                        }
                    }
                }
            }

            const blendRegions = blendMask.toTensor().expandDims(0).tile([1, 1, 1, 3]);
            return output.mul(tf.scalar(0.7)).add(blendRegions.mul(tf.tensor4d([0.3, 0, 0], [1, 1, 1, 3])));
        }) as tf.Tensor4D;
    }

    // creates Data URLS of tiles so we can compare them
    /*
    // In the processLargeTensor method, add these lines:
// Before processing:
await this.visualizeTiles(batchTiles, 'before');

// After processing:
await this.visualizeTiles(processedTiles, 'after');
*/

    async visualizeTiles(tiles: tf.Tensor4D[], stage: 'before' | 'after'): Promise<void> {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const tileSize = this.tileSize[0];
        const tilesPerRow = Math.ceil(Math.sqrt(tiles.length));
        canvas.width = tileSize * tilesPerRow;
        canvas.height = tileSize * Math.ceil(tiles.length / tilesPerRow);

        for (let i = 0; i < tiles.length; i++) {
            const x = (i % tilesPerRow) * tileSize;
            const y = Math.floor(i / tilesPerRow) * tileSize;

            const tileData = await tiles[i].squeeze().mul(255).toInt().array() as Array<unknown>;
            const imageData = new ImageData(tileSize, tileSize);

            for (let j = 0; j < tileData.length; j++) {
                //@ts-ignore
                for (let k = 0; k < tileData[j].length; k++) {
                    const idx = (j * tileSize + k) * 4;
                    //@ts-ignore
                    imageData.data[idx] = tileData[j][k][0];
                    //@ts-ignore
                    imageData.data[idx + 1] = tileData[j][k][1];
                    //@ts-ignore
                    imageData.data[idx + 2] = tileData[j][k][2];
                    imageData.data[idx + 3] = 255;
                }
            }

            ctx.putImageData(imageData, x, y);
        }

        console.log(`Tiles ${stage} processing:`);
        console.log(canvas.toDataURL());
    }


    static generateSampleInput(height: number, width: number): tf.Tensor4D {
        return tf.tidy(() => {
            const channels = 9;

            // Generate random data for each channel
            const channelData: tf.Tensor3D[] = [];
            for (let i = 0; i < channels; i++) {
                channelData.push(tf.randomUniform([height, width, 1]));
            }

            // Concatenate all channels
            return tf.concat(channelData, -1).expandDims(0) as tf.Tensor4D;
        });
    }
}

