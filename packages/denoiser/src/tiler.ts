import * as tf from '@tensorflow/tfjs';
import { formatBytes, isMobile } from './utils';

export class GPUTensorTiler {
    private model: tf.LayersModel;
    private tileSize: [number, number];
    private overlap: number;
    private batchSize: number;
    private blendingMask: tf.Tensor2D | null = null;
    private debugMode: boolean;

    //* Debugging
    timers: { [key: string]: number } = {};
    stats: { [key: string]: number } = {};

    constructor(model: tf.LayersModel, tileSize: number, overlap = 16, batchSize?: number, debugMode = false) {
        console.log('%c Denoiser: Using Tiling Output', 'background: green; color: white;');
        this.model = model;
        this.tileSize = [tileSize, tileSize];
        this.overlap = overlap;
        this.batchSize = batchSize || this.calculateBatchSize();
        this.debugMode = debugMode;
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
        if (this.debugMode) {
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
    async processLargeTensor(input: tf.Tensor4D): Promise<tf.Tensor4D> {
        this.logMemoryUsage('Start of processLargeTensor');
        this.startTimer('processLargeTensor');

        const [batchSize, height, width, channels] = input.shape;
        const [tileHeight, tileWidth] = this.tileSize;
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        const totalTiles = tilesY * tilesX;
        const batches = Math.ceil(totalTiles / this.batchSize);

        const processedTiles: tf.Tensor4D[] = [];
        this.startTimer('allbatches')
        for (let batch = 0; batch < batches; batch++) {
            this.startTimer(`processLargeTensorBatch${batch + 1}`);
            tf.tidy(() => {
                const start = batch * this.batchSize;
                const end = Math.min(start + this.batchSize, totalTiles);

                const batchTiles: tf.Tensor4D[] = [];
                for (let i = start; i < end; i++) {
                    const y = Math.floor(i / tilesX);
                    const x = i % tilesX;

                    const [startY, curHeight] = this.getTileDims(y, height, tileHeight, strideY);
                    const [startX, curWidth] = this.getTileDims(x, width, tileWidth, strideX);

                    const tile = input.slice([0, startY, startX, 0], [1, curHeight, curWidth, channels]);
                    const paddedTile = tile.pad([
                        [0, 0],
                        [0, tileHeight - curHeight],
                        [0, tileWidth - curWidth],
                        [0, 0]
                    ]);

                    batchTiles.push(paddedTile as tf.Tensor4D);
                }
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
                this.logMemoryUsage(`After splitting batch ${batch + 1}/${batches}`);
            });

            this.logMemoryUsage(`After processing batch ${batch + 1}/${batches}`);
            this.stopTimer(`processLargeTensorBatch${batch + 1}`);
            // Explicitly run garbage collection after each batch
            //tf.disposeVariables();
            await tf.nextFrame();  // Allow GPU to potentially free up memory

        }
        this.stopTimer('allbatches');
        this.stopTimer('processLargeTensor');
        const result = this.reassembleTilesWithBlending(processedTiles, [batchSize, height, width, 3]);
        processedTiles.forEach(tile => tf.dispose(tile));
        this.logMemoryUsage('End of processLargeTensor');
        this.logResults();
        return result;
    }

    private createBlendingMask(): tf.Tensor2D {
        if (this.blendingMask === null) {
            this.startTimer('createBlendingMask');
            this.blendingMask = tf.tidy(() => {
                const [tileHeight, tileWidth] = this.tileSize;
                const mask = tf.buffer([tileHeight, tileWidth]);

                for (let i = 0; i < tileHeight; i++) {
                    for (let j = 0; j < tileWidth; j++) {
                        const yWeight = Math.min(i, tileHeight - 1 - i) / this.overlap;
                        const xWeight = Math.min(j, tileWidth - 1 - j) / this.overlap;
                        mask.set(Math.min(yWeight, xWeight, 1), i, j);
                    }
                }

                return mask.toTensor();
            }) as tf.Tensor2D;
            this.stopTimer('createBlendingMask');
        }
        return this.blendingMask!;
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

        const blendingMask = this.createBlendingMask();


        let reassembled = tf.zeros(outputShape);
        let weights = tf.zeros([batchSize, height, width, 1]);

        let tileIndex = 0;
        for (let y = 0; y < tilesY; y++) {
            for (let x = 0; x < tilesX; x++) {
                this.startTimer(`reassembleTile${tileIndex + 1}`);
                const [paddedWeightedTile, paddedTileWeights] = tf.tidy(() => {
                    const [startY, curHeight] = this.getTileDims(y, height, tileHeight, strideY);
                    const [startX, curWidth] = this.getTileDims(x, width, tileWidth, strideX);

                    const tile = tiles[tileIndex];
                    const slicedTile = tile.slice([0, 0, 0, 0], [1, curHeight, curWidth, channels]);
                    const tileMask = blendingMask.slice([0, 0], [curHeight, curWidth]).expandDims(0).expandDims(-1);
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
                // because of the functional nature of tensorflow this is leaving tensors hanging
                // so we have to create, destroy the old one and then set the new one
                const newTile = reassembled.add(paddedWeightedTile);
                reassembled.dispose();
                reassembled = newTile;
                const newWeights = weights.add(paddedTileWeights);
                weights.dispose();
                weights = newWeights;
                // now they are added and the sources can be destroyed
                paddedWeightedTile.dispose();
                paddedTileWeights.dispose();
                this.stopTimer(`reassembleTile${tileIndex + 1}`);
                this.logMemoryUsage(`After adding tile ${tileIndex + 1}/${tiles.length}`, '#1098F7');
                tileIndex++;
            }
        }

        const toReturn = tf.tidy(() => reassembled.div(weights.add(1e-8)));
        // destroy reassambled and weights
        reassembled.dispose();
        weights.dispose();
        this.stopTimer('reassembleTilesWithBlending');
        this.logMemoryUsage('End of reassembleTilesWithBlending');

        return toReturn as tf.Tensor4D;
    }

    startTimer(name: string) {
        if (!this.debugMode) return;
        this.timers[`${name}In`] = performance.now();
    }
    stopTimer(name: string) {
        if (!this.debugMode) return;
        this.timers[`${name}Out`] = performance.now();
        this.stats[name] = this.timers[`${name}Out`] - this.timers[`${name}In`];
    }
    logResults() {
        if (!this.debugMode) return;
        console.log('Tiler Results :');
        console.table(this.stats);
    }
    dispose() {
        if (this.blendingMask) {
            this.blendingMask.dispose();
            this.blendingMask = null;
        }
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