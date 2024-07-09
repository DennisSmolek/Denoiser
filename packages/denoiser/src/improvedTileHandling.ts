import * as tf from '@tensorflow/tfjs';

export class GPUTensorTiler {
    private model: tf.LayersModel;
    private tileSize: [number, number];
    private overlap: number;
    private batchSize: number;

    constructor(model: tf.LayersModel, tileSize: number, overlap = 16, batchSize = 4) {
        this.model = model;
        this.tileSize = [tileSize, tileSize];
        this.overlap = overlap;
        this.batchSize = batchSize;
    }

    private getTileDims(index: number, total: number, size: number, stride: number): [number, number] {
        const start = index * stride;
        const end = Math.min(start + size, total);
        return [start, end - start];
    }

    async processLargeTensor(input: tf.Tensor4D): Promise<tf.Tensor4D> {
        const [batchSize, height, width, channels] = input.shape;
        const [tileHeight, tileWidth] = this.tileSize;
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        const tilesList: tf.Tensor4D[] = [];
        for (let y = 0; y < tilesY; y++) {
            for (let x = 0; x < tilesX; x++) {
                const [startY, curHeight] = this.getTileDims(y, height, tileHeight, strideY);
                const [startX, curWidth] = this.getTileDims(x, width, tileWidth, strideX);

                const tile = input.slice([0, startY, startX, 0], [1, curHeight, curWidth, channels]);

                const paddedTile = tile.pad([
                    [0, 0],
                    [0, tileHeight - curHeight],
                    [0, tileWidth - curWidth],
                    [0, 0]
                ]);

                tilesList.push(paddedTile);
            }
        }

        const processedTiles: tf.Tensor4D[] = [];
        for (let i = 0; i < tilesList.length; i += this.batchSize) {
            const batch = tilesList.slice(i, i + this.batchSize);
            const batchTensor = tf.concat(batch, 0);
            const processedBatch = this.model.predict(batchTensor) as tf.Tensor4D;

            // Split the processed batch back into individual tiles
            for (let j = 0; j < processedBatch.shape[0]; j++) {
                processedTiles.push(processedBatch.slice([j, 0, 0, 0], [1, -1, -1, -1]));
            }

            tf.dispose([batchTensor, processedBatch]);
        }

        return this.reassembleTilesWithBlending(processedTiles, [batchSize, height, width, 3]);
    }

    private createBlendingMask(): tf.Tensor2D {
        return tf.tidy(() => {
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
        });
    }

    private reassembleTilesWithBlending(tiles: tf.Tensor4D[], outputShape: [number, number, number, number]): tf.Tensor4D {
        const [batchSize, height, width, channels] = outputShape;
        const [tileHeight, tileWidth] = this.tileSize;
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        const blendingMask = this.createBlendingMask();

        return tf.tidy(() => {
            let reassembled = tf.zeros(outputShape);
            let weights = tf.zeros([batchSize, height, width, 1]);

            let tileIndex = 0;
            for (let y = 0; y < tilesY; y++) {
                for (let x = 0; x < tilesX; x++) {
                    const [startY, curHeight] = this.getTileDims(y, height, tileHeight, strideY);
                    const [startX, curWidth] = this.getTileDims(x, width, tileWidth, strideX);

                    const tile = tiles[tileIndex].slice([0, 0, 0, 0], [1, curHeight, curWidth, channels]);
                    const tileMask = blendingMask.slice([0, 0], [curHeight, curWidth]).expandDims(0).expandDims(-1);

                    const weightedTile = tile.mul(tileMask);
                    const tileWeights = tileMask;

                    reassembled = reassembled.add(
                        tf.pad(weightedTile, [[0, 0], [startY, height - startY - curHeight], [startX, width - startX - curWidth], [0, 0]])
                    );
                    weights = weights.add(
                        tf.pad(tileWeights, [[0, 0], [startY, height - startY - curHeight], [startX, width - startX - curWidth], [0, 0]])
                    );

                    tileIndex++;
                }
            }

            // Normalize by the accumulated weights
            return reassembled.div(weights.add(1e-8));
        });
    }

    dispose() {
        // No additional resources to dispose in this version
    }
}

// Debug function to compare two tiles
function compareTiles(tile1: tf.Tensor4D, tile2: tf.Tensor4D): boolean {
    const diff = tile1.sub(tile2).abs().max();
    return diff.dataSync()[0] < 1e-6;
}

// Debug function to create colored tiles
function createColoredTiles(shape: [number, number, number, number], tileSize: [number, number]): tf.Tensor4D {
    const [batchSize, height, width, channels] = shape;
    const [tileHeight, tileWidth] = tileSize;
    const tilesY = Math.ceil(height / tileHeight);
    const tilesX = Math.ceil(width / tileWidth);

    return tf.tidy(() => {
        const coloredTiles = tf.buffer(shape);

        for (let y = 0; y < tilesY; y++) {
            for (let x = 0; x < tilesX; x++) {
                const startY = y * tileHeight;
                const startX = x * tileWidth;
                const endY = Math.min(startY + tileHeight, height);
                const endX = Math.min(startX + tileWidth, width);

                const r = (y / tilesY);
                const g = (x / tilesX);
                const b = ((y + x) / (tilesY + tilesX));

                for (let i = startY; i < endY; i++) {
                    for (let j = startX; j < endX; j++) {
                        coloredTiles.set(r, 0, i, j, 0);
                        coloredTiles.set(g, 0, i, j, 1);
                        coloredTiles.set(b, 0, i, j, 2);
                    }
                }
            }
        }

        return coloredTiles.toTensor();
    });
}

// Test function
export async function testTiling() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [null, null, 3],
        kernelSize: 3,
        filters: 3,
        padding: 'same',
        activation: 'relu'
    }));

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    const tiler = new GPUTensorTiler(model, 256, 16, 4);

    // Create a colored input tensor for debugging
    const inputShape: [number, number, number, number] = [1, 1000, 1000, 3];
    const input = createColoredTiles(inputShape, [256, 256]);

    console.log('Input shape:', input.shape);

    // Process the colored input
    const output = await tiler.processLargeTensor(input);

    console.log('Output shape:', output.shape);

    // Compare input and output
    const inputData = input.dataSync();
    const outputData = output.dataSync();
    console.log('Input and output are identical:', tf.util.arraysEqual(inputData, outputData));

    // Compare tiles
    const inputTile1 = input.slice([0, 0, 0, 0], [1, 256, 256, 3]);
    const inputTile2 = input.slice([0, 256, 256, 0], [1, 256, 256, 3]);
    const outputTile1 = output.slice([0, 0, 0, 0], [1, 256, 256, 3]);
    const outputTile2 = output.slice([0, 256, 256, 0], [1, 256, 256, 3]);

    console.log('Input tiles are different:', !compareTiles(inputTile1, inputTile2));
    console.log('Output tiles are different:', !compareTiles(outputTile1, outputTile2));

    // Clean up
    tf.dispose([input, output, inputTile1, inputTile2, outputTile1, outputTile2]);
    model.dispose();
    tiler.dispose();
}

