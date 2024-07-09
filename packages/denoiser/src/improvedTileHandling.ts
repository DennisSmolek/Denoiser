import * as tf from '@tensorflow/tfjs';

export class GPUTensorTiler {
    private model: tf.LayersModel;
    private tileSize: [number, number];
    private overlap: number;
    private inputShape: [number, number, number, number];
    private precomputedMasks: tf.Tensor4D[];

    constructor(model: tf.LayersModel, tileSize: [number, number], inputShape: [number, number, number, number], overlap = 0) {
        this.model = model;
        this.tileSize = tileSize;
        this.overlap = overlap;
        this.inputShape = inputShape;
        this.precomputedMasks = this.computeMasks();
    }

    private computeMasks(): tf.Tensor4D[] {
        const [batchSize, height, width, channels] = this.inputShape;
        const [tileHeight, tileWidth] = this.tileSize;
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        const masks: tf.Tensor4D[] = [];

        for (let y = 0; y < tilesY; y++) {
            for (let x = 0; x < tilesX; x++) {
                const [startY, curHeight] = this.getTileDims(y, height, tileHeight, strideY);
                const [startX, curWidth] = this.getTileDims(x, width, tileWidth, strideX);

                const maskY = tf.range(0, height).less(startY + curHeight).greater(startY - 1);
                const maskX = tf.range(0, width).less(startX + curWidth).greater(startX - 1);
                const mask = maskY.expandDims(1).matMul(maskX.expandDims(0))
                    .expandDims(0).expandDims(-1)
                    .tile([batchSize, 1, 1, channels]) as tf.Tensor4D;

                masks.push(mask);
            }
        }

        return masks;
    }

    private getTileDims(index: number, total: number, size: number, stride: number): [number, number] {
        const start = index * stride;
        const end = Math.min(start + size, total);
        return [start, end - start];
    }

    async processLargeTensor(input: tf.Tensor4D): Promise<tf.Tensor4D> {
        tf.util.assert(
            tf.util.arraysEqual(input.shape, this.inputShape),
            () => `Input shape ${input.shape} does not match expected shape ${this.inputShape}`
        );

        const [, height, width, channels] = this.inputShape;
        const [tileHeight, tileWidth] = this.tileSize;
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        const processedTiles = tf.tidy(() => {
            const tilesList: tf.Tensor4D[] = [];
            for (let y = 0; y < tilesY; y++) {
                for (let x = 0; x < tilesX; x++) {
                    const [startY, curHeight] = this.getTileDims(y, height, tileHeight, strideY);
                    const [startX, curWidth] = this.getTileDims(x, width, tileWidth, strideX);

                    const tile = tf.slice(input, [0, startY, startX, 0], [1, curHeight, curWidth, channels]);

                    const paddedTile = tile.pad([
                        [0, 0],
                        [0, tileHeight - curHeight],
                        [0, tileWidth - curWidth],
                        [0, 0]
                    ]);

                    tilesList.push(paddedTile as tf.Tensor4D);
                }
            }

            const tiles = tf.concat(tilesList, 0);
            return this.model.predict(tiles) as tf.Tensor4D;
        });

        return this.reassembleTiles(processedTiles);
    }

    private reassembleTiles(tiles: tf.Tensor4D): tf.Tensor4D {
        const [batchSize, height, width, channels] = this.inputShape;
        const [tileHeight, tileWidth] = this.tileSize;
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        return tf.tidy(() => {
            let reassembled = tf.zeros(this.inputShape);

            let maskIndex = 0;
            for (let y = 0; y < tilesY; y++) {
                for (let x = 0; x < tilesX; x++) {
                    const tileIndex = y * tilesX + x;
                    const [startY, curHeight] = this.getTileDims(y, height, tileHeight, strideY);
                    const [startX, curWidth] = this.getTileDims(x, width, tileWidth, strideX);

                    const tile = tf.slice(tiles, [tileIndex, 0, 0, 0], [1, curHeight, curWidth, channels]);

                    reassembled = tf.where(
                        this.precomputedMasks[maskIndex],
                        tf.pad(tile, [[0, 0], [startY, height - startY - curHeight], [startX, width - startX - curWidth], [0, 0]]),
                        reassembled
                    );

                    maskIndex++;
                }
            }

            return reassembled;
        });
    }

    dispose() {
        this.precomputedMasks.forEach(mask => mask.dispose());
    }
}

// Example usage:
export async function testTiling() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [null, null, 3],
        kernelSize: 1,
        filters: 3,
        activation: 'linear'
    }));

    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    const inputShape: [number, number, number, number] = [1, 1000, 1000, 3];
    const tiler = new GPUTensorTiler(model, [256, 256], inputShape, 16);

    const input = tf.ones(inputShape);
    const output = await tiler.processLargeTensor(input);

    console.log('Input shape:', input.shape);
    console.log('Output shape:', output.shape);

    // Clean up
    input.dispose();
    output.dispose();
    model.dispose();
    tiler.dispose();
}