import * as tf from '@tensorflow/tfjs';

class GPUTensorTiler {
    private model: tf.LayersModel;
    private tileSize: [number, number];
    private overlap: number;

    constructor(model: tf.LayersModel, tileSize: [number, number], overlap = 0) {
        this.model = model;
        this.tileSize = tileSize;
        this.overlap = overlap;
    }

    async processLargeTensor(input: tf.Tensor4D): Promise<tf.Tensor4D> {
        const [batchSize, height, width, channels] = input.shape;
        const [tileHeight, tileWidth] = this.tileSize;

        // Calculate the effective stride (tile size minus overlap)
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;

        // Calculate the number of tiles, including partial tiles
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        // Function to get tile dimensions, accounting for edge cases
        const getTileDims = (index: number, total: number, size: number, stride: number) => {
            const start = index * stride;
            const end = Math.min(start + size, total);
            return [start, end - start];
        };

        // Extract and process tiles
        const processedTiles = tf.tidy(() => {
            const tilesList: tf.Tensor4D[] = [];
            for (let y = 0; y < tilesY; y++) {
                for (let x = 0; x < tilesX; x++) {
                    const [startY, curHeight] = getTileDims(y, height, tileHeight, strideY);
                    const [startX, curWidth] = getTileDims(x, width, tileWidth, strideX);

                    const tile = tf.slice(input, [0, startY, startX, 0], [1, curHeight, curWidth, channels]);

                    // Pad the tile if it's smaller than the expected size
                    const paddedTile = tile.pad([
                        [0, 0],
                        [0, tileHeight - curHeight],
                        [0, tileWidth - curWidth],
                        [0, 0]
                    ]);

                    tilesList.push(paddedTile as tf.Tensor4D);
                }
            }

            // Process all tiles in a single batch
            const tiles = tf.concat(tilesList, 0);
            return this.model.predict(tiles) as tf.Tensor4D;
        });

        // Reassemble the processed tiles
        return this.reassembleTiles(processedTiles, [batchSize, height, width, channels]);
    }

    private reassembleTiles(tiles: tf.Tensor4D, originalShape: [number, number, number, number]): tf.Tensor4D {
        const [batchSize, height, width, channels] = originalShape;
        const [tileHeight, tileWidth] = this.tileSize;

        // Calculate the effective stride (tile size minus overlap)
        const strideY = tileHeight - this.overlap;
        const strideX = tileWidth - this.overlap;

        // Calculate the number of tiles, including partial tiles
        const tilesY = Math.ceil(height / strideY);
        const tilesX = Math.ceil(width / strideX);

        return tf.tidy(() => {
            // Create a tensor for the reassembled image
            let reassembled = tf.zeros([batchSize, height, width, channels]);

            // Function to get tile dimensions, accounting for edge cases
            const getTileDims = (index: number, total: number, size: number, stride: number) => {
                const start = index * stride;
                const end = Math.min(start + size, total);
                return [start, end - start];
            };

            for (let y = 0; y < tilesY; y++) {
                for (let x = 0; x < tilesX; x++) {
                    const tileIndex = y * tilesX + x;
                    const [startY, curHeight] = getTileDims(y, height, tileHeight, strideY);
                    const [startX, curWidth] = getTileDims(x, width, tileWidth, strideX);

                    const tile = tf.slice(tiles, [tileIndex, 0, 0, 0], [1, curHeight, curWidth, channels]);

                    // Create a boolean mask for the current tile position
                    const mask = tf.buffer([batchSize, height, width, channels], 'bool');
                    for (let i = 0; i < curHeight; i++) {
                        for (let j = 0; j < curWidth; j++) {
                            mask.set(true, 0, startY + i, startX + j, 0);
                        }
                    }

                    // Use tf.where with the boolean mask to update the reassembled tensor
                    reassembled = tf.where(
                        mask.toTensor(),
                        tf.pad(tile, [[0, 0], [startY, height - startY - curHeight], [startX, width - startX - curWidth], [0, 0]]),
                        reassembled
                    );
                }
            }

            return reassembled;
        }) as tf.Tensor4D;
    }
}

// Example usage:
export async function testTiling() {
    // Create a simple passthrough model for demonstration
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [null, null, 3],
        kernelSize: 1,
        filters: 3,
        activation: 'linear'
    }));

    // Compile the model
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    // Create a large input tensor (1x1000x1000x3)
    const input = tf.ones([1, 1000, 1000, 3]) as tf.Tensor4D;

    // Create the GPUTensorTiler instance
    const tiler = new GPUTensorTiler(model, [256, 256], 16);

    // Process the large tensor
    const output = await tiler.processLargeTensor(input);

    console.log('Input shape:', input.shape);
    console.log('Output shape:', output.shape);

    // Clean up
    input.dispose();
    output.dispose();
    model.dispose();
}

