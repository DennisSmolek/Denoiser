import * as tf from '@tensorflow/tfjs-node';

import * as config from './config';
import * as util from './util';
import * as dataset from './dataset';
import * as model from './model';
import * as color from './color';
import * as result from './result';

// Inference class
class Infer {
    private result_cfg: config.ResultConfig;
    private device: tf.Device;
    private features: string[];
    private main_feature: string;
    private aux_features: string[];
    private all_channels: number[];
    private num_main_channels: number;
    private is_aux: boolean;
    private model: model.Model;
    private epoch: number;
    private transfer: color.TransferFunction | null;
    private aux_infers: { [key: string]: Infer };

    constructor(cfg: config.Config, device: tf.Device, result?: string, is_aux = false) {
        // Load the result config
        const result_dir = util.getResultDir(cfg, result);
        if (!util.isDirectory(result_dir)) {
            throw new Error('result does not exist');
        }
        this.result_cfg = util.loadConfig(result_dir);
        this.device = device;
        this.features = this.result_cfg.features;
        this.main_feature = util.getMainFeature(this.features);
        this.aux_features = util.getAuxFeatures(this.features);
        this.all_channels = util.getDatasetChannels(this.features);
        this.num_main_channels = util.getDatasetChannels(this.main_feature).length;
        this.is_aux = is_aux;

        // Initialize the model
        this.model = model.getModel(this.result_cfg);
        this.model.to(device);

        // Load the checkpoint
        const checkpoint = util.loadCheckpoint(result_dir, device, cfg.num_epochs, this.model);
        this.epoch = checkpoint['epoch'];

        // Infer in FP16 if the model was trained with mixed precision
        if (this.result_cfg.precision === 'mixed') {
            this.model = this.model.toHalf();
        }
        if (device.deviceType === 'cpu') {
            this.model = this.model.toFloat(); // CPU does not support FP16, so convert it back to FP32
        }

        // Initialize the transfer function
        if (!this.is_aux || this.main_feature !== 'z') {
            this.transfer = color.getTransferFunction(this.result_cfg);
        } else {
            this.transfer = null; // already applied to auxiliary depth
        }

        // Set the model to evaluation mode
        this.model = this.model.eval();

        // Initialize auxiliary feature inference
        this.aux_infers = {};
        if (this.aux_features.length > 0) {
            for (const aux_result of new Set(cfg.aux_results)) {
                const aux_infer = new Infer(cfg, device, aux_result, true);
                if (
                    aux_infer.main_feature !== this.aux_features ||
                    aux_infer.aux_features.length > 0
                ) {
                    throw new Error(`result ${aux_result} does not correspond to an auxiliary feature`);
                }
                this.aux_infers[aux_infer.main_feature] = aux_infer;
            }
        }
    }

    call(input: tf.Tensor, exposure = 1.): tf.Tensor {
        let image = input.clone();

        // Apply the transfer function to the main feature
        let color = image.slice([0, 0, 0], [-1, this.num_main_channels, -1]);
        if (this.main_feature === 'hdr') {
            color = color.mul(exposure);
        }
        if (this.transfer) {
            color = this.transfer.forward(color);
        }
        image = image.slice([0, 0, 0], [-1, this.num_main_channels, -1]).assign(color);

        // Pad the output
        const shape = image.shape;
        image = tf.pad(image, [
            [0, 0],
            [0, util.roundUp(shape[2], this.model.alignment) - shape[2]],
            [0, util.roundUp(shape[3], this.model.alignment) - shape[3]],
        ]);

        // Prefilter the auxiliary features
        for (const [aux_feature, aux_infer] of Object.entries(this.aux_infers)) {
            const aux_channels = util.getDatasetChannels(aux_feature);
            const aux_channel_indices = util.getChannelIndices(aux_channels, this.all_channels);
            let aux = image.slice([0, aux_channel_indices[0], 0], [-1, aux_channel_indices.length, -1]);
            aux = aux_infer.call(aux);
            image = image.slice([0, aux_channel_indices[0], 0], [-1, aux_channel_indices.length, -1]).assign(aux);
        }

        // Filter the main feature
        if (this.result_cfg.precision === 'mixed' && this.device.deviceType !== 'cpu') {
            image = image.toFloat();
        }
        if (this.main_feature === 'sh1') {
            // Iterate over x, y, z
            const sh1Images = [0, 3, 6].map((i) =>
                this.model.call(
                    tf.concat([image.slice([0, i, 0], [-1, 3, -1]), image.slice([0, 9, 0], [-1, -1, -1])], 1)
                )
            );
            image = tf.concat(sh1Images, 1);
        } else {
            image = this.model.call(image);
        }
        image = image.toFloat();

        // Unpad the output
        image = image.slice([0, 0, 0, 0], [-1, -1, shape[2], shape[3]]);

        // Sanitize the output
        image = tf.clipByValue(image, 0, Infinity);

        // Apply the inverse transfer function
        if (this.transfer) {
            image = this.transfer.inverse(image);
        }

        if (this.main_feature === 'hdr') {
            image = image.div(exposure);
        } else {
            image = tf.clipByValue(image, 0, 1);
        }

        return image;
    }
}

function main() {
    // Parse the command line arguments
    const cfg = util.parseArgs({ description: 'Performs inference on a dataset using the specified training result.' });

    // Initialize the TensorFlow.js backend
    tf.setBackend('tensorflow');

    // Initialize the inference function
    const device = tf.device(cfg.device);
    const infer = new Infer(cfg, device);
    console.log('Result:', cfg.result);
    console.log('Epoch:', infer.epoch);

    // Initialize the dataset
    const data_dir = util.getDataDir(cfg, cfg.input_data);
    const image_sample_groups = dataset.getImageSampleGroups(data_dir, infer.features);

    // Iterate over the images
    console.log();
    const output_dir = util.joinPath(cfg.output_dir, cfg.input_data);
    const metric_sum: { [key: string]: number } = {};
    let metric_count = 0;

    // Saves an image in different formats
    function saveImages(
        path: string,
        image: tf.Tensor,
        image_srgb: tf.Tensor,
        num_channels: { [key: string]: number },
        feature_ext = infer.main_feature
    ) {
        if (feature_ext === 'sh1') {
            // Iterate over x, y, z
            for (const [i, axis] of [[0, 'x'], [3, 'y'], [6, 'z']]) {
                saveImages(path, image.slice([0, i, 0], [-1, 3, -1]), image_srgb.slice([0, i, 0], [-1, 3, -1]), num_channels, 'sh1' + axis);
            }
            return;
        }

        const imageTensor = tf.tensorToNdarray(image);
        const imageSRGBTensor = tf.tensorToNdarray(image_srgb);
        const filename_prefix = path + '.' + feature_ext + '.';
        for (const format of cfg.format) {
            if (['exr', 'pfm', 'phm', 'hdr'].includes(format)) {
                // Transform to original range
                if (['sh1', 'nrm'].includes(infer.main_feature)) {
                    imageTensor.mul(2).sub(1); // [0..1] -> [-1..1]
                }
                util.saveImage(filename_prefix + format, imageTensor, num_channels[feature_ext]);
            } else {
                util.saveImage(filename_prefix + format, imageSRGBTensor, num_channels[feature_ext]);
            }
        }
    }

    for (const [group, input_names, target_name] of image_sample_groups) {
        // Create the output directory if it does not exist
        const output_group_dir = util.joinPath(output_dir, util.dirname(group));
        if (!util.isDirectory(output_group_dir)) {
            util.createDirectory(output_group_dir);
        }

        // Load metadata for the images if it exists
        let tonemap_exposure = 1.;
        const metadata = util.loadImageMetadata(util.joinPath(data_dir, group));
        if (metadata) {
            tonemap_exposure = metadata['exposure'];
            util.saveImageMetadata(util.joinPath(output_dir, group), metadata);
        }

        // Load the target image if it exists
        let target: tf.Tensor | null = null;
        let target_num_channels: number[] | null = null;
        if (target_name) {
            [target, target_num_channels] = util.loadImageFeatures(util.joinPath(data_dir, target_name), infer.main_feature);
            target = util.imageToTensor(target, true).to(device);
            const target_srgb = util.transformFeature(target, infer.main_feature, 'srgb', tonemap_exposure);
        }

        // Iterate over the input images
        for (const input_name of input_names) {
            process.stdout.write(input_name + '...');
            process.stdout.flush();

            // Load the input image
            const [input, input_num_channels] = util.loadImageFeatures(util.joinPath(data_dir, input_name), infer.features);

            // Compute the autoexposure value
            const exposure = infer.main_feature === 'hdr' ? util.autoexposure(input) : 1.;

            // Infer
            const inputTensor = util.imageToTensor(input, true).to(device);
            const output = infer.call(inputTensor, exposure);

            const inputTensorMain = inputTensor.slice([0, 0, 0], [-1, infer.num_main_channels, -1]);
            const input_srgb = util.transformFeature(inputTensorMain, infer.main_feature, 'srgb', tonemap_exposure);
            const output_srgb = util.transformFeature(output, infer.main_feature, 'srgb', tonemap_exposure);

            // Compute metrics
            let metric_str = '';
            if (target_name && cfg.metric) {
                for (const metric of cfg.metric) {
                    const value = util.compareImages(output_srgb, target_srgb, metric);
                    metric_sum[metric] = (metric_sum[metric] || 0) + value;
                    if (metric_str) {
                        metric_str += ', ';
                    }
                    metric_str += `${metric}=${value.toFixed(4)}`;
                }
                metric_count++;
            }

            // Save the input and output images
            const output_suffix = cfg.result || cfg.output_suffix;
            let output_name = input_name + '.' + output_suffix;
            if (cfg.num_epochs) {
                output_name += `_${infer.epoch}`;
            }
            if (cfg.save_all) {
                saveImages(util.joinPath(output_dir, input_name), inputTensor, input_srgb, input_num_channels);
            }
            saveImages(util.joinPath(output_dir, output_name), output, output_srgb, input_num_channels);

            // Print metrics
            if (metric_str) {
                metric_str = ' ' + metric_str;
            }
            console.log(metric_str);
        }

        // Save the target image if it exists
        if (cfg.save_all && target_name) {
            saveImages(util.joinPath(output_dir, target_name), target, target_srgb, target_num_channels);
        }
    }

    // Print summary
    if (metric_count > 0) {
        let metric_str = '';
        for (const metric of cfg.metric) {
            const value = metric_sum[metric] / metric_count;
            if (metric_str) {
                metric_str += ', ';
            }
            metric_str += `${metric}_avg=${value.toFixed(4)}`;
        }
        console.log();
        console.log(`${cfg.result}: ${metric_str} (${metric_count} images)`);
    }
}

main();