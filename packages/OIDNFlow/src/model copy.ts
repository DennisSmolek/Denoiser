// model.ts
import * as tf from '@tensorflow/tfjs';

function get_model(cfg: any): tf.LayersModel {
  const type = cfg.model;
  const in_channels = get_model_channels(cfg.features).length;
  const out_channels = get_model_channels(get_main_feature(cfg.features)).length;

  if (type === 'unet') {
    return UNet(in_channels, out_channels);
  } else if (type === 'unet_small') {
    return UNet(in_channels, out_channels, true);
  } else if (type === 'unet_large') {
    return UNetLarge(in_channels, out_channels);
  } else if (type === 'unet_xl') {
    return UNetLarge(in_channels, out_channels, true);
  } else {
    throw new Error('Invalid model');
  }
}

// Network layers
function Conv(in_channels: number, out_channels: number): tf.layers.Layer {
  return tf.layers.conv2d({
    filters: out_channels,
    kernelSize: 3,
    padding: 'same',
    activation: 'linear',
    inputShape: [null, null, in_channels]
  });
}

function relu(x: tf.Tensor): tf.Tensor {
  return tf.relu(x);
}

function pool(x: tf.Tensor): tf.Tensor {
  return tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(x) as tf.Tensor;
}

function upsample(x: tf.Tensor): tf.Tensor {
  return tf.layers.upSampling2d({ size: [2, 2] }).apply(x) as tf.Tensor;
}

function concat(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
  return tf.concat([a, b], -1);
}

// U-Net model
class UNet extends tf.LayersModel {
  constructor(in_channels: number, out_channels: number, small = false) {
    super();

    const ic = in_channels;
    const oc = out_channels;

    let ec1, ec2, ec3, ec4, ec5, dc4, dc3, dc2a, dc2b, dc1a, dc1b;

    if (small) {
      ec1 = 32;
      ec2 = 32;
      ec3 = 32;
      ec4 = 32;
      ec5 = 32;
      dc4 = 64;
      dc3 = 64;
      dc2a = 64;
      dc2b = 32;
      dc1a = 32;
      dc1b = 32;
    } else {
      ec1 = 32;
      ec2 = 48;
      ec3 = 64;
      ec4 = 80;
      ec5 = 96;
      dc4 = 112;
      dc3 = 96;
      dc2a = 64;
      dc2b = 64;
      dc1a = 64;
      dc1b = 32;
    }

    this.enc_conv0 = Conv(ic, ec1);
    this.enc_conv1 = Conv(ec1, ec1);
    this.enc_conv2 = Conv(ec1, ec2);
    this.enc_conv3 = Conv(ec2, ec3);
    this.enc_conv4 = Conv(ec3, ec4);
    this.enc_conv5a = Conv(ec4, ec5);
    this.enc_conv5b = Conv(ec5, ec5);
    this.dec_conv4a = Conv(ec5 + ec3, dc4);
    this.dec_conv4b = Conv(dc4, dc4);
    this.dec_conv3a = Conv(dc4 + ec2, dc3);
    this.dec_conv3b = Conv(dc3, dc3);
    this.dec_conv2a = Conv(dc3 + ec1, dc2a);
    this.dec_conv2b = Conv(dc2a, dc2b);
    this.dec_conv1a = Conv(dc2b + ic, dc1a);
    this.dec_conv1b = Conv(dc1a, dc1b);
    this.dec_conv0 = Conv(dc1b, oc);

    this.alignment = 16;
  }

  call(input: tf.Tensor): tf.Tensor {
    let x = relu(this.enc_conv0.apply(input)) as tf.Tensor;

    x = relu(this.enc_conv1.apply(x)) as tf.Tensor;
    const pool1 = pool(x) as tf.Tensor;

    x = relu(this.enc_conv2.apply(pool1)) as tf.Tensor;
    const pool2 = pool(x) as tf.Tensor;

    x = relu(this.enc_conv3.apply(pool2)) as tf.Tensor;
    const pool3 = pool(x) as tf.Tensor;

    x = relu(this.enc_conv4.apply(pool3)) as tf.Tensor;
    x = pool(x) as tf.Tensor;

    x = relu(this.enc_conv5a.apply(x)) as tf.Tensor;
    x = relu(this.enc_conv5b.apply(x)) as tf.Tensor;

    x = upsample(x) as tf.Tensor;
    x = concat(x, pool3) as tf.Tensor;
    x = relu(this.dec_conv4a.apply(x)) as tf.Tensor;
    x = relu(this.dec_conv4b.apply(x)) as tf.Tensor;

    x = upsample(x) as tf.Tensor;
    x = concat(x, pool2) as tf.Tensor;
    x = relu(this.dec_conv3a.apply(x)) as tf.Tensor;
    x = relu(this.dec_conv3b.apply(x)) as tf.Tensor;

    x = upsample(x) as tf.Tensor;
    x = concat(x, pool1) as tf.Tensor;
    x = relu(this.dec_conv2a.apply(x)) as tf.Tensor;
    x = relu(this.dec_conv2b.apply(x)) as tf.Tensor;

    x = upsample(x) as tf.Tensor;
    x = concat(x, input) as tf.Tensor;
    x = relu(this.dec_conv1a.apply(x)) as tf.Tensor;
    x = relu(this.dec_conv1b.apply(x)) as tf.Tensor;

    x = this.dec_conv0.apply(x) as tf.Tensor;

    return x;
  }
}

// U-Net model: large
class UNetLarge extends tf.LayersModel {
  constructor(in_channels: number, out_channels: number, xl = false) {
    super();

    const ic = in_channels;
    const oc = out_channels;

    let ec1, ec2, ec3, ec4, ec5, dc4, dc3, dc2, dc1;

    if (xl) {
      ec1 = 96;
      ec2 = 128;
      ec3 = 192;
      ec4 = 256;
      ec5 = 384;
      dc4 = 256;
      dc3 = 192;
      dc2 = 128;
      dc1 = 96;
    } else {
      ec1 = 64;
      ec2 = 96;
      ec3 = 128;
      ec4 = 192;
      ec5 = 256;
      dc4 = 192;
      dc3 = 128;
      dc2 = 96;
      dc1 = 64;
    }

    this.enc_conv1a = Conv(ic, ec1);
    this.enc_conv1b = Conv(ec1, ec1);
    this.enc_conv2a = Conv(ec1, ec2);
    this.enc_conv2b = Conv(ec2, ec2);
    this.enc_conv3a = Conv(ec2, ec3);
    this.enc_conv3b = Conv(ec3, ec3);
    this.enc_conv4a = Conv(ec3, ec4);
    this.enc_conv4b = Conv(ec4, ec4);
    this.enc_conv5a = Conv(ec4, ec5);
    this.enc_conv5b = Conv(ec5, ec5);
    this.dec_conv4a = Conv(ec5 + ec3, dc4);
    this.dec_conv4b = Conv(dc4, dc4);
    this.dec_conv3a = Conv(dc4 + ec2, dc3);
    this.dec_conv3b = Conv(dc3, dc3);
    this.dec_conv2a = Conv(dc3 + ec1, dc2);
    this.dec_conv2b = Conv(dc2, dc2);
    this.dec_conv1a = Conv(dc2 + ic, dc1);
    this.dec_conv1b = Conv(dc1, dc1);
    this.dec_conv1c = Conv(dc1, oc);

    this.alignment = 16;
  }

  call(input: tf.Tensor): tf.Tensor {
    let x = relu(this.enc_conv1a.apply(input)) as tf.Tensor;
    x = relu(this.enc_conv1b.apply(x)) as tf.Tensor;
    const pool1 = pool(x) as tf.Tensor;

    x = relu(this.enc_conv2a.apply(pool1)) as tf.Tensor;
    x = relu(this.enc_conv2b.apply(x)) as tf.Tensor;
    const pool2 = pool(x) as tf.Tensor;

    x = relu(this.enc_conv3a.apply(pool2)) as tf.Tensor;
    x = relu(this.enc_conv3b.apply(x)) as tf.Tensor;
    const pool3 = pool(x) as tf.Tensor;

    x = relu(this.enc_conv4a.apply(pool3)) as tf.Tensor;
    x = relu(this.enc_conv4b.apply(x)) as tf.Tensor;
    x = pool(x) as tf.Tensor;

    x = relu(this.enc_conv5a.apply(x)) as tf.Tensor;
    x = relu(this.enc_conv5b.apply(x)) as tf.Tensor;

    x = upsample(x) as tf.Tensor;
    x = concat(x, pool3) as tf.Tensor;
    x = relu(this.dec_conv4a.apply(x)) as tf.Tensor;
    x = relu(this.dec_conv4b.apply(x)) as tf.Tensor;

    x = upsample(x) as tf.Tensor;
    x = concat(x, pool2) as tf.Tensor;
    x = relu(this.dec_conv3a.apply(x)) as tf.Tensor;
    x = relu(this.dec_conv3b.apply(x)) as tf.Tensor;

    x = upsample(x) as tf.Tensor;
    x = concat(x, pool1) as tf.Tensor;
    x = relu(this.dec_conv2a.apply(x)) as tf.Tensor;
    x = relu(this.dec_conv2b.apply(x)) as tf.Tensor;

    x = upsample(x) as tf.Tensor;
    x = concat(x, input) as tf.Tensor;
    x = relu(this.dec_conv1a.apply(x)) as tf.Tensor;
    x = relu(this.dec_conv1b.apply(x)) as tf.Tensor;
    x = relu(this.dec_conv1c.apply(x)) as tf.Tensor;

    return x;
  }
}
