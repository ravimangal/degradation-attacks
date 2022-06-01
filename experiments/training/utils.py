import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


def data_augmentation(
        flip=True,
        saturation=(0.5, 1.2),
        contrast=(0.8, 1.2),
        zoom=0.1,
        noise=None,
        cutout=None,
):
    def augment(x, y):
        batch_size = tf.shape(x)[0]
        input_shape = x.shape[1:]

        # Horizontal flips
        if flip:
            x = tf.image.random_flip_left_right(x)

        # Randomly adjust the saturation and contrast.
        if saturation is not None and input_shape[-1] == 3:
            x = tf.image.random_saturation(
                x, lower=saturation[0], upper=saturation[1])

        if contrast is not None:
            x = tf.image.random_contrast(
                x, lower=contrast[0], upper=contrast[1])

        # Randomly zoom.
        if zoom is not None:
            widths = tf.random.uniform([batch_size], 1. - zoom, 1.)
            top_corners = tf.random.uniform(
                [batch_size, 2], 0, 1. - widths[:, None])
            bottom_corners = top_corners + widths[:, None]
            boxes = tf.concat((top_corners, bottom_corners), axis=1)

            x = tf.image.crop_and_resize(
                x, boxes,
                box_indices=tf.range(batch_size),
                crop_size=input_shape[0:2])

        if noise is not None:
            x = x + tf.random.normal(tf.shape(x), stddev=noise)

        if cutout is not None:
            x = cutout_augmentation(x, mask_size=cutout)

        return x, y

    return augment


def sample_beta_distribution(n, alpha, beta):
    gamma_1 = tf.random.gamma(shape=(n,), alpha=alpha)
    gamma_2 = tf.random.gamma(shape=(n,), alpha=beta)

    return gamma_1 / (gamma_1 + gamma_2)


def mixup_augmentation(D_1, D_2, alpha=0.2):
    X_1, y_1 = D_1
    X_2, y_2 = D_2

    batch_size = tf.shape(X_1)[0]
    input_dims = (1 for _ in range(len(tf.shape(X_1[1:]))))

    lam = sample_beta_distribution(batch_size, alpha, alpha)

    # Enable broadcasting to shape of X.
    lam_x = tf.reshape(lam, (batch_size, *input_dims))

    return (
        X_1 * lam_x + X_2 * (1. - lam_x),
        y_1 * lam[:, None] + y_2 * (1. - lam[:, None]))


def cutout_augmentation(x, mask_size=8):
    nhwc = tf.shape(x)
    N = nhwc[0]
    h = nhwc[1]
    w = nhwc[2]

    mask_corners_y = tf.random.uniform(
        (N,), 0, h - mask_size + 1, dtype='int32')[:, None, None, None]
    mask_corners_x = tf.random.uniform(
        (N,), 0, w - mask_size + 1, dtype='int32')[:, None, None, None]

    inds = tf.reshape(tf.range(h * w), (h, w))[None, :, :, None]

    mask = tf.cast(
        tf.logical_and(
            tf.logical_and(
                mask_corners_x <= inds % w,
                inds % w < mask_corners_x + mask_size),
            tf.logical_and(
                mask_corners_y <= inds // w,
                inds // w < mask_corners_y + mask_size)),
        'float32')

    return (1. - mask) * x + mask * 0.5


def get_data(dataset, batch_size, augmentation=None, mixup=False):
    # Get the augmentation.
    if augmentation is None or augmentation.lower() == 'none':
        augmentation = data_augmentation(
            flip=False, saturation=None, contrast=None, zoom=None)

    elif augmentation == 'all':
        augmentation = data_augmentation()

    elif augmentation == 'standard':
        augmentation = data_augmentation(
            saturation=None, contrast=None, zoom=0.25)

    elif augmentation == 'standard_cutout':
        augmentation = data_augmentation(
            saturation=None, contrast=None, zoom=0.25, cutout=8)

    elif augmentation == 'no_flip':
        augmentation = data_augmentation(
            flip=False, saturation=None, contrast=None, zoom=0.25)

    elif augmentation.startswith('rs'):
        flip = data_augmentation.startswith('rs_no_flip')

        if augmentation.startswith('rs.'):
            noise = float(augmentation.split('rs.')[1])

        elif augmentation.startswith('rs_no_flip.'):
            noise = float(augmentation.split('rs_no_flip.')[1])

        else:
            noise = 0.125

        augmentation = data_augmentation(
            flip=flip, saturation=None, contrast=None, noise=noise)

    else:
        raise ValueError(f'unknown augmentation type: {augmentation}')

    # Load the data.
    if dataset == 'tiny-imagenet':

        if 'TINY_IMAGENET_LOCATION' in os.environ:
            tiny_imagenet_dir = os.environ['TINY_IMAGENET_LOCATION']

        else:
            raise ValueError(
                'to use tiny-imagenet you must set the '
                '"TINY_IMAGENET_LOCATION" environment variable')

        x_tr = np.load(f'{tiny_imagenet_dir}/TinyImagenet_train_x.npy')
        x_te = np.load(f'{tiny_imagenet_dir}/TinyImagenet_test_x.npy')
        y_tr = np.load(f'{tiny_imagenet_dir}/TinyImagenet_train_y.npy')
        y_te = np.load(f'{tiny_imagenet_dir}/TinyImagenet_test_y.npy')

        train = (Dataset.from_tensor_slices((x_tr, y_tr))
                 .cache()
                 .batch(batch_size)
                 .map(
            augmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)
                 .prefetch(tf.data.experimental.AUTOTUNE))

        test = (Dataset.from_tensor_slices((x_te, y_te))
                .cache()
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))

        metadata = lambda: None
        metadata.features = {
            'image': lambda: None,
            'label': lambda: None,
        }
        metadata.features['label'].num_classes = 200
        metadata.features['image'].shape = (64, 64, 3)

    elif dataset.startswith('cifar10_wti'):

        ratio = float(dataset.split(',')[1])

        if 'TINY_IMAGENET_LOCATION' in os.environ:
            tiny_imagenet_dir = os.environ['TINY_IMAGENET_LOCATION']

        else:
            raise ValueError(
                'to use tiny-imagenet you must set the '
                '"TINY_IMAGENET_LOCATION" environment variable')

        x_aug_tr = np.load(
            f'{tiny_imagenet_dir}/TinyImagenet_downsized_train_x.npy')
        y_aug_tr = np.load(
            f'{tiny_imagenet_dir}/TinyImagenet_downsized_train_y.npy')

        random_ix = np.random.permutation(x_aug_tr.shape[0])
        x_aug_tr = x_aug_tr[random_ix]
        y_aug_tr = y_aug_tr[random_ix]
        x_aug_tr = x_aug_tr[:int(ratio * x_aug_tr.shape[0])]
        y_aug_tr = y_aug_tr[:int(ratio * y_aug_tr.shape[0])]

        tfds_dir = os.environ['TFDS_DIR'] if 'TFDS_DIR' in os.environ else None
        (train, test), _ = tfds.load(
            'cifar10',
            data_dir=tfds_dir,
            split=['train', 'test'],
            with_info=True,
            shuffle_files=True,
            as_supervised=True)

        x_train = []
        y_train = []

        # Iterate over a dataset
        for image, label in tfds.as_numpy(train.batch(1000)):
            x_train.append(image)
            y_train.append(label)

        x_train = np.vstack(x_train)
        y_train = np.hstack(y_train)

        x_train = x_train.astype(np.float32) / 255.

        x_tr = np.concatenate((x_train, x_aug_tr), axis=0)
        y_tr = np.concatenate((y_train, y_aug_tr))

        train = (Dataset.from_tensor_slices((x_tr, y_tr))
                 .cache()
                 .batch(batch_size)
                 .map(
            augmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)
                 .shuffle(10000)
                 .prefetch(tf.data.experimental.AUTOTUNE))

        test = (test
                .map(
            lambda x, y: (tf.cast(x, 'float32') / 255., y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)
                .cache()
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))

        metadata = lambda: None
        metadata.features = {
            'image': lambda: None,
            'label': lambda: None,
        }
        metadata.features['label'].num_classes = 10
        metadata.features['image'].shape = (32, 32, 3)

    else:
        tfds_dir = os.environ['TFDS_DIR'] if 'TFDS_DIR' in os.environ else None

        split = ['train', 'test']

        (train, test), metadata = tfds.load(
            dataset,
            data_dir=tfds_dir,
            split=split,
            with_info=True,
            shuffle_files=True,
            as_supervised=True)

        train = (train
                 .map(
            lambda x, y: (tf.cast(x, 'float32') / 255., y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)
                 .cache()
                 .batch(batch_size)
                 .map(
            augmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)
                 .prefetch(tf.data.experimental.AUTOTUNE))

        test = (test
                .map(
            lambda x, y: (tf.cast(x, 'float32') / 255., y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)
                .cache()
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))

    if mixup:
        train_2, _, _ = get_data(
            dataset, batch_size, augmentation=augmentation, mixup=False)

        train = Dataset.zip((train, train_2)).map(
            mixup_augmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)

    return train, test, metadata


def get_optimizer(optimizer, lr, gradient_transformers=None):
    regular_kwargs = {'learning_rate': lr}
    if gradient_transformers is not None:
        regular_kwargs['gradient_transformers'] = gradient_transformers

    if optimizer == 'adam':
        try:
            return Adam(**regular_kwargs)
        except:
            raise ValueError(
                f'You may need at least Tensorflow 2.4 to use some features. '
                f'The current version is {tf.__version__}')

    elif optimizer.startswith('sgd'):
        try:
            if optimizer.startswith('sgd.'):
                return SGD(
                    momentum=float(optimizer.split('sgd.')[1]),
                    **regular_kwargs)

            else:
                return SGD(**regular_kwargs)
        except:
            raise ValueError(
                f'You may need at least Tensorflow 2.4 to use some features. '
                f'The current version is {tf.__version__}')

    else:
        raise ValueError(f'unknown optimizer: {optimizer}')
