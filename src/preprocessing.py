import tensorflow as tf
import numpy as np
import os


IMG_SIZE = (224, 224)
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]


def load_and_preprocess_image(image_path):
    """
    Load a single image from disk, resize and normalize it.
    Used during inference and retraining.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def preprocess_uploaded_image(image_bytes):
    """
    Preprocess an image received as raw bytes from an API upload.
    Returns a batch-ready tensor.
    """
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image


def build_inference_dataset(image_dir, batch_size=32):
    """
    Build a tf.data pipeline from a directory of images for batch inference.
    Does not apply augmentation — only resizes and normalizes.
    """
    image_paths = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(image_dir, fname))

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, image_paths


def build_retraining_dataset(data_dir, batch_size=32, validation_split=0.2, seed=42):
    """
    Build train and validation tf.data pipelines from a directory of class folders.
    Applies augmentation to training data only.
    Expected directory structure:
        data_dir/
            ClassName1/
            ClassName2/
            ...
    """
    all_paths = []
    all_labels = []

    class_names = sorted(os.listdir(data_dir))
    class_indices = {cls: idx for idx, cls in enumerate(class_names)}
    num_classes = len(class_names)

    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = os.listdir(cls_path)
        np.random.seed(seed)
        np.random.shuffle(images)
        for img in images:
            all_paths.append(os.path.join(cls_path, img))
            all_labels.append(class_indices[cls])

    split_idx = int(len(all_paths) * (1 - validation_split))
    train_paths = all_paths[:split_idx]
    train_labels = all_labels[:split_idx]
    val_paths = all_paths[split_idx:]
    val_labels = all_labels[split_idx:]

    def load_labeled(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, depth=num_classes)
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.1)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(buffer_size=500, seed=seed)
        .map(load_labeled, num_parallel_calls=AUTOTUNE)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(load_labeled, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_dataset, val_dataset, class_names
