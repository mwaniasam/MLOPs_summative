import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = (224, 224)
MODEL_PATH = "models/coffeeguard_model.h5"


def build_model(num_classes=NUM_CLASSES, freeze_base=True):
    """
    Build the CoffeeGuard classification model.
    Uses MobileNetV2 pretrained on ImageNet as the base with a custom head.

    Args:
        num_classes: number of output classes
        freeze_base: if True, freeze the entire MobileNetV2 base for phase 1 training
    """
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = not freeze_base

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model


def unfreeze_top_layers(model, base_model, num_layers=30, learning_rate=0.00001):
    """
    Unfreeze the top N layers of the base model for fine-tuning phase.
    Recompiles at a lower learning rate.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_callbacks(model_path=MODEL_PATH, phase=1):
    """
    Return training callbacks for a given phase.
    Phase 1 uses shorter patience. Phase 2 uses longer patience.
    """
    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    if phase == 1:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    else:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )

    return [early_stopping, reduce_lr, checkpoint]


def retrain(new_data_dir, model_path=MODEL_PATH, epochs=5):
    """
    Retrain the existing saved model on newly uploaded images.
    Loads the current model as a pretrained base and fine-tunes on new data.

    Args:
        new_data_dir: path to directory containing class subfolders of new images
        model_path: path to the saved model file
        epochs: number of fine-tuning epochs
    """
    import numpy as np
    import os

    print(f"Loading existing model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Build dataset manually to ensure all 5 classes are always used
    all_paths = []
    all_labels = []

    for idx, cls in enumerate(CLASSES):
        cls_path = os.path.join(new_data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for img in os.listdir(cls_path):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(cls_path, img))
                all_labels.append(idx)

    if not all_paths:
        raise ValueError("No valid images found in upload directory")

    print(f"Retraining on {len(all_paths)} images across classes: {list(set(all_labels))}")

    all_paths_t = tf.constant(all_paths, dtype=tf.string)
    all_labels_t = tf.constant(all_labels, dtype=tf.int32)

    def load_and_preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, depth=len(CLASSES))
        return image, label

    dataset = (
        tf.data.Dataset.from_tensor_slices((all_paths_t, all_labels_t))
        .shuffle(buffer_size=len(all_paths))
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"Fine-tuning for {epochs} epochs at learning rate 0.00001")

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        dataset,
        epochs=epochs,
        verbose=1,
        callbacks=[
            ModelCheckpoint(model_path, monitor="accuracy", save_best_only=True, verbose=1)
        ]
    )

    model.save(model_path)
    print(f"Retrained model saved to {model_path}")
    return history


def load_model(model_path=MODEL_PATH):
    """
    Load the saved model from disk.
    """
    return tf.keras.models.load_model(model_path)
