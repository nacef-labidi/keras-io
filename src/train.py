"""Data preprocessing and model training using MultiWorkerMirroredStrategy"""
import datetime
import os
import logging
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from time import perf_counter

from pathlib import Path  
 
logging.basicConfig(level=logging.DEBUG) 

DATASET_NAME = "cifar100"
DATASET_IMG_SIZE = 224
DATASET_NUM_CLASSES = 120

DATASET_BATCH_SIZE = 64

MODEL_TOP_LAYER_DROPOUT = 0.2
MODEL_TOP_LAYER_LR = 0.01
MODEL_TOP_LAYER_EPOCHS = 5

MODEL_TOP_20_LAYERS_LR = 0.0001
MODEL_TOP_20_LAYERS_EPOCHS = 5

MODEL_SAVE_PATH = ".models/default"


def encode_labels(images, labels):
    """Encodes a batch of labels as one-hot vectors"""

    labels = tf.one_hot(labels, DATASET_NUM_CLASSES)
    return images, labels


def augment_images():
    """Creates an image augmentation model with random rotations,
    translations, flips, and contrast changes"""

    return Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ],
        name="image_augmentation",
    )


def build_model():
    """Builds the model from EfficientNetB0 with pre-trained imagenet weights
    to only train the top layer"""

    inputs = layers.Input(shape=(DATASET_IMG_SIZE, DATASET_IMG_SIZE, 3))
    x = augment_images()(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(MODEL_TOP_LAYER_DROPOUT, name="top_dropout")(x)
    outputs = layers.Dense(DATASET_NUM_CLASSES, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=MODEL_TOP_LAYER_LR)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def unfreeze_model(model):
    """Unfreezes the top 20 layers of the model to finish the transfer learning"""
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=MODEL_TOP_20_LAYERS_LR)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


if __name__ == "__main__":
    INPUT_PATH = os.environ.get("AICHOR_INPUT_PATH", "input")
    OUTPUT_PATH = os.environ.get("AICHOR_OUTPUT_PATH", "output")
    AICHOR_EXPERIMENT_ID = os.environ.get("AICHOR_EXPERIMENT_ID", "id")
    LOGS_PATH = os.environ.get("AICHOR_LOGS_PATH", os.path.join(OUTPUT_PATH, "logs", AICHOR_EXPERIMENT_ID))

    # Path(os.environ["ICHOR_OUTPUT_DATASET"]).parent.mkdir(exist_ok=True) 
    # Path(os.environ["ICHOR_OUTPUT_DATASET"]).mkdir(exist_ok=True) 
    # Path(os.environ["ICHOR_LOGS"]).parent.mkdir(exist_ok=True)
    # Path(os.environ["ICHOR_LOGS"]).mkdir(exist_ok=True)

    # use all available GPUs (use CPUs if no GPUs are available)
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # load and split dataset directly from tensorflow_datasets
    (training_set, validation_set) = tfds.load(
        DATASET_NAME,
        split=["train", "test"],
        as_supervised=True,
        # download=False,
        # data_dir=INPUT_PATH
    )

    num_training_examples = training_set.cardinality().numpy()
    num_validation_examples = validation_set.cardinality().numpy()

    # resize training and testing datasets
    size = (DATASET_IMG_SIZE, DATASET_IMG_SIZE)
    training_set = training_set.map(
        lambda image, label: (tf.image.resize(image, size), label)
    )
    validation_set = validation_set.map(
        lambda image, label: (tf.image.resize(image, size), label)
    )

    # encode labels and batch datasets
    training_set = training_set.map(
        encode_labels, num_parallel_calls=tf.data.AUTOTUNE
    )
    training_set = training_set.batch(
        batch_size=DATASET_BATCH_SIZE,
        drop_remainder=True
    ).prefetch(tf.data.AUTOTUNE).repeat()
    validation_set = validation_set.map(encode_labels)
    validation_set = validation_set.batch(
        batch_size=DATASET_BATCH_SIZE,
        drop_remainder=True
    ).repeat()

    # build model and train its top layer
    with strategy.scope():
        model = build_model()

    training_start = perf_counter()

    # log_dir = os.path.join(LOGS_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # log_dir = "s3://tensorboard-2309057bc0e04398-outputs/logs/" + EXPERIMENT_ID + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"az://{os.environ['TF_AZURE_STORAGE_ACCOUNT_NAME']}/{os.environ['AICHOR_OUTPUT_BUCKET_NAME']}/logs/{os.environ['AICHOR_EXPERIMENT_ID']}"
    output_path = f"az://{os.environ['TF_AZURE_STORAGE_ACCOUNT_NAME']}/{os.environ['AICHOR_OUTPUT_BUCKET_NAME']}/output/{os.environ['AICHOR_EXPERIMENT_ID']}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        training_set,
        epochs=MODEL_TOP_LAYER_EPOCHS,
        steps_per_epoch=num_training_examples//DATASET_BATCH_SIZE,
        validation_data=validation_set,
        validation_steps=num_validation_examples//DATASET_BATCH_SIZE,
        callbacks=[tensorboard_callback]
    )

    # unfreeze some non-top layers and train them
    unfreeze_model(model)

    model.fit(
        training_set,
        epochs=MODEL_TOP_20_LAYERS_EPOCHS,
        steps_per_epoch=num_training_examples//DATASET_BATCH_SIZE,
        validation_data=validation_set,
        validation_steps=num_training_examples//DATASET_BATCH_SIZE,
        callbacks=[tensorboard_callback]
    )

    logging.debug(f"Training time: {str(perf_counter() - training_start)}")

    model.save(output_path)
