import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 224
BATCH = 32
EPOCHS = 2
MODEL_PATH = "model.keras"

def preprocess(example):
    img = tf.image.resize(example["image"], (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    label = example["label"]
    return img, label

def main():
    # تحميل PlantVillage من TFDS وتقسيمه
    (train_ds, val_ds, test_ds), info = tfds.load(
        "plant_village",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        with_info=True,
    )

    class_names = info.features["label"].names
    num_classes = info.features["label"].num_classes
    print("Num classes:", num_classes)

    train = (train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
             .shuffle(2000)
             .batch(BATCH)
             .prefetch(tf.data.AUTOTUNE))

    val = (val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH)
           .prefetch(tf.data.AUTOTUNE))

    test = (test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))

    # موديل سريع: MobileNetV2
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train, validation_data=val, epochs=EPOCHS)

    loss, acc = model.evaluate(test)
    print("Test accuracy:", acc)

    model.save(MODEL_PATH)

    with open("classes.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print("Saved model ->", MODEL_PATH)
    print("Saved classes -> classes.txt")

if __name__ == "__main__":
    main()
