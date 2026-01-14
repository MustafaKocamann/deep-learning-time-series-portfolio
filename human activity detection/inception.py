from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Dense,
    GlobalAveragePooling1D, Activation, BatchNormalization, Concatenate
)

def inception_module(x, filters=32):
    # 1x1 konvolüsyon kolu
    conv1 = Conv1D(filters, kernel_size=1, padding="same", activation="relu")(x)

    # 3x3 konvolüsyon kolu
    conv3 = Conv1D(filters, kernel_size=3, padding="same", activation="relu")(x)

    # 5x5 konvolüsyon kolu
    conv5 = Conv1D(filters, kernel_size=5, padding="same", activation="relu")(x)

    # Max pooling + 1x1 konvolüsyon kolu
    pool = MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
    pool_conv = Conv1D(filters, kernel_size=1, padding="same", activation="relu")(pool)

    # Tüm kolları birleştir
    output = Concatenate(axis=-1)([conv1, conv3, conv5, pool_conv])
    return output

def build_inception_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = inception_module(inputs, filters=32)
    x = BatchNormalization()(x)

    x = inception_module(x, filters=32)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = build_inception_model((128, 9), 6)
    model.summary()
