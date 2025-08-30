from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense

def build_model():
    inputs = Input(shape=(64,64,3))
    # Block 1
    x = Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2,2)(x)
    x = Dropout(0.2)(x)
    # Block 2
    x = Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2,2)(x)
    x = Dropout(0.3)(x)
    # Block 3
    x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2,2)(x)
    x = Dropout(0.4)(x)
    # Block 4
    x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    # Dense layers
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

