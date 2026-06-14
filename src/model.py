import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetV2L, EfficientNetB7, InceptionResNetV2, DenseNet201, NASNetLarge
from src.config import IMG_SIZE, DENSE_UNITS, DROPOUT, MODEL_DIR, OUTPUT_DIR, PATIENCE

ARCHITECTURES = {
    'EfficientNetV2L': EfficientNetV2L,
    'EfficientNetB7': EfficientNetB7,
    'InceptionResNetV2': InceptionResNetV2,
    'DenseNet201': DenseNet201,
    'NASNetLarge': NASNetLarge,
}

ARCH_INPUT_SIZE = {
    'EfficientNetV2L': 224,
    'EfficientNetB7': 224,
    'InceptionResNetV2': 224,
    'DenseNet201': 224,
    'NASNetLarge': 331,
}

def build_model(arch_name):
    input_size = ARCH_INPUT_SIZE[arch_name]
    base_fn = ARCHITECTURES[arch_name]

    base = base_fn(input_shape=(input_size, input_size, 3), include_top=False, weights='imagenet')
    base.trainable = False  

    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    if input_size != IMG_SIZE:
        x = layers.Resizing(input_size, input_size)(inputs)
    else:
        x = inputs
        
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name=arch_name)
    return model, base

def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

def get_callbacks(arch_name, stage):
    model_path = os.path.join(MODEL_DIR, f'{arch_name}_best.keras')
    return [
        callbacks.ModelCheckpoint(model_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=PATIENCE, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1),
        callbacks.CSVLogger(os.path.join(OUTPUT_DIR, f'{arch_name}_stage{stage}_history.csv')),
    ]