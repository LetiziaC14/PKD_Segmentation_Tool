import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

smooth = 1.0

# --- Funzioni di Loss e Metriche ---

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred) # Versione moderna per minimizzare la loss

# --- Architettura del Modello ---

def build_modern_autokidneycyst_model(
    img_rows=256, img_cols=256, num_classes=1, input_channels=2,
):
    # Formato moderno standard (channels_last): (H, W, Channels)
    inputs = Input((img_rows, img_cols, input_channels))
    
    # Livello 1 (Kernel 7x7)
    conv1 = Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Conv2D(32, (7, 7), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
    # Livello 2 (Kernel 5x5)
    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Livello 3 (Kernel 3x3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Livello 4 (Kernel 3x3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Livello 5 (Kernel 3x3)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.1)(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # Bottleneck (Livello 6)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
    conv6 = Dropout(0.1)(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    # --- Decoder (con Additive Skip Connections) ---

    up7 = Add()([UpSampling2D(size=(2, 2))(conv6), conv5])
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.1)(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Add()([UpSampling2D(size=(2, 2))(conv7), conv4])
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Add()([UpSampling2D(size=(2, 2))(conv8), conv3])
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = BatchNormalization(axis=-1)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    up10 = Add()([UpSampling2D(size=(2, 2))(conv9), conv2])
    conv10 = Conv2D(64, (5, 5), activation='relu', padding='same')(up10)
    conv10 = Dropout(0.1)(conv10)
    conv10 = BatchNormalization(axis=-1)(conv10)
    conv10 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv10)
    
    up11 = Add()([UpSampling2D(size=(2, 2))(conv10), conv1])
    conv11 = Conv2D(32, (7, 7), activation='relu', padding='same')(up11)
    conv11 = Dropout(0.1)(conv11)
    conv11 = BatchNormalization(axis=-1)(conv11)
    conv11 = Conv2D(32, (7, 7), activation='relu', padding='same')(conv11)
    
    # Output layer
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    conv12 = Conv2D(num_classes, (1, 1), activation=activation)(conv11)

    # Compilazione del modello
    model = Model(inputs=inputs, outputs=conv12)
    
    # Utilizziamo i parametri esatti citati nel testo del paper
    opt = Adam(learning_rate=1e-3) # Nel TF moderno il decay si gestisce tramite LearningRateScheduler
    if num_classes == 1:
        loss = dice_coef_loss
        metrics = [dice_coef]
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [dice_coef, tf.keras.metrics.CategoricalAccuracy(name='cat_acc')]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model

