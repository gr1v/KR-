import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# 1. GENERATOR - Генератор изображений
# ============================================================================

class Generator(keras.Model):
    """
    Генератор преобразует случайный шум в реалистичные изображения цифр.
    
    Архитектура:
    - Dense слои для расширения латентного вектора
    - Reshape для преобразования в тензор
    - Conv2DTranspose для увеличения разрешения
    - BatchNormalization для стабилизации обучения
    """
    
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Плотные слои для обработки шума
        self.dense1 = layers.Dense(256)
        self.bn1 = layers.BatchNormalization()
        
        self.dense2 = layers.Dense(512)
        self.bn2 = layers.BatchNormalization()
        
        self.dense3 = layers.Dense(1024)
        self.bn3 = layers.BatchNormalization()
        
        self.dense4 = layers.Dense(28 * 28 * 1)
        
        # Conv2DTranspose для увеличения разрешения (7x7 -> 28x28)
        self.conv_transpose1 = layers.Conv2DTranspose(
            filters=64,
            kernel_size=4,
            strides=2,
            padding='same'
        )
        self.bn4 = layers.BatchNormalization()
        
        self.conv_transpose2 = layers.Conv2DTranspose(
            filters=32,
            kernel_size=4,
            strides=2,
            padding='same'
        )
        self.bn5 = layers.BatchNormalization()
        
        self.conv_transpose3 = layers.Conv2DTranspose(
            filters=1,
            kernel_size=4,
            strides=1,
            padding='same',
         
