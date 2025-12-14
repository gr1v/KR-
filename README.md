[GAN_blocks.md](https://github.com/user-attachments/files/24154385/GAN_blocks.md)Задание 8: GAN для генерации изображений
=
### Задача: создать Generative Adversarial Network для синтеза изображений цифр.

Требования:

Generator: Dense слои + Reshape + Conv2DTranspose

Discriminator: Conv2D слои + Flatten + Dense

45

Minimax loss для состязательного обучения

Использовать batch normalization

### Что нужно дополнить:

1. Архитектуру Generator с Conv2DTranspose

 2. Архитектуру Discriminator с Conv2D

 3. Batch normalization слои

 4. Функцию train_step с обучением обеих сетей

 5. Сохранение сгенерированных изображений

 6. Графики loss кривых

ИМПОРТ БИБЛИОТЕК
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
```

ИНИЦИАЛИЗАЦИЯ GAN
```python
class GAN:
    """Generative Adversarial Network для MNIST"""
    
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
        
        self.g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        self.d_losses = []
        self.g_losses = []
```

АРХИТЕКТУРА ГЕНЕРАТОРА
```python
def _build_generator(self):
    model = keras.Sequential([
        layers.Input(shape=(self.latent_dim,)),
        
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Dense(7*7*64),
        layers.BatchNormalization(),
        layers.Reshape((7, 7, 64)),
        
        layers.Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding='same'),
        layers.Activation('tanh')
    ])
    return model
```

АРХИТЕКТУРА ДИСКРИМИНАТОРА
```python
def _build_discriminator(self):
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Flatten(),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

ПРОВЕРКА РАЗМЕРОВ
```python
def verify_shapes(self):
    print("\n📏 ПРОВЕРКА РАЗМЕРОВ:")
    print("=" * 60)
    
    test_noise = tf.random.normal([1, self.latent_dim])
    gen_output = self.generator(test_noise, training=False)
    
    print(f"✓ Generator input:  (1, {self.latent_dim})")
    print(f"✓ Generator output: {gen_output.shape}")
    
    if gen_output.shape != (1, 28, 28, 1):
        print(f"❌ ОШИБКА! Generator должен выводить (1, 28, 28, 1)")
        return False
    
    disc_output = self.discriminator(gen_output, training=False)
    
    print(f"✓ Discriminator input:  {gen_output.shape}")
    print(f"✓ Discriminator output: {disc_output.shape}")
    
    if disc_output.shape != (1, 1):
        print(f"❌ ОШИБКА! Discriminator должен выводить (1, 1)")
        return False
    
    print("=" * 60)
    print("✅ ВСЕ РАЗМЕРЫ ПРАВИЛЬНЫЕ!\n")
    return True
```

ОДИН ШАГ ОБУЧЕНИЯ
```python
@tf.function
def train_step(self, real_images):
    batch_size = tf.shape(real_images)[0]
    
    with tf.GradientTape() as tape:
        noise = tf.random.normal([batch_size, self.latent_dim])
        fake_images = self.generator(noise, training=True)
        
        real_predictions = self.discriminator(real_images, training=True)
        fake_predictions = self.discriminator(fake_images, training=True)
        
        real_loss = self.loss_fn(tf.ones_like(real_predictions), real_predictions)
        fake_loss = self.loss_fn(tf.zeros_like(fake_predictions), fake_predictions)
        d_loss = real_loss + fake_loss
    
    d_gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(
        zip(d_gradients, self.discriminator.trainable_weights)
    )
    
    with tf.GradientTape() as tape:
        noise = tf.random.normal([batch_size, self.latent_dim])
        fake_images = self.generator(noise, training=True)
        fake_predictions = self.discriminator(fake_images, training=True)
        g_loss = self.loss_fn(tf.ones_like(fake_predictions), fake_predictions)
    
    g_gradients = tape.gradient(g_loss, self.generator.trainable_weights)
    self.g_optimizer.apply_gradients(
        zip(g_gradients, self.generator.trainable_weights)
    )
    
    return d_loss, g_loss
```

ЦИКЛ ОБУЧЕНИЯ
```python
def train(self, X_train, epochs=50, batch_size=128):
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)
    
    print("=" * 70)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ GAN")
    print("=" * 70)
    print(f"📊 Параметры:")
    print(f"   • Эпохи: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Датасет: {len(X_train)} изображений")
    print("=" * 70)
    
    for epoch in range(epochs):
        epoch_d_loss = []
        epoch_g_loss = []
        
        for real_images in train_dataset:
            d_loss, g_loss = self.train_step(real_images)
            epoch_d_loss.append(float(d_loss))
            epoch_g_loss.append(float(g_loss))
        
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        
        self.d_losses.append(avg_d_loss)
        self.g_losses.append(avg_g_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
    
    print("=" * 70)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
```

ГЕНЕРИРОВАНИЕ ИЗОБРАЖЕНИЙ
```python
def generate_images(self, num_images=10):
    noise = tf.random.normal([num_images, self.latent_dim])
    return self.generator(noise, training=False)
```

ЭКСПОРТ РЕЗУЛЬТАТОВ
```python
def export_results(self, filename='gan_results.json'):
    results = {
        'epochs': len(self.d_losses),
        'd_losses': [float(x) for x in self.d_losses],
        'g_losses': [float(x) for x in self.g_losses],
        'learning_rate': 0.0002,
        'batch_size': 128,
        'dataset_size': 10000,
        'timestamp': datetime.now().isoformat()
    }
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Результаты сохранены в {filename}")
```

СОХРАНЕНИЕ МОДЕЛЕЙ
```python
def save_models(self):
    self.generator.save('generator.h5')
    self.discriminator.save('discriminator.h5')
    print("✅ Модели сохранены: generator.h5, discriminator.h5")
```

ГРАФИК ПОТЕРЬ
```python
def plot_losses(d_losses, g_losses):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GAN Training Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(d_losses, label='Discriminator Loss', color='#FF6B6B', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Discriminator Loss (Raw)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    ax = axes[0, 1]
    ax.plot(g_losses, label='Generator Loss', color='#4ECDC4', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Generator Loss (Raw)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    ax = axes[1, 0]
    window = max(3, len(d_losses) // 10)
    d_smooth = pd.Series(d_losses).rolling(window=window, center=True).mean()
    g_smooth = pd.Series(g_losses).rolling(window=window, center=True).mean()
    
    ax.plot(d_smooth, label='D Loss (Smoothed)', color='#FF6B6B', linewidth=2.5)
    ax.plot(g_smooth, label='G Loss (Smoothed)', color='#4ECDC4', linewidth=2.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Both Losses (Smoothed)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    ax = axes[1, 1]
    ax.axis('off')
    
    d_improvement = ((d_losses[-1] - d_losses[0]) / d_losses[0] * 100)
    g_improvement = ((g_losses[-1] - g_losses[0]) / g_losses[0] * 100)
    
    stats_text = f"""
📊 TRAINING STATISTICS

Discriminator Loss:
  • Initial: {d_losses[0]:.4f}
  • Final:   {d_losses[-1]:.4f}
  • Average: {sum(d_losses)/len(d_losses):.4f}

Generator Loss:
  • Initial: {g_losses[0]:.4f}
  • Final:   {g_losses[-1]:.4f}
  • Improvement: {g_improvement:+.1f}%"""
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('gan_training_loss.png', dpi=150, bbox_inches='tight')
    print("✅ Loss graph saved to gan_training_loss.png")
    plt.show()
```

ГРАФИК ОБРАЗЦОВ
```python
def plot_generated_samples(gan, num_samples=16):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Generated MNIST Digits', fontsize=14, fontweight='bold')
    
    generated = gan.generate_images(num_samples)
    generated = (generated.numpy() + 1) / 2
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=150, bbox_inches='tight')
    print("✅ Generated samples saved to generated_samples.png")
    plt.show()
```

ЗАГРУЗКА ДАННЫХ
```python
def load_and_preprocess_mnist():
    print("📊 Загрузка MNIST датасета...")
    (X_train, _), _ = keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32) / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=-1)
    print(f"✅ Данные загружены: {X_train.shape}\n")
    return X_train
```

ГЛАВНАЯ ПРОГРАММА
```python
if __name__ == "__main__":
    
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "🤖 GAN для MNIST с ГРАФИКАМИ! 🤖" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    X_train = load_and_preprocess_mnist()
    
    print("🏗️ Создание GAN...")
    gan = GAN(latent_dim=100)
    
    if not gan.verify_shapes():
        print("❌ ОШИБКА В АРХИТЕКТУРЕ!")
        exit(1)
    
    print("🚀 Начинаем обучение...\n")
    gan.train(X_train[:10000], epochs=50, batch_size=128)
    
    print("\n💾 Сохранение результатов...")
    gan.export_results('gan_results.json')
    gan.save_models()
    
    print("\n" + "=" * 70)
    print("📈 ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 70)
    
    print("\n1️⃣ Графики потерь обучения...")
    plot_losses(gan.d_losses, gan.g_losses)
    
    print("\n2️⃣ Сгенерированные цифры...")
    plot_generated_samples(gan, num_samples=16)
    
    print("\n" + "=" * 70)
    print("✅ ОБУЧЕНИЕ И ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНЫ!")
    print("=" * 70)
    print("\n📁 Созданные файлы:")
    print("  ✓ gan_results.json - результаты обучения")
    print("  ✓ gan_training_loss.png - графики потерь 📊")
    print("  ✓ generated_samples.png - сгенерированные цифры 🎨")
    print("  ✓ generator.h5 - модель генератора")
    print("  ✓ discriminator.h5 - модель дискриминатора")
    print("\n✨ Загрузите gan_results.json в GAN_Browser_App.html для веб-визуализации!")
    print("=" * 70)
```
 
# Ответ на контрольный вопрос номер 8

### Опишите алгоритм Graham Scan для построения выпуклой оболочки. Какова его временная сложность?

### Graham Scan 
— классический геометрический алгоритм для построения минимальной выпуклой оболочки конечного множества точек на плоскости. Основан на идее "обхода" точек в порядке возрастания полярного угла относительно крайней точки.

Шаги:

Найти точку P0 с минимальной y (при равных y — минимальную x)

Остальные точки отсортировать по полярному углу относительно P0

Поместить P0 и первые 2 точки в стек

Для каждой следующей точки:

Пока 3 последние точки стека образуют не левый поворот (векторное произведение ≤ 0)

Удалить предпоследнюю точку

Добавить текущую точку

Сложность: O(n log n) 

