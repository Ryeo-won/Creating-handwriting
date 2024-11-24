import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 폴더 아래 있는 모든 이미지를 불러오는 함수 정의
def load_images_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img = Image.open(os.path.join(root, filename)).convert('L')
                img = img.resize((28, 28))
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=-1)
                images.append(img)
    return np.array(images).astype(np.float32)

# 폴더에서 이미지 불러오기
img_folder = './myletter'
X_train = load_images_from_folder(img_folder)

# 랜덤 시드 설정
np.random.seed(42)
tf.random.set_seed(42)

# 생성자
codings_size = 100
generator = keras.models.Sequential([
    keras.layers.Dense(7 * 7 * 256, input_shape=[codings_size]),
    keras.layers.Reshape([7, 7, 256]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(128, kernel_size=5, strides=1, padding="SAME", activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME", activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding="SAME", activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=1, padding="SAME", activation="tanh"),
])

# 판별자
discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME", activation=keras.layers.LeakyReLU(0.2), input_shape=[28, 28, 1]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME", activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(256, kernel_size=5, strides=2, padding="SAME", activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])

# GAN 모델 정의
gan = keras.models.Sequential([generator, discriminator])

# 판별자 컴파일
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])  # 정확도(metrics) 추가
discriminator.trainable = False

# GAN 컴파일
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

# 데이터셋 생성
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

# 이미지 플롯 함수 정의
def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

# GAN 학습 함수 정의
def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    global generated_images
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        for X_batch in dataset:
            X_batch = tf.cast(X_batch, tf.float32)
            # 판별자 학습 단계
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            d_loss, d_acc = discriminator.train_on_batch(X_fake_and_real, y1)
            # 생성자 학습 단계
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

    plot_multiple_images(generated_images, 8)
    plt.show()

# GAN 학습
train_gan(gan, dataset, batch_size, codings_size, n_epochs=200)
