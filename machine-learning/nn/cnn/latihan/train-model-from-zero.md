---
description: >-
  Kali ini kita akan mencoba train model dari 0, dengan model CNN untuk image
  classification
---

# Train Model From Zero

### Load Library

```python
import numpy as np
import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
```

### Load Dataset Horse vs Human

```python
tfds.disable_progress_bar()

train_ds, validation_ds, test_ds = tfds.load(
    # Load dataset horses_or_humans
    "horses_or_humans",
    # Membagi dataset menjadi 3 category, 80% training, 10% validation, 10% test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    # Label akan disertakan ketika load datase
    as_supervised=True,
)
print(f"Number of training samples: {train_ds.cardinality()}")
print(f"Number of validation samples: {validation_ds.cardinality()}")
print(f"Number of test samples: {test_ds.cardinality()}")

# Output jumlah pembagian dataset
# Number of training samples: 411
# Number of validation samples: 103
# Number of test samples: 102
```

### Menampilkan Contoh Sample Dataset

```python
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")
```

<figure><img src="../../../../.gitbook/assets/image.png" alt=""><figcaption><p>Sample Dataset</p></figcaption></figure>

### Resize Dataset

```python
# Declare resize image
# Hal ini dilakukan agar ketika proses training model, 
# dataset image memiliki size atau ukuran yang sama
resize_fn = keras.layers.Resizing(150, 150)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))
```

### Augumentasi Image

```python
# Augumentasi image dilakukan agar dataset memiliki variasi gambar yang berbeda, 
# augumentasi ini sangat berguna, apabila dataset yang dimiliki sangat sedikit.
# dengan augumentasi ini diharapkan agar ketika training berjalan dapat mengenal banyak kondisi dataset
# augumentasi ini hanya dilakukan di dataset train saja, untuk di validasi serta test tidak dilakukan augumentasi image
augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x
    
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
```

### Tuning Dataset

```python
from tensorflow import data as tf_data

batch_size = 64

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
```

### Menampilkan Hasil Augumentasi Image

```python
for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(np.expand_dims(first_image, 0))
        plt.imshow(np.array(augmented_image[0]).astype("int32"))
        plt.title(int(labels[0]))
        plt.axis("off")
```

<figure><img src="../../../../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

### Membaungun Sebuah Model&#x20;

<pre class="language-python"><code class="lang-python"># Input dataset yang akan ditraining disesuaikan dengan ukuran image pada datase
inputs = keras.Input(shape=(150, 150, 3))
# Conv2D ini dilakukan untuk ektraksi fitur
# CONV2D yang bekerja dengan data dua dimensi, seperti gambar hitam putih atau satu saluran warna. 
# Jenis ini adalah konvolusi yang paling sering digunakan dalam pengolahan gambar. 
# CONV2D dapat membantu kita mengidentifikasi fitur visual dalam gambar, 
# seperti garis tepi (edges), sudut, atau pola yang lebih kompleks.

# Setiap layer Conv2D menggunakan ReLU (Rectified Linear Unit) sebagai fungsi aktivasi,
# yang membantu memperkenalkan non-linearitas dalam model.

# Padding 'same' menentukan bahwa ukuran output dari setiap layer konvolusi sama dengan ukuran input sehingga tidak ada informasi yang hilang.
x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
# ayer Batch Normalization. Layer ini membantu mempercepat proses pelatihan dan 
# membuat proses lebih stabil dengan menormalisasi output dari layer sebelumnya.
<strong>x = keras.layers.BatchNormalization()(x)
</strong># Max Pooling Layer untuk mengurangi dimensi spasial dari setiap feature map.
# Parameter (2, 2) menentukan ukuran jendela pooling, yang dalam hal ini adalah 2 × 2 piksel.
# Max pooling mengambil nilai maksimum dari setiap jendela pooling untuk mengurangi ukuran feature map dan membuat representasi fitur lebih invarian terhadap translasi kecil.
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(32, 4, padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Dense(32, activation='relu')(x)
# Flatten Layer untuk mengubah output dari layer-layer sebelumnya menjadi bentuk vektor satu dimensi.
# Ini diperlukan karena layer-layer dense (fully connected) membutuhkan input berupa vektor, bukan matriks atau tensor.
x = keras.layers.Flatten()(x)
# kita menambahkan layer Dense (fully connected) dengan 32 neuron.
x = keras.layers.Dense(64, activation='relu')(x)
# Penambahan dropout dimaksudkan agar mengurangi overfitting, 
# dengan cara mengabaikan sejumlah neuron, value 0.1 berarti 10% dari total neuron 
# akan dinonaktifkan secara acak selama proses training
x = keras.layers.Dropout(0.1)(x)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
# Layer Fully Connected, memiliki 2 output neuron, dengan fungsi activasi softmax,
# Untuk tugas classification
outputs = keras.layers.Dense(2, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.summary()
</code></pre>

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_15 (InputLayer)     │ (None, 150, 150, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_31 (Conv2D)              │ (None, 150, 150, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_30          │ (None, 150, 150, 32)   │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_33 (MaxPooling2D) │ (None, 75, 75, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_32 (Conv2D)              │ (None, 75, 75, 32)     │        16,416 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_31          │ (None, 75, 75, 32)     │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_34 (MaxPooling2D) │ (None, 37, 37, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_15 (Flatten)            │ (None, 43808)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_39 (Dense)                │ (None, 64)             │     2,803,776 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_21 (Dropout)            │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_40 (Dense)                │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_22 (Dropout)            │ (None, 32)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_41 (Dense)                │ (None, 2)              │            66 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

### Manambahkan Fungsi Callback Serta Early Stopper Mencegah Overfitting

```python
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Ensure logs is not None and contains accuracy metrics
        if logs and logs.get('accuracy') >= 0.98 and logs.get('val_accuracy') >= 0.98:
            print("\nAkurasi telah mencapai >=98%!")
            self.model.stop_training = True

# Instantiate the callback
callbacks = MyCallback()

early_stopping = EarlyStopping(
    monitor='accuracy',  # Monitor the validation loss
    patience=10,          # Stop after 5 epochs with no improvement
    restore_best_weights=True  # Restore the best weights
)
```

### Compile Serta Trining Model

```python
model.compile(
    # Fungsi optimize menggunkana adam
    optimizer=keras.optimizers.Adam(),
    # Menambahkan fungsi loss SparseCategoricalCrossentropy untuk classification object 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

epochs = 70
print("Fitting the top layer of the model")
model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[callbacks, early_stopping])
```

<figure><img src="../../../../.gitbook/assets/image (4).png" alt=""><figcaption></figcaption></figure>

### Menmpilkan Grafik Hasil Selama Training

```python
acc = histor_model.history['accuracy']
val_acc = histor_model.history['val_accuracy']
loss = histor_model.history['loss']
val_loss = histor_model.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
 
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title('Training and Validaion Loss')
plt.show()
```

<figure><img src="../../../../.gitbook/assets/image (5).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../../.gitbook/assets/image (6).png" alt=""><figcaption></figcaption></figure>

### Evaluasi Model

```python
print("Test dataset evaluation")
model.evaluate(test_ds)
# Output dari hasil evaluate, model memiliki akurasi 98% dan loss 17%
```

<figure><img src="../../../../.gitbook/assets/image (7).png" alt=""><figcaption></figcaption></figure>
