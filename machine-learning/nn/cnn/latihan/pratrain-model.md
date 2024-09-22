---
description: >-
  Contoh latihan pretrain model implement CNN dan Xception model untuk pretrain,
  dataset terdiri dari 2 categori dalam 1 dataset, human dan horse.
---

# Pretrain Model

Deklarasi library yang digunakan untuk pretrain model

```python
import numpy as np
import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
```

Load dataset horses dan human, dataset dibagi menjadi 3 antara lain:

* Train -> Digunakan dalam proses pelatihan model. Dengan banyak variasi gambar , model dapat mempelajari berbagai ekstraksi fitur."
* Validation -> Digunakan ketika proses training berjalan, setiap sekali iterasi setelah training berjalan akan melakukan evaluasi model
* Test -> Digunakan ketika pelatihan model telah selesai, untuk mengukur akurasi model kembali dalam mengenal object gambar

Untuk persentase training-nya dibagi jadi 3, 80% untuk Training, 10% untuk validasi, 10% untuk test.

```python
tfds.disable_progress_bar()

train_ds, validation_ds, test_ds = tfds.load(
    "horses_or_humans",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
) # Only keep examples with label < 2

print(f"Number of training samples: {train_ds.cardinality()}")
print(f"Number of validation samples: {validation_ds.cardinality()}")
print(f"Number of test samples: {test_ds.cardinality()}")
```

Menampilkan sample gambar berserta label dalam suatu dataset.\
Untuk horse dilabelkan dengan angka 0 serta human dilabelkan dalam angka 1.

```python
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")
```

<figure><img src="../../../../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

Selanjutnya resize tinggi lebar image, tinggi lebar image ini perlu sesuai dengan value input shape ketika meracik komponen untuk train model.

```python
resize_fn = keras.layers.Resizing(150, 150)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))
```

Selanjutnya image augumentasi, dengan image augumentasi ini diharapkan menambah beberapa variasi image, sehingga model dapat berlatih dengan variasi gambar dengan kondisi berbeda-beda.

Image augumentasina sangat berguna sekali ketika kita memiliki jumlah dataset yang sangat sedikit.

```python
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

Menampilkan sample hasil augumentasi image

<figure><img src="../../../../.gitbook/assets/image (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

Set batch size serta optimize loading speed ketika pelatihan model berjalan

```python
from tensorflow import data as tf_data

batch_size = 64

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
```

Selanjutnya load model Xception serta beberapa parameter yang perlu diisi seperti input\_shape dengan value (150, 150, 3) angka 150, 150 berarti proses pelatihan model memilik requirement ukuran image tinggi = 150, lebar = 150, serta 3 mewaliki jenis gambar-nya, untuk value = 3 berarti gambar tersebut jenisinya RGB.

Ada beberpa keriteria lain yang perlu diperhatikan antara lain

* Freeze base model dengan memberi value \*base\_model.trainable = False\* hal ini agar ketika pelatihan model berlangsung, base model yang telah dilatih tidak dilatih kembali, karena tujuan pretrain model sendiri yaitu menggunakan base model yang telah mengerti cara belajar mengenali object, digunakan kembali untuk melatih model yang baru dalam mempelajari object yang baru, biasanya yang dilatih pada pretrain ini yaitu hanya output-nya saja.\
  Jika diilustrasikan dalam kehidupan sehari-hari misalnya bayi telah sukses belajar berjalan maka untuk belajar berlari perlu mengerti konsep berjalan dulu yang digunakan untuk belajar berlari dengan hanya menambah speed berjalan dengan lebih cepat.
* Manambah layer output untuk melatih model dalam mengklasifikasi gambar.

```python
base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(2, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.summary(show_trainable=True)
```

Berikut hasil execute code diatas, menghasilkan parameter model yang akan di training.

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃ Layer (type)                ┃ Output Shape          ┃    Param # ┃ Trai… ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│ input_layer_7 (InputLayer)  │ (None, 150, 150, 3)   │          0 │   -   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ rescaling_3 (Rescaling)     │ (None, 150, 150, 3)   │          0 │   -   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ xception (Functional)       │ (None, 5, 5, 2048)    │ 20,861,480 │   N   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ global_average_pooling2d_3  │ (None, 2048)          │          0 │   -   │
│ (GlobalAveragePooling2D)    │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dropout_3 (Dropout)         │ (None, 2048)          │          0 │   -   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dense_3 (Dense)             │ (None, 2)             │      4,098 │   Y   │
└─────────────────────────────┴───────────────────────┴────────────┴───────┘
```

Setalah setting merancang model untuk di pretrain, langkah selanjutnya, membuat beberapa fungsi untuk mengevaluasi model serta untuk menghentikan proses train model dengan ketentuan yang dibuat, fungsi tersebut memiliki tujuan mencegah overfitting serta jika kondisi model terpenuhi karena overfitting atau karena lain hal, maka train model akan dihentikan serta akan merollback ke versi model terbaik selama training berlangsung.

```python
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Ensure logs is not None and contains accuracy metrics
        if logs and logs.get('accuracy') >= 0.95 and logs.get('val_accuracy') >= 0.95:
            print("\nAkurasi telah mencapai >=95%!")
            self.model.stop_training = True

# Instantiate the callback
callbacks = MyCallback()

early_stopping = EarlyStopping(
    monitor='accuracy',  # Monitor the validation loss
    patience=5,          # Stop after 5 epochs with no improvement
    restore_best_weights=True  # Restore the best weights
)
```

Setelah semua configurasi pretrain model telah dilakukan, langkah berikutnya memanggil fungsi compile, beberapa configurasi tambahan diperlukan hal tersebut untuk meningkatkan optimasi model, mencegah overfitting antara lain penambahan fungsi optimizer, loss.

```python
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
```

Selanjutnya memulai training model dengan mengexecute fungsi fit, serta memberi nilai epoch sebagai nilai untuk jumlah iterasi training model

```
epochs = 15
print("Fitting the top layer of the model")
history_pretrain = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[callbacks, early_stopping])
```

Dari hasil evaluasi terlihat model telah mencapai akurasi model serta validasi akurasi lebih dari 95%

```
Epoch 1/15
7/7 ━━━━━━━━━━━━━━━━━━━━ 30s 4s/step - accuracy: 0.6723 - loss: 0.6647 - val_accuracy: 1.0000 - val_loss: 0.1282 
Epoch 2/15 
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.9652 - loss: 0.1627 
Akurasi telah mencapai >=95%! 
7/7 ━━━━━━━━━━━━━━━━━━━━ 21s 3s/step - accuracy: 0.9659 - loss: 0.1593 - val_accuracy: 1.0000 - val_loss: 0.0422
```

Setelah training dilakukan, langkah selanjutnya melakukan evaluasi model, hal tersebut dilakukan untut memastikan tingkat akurasi model yg lebih riil

```python
print("Test dataset evaluation")
model.evaluate(test_ds)
```

Hasil compile terlihat tingkat akurasi model sebesar 100%, atau model dalam mendeteksi semua test tidak ada yang keliru, kemudian loss dibawah 0.5%

```
Test dataset evaluation
2/2 ━━━━━━━━━━━━━━━━━━━━ 3s 1s/step - accuracy: 1.0000 - loss: 0.0418
[0.04351438209414482, 1.0]
```

### Fine Tuning Model

Fine tuning model dilakukan agar model tidak lupa dengan tugas selanjutnya, kalo diasumsikan jika model sudah bisa berjalan kemudia kita pretrain model agara model bisa berlari juga, setelah bisa berlari maka model tetep bisa berjalan juga.

Fine tuning model dilakukan dengan cara unfreeze semua layer pada base model, serta mengcompile ulang model

```
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary(show_trainable=True)

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

epochs = 1
print("Fitting the end-to-end model")
history_finetuning = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
```

```
Fitting the end-to-end model
7/7 ━━━━━━━━━━━━━━━━━━━━ 128s 13s/step - accuracy: 0.8045 - loss: 0.4751 - val_accuracy: 1.0000 - val_loss: 0.0343
```

Lakukan evaluasi model kembali

```python
print("Test dataset evaluation")
model.evaluate(test_ds)

y_pred = model.predict(validation_ds)
```

Test dataset evaluation \
2/2 ━━━━━━━━━━━━━━━━━━━━ 3s 1s/step - accuracy: 1.0000 - loss: 0.0342 \
2/2 ━━━━━━━━━━━━━━━━━━━━ 5s 2s/step

Hasil prediksi kemudian diubah menjadi label kelas biner (0 atau 1) berdasarkan _threshold_ 0.5. Setelah itu, dilakukan pencetakan matriks kebingungan (_confusion matrix_) yang menampilkan seberapa baik model dalam memprediksi setiap kelas.&#x20;

Selain itu, dilakukan juga pencetakan _classification report_ yang memberikan informasi lebih detail tentang kinerja model, termasuk presisi, recall, dan F1-Score untuk setiap kelas. Ini membantu dalam mengevaluasi secara komprehensif kinerja model pada dataset pengujian.

```python
preds_1 = y_pred.copy()

# convert a 2D array with one-hot encoded labels (like [[0., 1.], [0., 1.], [1., 0.]]) to a 1D array of class labels (like [1, 1, 0])
preds_1 = np.argmax(preds_1, axis=1)

# Get validation labels
y_true = np.concatenate([y for x, y in validation_ds], axis = 0)

# Print classification report
print(classification_report(y_true, preds_1, target_names=['horse', 'human']))

# Print Confusion Matrix
cm = pd.DataFrame(data=confusion_matrix(y_true, preds_1, labels=[0, 1]),index=['horse', 'human'], columns=["Horse", "Human"])
sns.heatmap(cm,annot=True,fmt="d")
```

Dari _classification report_ di dapat dilihat bahwa model memiliki **akurasi sebesar 100%**, yang berarti sebagian besar dari prediksinya benar. Namun, kita perlu melihat lebih dalam untuk memahami kinerja model secara lebih rinci.

Precision mengukur seberapa baik model dalam memprediksi positif secara benar. Untuk kelas "Horse", precisionnya adalah 100%, yang berarti dari semua kasus yang diprediksi sebagai "Horse", 100% di antaranya benar-benar "Horse". Untuk kelas "Human", precisionnya adalah 100%, yang berarti dari semua kasus yang diprediksi sebagai "Human", 100% di antaranya benar-benar "Human". Precision yang tinggi menunjukkan bahwa model cenderung memberikan sedikit _false positive._

_Recall_ mengukur seberapa baik model dalam menemukan semua kasus yang sebenarnya positif. Untuk kelas "Horse", recall-nya adalah 100%. Ini berarti dari semua kasus "Horse" yang sebenarnya, model berhasil mengidentifikasi 100% di antaranya. Untuk kelas "Human", recall-nya adalah 100%. Ini berarti dari semua kasus "Human" yang sebenarnya, model berhasil mengidentifikasi 100% di antaranya. Recall yang tinggi menunjukkan bahwa model cenderung tidak memberikan _false negative._

F1-score adalah rata-rata harmonik dari precision dan recall yang memberikan gambaran keseluruhan tentang kinerja model. F1-score untuk kelas "Horse" adalah 100%, sementara untuk kelas "Human" adalah 100%.

Hasil evaluasi menunjukkan bahwa model memiliki kinerja yang baik dalam membedakan antara gambar yang menunjukkan klasifikasi human dan horse. Namun, penting untuk dipahami bahwa interpretasi hasil evaluasi ini bisa berbeda tergantung pada kebutuhan spesifik aplikasi.

```
              precision    recall  f1-score   support

       horse       1.00      1.00      1.00        50
       human       1.00      1.00      1.00        53

    accuracy                           1.00       103
   macro avg       1.00      1.00      1.00       103
weighted avg       1.00      1.00      1.00       103
```

<figure><img src="../../../../.gitbook/assets/image (8).png" alt=""><figcaption></figcaption></figure>

### Referensi

{% embed url="https://keras.io/guides/transfer_learning/" %}

{% embed url="https://learning.oreilly.com/library/view/ai-and-machine/9781492078180/ch04.html#loading_specific_versions" %}

