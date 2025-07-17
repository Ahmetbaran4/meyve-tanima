import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Ayarlar
image_size = (224, 224)  # Daha büyük çözünürlük
batch_size = 32
epochs = 50
dropout_rate = 0.4

# CSV'den oku
df = pd.read_csv("veriseti.csv")  # içinde: filepath,label
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Veri artırma ve ön işleme
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Eğitim veri jeneratörü
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Doğrulama veri jeneratörü
val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filepath",
    y_col="label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# MobilNetV2 transfer learning modeli
base_model = MobileNetV2(input_shape=(image_size[0], image_size[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False  # İlk eğitimde dondur

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(dropout_rate)(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Derleme
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Erken durdurma
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Eğitim
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stop]
)

# Modeli kaydet (Keras formatında)
model.save("meyve_modeli.keras")

# Sınıf etiketlerini dosyaya kaydet
with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=4)

# --- Test sonuçları ---
val_generator.reset()
predictions = model.predict(val_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes
class_labels = list(train_generator.class_indices.keys())

print("\n🔍 Sınıflandırma Raporu:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")
plt.title("Karışıklık Matrisi")
plt.tight_layout()
plt.show()

# --- Accuracy Yüzdelik Olarak Yazdır ---
loss, accuracy = model.evaluate(val_generator, verbose=0)
accuracy_percent = accuracy * 100
print(f"\n✅ Doğruluk (Accuracy): %{accuracy_percent:.2f}")


print("Sınıf etiketleri:", train_generator.class_indices)
