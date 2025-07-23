# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import os, shutil
import warnings
warnings.filterwarnings('ignore')

# Trực quan hóa data
train_path = "../Data/Vegetable Images/train"
validation_path = "../Data/Vegetable Images/validation"
test_path = "../Data/Vegetable Images/test"

image_categories = os.listdir('../Data/Vegetable Images/train')

def plot_figures(image_categories):
    plt.figure(figsize=(12, 12))
    for i, cat in enumerate(image_categories):
        image_path = train_path + '/' + cat
        images_in_folder = os.listdir(image_path)
        first_image_of_folder = images_in_folder[0]
        first_image_path = image_path + '/' + first_image_of_folder
        img = image.load_img(first_image_path)
        img_arr = image.img_to_array(img)/255.0
        plt.subplot(4, 4, i+1)
        plt.imshow(img_arr)
        plt.title(cat)
        plt.axis('off')
    plt.show()

plot_figures(image_categories)

# Prepare the Dataset
# 1. Train Set với Data Augmentation
train_gen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_image_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# 2. Validation Set
val_gen = ImageDataGenerator(rescale=1.0/255.0)
val_image_generator = val_gen.flow_from_directory(
    validation_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# 3. Test Set
test_gen = ImageDataGenerator(rescale=1.0/255.0)
test_image_generator = test_gen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

class_map = dict([(v, k) for k, v in train_image_generator.class_indices.items()])
print(class_map)

# Build mô hình CNN
model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=[150, 150, 3]))
model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(15, activation='softmax'))

model.summary()

# Train model
early_stopping = keras.callbacks.EarlyStopping(patience=5)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(
    train_image_generator,
    epochs=100,
    verbose=1,
    validation_data=val_image_generator,
    steps_per_epoch=15000//32,
    validation_steps=3000//32,
    callbacks=[early_stopping]
)

# Biểu đồ loss và accuracy
h = hist.history
plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
plt.plot(h['loss'], c='red', label='Training Loss')
plt.plot(h['val_loss'], c='red', linestyle='--', label='Validation Loss')
plt.plot(h['accuracy'], c='blue', label='Training Accuracy')
plt.plot(h['val_accuracy'], c='blue', linestyle='--', label='Validation Accuracy')
plt.xlabel("Number of Epochs")
plt.legend(loc='best')
plt.show()

# Đánh giá mô hình
print("Evaluation on Test Set (Loss and Accuracy):")
model.evaluate(test_image_generator)

# Đánh giá precision, recall, f1-score
y_true = []
y_pred = []
test_image_generator.reset()
steps = len(test_image_generator)
for i in range(steps):
    images, labels = next(test_image_generator)
    predictions = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# Lấy classification report dưới dạng dictionary
report = classification_report(y_true, y_pred, target_names=list(class_map.values()), output_dict=True)

# Trích xuất precision, recall, f1-score cho từng lớp
labels = list(class_map.values())
precision = [report[label]['precision'] for label in labels]
recall = [report[label]['recall'] for label in labels]
f1_score = [report[label]['f1-score'] for label in labels]

# In báo cáo dạng text
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(class_map.values())))

# Vẽ biểu đồ cột bằng Matplotlib
plt.figure(figsize=(15, 6))
x = np.arange(len(labels))
width = 0.25  # Độ rộng cột

plt.bar(x - width, precision, width, label='Precision', color='cyan')
plt.bar(x, recall, width, label='Recall', color='pink')
plt.bar(x + width, f1_score, width, label='F1-Score', color='skyblue')

plt.xlabel('Classes')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1-Score by Class')
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylim(0, 1)  # Giới hạn trục y từ 0 đến 1
plt.legend()
plt.tight_layout()
plt.show()

# Save file model
model.save('../SavedModels/model_manyParam.h5')

# Load file model
model = models.load_model('../SavedModels/model_manyParam.h5')

# Hàm đánh giá dữ liệu thực tế
def generate_predictions_subplot(folder_actual_test_path):
    folder_actual_test = os.listdir(folder_actual_test_path)
    num_images = len(folder_actual_test)
    cols = 4
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, sub_folder in enumerate(folder_actual_test):
        sub_folder_path = os.path.join(folder_actual_test_path, sub_folder)
        list_images_folder = os.listdir(sub_folder_path)
        if not list_images_folder:
            continue
        ind_ran = np.random.randint(0, len(list_images_folder))
        first_image_of_folder = list_images_folder[ind_ran]
        first_image_path = os.path.join(sub_folder_path, first_image_of_folder)
        test_img = image.load_img(first_image_path, target_size=(150, 150))
        test_img_arr = image.img_to_array(test_img) / 255.0
        test_img_input = np.expand_dims(test_img_arr, axis=0)
        predicted_label = np.argmax(model.predict(test_img_input))
        predicted_vegetable = class_map[predicted_label]
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(test_img_arr)
        color = 'green' if predicted_vegetable == sub_folder else 'red'
        plt.title(f"Predict: {predicted_vegetable}\nActual: {sub_folder}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

folder_actual_test_path = '../Data/ActualData/validation'
generate_predictions_subplot(folder_actual_test_path)