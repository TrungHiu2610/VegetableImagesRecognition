# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
        
        # Load images for the ith category
        image_path = train_path + '/' + cat
        images_in_folder = os.listdir(image_path)
        first_image_of_folder = images_in_folder[0]
        first_image_path = image_path + '/' + first_image_of_folder
        img = image.load_img(first_image_path)
        img_arr = image.img_to_array(img)/255.0
        
        
        # Create Subplot and plot the images
        plt.subplot(4, 4, i+1)
        plt.imshow(img_arr)
        plt.title(cat)
        plt.axis('off')
        
    plt.show()

#show 1 vài ảnh trong dataset
plot_figures(image_categories)

# Prepare the Dataset

# 1. Train Set
train_gen = ImageDataGenerator(rescale = 1.0/255.0) # Normalise the data
train_image_generator = train_gen.flow_from_directory(
                                            train_path,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='categorical')

# 2. Validation Set
val_gen = ImageDataGenerator(rescale = 1.0/255.0) # Normalise the data
val_image_generator = train_gen.flow_from_directory(
                                            validation_path,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='categorical')

# 3. Test Set
test_gen = ImageDataGenerator(rescale = 1.0/255.0) # Normalise the data
test_image_generator = train_gen.flow_from_directory(
                                            test_path,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='categorical')

class_map = dict([(v, k) for k, v in train_image_generator.class_indices.items()])
print(class_map)

# class_map = {0: 'Bean', 
#              1: 'Bitter_Gourd', 
#              2: 'Bottle_Gourd', 
#              3: 'Brinjal', 
#              4: 'Broccoli', 
#              5: 'Cabbage', 
#              6: 'Capsicum', 
#              7: 'Carrot', 
#              8: 'Cauliflower',
#              9: 'Cucumber', 
#              10: 'Papaya', 
#              11: 'Potato', 
#              12: 'Pumpkin', 
#              13: 'Radish', 
#              14: 'Tomato'}

# Build mô hình CNN

model = Sequential() 

model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=[150, 150, 3]))
model.add(MaxPooling2D(2, ))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(2))

# Flatten 
model.add(Flatten())

# Add the fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(15, activation='softmax'))

model.summary()

# Train model
early_stopping = keras.callbacks.EarlyStopping(patience=5) # Set up callbacks
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_image_generator, 
                 epochs=100, 
                 verbose=1, 
                 validation_data=val_image_generator, 
                 steps_per_epoch = 15000//32, 
                 validation_steps = 3000//32, 
                 callbacks=early_stopping)

# Bieu do loss va acc
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

# Danh gia mo hinh
model.evaluate(test_image_generator)

# Save file model
model.save('../SavedModels/model_manyParam.h5')

#Load file model
model = load_model('../SavedModels/model_manyParam.h5')

#Ham danh gia du lieu thuc te
def generate_predictions_subplot(folder_actual_test_path):
    folder_actual_test = os.listdir(folder_actual_test_path)

    # Tính số lượng ảnh (số thư mục)
    num_images = len(folder_actual_test)
    cols = 4  # số cột trong plot
    rows = (num_images + cols - 1) // cols  

    plt.figure(figsize=(4 * cols, 4 * rows))

    for idx, sub_folder in enumerate(folder_actual_test):
        sub_folder_path = os.path.join(folder_actual_test_path, sub_folder)
        list_images_folder = os.listdir(sub_folder_path)

        # Bỏ qua thư mục rỗng nếu có
        if not list_images_folder:
            continue

        # Lấy ảnh random
        ind_ran = np.random.randint(0,len(list_images_folder))
        first_image_of_folder = list_images_folder[ind_ran] 
        first_image_path = os.path.join(sub_folder_path, first_image_of_folder)

        # Load ảnh và xử lý
        test_img = image.load_img(first_image_path, target_size=(150, 150))
        test_img_arr = image.img_to_array(test_img) / 255.0
        test_img_input = np.expand_dims(test_img_arr, axis=0)

        # Dự đoán
        predicted_label = np.argmax(model.predict(test_img_input))
        predicted_vegetable = class_map[predicted_label]

        # Vẽ subplot
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(test_img_arr)
        color = 'green' if predicted_vegetable == sub_folder else 'red'
        plt.title(f"Predict: {predicted_vegetable}\nActual: {sub_folder}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Gọi hàm
folder_actual_test_path = '../Data/ActualData/validation'
generate_predictions_subplot(folder_actual_test_path)
