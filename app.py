import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cấu hình Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load mô hình
model = load_model('SavedModels/model_manyParam.h5')

# Danh sách class (đặt đúng thứ tự label)
class_map = {
    0: 'Bean', 
    1: 'Bitter_Gourd', 
    2: 'Bottle_Gourd', 
    3: 'Brinjal', 
    4: 'Broccoli', 
    5: 'Cabbage', 
    6: 'Capsicum', 
    7: 'Carrot', 
    8: 'Cauliflower',
    9: 'Cucumber', 
    10: 'Papaya', 
    11: 'Potato', 
    12: 'Pumpkin', 
    13: 'Radish', 
    14: 'Tomato'
}


# Hàm xử lý ảnh
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_map[np.argmax(predictions)]
    return predicted_class

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Xử lý upload ảnh
@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    results = []

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)

            label = predict_image(temp_path)

            # Tạo thư mục theo tên label nếu chưa có
            label_folder = os.path.join(app.config['UPLOAD_FOLDER'], label)
            os.makedirs(label_folder, exist_ok=True)

            # Lưu file vào folder tương ứng
            final_path = os.path.join(label_folder, filename)
            os.rename(temp_path, final_path)

            results.append((filename, label, final_path))

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
