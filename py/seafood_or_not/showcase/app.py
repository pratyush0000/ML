from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('seafood_classifier_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to match training input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    prediction = model.predict(img_array)
    return 'Seafood' if prediction[0][0] > 0.5 else 'Normal Meat'  # Adjust based on your model's output

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def uploader_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join('static/uploaded_images', file.filename)
        file.save(file_path)
        prediction = predict_image(file_path)
        return render_template('result.html', prediction=prediction, img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
