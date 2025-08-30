from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model import build_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = build_model()
model.load_weights('model_anemia.h5')  # Optional

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_class(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img, verbose=0)
    classes = ['Anemic', 'Non-Anemic']
    return classes[int(prediction.round())]

@app.route("/", methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            result = predict_class(filepath)
            return render_template("index.html", result=result, image_path=filepath)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
