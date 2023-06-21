from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']

        # Ensure a file was selected
        if file.filename == '':
            return 'No selected file'
        
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            img = image.load_img(file_path, target_size=(28, 28), color_mode="grayscale")
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img /= 255.0

            pred = model.predict(img)
            digit = np.argmax(pred)

            return render_template('result.html', digit=digit, image_file=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
