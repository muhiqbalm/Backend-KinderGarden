import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from flask import request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
CORS(app)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/


@app.route('/', methods=['GET'])
def index():
    return "gege!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.methods == 'POST':
        # Get the file from post request
        f = request.body['file']
        print(f)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                         'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        return result
    return None

@app.route("/data", methods=["GET", "POST"])
# @jit
def main():
    model = load_model("./models")
    print('Model loaded. Check http://127.0.0.1:5000/')

    def model_predict(img_path, model):
        img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
        show_img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = np.array(x, 'float32')
        x /= 255
        preds = model.predict(x)
        return preds

    test = "test.JPG"
    print("cek1")

    file = request.files['file']
    file.save('uploaded_image.jpg')

    preds = model_predict('uploaded_image.jpg', model)
    print(preds[0])

    disease_class = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                         'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    
    a = preds[0]
    ind=np.argmax(a)
    print('Prediction:', disease_class[ind])
    result=disease_class[ind]
    confidence = a[ind] * 100  # Menghitung confidence dalam persentase
    return {"predict": result, "confidence": confidence}


@app.route("/upload", methods=["POST"])
def handle_upload():
    try:
        model = load_model("./models")
        print('Model loaded. Check http://127.0.0.1:5000/')

        def model_predict(img_path, model):
            img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
            show_img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = np.array(x, 'float32')
            x /= 255
            preds = model.predict(x)
            return preds
    
        file = request.files['file']
        file.save('uploaded_image.jpg')  # Simpan file gambar di server

        # Lakukan prediksi menggunakan file gambar yang diunggah
        preds = model_predict('uploaded_image.jpg', model)
        print(preds[0])

        disease_class = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                         'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    
        ind = np.argmax(preds[0])
        prediction = disease_class[ind]
        return {"predict": prediction}
    except Exception as e:
        print(str(e))
        return {"error": "Failed to process the uploaded image."}

if __name__ == '_main_':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()