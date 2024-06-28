from flask import Flask, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import subprocess
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import time

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='androzoo_model_binary.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize Flask app
app = Flask(__name__)

# Directory to save uploaded APK files and output images
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route("/")
def index():
    return "<h1>ClaMal App Demo!</h1>"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert APK to image
        image_path = convert_apk_to_image(file_path, app.config['OUTPUT_FOLDER'])

        # Classify the image
        result = classify_image(image_path)

        return jsonify({'result': result})

def convert_apk_to_image(apk_path, output_folder_path):
    # Call the provided Python script to convert APK to image
    command = f'python convert_apk_to_image.py {apk_path} {output_folder_path}'
    subprocess.run(command, shell=True)

    # Assuming the script saves the output image with '_Fused.jpg' suffix
    base_filename = os.path.basename(apk_path)
    # image_path = os.path.join(output_folder_path, base_filename.replace('.apk', '_Fused.jpg'))
    # Ensure the images are saved before reading them
    while not (os.path.exists(os.path.join(output_folder_path, base_filename, '_Fused.jpg'))):
        time.sleep(0.1)
    image_path = os.path.join(output_folder_path, base_filename, '_Fused.jpg')
    return image_path

def classify_image(image_path):
    # Prepare the input data
    input_data = image.load_img(image_path, target_size=(299, 299))
    input_data = image.img_to_array(input_data)
    input_data = np.expand_dims(input_data, axis = 0)

    # Set the input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the output tensor and process the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probability = output_data[0][0]

    # Interpret the result
    threshold = 0.5  # Common threshold for binary classification
    if probability > threshold:
        predicted_class = 1
    else:
        predicted_class = 0

    # Print detailed prediction
    class_names = ['Benign', 'Malware']
    probability = probability if (probability > 0.5) else (1 - probability)
    result = "This image most likely belongs to {} with a {:.2f}% confidence.".format(class_names[predicted_class], 100 * probability)
    print(result)

    return str(result)

    # # Load and preprocess the image
    # img = Image.open(image_path)
    # img = img.resize((224, 224))  # Resize image to match the input shape of your model
    # img_array = np.array(img) / 255.0  # Normalize the image
    # img = img.astype(np.float32)  # Convert to FLOAT32
    # img_array = np.expand_dims(img_array, axis=0)  # Create batch axis

    # # Set the tensor to point to the input data to be inferred
    # interpreter.set_tensor(input_details[0]['index'], img_array)

    # # Run the inference
    # interpreter.invoke()

    # # The function `get_tensor` returns a copy of the tensor data.
    # # Use `tensor()` in order to get a pointer to the tensor.
    # # output_data = interpreter.get_tensor(output_details[0]['index'])
    # # predicted_class = np.argmax(output_data[0])

    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # predicted_class_index = np.argmax(output_data)
    # predicted_class_confidence = np.max(output_data)

    # class_names = ['Benign', 'Malware']
    # result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[predicted_class_index], 100 * predicted_class_confidence)

    # # return str(predicted_class)
    # return str(result)

if __name__ == '__main__':
    app.run(debug=True)
