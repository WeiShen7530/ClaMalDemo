from flask import Flask, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import subprocess
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import time
from flask_cors import CORS  # Import CORS

from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from androguard.core.bytecodes.apk import APK
from skimage.feature import graycomatrix
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
import sys

from huggingface_hub import from_pretrained_keras
model = from_pretrained_keras("duypn99/uitObfAMC12Class")
infer = model.signatures['serving_default']

class_names = ['Mã độc quảng cáo', 'Mã độc ngân hàng', 'Ứng dụng lành tính', 'Mã độc', 'Mã độc quảng cáo bị làm rối mã nguồn', 'Mã độc ngân hàng bị làm rối mã nguồn', 'Ứng dụng lành tính bị làm rối mã nguồn', 'Mã độc bị làm rối mã nguồn', 'Mã độc rủi ro bị làm rối mã nguồn', 'Mã độc SMS bị làm rối mã nguồn', 'Mã độc rủi ro', 'Mã độc SMS']

def classify_image(image_path):
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Convert the image to a tensor
    input_tensor = tf.convert_to_tensor(test_image)

    # Prepare the input for the serving function (as a dictionary)
    input_data = {"inputs": input_tensor}

    # Make a prediction
    prediction = infer(**input_data)
    output_tensor = prediction['output_0']

    # Convert tensor to numpy array
    output_array = output_tensor.numpy()

    # Find the index of the class with the highest probability
    predicted_class_index = np.argmax(output_array, axis=-1)
    probability = output_array[0, predicted_class_index[0]]

    prediction = {
        "class": class_names[predicted_class_index[0]],
        "confidence": 100 * probability
    }

    return prediction

# # Load the TensorFlow Lite model
# interpreter = tf.lite.Interpreter(model_path='androzoo_model_binary.tflite')
# interpreter.allocate_tensors()

# # Get input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Directory to save uploaded APK files and output images
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route("/")
def home():
    # test_img_path = '0ae6dd4d3d7c4710ae726ff7c3d64441be44b83cf9aa0ef29ddad40a22d68f29_obfuscated.apk.jpg'
    # result = classify_image(test_img_path)

    # response = {
    #     "Dự đoán": result['class'],
    #     "Độ tự tin": f"{result['confidence']:.10f}%"
    # }

    # return response
    return "<h1>uitObfAMC - An Obfuscated Android Malware Classification System!</h1>"

@app.route("/about")
def about():
    return "<h1>Contact me: duypn@uit.edu.vn</h1>"

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

        response = {
            "Dự đoán": result['class'],
            "Độ tự tin": f"{result['confidence']:.10f}%"
        }

        return response
        # return jsonify({'result': result})

def compute_glmi(grayscale_image):
    grayscale_image_np = np.array(grayscale_image)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    combined_image = np.zeros((256, 256), dtype=np.float32)

    for i in range(len(angles)):
        glcm = graycomatrix(grayscale_image_np, distances, np.array([angles[i]]), levels=256, symmetric=True, normed=True)
        image_2D = np.squeeze(glcm)
        image_resized = resize(image_2D, (128, 128))
        row_start = (i // 2) * 128
        row_end = row_start + 128
        col_start = (i % 2) * 128
        col_end = col_start + 128
        combined_image[row_start:row_end, col_start:col_end] = image_resized

    return combined_image

def calculate_markov_matrix(byte_sequence):
    markov_matrix = np.zeros((256, 256), dtype=np.float32)
    row_col_pairs = np.column_stack((byte_sequence[:-1], byte_sequence[1:]))
    np.add.at(markov_matrix, (row_col_pairs[:, 0], row_col_pairs[:, 1]), 1)
    row_sums = markov_matrix.sum(axis=1)
    markov_matrix /= row_sums[:, np.newaxis]
    markov_matrix *= 255
    return markov_matrix.astype(np.uint8)

def calculate_entropy(block):
    value, counts = np.unique(block, return_counts=True)
    probabilities = counts / len(block)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

def process_apk(apk_path, output_folder_path):
    try:
        apk = APK(apk_path)
        bytecodes = apk.get_dex()
        byte_sequence = np.frombuffer(bytecodes, dtype=np.uint8)
        entropy_byte_sequence = bytearray(bytecodes)
        glmi_byte_sequence = np.frombuffer(bytecodes, dtype=np.uint8) # Because crop 256x256 for GLMI
        base_filename = os.path.basename(apk_path)

        # Calculate the GLMI image
        required_padding = 256 * 256 - len(glmi_byte_sequence)
        if (required_padding > 0):
            glmi_byte_sequence = np.pad(glmi_byte_sequence, (0, required_padding), mode='constant')
        size = 256
        byte_image = glmi_byte_sequence[:size*size].reshape(size, size)
        grayscale_image = Image.fromarray(byte_image, 'L')
        resized_image = grayscale_image.resize((128, 128))
        resized_image_path = os.path.join(output_folder_path, base_filename + '_Grayscale.jpg')
        resized_image.save(resized_image_path)
        resized_image = Image.open(resized_image_path)
        glmi_image = compute_glmi(resized_image)
        glmi_image_np = np.array(glmi_image)
        final_glmi_img = Image.fromarray(glmi_image_np, 'L')
        glmi_image_path = os.path.join(output_folder_path, base_filename + '_GLMI.jpg')
        final_glmi_img.save(glmi_image_path)

        # Calculate the Markov image
        markov_matrix = calculate_markov_matrix(byte_sequence)
        markov_image_path = os.path.join(output_folder_path, base_filename + '_Markov.jpg')
        Image.fromarray(markov_matrix, mode='L').save(markov_image_path)

        # Calculate the Entropy graph image
        block_size = 128
        num_blocks = math.ceil(len(entropy_byte_sequence) / block_size)
        padding = num_blocks * block_size - len(entropy_byte_sequence)
        padded = entropy_byte_sequence + bytearray(padding)
        blocks = [padded[i:i+block_size] for i in range(0, len(padded), block_size)]
        entropies = [calculate_entropy(block) for block in blocks]
        plt.figure(figsize=(2.56, 2.56))
        plt.plot(entropies, color='black')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_folder_path + '/' + base_filename + '_Entropy.jpg', format='jpg', dpi=100)
        plt.close()
        img = Image.open(output_folder_path + '/' + base_filename + '_Entropy.jpg')
        final_entropy_img = img.convert("L")
        entropy_image_path = os.path.join(output_folder_path, base_filename + '_Entropy.jpg')
        final_entropy_img.save(entropy_image_path)

        # Ensure the images are saved before reading them
        while not (os.path.exists(markov_image_path) and os.path.exists(entropy_image_path) and os.path.exists(glmi_image_path)):
            time.sleep(0.1)

        # Create the merged image
        markov_img = np.array(Image.open(markov_image_path))
        entropy_img = np.array(Image.open(entropy_image_path))
        glmi_img = np.array(Image.open(glmi_image_path))

        merged_img = np.stack((markov_img, entropy_img, glmi_img), axis=-1)
        merged_image = Image.fromarray(merged_img)
        merged_image_path = os.path.join(output_folder_path, base_filename + '_Fused.jpg')
        merged_image.save(merged_image_path)

        print(f"Processed {apk_path}")
    except Exception as e:
        print(f"Error processing {apk_path}: {str(e)}")

def convert_apk_to_image(apk_path, output_folder_path):
    # # Call the provided Python script to convert APK to image
    # command = f'python convert_apk_to_image.py {apk_path} {output_folder_path}'
    # subprocess.run(command, shell=True)

    process_apk(apk_path, output_folder_path)

    # Assuming the script saves the output image with '_Fused.jpg' suffix
    base_filename = os.path.basename(apk_path)

    # Ensure the images are saved before reading them
    while not (os.path.exists(os.path.join(output_folder_path, base_filename, '_Fused.jpg'))):
        time.sleep(0.1)
    image_path = os.path.join(output_folder_path, base_filename, '_Fused.jpg')
    return image_path

# def classify_image(image_path):
#     # Prepare the input data
#     input_data = image.load_img(image_path, target_size=(299, 299))
#     input_data = image.img_to_array(input_data)
#     input_data = np.expand_dims(input_data, axis = 0)

#     # Set the input tensor and run inference
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()

#     # Get the output tensor and process the output
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     probability = output_data[0][0]

#     # Interpret the result
#     threshold = 0.5  # Common threshold for binary classification
#     if probability > threshold:
#         predicted_class = 1
#     else:
#         predicted_class = 0

#     # Print detailed prediction
#     class_names = ['Benign', 'Malware']
#     probability = probability if (probability > 0.5) else (1 - probability)
#     result = "This app most likely belongs to {} with a {:.4f}% confidence.".format(class_names[predicted_class], 100 * probability)
#     print(result)

#     return str(result)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=False, host='0.0.0.0')
