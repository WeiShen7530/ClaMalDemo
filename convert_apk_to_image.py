import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from androguard.core.bytecodes.apk import APK
from skimage.feature import graycomatrix
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
import time
import sys

# Function to compute GLMI
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

# Function to compute Markov matrix
def calculate_markov_matrix(byte_sequence):
    markov_matrix = np.zeros((256, 256), dtype=np.float32)
    row_col_pairs = np.column_stack((byte_sequence[:-1], byte_sequence[1:]))
    np.add.at(markov_matrix, (row_col_pairs[:, 0], row_col_pairs[:, 1]), 1)
    row_sums = markov_matrix.sum(axis=1)
    markov_matrix /= row_sums[:, np.newaxis]
    markov_matrix *= 255
    return markov_matrix.astype(np.uint8)

# Function to compute entropy
def calculate_entropy(block):
    value, counts = np.unique(block, return_counts=True)
    probabilities = counts / len(block)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

# Main processing function
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
        if required_padding > 0:
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

if __name__ == "__main__":
    apk_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    process_apk(apk_path, output_folder_path)