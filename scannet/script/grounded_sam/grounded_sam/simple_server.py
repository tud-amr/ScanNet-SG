from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
from grounded_sam_simple_demo import GroundedSam
import cv2
import numpy as np


# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

grounded_sam = GroundedSam()

image_path = "/home/cc/chg_ws/isaac_lab/git/Grounded-Segment-Anything/assets/annotated_image.jpg"
def load_image(image_path):
    """
    Load an image from the given path and return it as a binary stream.
    :param image_path: Path to the image file.
    :return: BytesIO object containing the image.
    """
    try:
        with Image.open(image_path) as image:
            img_io = BytesIO()
            image.save(img_io, 'JPEG')
            img_io.seek(0)
            return img_io
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {str(e)}")
    



@app.route('/process', methods=['POST'])
def process():
    # Check if image and prompts are provided
    if 'image' not in request.files or 'prompts' not in request.form:
        return jsonify({'error': 'Image and prompts are required'}), 400

    # Retrieve the image
    image_file = request.files['image']
    image_filename = image_file.filename

    # Retrieve the prompts
    prompts = request.form.getlist('prompts')

    # Show the image
    image = Image.open(image_file)
    # image.show()

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    img_io2 = BytesIO()
    image.save(img_io2, 'JPEG')
    img_io2.seek(0)


    box_threshold = 0.25
    text_threshold = 0.25
    nms_threshold = 0.3

    result_image, result_mask, class_ids, confidences = grounded_sam.infer(img_io2, prompts, box_threshold, text_threshold, nms_threshold, image_type="bytesio")

    # Convert the image to base64
    _, buffer = cv2.imencode('.jpg', result_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Convert the masks to base64
    mask_uint8 = result_mask.astype(np.uint8) * 255
    masks_buffer = [cv2.imencode('.jpg', mask)[1] for mask in mask_uint8]
    encoded_masks = [base64.b64encode(mask).decode('utf-8') for mask in masks_buffer]

    class_names = [prompts[class_id] for class_id in class_ids]
    confidences = list(map(str, confidences))

    # Log or process data (for now, just printing to console)
    print(f"Received image: {image_filename}")
    print(f"Received prompts: {prompts}")

    response_text = "Segmentation successful!"

    return jsonify({'image':encoded_image, 'masks':encoded_masks, 'classes': class_names, 'confidences': confidences}), 200, {
        'X-Text-Response': response_text  # Sending text in a custom header
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

    # Client entry address: http://145.94.60.29:8000/process
