import os
import cv2
import numpy as np
import requests
from io import BytesIO
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Google Drive Direct Links (converted to direct download format)
MODEL_URLS = {
    "prototxt": "https://drive.google.com/uc?id=1tGAEwt-ZaH58b-DXP9l5hvYxDM5IT0mz",
    "caffemodel": "https://drive.google.com/uc?id=1qVB_MNdZ1l0mg-mqSCqt6TQaB-O4BvSw",
    "npy": "https://drive.google.com/uc?id=1FMVKzrlQL1qWTR1YbvPz4_xaLHNl8oib"
}

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def load_model_from_drive():
    """Load model files directly from Google Drive"""
    print("Loading model files from Google Drive...")
    
    # Load prototxt
    prototxt_response = requests.get(MODEL_URLS["prototxt"])
    prototxt_content = prototxt_response.content.decode('utf-8')
    
    # Load caffemodel
    caffemodel_response = requests.get(MODEL_URLS["caffemodel"], stream=True)
    caffemodel_content = BytesIO()
    for chunk in caffemodel_response.iter_content(chunk_size=8192):
        caffemodel_content.write(chunk)
    caffemodel_content.seek(0)
    
    # Load numpy array
    npy_response = requests.get(MODEL_URLS["npy"])
    pts_in_hull = np.load(BytesIO(npy_response.content))
    
    # Create the network
    net = cv2.dnn.readNetFromCaffe(
        prototxt_content.encode('utf-8'), 
        caffemodel_content.getvalue()
    )
    
    # Add the cluster centers
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    print("Model loaded successfully")
    return net

# Initialize the model when the app starts
net = load_model_from_drive()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def colorize_image(image_path, output_path):
    """Colorize the input image"""
    # Read and preprocess image
    img = cv2.imread(image_path)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Resize for network
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50  # mean subtraction
    
    # Predict ab channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
    
    # Combine with original L channel
    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    
    # Convert to BGR and save
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    # Resize and save
    img = cv2.resize(img, (640, 640))
    colorized = cv2.resize(colorized, (640, 640))
    result = cv2.hconcat([img, colorized])
    cv2.imwrite(output_path, result)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
            
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file')
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            output_filename = 'colorized_' + filename
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            
            colorize_image(input_path, output_path)
            
            return render_template('result.html', image=output_filename)
    
    return render_template('index.html')

@app.route('/static/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)