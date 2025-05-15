import os
import cv2
import numpy as np
import requests
import tempfile
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Google Drive Direct Links
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

def load_model():
    """Load model files from Google Drive using temporary files"""
    print("Loading model files...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download prototxt
        prototxt_path = os.path.join(temp_dir, "colorization_deploy_v2.prototxt")
        with requests.get(MODEL_URLS["prototxt"], stream=True) as r:
            r.raise_for_status()
            with open(prototxt_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Download caffemodel
        caffemodel_path = os.path.join(temp_dir, "colorization_release_v2.caffemodel")
        with requests.get(MODEL_URLS["caffemodel"], stream=True) as r:
            r.raise_for_status()
            with open(caffemodel_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Download numpy array
        npy_path = os.path.join(temp_dir, "pts_in_hull.npy")
        with requests.get(MODEL_URLS["npy"], stream=True) as r:
            r.raise_for_status()
            with open(npy_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Load the cluster centers
        pts_in_hull = np.load(npy_path)
        
        # Load the model
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        
        # Add the cluster centers
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    print("Model loaded successfully")
    return net

# Initialize the model when the app starts
net = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def colorize_image(image_path, output_path):
    """Colorize the input image"""
    img = cv2.imread(image_path)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
    
    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
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