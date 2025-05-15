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

def download_file(url, path):
    """Download file with progress and error handling"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        with open(path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def load_model():
    """Load model with proper error handling"""
    print("Initializing colorization model...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        prototxt_path = os.path.join(temp_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(temp_dir, "model.caffemodel")
        npy_path = os.path.join(temp_dir, "pts_in_hull.npy")
        
        # Download all required files
        if not all([
            download_file(MODEL_URLS["prototxt"], prototxt_path),
            download_file(MODEL_URLS["caffemodel"], caffemodel_path),
            download_file(MODEL_URLS["npy"], npy_path)
        ]):
            raise RuntimeError("Failed to download model files")
        
        # Verify files were downloaded
        if not all(os.path.exists(p) for p in [prototxt_path, caffemodel_path, npy_path]):
            raise FileNotFoundError("Some model files are missing")
        
        # Load cluster centers
        try:
            pts_in_hull = np.load(npy_path)
            print("Successfully loaded cluster centers")
        except Exception as e:
            raise RuntimeError(f"Failed to load cluster centers: {str(e)}")
        
        # Load the model
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            if net.empty():
                raise RuntimeError("Failed to load model: network is empty")
            print("Successfully loaded Caffe model")
        except Exception as e:
            raise RuntimeError(f"Failed to load Caffe model: {str(e)}")
        
        # Add cluster centers to the model
        try:
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
            net.getLayer(class8).blobs = [pts.astype("float32")]
            net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            print("Successfully configured model layers")
        except Exception as e:
            raise RuntimeError(f"Failed to configure model layers: {str(e)}")
    
    return net

# Initialize model at startup
try:
    net = load_model()
    print("Model initialized successfully!")
except Exception as e:
    print(f"Fatal error during model initialization: {str(e)}")
    exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def colorize_image(image_path, output_path):
    """Colorize image with error handling"""
    try:
        # Read and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read input image")
        
        scaled = img.astype("float32") / 255.0
        lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        
        # Prepare L channel
        resized = cv2.resize(lab_img, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        
        # Colorization
        net.setInput(cv2.dnn.blobFromImage(L))
        ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
        
        # Combine channels
        L = cv2.split(lab_img)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
        
        # Create comparison image
        img = cv2.resize(img, (640, 640))
        colorized = cv2.resize(colorized, (640, 640))
        result = cv2.hconcat([img, colorized])
        cv2.imwrite(output_path, result)
        return True
    except Exception as e:
        print(f"Error during colorization: {str(e)}")
        return False

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
            output_path = os.path.join(app.config['RESULT_FOLDER'], f'colorized_{filename}')
            
            try:
                file.save(input_path)
                if colorize_image(input_path, output_path):
                    return render_template('result.html', image=f'colorized_{filename}')
                else:
                    return render_template('index.html', error='Colorization failed')
            except Exception as e:
                return render_template('index.html', error=f'Processing error: {str(e)}')
    
    return render_template('index.html')

@app.route('/static/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)