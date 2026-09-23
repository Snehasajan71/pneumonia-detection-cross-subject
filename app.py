import os
import sys
import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from http.server import HTTPServer, SimpleHTTPRequestHandler
from dataset_loader import PneumoniaDatasetLoader
from best_prediction import BestPredictionEngine

PORT = 8050
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

class PneumoniaAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open(os.path.join(STATIC_DIR, 'index.html'), 'rb') as f:
                self.wfile.write(f.read())
            return

        elif self.path == '/api/cv_summary':
            cv_path = os.path.join(STATIC_DIR, 'cv_results.json')
            if os.path.exists(cv_path):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                with open(cv_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self._send_json({'error': 'CV summary not found. Please run train_pipeline.py first.'}, status=404)
            return

        elif self.path == '/api/samples':
            samples = self._get_sample_images()
            self._send_json(samples)
            return

        # Serve static assets
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/predict':
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            
            try:
                data = json.loads(post_body.decode('utf-8'))
                engine = BestPredictionEngine(checkpoint_dir=CHECKPOINT_DIR, img_size=(150, 150))

                if 'image_data' in data and data['image_data']:
                    # Base64 image upload
                    header, b64_str = data['image_data'].split(',', 1) if ',' in data['image_data'] else ('', data['image_data'])
                    img_bytes = base64.b64decode(b64_str)
                    img = Image.open(BytesIO(img_bytes)).convert('L').resize((150, 150))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    arr = np.expand_dims(arr, axis=0)
                    result = engine.predict_image(arr)

                elif 'sample_path' in data and data['sample_path']:
                    filepath = data['sample_path']
                    if os.path.exists(filepath):
                        result = engine.predict_image(filepath)
                    else:
                        result = engine.predict_image(self._create_dummy_image())
                else:
                    result = engine.predict_image(self._create_dummy_image())

                self._send_json(result)

            except Exception as e:
                self._send_json({'error': str(e)}, status=500)
            return

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _get_sample_images(self):
        loader = PneumoniaDatasetLoader(img_size=(150, 150))
        records = loader.load_dataset_index(DATA_DIR)
        if len(records) == 0:
            loader.generate_synthetic_dataset(DATA_DIR, num_subjects=20, images_per_subject=4)
            records = loader.load_dataset_index(DATA_DIR)

        samples = []
        normal_samples = [r for r in records if r['category'] == 'NORMAL'][:3]
        pneumonia_samples = [r for r in records if r['category'] == 'PNEUMONIA'][:3]

        for r in normal_samples + pneumonia_samples:
            samples.append({
                'filename': r['filename'],
                'category': r['category'],
                'subject_id': r['subject_id'],
                'filepath': r['filepath']
            })
        return samples

    def _create_dummy_image(self):
        arr = np.random.uniform(0.2, 0.8, (1, 150, 150)).astype(np.float32)
        return arr

def start_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, PneumoniaAppHandler)
    print(f"===========================================================")
    print(f" PNEUMONIA CROSS-SUBJECT CV WEB DASHBOARD RUNNING")
    print(f" Access URL: http://localhost:{PORT}")
    print(f"===========================================================")
    httpd.serve_forever()

if __name__ == '__main__':
    start_server()
