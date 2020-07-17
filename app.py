
from flask import Flask, render_template, request, send_file, jsonify
import io

import model

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    #return "Hello Flask!"
    return render_template('upload.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    img_bytes = request.files["file"].read()
    mask = model.get_mask_image(img_bytes)
    return serve_pil_image(mask)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      return 'file uploaded successfully'

@app.route('/maoe', methods=['POST'])
def getMAOEs():
    img_bytes = request.files["img"].read()
    maoe = model.get_maoe(img_bytes)
    return jsonify(maoe)

def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', attachment_filename="mask.png")

if __name__ == "__main__":
    #app.run(debug=False, host="127.0.40.0", port=8000)
    app.run(debug=True)