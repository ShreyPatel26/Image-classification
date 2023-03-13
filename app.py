from flask import Flask, jsonify, request
from keras.models import load_model
from keras.utils import load_img
import numpy as np
app = Flask(__name__)


model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = load_img(img_path, target_size=(150,150))
	i = i.resize((150,150))
	#i = img_to_array(i)/255.0
	i = np.expand_dims(i, axis=0)
	
	#p = model.predict(i)
	p=model.predict(i) 
	classes_x=np.argmax(p,axis=1)
	if p[0][0] < 0.5:
		return "healthy"
	else:
		return "diseased"

@app.route("/", methods=['GET'])
def main():
	return {'message': 'Welcome to our Project!'}


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has a file part.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an empty file.
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the file to disk.
    file.save(file.filename)
    p = predict_label(file.filename)

    return jsonify({'message': p}), 200


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
