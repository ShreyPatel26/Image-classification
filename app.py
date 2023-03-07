from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import load_img, img_to_array
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

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename
		img.save(img_path)
		p = predict_label(img_path)
	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ == '__main__':
	# app.run(debug = True)
    app.run(threaded=True, port=5000)
