from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)  # initializing a flask app
CORS(app)  # enabling CORS


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(filename=self.filename)

@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    # os.system('python main.py')
    os.system("dvc repro --force")
    return "Training done successfully!"

@app.route('/predict', methods=['POST']) 
@cross_origin()
def predictRoute():
    try:
        if request.method == 'POST':
            image_data = request.json['image']
            decodeImage(image_data, filename=ClientApp().filename)
            prediction = ClientApp().classifier.predict()
            return jsonify(str(prediction))
    except Exception as e:
        print(e)
        return "Error Occurred! Please try again."
    
if __name__ == "__main__":
    clApp = ClientApp()
    # app.run(host='0.0.0.0', port=8080, debug=True)#local host
    # app.run(host='0.0.0.0',port=8080,debug=True) # for AWS deployment use port=8080
    app.run(host='0.0.0.0', port=80, debug=True) # for AZURE deployment use port=80