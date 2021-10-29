# Imports for data pipeline
import os
import json
from sentiment_extraction import polarity
from sentiment_extraction import subjectivity
from toxicity_analysis import text_analysis

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Defining the flask app
app = Flask(__name__)
CORS(app)
PORT = os.getenv('PORT',8000)

@app.route("/endpoint", methods = ["GET"])
def endpoint():
    text = request.args.get("text")
    pol = polarity(text)
    sub = subjectivity(text)
    toxicity = ["none"]
    return {"toxicity" : toxicity}

@app.route('/', methods=['GET', 'POST'])
def main():
    # Get the form data from post request
    if request.method == 'POST':
        text = request.form['text_data']
        pol = polarity(text)
        sub = subjectivity(text)
        toxicity = ["none"]
        toxicity = text_analysis(text)
        output = json.dumps(toxicity)
        output2 = json.loads(output)
        final_output = output2['result']
        print(final_output)
        return render_template("index.html", toxicity_class = final_output)
    return render_template("index.html", toxicity_class = None)

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = PORT)
