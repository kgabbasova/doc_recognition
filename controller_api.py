import os
import time
import main

from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Api, Resource, reqparse, abort

UPLOAD_DIRECTORY = "data/api_uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

api = Flask(__name__)




@api.route("/files", methods=["POST"])
def post_file():
    """Upload a file."""

    file = request.files.get('file')
    if file:
        classifier = main.DocClassifier
        filename = str(time.time()) + file.filename
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        file.save(path)
        result = classifier.predict(classifier, filename)
    # Return 201 CREATED
    return result, 201


if __name__ == "__main__":
    main.DocClassifier.train(main)
    api.run(debug=True, port=5000)
