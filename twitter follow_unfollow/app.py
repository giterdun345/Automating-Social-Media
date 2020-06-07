import os
import flask
from flask import Flask, render_template, redirect, request, flash, url_for, jsonify
from werkzeug.utils import secure_filename
from . import predictSingle


app = flask.Flask(__name__)
app.config['SECRET_KEY'] = '290d1c47c5a94841bf35023c7ef8b7c7'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/twitter/following-buttons', methods=["POST"])
def detection():
    if request.method == "POST":
        
        image_follow = request.files.get("image_follow")
        image_unfollow = request.files.get("image_unfollow")
        
        if not image_follow and not image_unfollow:
            return jsonify({"message": "No file part"})
        elif image_follow and image_unfollow:
            return jsonify({"message": "Two types of images detected! One type image at a time."})
        elif image_follow:
            image_key = "image_follow"
            image = image_follow
        else:
            image_key = "image_unfollow"
            image = image_unfollow
        
        if image.filename == '':
            return jsonify({"message": "No Selected Image"})
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save("uploads/"+filename)
            result = predictSingle("./uploads/"+filename, image_key)
            return jsonify({"Cordinates": result})

@app.route('/', methods=["GET"])
def hello_world():
    if request.method == "GET":
        return "Hello, world"


if __name__ == '__main__':
    app.run(debug=True)
