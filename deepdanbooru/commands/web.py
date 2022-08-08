import os
import six
import tensorflow as tf

import deepdanbooru as dd

from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage


class WebUpload(Resource):
    def __init__(self, **kwargs) -> None:
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("img", required=True, type=FileStorage, location="files")
        self.model = kwargs["model"]
        self.tags = kwargs["tags"]
        self.tags_character = kwargs["tags_character"]
        self.tags_system = ["rating:safe", "rating:questionable", "rating:explicit"]
        self.threshold = kwargs["threshold"]
        self.verbose = kwargs["verbose"]

    def post(self):
        imgFile = self.parser.parse_args().get("img")
        res = {"general": {}, "character": {}, "system": {}}
        for tag, score in dd.commands.evaluate_image(six.BytesIO(imgFile.read()), self.model, self.tags, self.threshold):
            if tag in self.tags_character:
                res["character"][tag] = float(score)
            elif tag in self.tags_system:
                res["system"][tag] = float(score)
            else:
                res["general"][tag] = float(score)
        return res


def web_upload(
    project_path,
    model_path,
    tags_path,
    threshold,
    allow_gpu,
    compile_model,
    port,
    verbose,
):
    if not allow_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if not model_path and not project_path:
        raise Exception("You must provide project path or model path.")

    if not tags_path and not project_path:
        raise Exception("You must provide project path or tags path.")

    if model_path:
        if verbose:
            print(f"Loading model from {model_path} ...")
        model = tf.keras.models.load_model(model_path, compile=compile_model)
    else:
        if verbose:
            print(f"Loading model from project {project_path} ...")
        model = dd.project.load_model_from_project(
            project_path, compile_model=compile_model
        )

    if tags_path:
        if verbose:
            print(f"Loading tags from {tags_path} ...")
        tags = dd.data.load_tags(tags_path)
    else:
        if verbose:
            print(f"Loading tags from project {project_path} ...")
        tags = dd.project.load_tags_from_project(project_path)
        tags_character = dd.project.load_tags_character_from_project(project_path)

    app = Flask(__name__)
    CORS(app)
    api = Api(app)
    api.add_resource(WebUpload, "/upload", resource_class_kwargs={"model": model, "tags": tags, "tags_character": tags_character, "threshold": threshold, "verbose": verbose})
    app.run(host="0.0.0.0", port=port)
