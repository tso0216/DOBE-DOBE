from flask import Blueprint, current_app, jsonify, render_template, request

views = Blueprint('views', __name__)


def dashboard():
    return current_app.extensions["poi_dashboard"]


@views.route('/')
def home():
    return render_template("home.html")


@views.get('/api/bootstrap')
def bootstrap():
    return jsonify(dashboard().bootstrap())


@views.get('/api/patch/<int:patch_id>')
def patch(patch_id):
    try:
        return jsonify(dashboard().patch(patch_id))
    except ValueError as error:
        return jsonify(error=str(error)), 400


@views.get('/api/models/<model_key>/embedding')
def embedding(model_key):
    try:
        return jsonify(dashboard().embedding(model_key))
    except (KeyError, FileNotFoundError, RuntimeError) as error:
        return jsonify(error=str(error)), 400


@views.post('/api/simulate')
def simulate():
    payload = request.get_json(silent=True) or {}
    try:
        result = dashboard().simulate(
            model_key=str(payload.get("model", "ddae_fsce")),
            patch_id=int(payload.get("patch_id", 0)),
            category=int(payload.get("category", 0)),
            amount=int(payload.get("amount", 0)),
        )
        return jsonify(result)
    except (TypeError, ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        return jsonify(error=str(error)), 400
