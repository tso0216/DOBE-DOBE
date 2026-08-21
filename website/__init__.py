from pathlib import Path

from flask import Flask, jsonify

from .dashboard import DashboardData


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        JSON_SORT_KEYS=False,
        TEMPLATES_AUTO_RELOAD=True,
    )
    if test_config:
        app.config.update(test_config)

    root = Path(__file__).resolve().parent.parent
    app.extensions["poi_dashboard"] = DashboardData(root)

    from .views import views

    app.register_blueprint(views)

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify(error="找不到要求的頁面或 API"), 404

    return app

