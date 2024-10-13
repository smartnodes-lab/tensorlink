from flask import Flask, jsonify, request
from flask_cors import CORS


def create_endpoint(smart_node):
    app = Flask(__name__)
    CORS(app)

    @app.route("/roles", methods=["GET"])
    def post_node_info():
        response = smart_node.get_self_info()
        return jsonify(response), 200

    # @app.route("/roles", methods=["GET"])
    # def get_connected_nodes():
    #     roles = list(smart_node.roles.keys())
    #     return jsonify({"connected_nodes": roles})
    #
    # @app.route("/jobs", methods=["POST"])
    # def upload_job_info():
    #     data = request.get_json()
    #     job_id = data.get("job_id")
    #     job_info = data.get("job_info")
    #     smart_node.jobs.append({job_id: job_info})
    #     return (
    #         jsonify({"message": "Job info uploaded successfully", "job_id": job_id}),
    #         200,
    #     )
    #
    # @app.route("/jobs", methods=["POST"])
    # def upload_job_info():
    #     data = request.get_json()
    #     job_id = data.get("job_id")
    #     job_info = data.get("job_info")
    #     smart_node.jobs.append({job_id: job_info})
    #     return (
    #         jsonify({"message": "Job info uploaded successfully", "job_id": job_id}),
    #         200,
    #     )

    return app
