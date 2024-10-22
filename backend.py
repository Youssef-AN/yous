import logging
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["POST", "GET", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# Enable logging of requests and responses
logging.basicConfig(level=logging.DEBUG)

is_detecting = False

@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({"message": "GET request successful!"}), 200


@app.route('/toggle-detection', methods=['POST'])
def toggle_detection():
    global is_detecting
    is_detecting = not is_detecting
    status = "started" if is_detecting else "stopped"
    app.logger.info(f"Detection {status}")
    return jsonify({"message": f"Detection {status}", "status": is_detecting})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
