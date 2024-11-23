import os
from flask import Flask, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define a simple route for testing
@app.route("/")
def home():
    return jsonify({"message": "Flask server is running!"})

# Define additional routes if needed
@app.route("/api/process-image", methods=["POST"])
def process_image():
    return jsonify({"message": "This is where image processing logic will go."})

# Run the app
if __name__ == "__main__":
    # Use dynamic port for Heroku or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
