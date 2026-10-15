import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Inyong routes at model loading code dito...
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "TANGHAP API is running!"})
if __name__ == "__main__":
    # Babasahin ang PORT na ibibigay ng Render, gagamit ng 5000 kung local
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)