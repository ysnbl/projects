import json
from flask import Flask, request, jsonify
from fuzzywuzzy import fuzz
from huggingface_hub import InferenceClient

# Load brand list
with open("brand_list.json", "r", encoding="utf-8") as f:
    BRANDS = json.load(f)

app = Flask(__name__)
client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct")

def match_brand(store_name):
    best_match = None
    best_score = 0

    for brand in BRANDS:
        score = fuzz.ratio(store_name.lower(), brand["name"].lower())
        if score > best_score:
            best_score = score
            best_match = brand

    if best_score >= 90:
        return {"match": best_match["name"], "id": best_match["id"], "score": best_score}
    else:
        return {"match": "not found", "score": best_score}

@app.route("/match", methods=["GET"])
def match():
    store_name = request.args.get("store")
    if not store_name:
        return jsonify({"error": "Missing 'store' query parameter"}), 400

    result = match_brand(store_name)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
