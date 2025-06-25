from flask import Flask, request, jsonify
import pandas as pd
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from fuzzywuzzy import process
import os

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# Load brand list
BRAND_LIST_PATH = "brand_list.json"
with open(BRAND_LIST_PATH, "r", encoding="utf-8") as f:
    brand_df = pd.read_json(f)

# Hugging Face client
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=os.getenv("HF_TOKEN")
)

def call_llama(prompt, max_tokens=400, temperature=0.2):
    try:
        completion = client.text_generation(prompt=prompt, max_new_tokens=max_tokens, temperature=temperature)
        return completion.strip()
    except Exception as e:
        return None

def extract_best_brand_match(user_input, threshold=85):
    # Fuzzy match
    choices = brand_df["BRAND NAME"].tolist()
    best_match, score = process.extractOne(user_input, choices)

    if score >= threshold:
        match_row = brand_df[brand_df["BRAND NAME"] == best_match].iloc[0]
        return {
            "brand_name": best_match,
            "brand_id": match_row["BRAND ID"],
            "score": score
        }
    return None

@app.route('/match-brand', methods=['GET'])
def match_brand():
    store_name = request.args.get("store")

    if not store_name:
        return jsonify({"error": "Missing 'store' query parameter"}), 400

    print(f"🔍 Searching for: {store_name}")

    result = extract_best_brand_match(store_name)

    if result:
        return jsonify({
            "success": True,
            "matched_brand": result["brand_name"],
            "brand_id": result["brand_id"],
            "confidence_score": result["score"]
        })

    # Fallback: Ask LLM
    brand_names = brand_df["BRAND NAME"].tolist()
    brand_str = "\n".join(f"- {b}" for b in brand_names[:100])  # limit to top 100 for speed

    prompt = f"""
You are a helpful AI for fuzzy brand name matching. A user searched for "{store_name}".
Which brand from this list is the best match (or say NONE)?

{brand_str}

Respond ONLY in this JSON format:
{{"match": "matched brand name or NONE"}}
"""
    response = call_llama(prompt)
    try:
        match_json = json.loads(response)
        match_name = match_json.get("match")
        if match_name and match_name != "NONE":
            match_row = brand_df[brand_df["BRAND NAME"] == match_name].iloc[0]
            return jsonify({
                "success": True,
                "matched_brand": match_name,
                "brand_id": match_row["BRAND ID"],
                "source": "llm"
            })
    except Exception as e:
        print(f"Error parsing LLM: {e}")

    return jsonify({"success": False, "error": "Brand not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
