from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
import requests
import asyncio
import os
import re
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

app = Flask(__name__)
client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ.get("HF_TOKEN"),
)

async def render_js_content(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=15000)
            content = await page.content()
            await browser.close()
            return content
    except:
        return None

def extract_html_content(url):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.text
    except:
        return None
    return None

def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    [s.extract() for s in soup(["script", "style"])]
    return soup.get_text(separator=" ", strip=True)

def call_llama(prompt):
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling LLaMA: {str(e)}"

@app.route("/store-info", methods=["GET"])
def store_info():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    html = extract_html_content(url)
    if not html:
        html = asyncio.run(render_js_content(url))
        if not html:
            return jsonify({"error": "Failed to fetch page content."}), 500

    text = extract_text_from_html(html)[:5000]  # Keep within token limit

    prompt = f"""
RESPOND IN JSON ONLY. NO EXPLANATION.

Extract the following fields from this webpage text:
- store_name (not the mall name)
- description
- phone
- hours
- categories (as an array)

If a field is missing, return null for it. Format:
{{"store_name": ..., "description": ..., "phone": ..., "hours": ..., "categories": [...]}}

TEXT:
{text}
"""

    response = call_llama(prompt)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return jsonify(eval(match.group(0))) if match else jsonify({"error": "No JSON returned"})
    except Exception as e:
        return jsonify({"error": str(e), "raw": response}), 500


@app.route('/check-url', methods=['GET'])
def check_url():
    url = request.args.get('url')
    
    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return jsonify({'result': url})
        else:
            return jsonify({'result': ''})
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return jsonify({'result': ''})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

