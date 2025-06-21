from flask import Flask, request, jsonify
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
import os
import time
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
app = Flask(__name__)

client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ.get("HF_TOKEN"),
)

def extract_sitemap_urls(url):
    urls = set()
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            found = re.findall(r"<loc>(.*?)</loc>", res.text)
            urls.update(found)
    except:
        pass
    return urls

def crawl_site(homepage):
    parsed = urlparse(homepage)
    base = f"{parsed.scheme}://{parsed.netloc}"
    urls = set()

    # Try robots.txt
    try:
        robots = requests.get(urljoin(base, "/robots.txt")).text
        sitemap_links = re.findall(r"sitemap:\s*(https?://[^\s]+)", robots, re.IGNORECASE)
        for link in sitemap_links:
            urls.update(extract_sitemap_urls(link))
    except:
        pass

    # Try default sitemap
    urls.update(extract_sitemap_urls(urljoin(base, "/sitemap.xml")))
    urls.update(extract_sitemap_urls(urljoin(base, "/sitemap_index.xml")))

    # Try homepage crawl
    try:
        soup = BeautifulSoup(requests.get(homepage).text, "html.parser")
        for link in soup.find_all("a", href=True):
            full_url = urljoin(base, link["href"])
            if parsed.netloc in urlparse(full_url).netloc:
                urls.add(full_url)
    except:
        pass

    return sorted(urls)

def call_llama(prompt, max_tokens=400):
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling LLaMA: {str(e)}"

def extract_json(text):
    try:
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        return json.loads(text[json_start:json_end])
    except:
        return None

@app.route("/find-store-root", methods=["GET"])
def find_store_root():
    homepage = request.args.get("url")
    if not homepage:
        return jsonify({"error": "Missing ?url parameter"}), 400

    all_urls = crawl_site(homepage)

    prompt = f"""RESPOND IN JSON ONLY. NO EXPLANATIONS.

These are URLs from a mall website. Identify the most likely root path used to build individual store pages.

Examples of roots: https://examplemall.com/stores/, https://examplemall.com/store-directory/

Exclude news, blog, contact, about, events, deals, promotions...

URLs:
{chr(10).join(all_urls[:100])}

Return:
{{"store_roots": "url",}}"""

    llama_response = call_llama(prompt)
    parsed = extract_json(llama_response)

    return jsonify({
        "homepage": homepage,
        "total_urls": len(all_urls),
        "store_root_candidates": parsed.get("store_roots", []) if parsed else [],
        "llama_response": llama_response
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
