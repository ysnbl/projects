from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import re
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ.get("HF_TOKEN"),
)

DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"


def call_llama(prompt, max_tokens=500, temperature=0.1, retries=3):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Llama call failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                return None


def search_duckduckgo(query, max_results=10):
    try:
        response = requests.post(DUCKDUCKGO_SEARCH_URL, data={'q': query}, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for a in soup.find_all('a', class_='result__a', href=True):
            url = a['href']
            title = a.get_text(strip=True)
            results.append((title, url))
            if len(results) >= max_results:
                break
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []


def score_results(results, mall_name):
    scored = []
    for title, url in results:
        domain = urlparse(url).netloc.lower()
        score = 0
        if any(ext in domain for ext in ['.com', '.ca', '.org']):
            score += 2
        if mall_name.lower().replace(" ", "") in domain:
            score += 3
        if any(kw in title.lower() for kw in ['home', 'official site']):
            score += 2
        if "mall" in title.lower():
            score += 1
        scored.append({'url': url, 'title': title, 'score': score})
    return sorted(scored, key=lambda x: x['score'], reverse=True)


@app.route('/find-homepage', methods=['GET'])
def find_homepage():
    mall = request.args.get('mall')
    address = request.args.get('address')

    if not mall or not address:
        return jsonify({'error': 'Missing mall or address parameter'}), 400

    query = f"{mall} {address}"
    results = search_duckduckgo(query)
    if not results:
        return jsonify({'homepage': None, 'reason': 'No search results'})

    ranked = score_results(results, mall)
    top = ranked[0] if ranked and ranked[0]['score'] >= 3 else None

    return jsonify({
        'query': query,
        'homepage': top['url'] if top else None,
        'confidence_score': top['score'] if top else 0,
        'top_candidates': ranked[:3]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
