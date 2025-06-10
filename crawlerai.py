from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

app = Flask(__name__)

def discover_site_urls(homepage):
    parsed = urlparse(homepage)
    base = f"{parsed.scheme}://{parsed.netloc}"
    urls = set()

    # robots.txt
    try:
        robots = requests.get(urljoin(base, '/robots.txt')).text
        urls.update(re.findall(r'(https?://[^\s]+)', robots))
    except:
        pass

    # sitemap.xml
    try:
        sitemap = requests.get(urljoin(base, '/sitemap.xml')).text
        urls.update(re.findall(r'<loc>(.*?)</loc>', sitemap))
    except:
        pass

    # homepage crawl
    try:
        soup = BeautifulSoup(requests.get(homepage).text, 'html.parser')
        for link in soup.find_all('a', href=True):
            full_url = urljoin(base, link['href'])
            if parsed.netloc in urlparse(full_url).netloc:
                urls.add(full_url)
    except:
        pass

    return sorted(urls)

@app.route('/discover', methods=['GET'])
def discover():
    homepage = request.args.get('url')
    if not homepage:
        return jsonify({'error': 'Missing url parameter'}), 400

    urls = discover_site_urls(homepage)
    return jsonify({'discovered_urls': urls})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
