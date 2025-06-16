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

    # Try robots.txt
    try:
        robots = requests.get(urljoin(base, '/robots.txt')).text
        urls.update(re.findall(r'(https?://[^\s]+)', robots))
    except:
        pass

    # Try sitemap.xml
    #try:
     #   sitemap = requests.get(urljoin(base, '/sitemap.xml')).text
      #  urls.update(re.findall(r'<loc>(.*?)</loc>', sitemap))
    #except:
     #   pass

sitemap_paths = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/site-map.xml",
    "/sitemap.html",
    "/sitemap-en.xml",
    "/sitemap-es.xml",
    "/sitemap1.xml",
    "/sitemap1_index.xml",
    "/sitemap_index1.xml",
    "/sitemap/sitemap.xml",
    "/sitemap/sitemap-index.xml",
    "/sitemap_index/sitemap.xml",
]

for variant in sitemap_variants:
    try:
        sitemap = requests.get(urljoin(base, variant), timeout=5).text
        urls.update(re.findall(r'<loc>(.*?)</loc>', sitemap))
    except:
        pass


    
    # Try homepage crawl
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
    filter_string = request.args.get('filter', '')  # optional

    if not homepage:
        return jsonify({'error': 'Missing url parameter'}), 400

    all_urls = discover_site_urls(homepage)

    if filter_string:
        filtered_urls = [url for url in all_urls if filter_string in url]
    else:
        filtered_urls = all_urls

    return jsonify({'discovered_urls': filtered_urls})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
