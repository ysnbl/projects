from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import time
from datetime import datetime

import ssl
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

load_dotenv()
app = Flask(__name__)

client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ.get("HF_TOKEN"),
)

# Rate limiting globals
last_request_time = 0
min_request_interval = 1.0  # 1 second between requests

class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')  # reduce strictness to avoid handshake errors
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

def rate_limit():
    """Add delay between requests to avoid overwhelming APIs"""
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    
    if time_since_last < min_request_interval:
        sleep_time = min_request_interval - time_since_last
        print(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    last_request_time = time.time()

def extract_json_from_response(response_text):
    """Extract JSON from Llama response, handling markdown code blocks"""
    
    # Remove markdown code blocks if present
    response_text = response_text.strip()
    
    # Remove ```json and ``` markers
    if response_text.startswith('```json'):
        response_text = response_text[7:]  # Remove ```json
    if response_text.startswith('```'):
        response_text = response_text[3:]   # Remove ```
    if response_text.endswith('```'):
        response_text = response_text[:-3]  # Remove trailing ```
    
    response_text = response_text.strip()
    
    # Find JSON object bounds
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    
    if json_start >= 0 and json_end > json_start:
        json_text = response_text[json_start:json_end]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Attempted to parse: {json_text[:200]}...")
            return None
    
    print("No valid JSON found in response")
    return None

session = requests.Session()
session.mount('https://', TLSAdapter())

def safe_request(url, timeout=10, retries=2):
    """Make HTTP request with retry logic, return None on failure"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout, headers=headers)
            if response.status_code == 200:
                return response
            else:
                print(f"HTTP {response.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"Request attempt {attempt + 1} failed for {url}: {str(e)[:100]}...")
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 3  # 3, 6 seconds
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    print(f"❌ All attempts failed for {url} - returning None")
    return None

def call_llama(prompt, max_tokens=1000, temperature=0.1, retries=3):
    """Centralized Llama API call method with retry logic"""
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response_text = completion.choices[0].message.content.strip()
            return {
                "success": True,
                "response": response_text,
                "error": None
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"Llama call attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return {
                    "success": False,
                    "response": None,
                    "error": error_msg
                }

def crawl_website(homepage):
    """Crawl website and get all URLs - PROPERLY"""
    parsed = urlparse(homepage)
    base = f"{parsed.scheme}://{parsed.netloc}"
    urls = set()
    
    print(f"Crawling {base}...")
    
    # Try robots.txt for sitemaps
    try:
        robots = requests.get(urljoin(base, '/robots.txt'), timeout=10).text
        sitemap_refs = re.findall(r'sitemap:\s*(https?://[^\s]+)', robots, re.IGNORECASE)
        for sitemap_url in sitemap_refs:
            urls.update(extract_sitemap_urls(sitemap_url))
    except:
        pass

    # Try default sitemap.xml
    try:
        urls.update(extract_sitemap_urls(urljoin(base, '/sitemap.xml')))
    except:
        pass
        
    # Try sitemap index
    try:
        urls.update(extract_sitemap_urls(urljoin(base, '/sitemap_index.xml')))
    except:
        pass

    # Try homepage
    try:
        soup = BeautifulSoup(requests.get(homepage, timeout=10).text, 'html.parser')
        for link in soup.find_all('a', href=True):
            full_url = urljoin(base, link['href'])
            if parsed.netloc in urlparse(full_url).netloc:
                urls.add(full_url)
    except:
        pass

    print(f"Found {len(urls)} total URLs")
    return sorted(urls)

def extract_sitemap_urls(sitemap_url):
    """Extract URLs from sitemap"""
    urls = set()
    try:
        response = requests.get(sitemap_url, timeout=10)
        if response.status_code == 200:
            content = response.text
            if '<sitemapindex' in content:
                # Sitemap index - get nested sitemaps
                nested = re.findall(r'<loc>(.*?)</loc>', content)
                for nested_url in nested:
                    urls.update(extract_sitemap_urls(nested_url))
            else:
                # Regular sitemap - get page URLs
                urls.update(re.findall(r'<loc>(.*?)</loc>', content))
    except:
        pass
    return urls

def find_directory_pages(urls):
    """Find store directory pages using pure Llama approach with chunking"""
    
    print(f"Looking for directory pages in {len(urls)} URLs using Llama...")
    
    # Step 1: Send chunks of 300 URLs to Llama, get top 10 from each
    chunk_size = 5000
    all_candidates = []
    
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(urls) + chunk_size - 1) // chunk_size
        
        print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} URLs)...")
        
        prompt = f"""RESPOND WITH ONLY JSON. NO EXPLANATIONS.

            Find TOP 10 URLs that are likely to be STORE, MEMBERS, BRANDS or SHOPS listing pages (list multiple stores/shops).

            EXCLUDE: news, blog, events, contact, about, individual stores, other companies, hotels, ...

            We are only interested in shops listing page or pages 

            You should be multilingual in your analysis 

            URLs:
            {chr(10).join(f"{j+1}. {url}" for j, url in enumerate(chunk))}

            JSON ONLY:
            {{"directory_candidates": ["url1", "url2", "url3", "url4", "url5", "url6", "url7", "url8", "url9", "url10"]}}"""

        try:
            # Use centralized Llama call
            llama_result = call_llama(prompt, max_tokens=500, temperature=0.0)
            
            if not llama_result["success"]:
                print(f"  Chunk {chunk_num} failed: {llama_result['error']}")
                continue
            
            response = llama_result["response"]
            print(f"  Llama response: {response[:100]}...")
            
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response[json_start:json_end]
                result = json.loads(json_text)
                chunk_candidates = result.get("directory_candidates", [])
                
                print(f"  Llama selected from chunk {chunk_num}:")
                for j, candidate in enumerate(chunk_candidates):
                    print(f"    {j+1}. {candidate}")
                
                # Validate URLs exist in original chunk
                valid_candidates = [url for url in chunk_candidates if url in chunk]
                invalid_candidates = [url for url in chunk_candidates if url not in chunk]
                
                if invalid_candidates:
                    print(f"  WARNING: Llama hallucinated {len(invalid_candidates)} URLs:")
                    for invalid in invalid_candidates:
                        print(f"    FAKE: {invalid}")
                
                all_candidates.extend(valid_candidates)
                print(f"  Added {len(valid_candidates)} valid candidates from chunk {chunk_num}")
            else:
                print(f"  No valid JSON in chunk {chunk_num}")
                
        except Exception as e:
            print(f"  Chunk {chunk_num} JSON parsing failed: {e}")
    
    print(f"\nStep 1 complete: {len(all_candidates)} total candidates from all chunks")
    
    if not all_candidates:
        return []
    
    # Step 2: Send all candidates to Llama for final selection
    print(f"\nStep 2: Final Llama analysis of {len(all_candidates)} candidates...")
    print("all_candidates: ", all_candidates)
    final_prompt = f"""RESPOND WITH ONLY JSON. NO EXPLANATIONS.

Select ONLY store/member/shops listing pages. EXCLUDE news, events, blogs.

{chr(10).join(f"{i+1}. {url}" for i, url in enumerate(all_candidates))}

JSON ONLY:
{{"final_directories": ["url1", "url2"]}}"""

    try:
        # Use centralized Llama call
        llama_result = call_llama(final_prompt, max_tokens=800, temperature=0.0)
        
        if not llama_result["success"]:
            print(f"Final Llama call failed: {llama_result['error']}")
            return []
        
        response = llama_result["response"]
        print(f"Final Llama response: {response[:200]}...")
        
        # Extract JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = response[json_start:json_end]
            result = json.loads(json_text)
            final_directories = result.get("final_directories", [])
            
            print(f"\nLlama final selection:")
            for i, directory in enumerate(final_directories):
                print(f"  {i+1}. {directory}")
            
            print(f"\nFinal result: {len(final_directories)} directory pages found")
            
            return final_directories
        else:
            print("No valid JSON in final response")
            return []
            
    except Exception as e:
        print(f"Final JSON parsing failed: {e}")
        return []

def find_store_roots(urls):
    """Find root URLs where individual store pages are built from"""
    
    print(f"Looking for store root patterns in {len(urls)} URLs...")
    
    # Analyze URL patterns to find roots
    potential_roots = set()
    individual_stores = []
    
    # Group URLs by their base patterns
    pattern_groups = {}
    
    for url in urls:
        parsed = urlparse(url)
        path_segments = [seg for seg in parsed.path.split('/') if seg]
        
        # Look for URLs that might be individual store pages
        if len(path_segments) >= 2:
            # Check if this looks like an individual store page
            potential_base = '/'.join(path_segments[:-1])  # Remove last segment
            last_segment = path_segments[-1]
            
            # Skip obvious non-store patterns
            skip_patterns = ['news', 'blog', 'events', 'contact', 'about', 'policy']
            if any(skip in url.lower() for skip in skip_patterns):
                continue
                
            # Look for store-like patterns
            store_indicators = [
                'member', 'miembro', 'membre', 'membro',
                'store', 'shop', 'tienda', 'magasin', 'negozio', 'loja',
                'brand', 'marca', 'marque'
            ]
            
            if any(indicator in potential_base.lower() for indicator in store_indicators):
                if potential_base not in pattern_groups:
                    pattern_groups[potential_base] = []
                pattern_groups[potential_base].append(url)
    
    # Find patterns with multiple individual pages (indicating it's a root)
    for base_pattern, urls_in_pattern in pattern_groups.items():
        if len(urls_in_pattern) >= 2:  # At least 2 individual pages
            # Reconstruct the full root URL
            sample_url = urls_in_pattern[0]
            parsed = urlparse(sample_url)
            root_url = f"{parsed.scheme}://{parsed.netloc}/{base_pattern.strip('/')}/"
            potential_roots.add(root_url)
            individual_stores.extend(urls_in_pattern)
    
    print(f"Found {len(potential_roots)} potential root patterns:")
    for i, root in enumerate(sorted(potential_roots)):
        print(f"  {i+1}. {root}")
    
    # Use Llama to validate and refine the roots
    if potential_roots:
        roots_list = sorted(potential_roots)
        
        prompt = f"""RESPOND WITH ONLY JSON. NO EXPLANATIONS.

            These are potential store root URLs where individual store pages are built from.
            Select the ones that are definitely ROOT URLs for stores/shops/brands.

            Potential roots:
            {chr(10).join(f"{i+1}. {url}" for i, url in enumerate(roots_list))}

            JSON ONLY:
            {{"store_roots": ["url1", "url2", "url3"]}}"""

        llama_result = call_llama(prompt, max_tokens=400, temperature=0.0)
        
        if llama_result["success"]:
            try:
                response = llama_result["response"]
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = response[json_start:json_end]
                    result = json.loads(json_text)
                    validated_roots = result.get("store_roots", [])
                    
                    print(f"\nLlama validated {len(validated_roots)} roots:")
                    for i, root in enumerate(validated_roots):
                        print(f"  {i+1}. {root}")
                    
                    return validated_roots
                    
            except Exception as e:
                print(f"Error parsing Llama response: {e}")
    
    # Fallback: return pattern-detected roots
    return sorted(potential_roots)

@app.route('/filter-links', methods=['GET'])
def filter_links():
    """
    Crawl entire website and filter ALL discovered URLs containing a root pattern
    Expected: GET /filter-links?url=https://example.com&root=/pattern/to/match/
    """
    url = request.args.get('url')
    root = request.args.get('root')
    
    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400
    if not root:
        return jsonify({'error': 'Missing root parameter'}), 400

    try:
        print(f"🔍 Full site crawling: {url}")
        print(f"🎯 Filtering for root pattern: {root}")
        
        # Use the same comprehensive crawling as discover-roots
        all_urls = crawl_website(url)
        
        print(f"📊 Found {len(all_urls)} total URLs from comprehensive crawling")
        
        # Filter URLs containing the root pattern
        filtered_links = []
        for discovered_url in all_urls:
            # Filter: keep only URLs containing the root pattern
            if root.lower() in discovered_url.lower():
                filtered_links.append(discovered_url)
        
        # Remove duplicates and sort
        unique_filtered_links = list(set(filtered_links))
        unique_filtered_links.sort()
        
        print(f"✅ Found {len(unique_filtered_links)} URLs containing '{root}'")
        
        return jsonify({
            'success': True,
            'crawled_url': url,
            'root_filter': root,
            'total_urls_discovered': len(all_urls),
            'filtered_links': unique_filtered_links,
            'filtered_count': len(unique_filtered_links)
        })
        
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error crawling website: {str(e)}',
            'crawled_url': url,
            'root_filter': root
        }), 500

@app.route('/discover-roots', methods=['GET'])
def discover_roots():
    """Find root URLs where individual store pages are built from"""
    homepage = request.args.get('url')
    if not homepage:
        return jsonify({'error': 'Missing url parameter'}), 400

    # Crawl website
    all_urls = crawl_website(homepage)
    
    # Find store roots
    store_roots = find_store_roots(all_urls)
    
    return jsonify({
        'website': homepage,
        'total_urls': len(all_urls),
        'store_roots': store_roots,
        'roots_count': len(store_roots),
        'usage': 'Append store names to these roots to build individual store URLs'
    })

@app.route('/crawl-only', methods=['GET'])
def crawl_only():
    """Only crawl website and return all URLs (no Llama analysis)"""
    homepage = request.args.get('url')
    if not homepage:
        return jsonify({'error': 'Missing url parameter'}), 400

    # Just crawl website
    all_urls = crawl_website(homepage)
    
    return jsonify({
        'website': homepage,
        'total_urls': len(all_urls),
        'all_urls': all_urls
    })

@app.route('/llama', methods=['POST'])
def llama_endpoint():
    """Expose Llama via endpoint - send prompt, get response"""
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing prompt in request body'}), 400
    
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 1000)
    temperature = data.get('temperature', 0.1)
    
    # Use centralized Llama call
    llama_result = call_llama(prompt, max_tokens, temperature)
    
    if llama_result["success"]:
        return jsonify({
            'prompt': prompt,
            'response': llama_result["response"],
            'model': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
            'max_tokens': max_tokens,
            'temperature': temperature
        })
    else:
        return jsonify({'error': f'Llama API call failed: {llama_result["error"]}'}), 500


@app.route('/discover', methods=['GET'])
def discover():
    """Find store directory pages"""
    homepage = request.args.get('url')
    filter_text = request.args.get('filter')  # New optional filter argument

    if not homepage:
        return jsonify({'error': 'Missing url parameter'}), 400

    # Crawl website
    all_urls = crawl_website(homepage)

    # Find directory pages
    directories = find_directory_pages(all_urls)

    # Filter results if filter_text is provided
    if filter_text:
        directories = [url for url in directories if filter_text.lower() in url.lower()]

    return jsonify({
        'website': homepage,
        'total_urls': len(all_urls),
        'discovered_urls': directories,
        'directory_count': len(directories),
        'filtered_by': filter_text if filter_text else None
    })


@app.route('/parse-shop', methods=['GET'])
def parse_shop():
   """
   Extract structured information from a shop/store page
   Expected: GET /parse-shop?url=https://example.com/store/shop-name
   """
   rate_limit()  # Add rate limiting
   
   url = request.args.get('url')
   
   if not url:
       return jsonify({'error': 'Missing url parameter'}), 400

   try:
       print(f"🏪 Parsing shop page: {url}")
       
       # Get HTML content using our safe_request function
       response = safe_request(url, timeout=15)
       
       if not response:
           return jsonify({
               'success': False,
               'error': 'Failed to fetch shop page (network/DNS issue)',
               'shop_url': url
           }), 200  # Return 200 to not break n8n
       
       # Parse HTML and clean it
       soup = BeautifulSoup(response.content, 'html.parser')
       
       # Remove noise elements
       for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
           element.decompose()
       
       # Get clean text
       clean_text = soup.get_text(separator=' ', strip=True)
       
       # Limit text length for LLM (keep it reasonable)
       if len(clean_text) > 3000:
           clean_text = clean_text[:3000] + "..."
       
       print(f"📝 Extracted {len(clean_text)} characters of clean text")
       
       # Use Llama to extract structured information
       prompt = f"""Extract store/shop information from this webpage text. Return JSON only.

           Store page text:
           {clean_text}

           Extract these fields if available (use null if not found):
           - store_name: The name of the store/shop
           - description: Brief description of what they sell or do
           - phone: Phone number
           - hours: Opening hours or schedule
           - website: Store website URL
           - email: Email address
           - location: Floor, unit number, or specific location within mall
           - categories: Array of what they sell (clothing, food, electronics, etc.)
           - services: Array of services they offer

       JSON ONLY:
       {{"store_name": "...", "description": "...", "phone": "...", "hours": "...", "website": "...", "email": "...", "location": "...", "categories": [...], "services": [...]}}"""

       print(f"🤖 Sending to Llama for information extraction...")
       
       # Call Llama using our centralized function
       llama_result = call_llama(prompt, max_tokens=800, temperature=0.0)
       
       if not llama_result["success"]:
           return jsonify({
               'success': False,
               'error': f'Llama processing failed: {llama_result["error"]}',
               'shop_url': url
           }), 200  # Return 200 to not break n8n
       
       response_text = llama_result["response"]
       print(f"🤖 Llama response: {response_text[:200]}...")
       
       # Extract JSON using our improved function that handles ```json blocks
       extracted_info = extract_json_from_response(response_text)
       
       if extracted_info:
           print(f"✅ Successfully extracted shop information")
           
           return jsonify({
               'success': True,
               'shop_url': url,
               'extracted_info': extracted_info,
               'raw_text_length': len(clean_text)
           })
       else:
           print(f"❌ Failed to parse JSON from Llama response")
           return jsonify({
               'success': False,
               'error': 'Failed to parse Llama JSON response',
               'shop_url': url,
               'raw_response': response_text[:500]  # Truncate for safety
           }), 200  # Return 200 to not break n8n
       
   except Exception as e:
       print(f"💥 Unexpected error: {str(e)}")
       return jsonify({
           'success': False,
           'error': f'Unexpected error: {str(e)}',
           'shop_url': url
       }), 200  # Return 200 to not break n8n

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
