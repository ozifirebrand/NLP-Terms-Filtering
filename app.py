from flask import Flask, request, jsonify
import logging
from collections import defaultdict

app = Flask(__name__)

import requests
from bs4 import BeautifulSoup
import nltk
import re
from collections import defaultdict, OrderedDict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import logging
from bs4.element import Tag
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import warnings

# Install and import the contractions library
try:
    import contractions
except ImportError:
    print("Contractions library not found. Installing...")
    import subprocess
    subprocess.call(['pip', 'install', 'contractions'])
    import contractions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize spaCy's English model for NER and POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.info("Downloading 'en_core_web_sm' model for spaCy as it was not found.")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define essential stopwords to retain within phrases (if any)
ESSENTIAL_STOPWORDS = set()

# Define patterns to identify cookie banners
COOKIE_BANNER_IDENTIFIERS = [
    re.compile(r'.*\bcookie\b.*', re.IGNORECASE),
    re.compile(r'.*\bconsent\b.*', re.IGNORECASE),
    re.compile(r'.*\bprivacy\b.*', re.IGNORECASE),
    re.compile(r'.*\bgdpr\b.*', re.IGNORECASE),
    re.compile(r'.*\bterms\b.*', re.IGNORECASE)
]

# Define patterns to identify review sections
REVIEW_IDENTIFIERS = [
    re.compile(r'.*\breview\b.*', re.IGNORECASE),
    re.compile(r'.*\btestimonial\b.*', re.IGNORECASE),
    re.compile(r'.*\bfeedback\b.*', re.IGNORECASE),
    re.compile(r'.*\bratings\b.*', re.IGNORECASE)
]

def download_nltk_resources():
    resources = ['punkt', 'wordnet', 'omw-1.4', 'stopwords']
    for resource in resources:
        try:
            if resource == 'punkt':
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            logging.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

def is_cookie_banner(tag):
    if not isinstance(tag, Tag) or tag.attrs is None:
        return False
    for attr in ['class', 'id', 'role', 'aria-label']:
        attr_values = tag.get(attr)
        if attr_values:
            if isinstance(attr_values, list):
                attr_values = ' '.join(attr_values)
            for pattern in COOKIE_BANNER_IDENTIFIERS:
                if pattern.match(attr_values):
                    return True
    return False

def is_review_section(tag):
    if not isinstance(tag, Tag) or tag.attrs is None:
        return False
    for attr in ['class', 'id', 'role', 'aria-label']:
        attr_values = tag.get(attr)
        if attr_values:
            if isinstance(attr_values, list):
                attr_values = ' '.join(attr_values)
            for pattern in REVIEW_IDENTIFIERS:
                if pattern.match(attr_values):
                    return True
    return False

def scrape_website(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        site_name_meta = soup.find('meta', property='og:site_name')
        site_name = site_name_meta['content'].strip().lower() if site_name_meta and site_name_meta.get('content') else ''

        excluded_tags = ['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'button', 'input']
        for tag in soup.find_all(excluded_tags):
            tag.decompose()

        # Remove cookie banners and review sections
        for tag in soup.find_all(True):
            if is_cookie_banner(tag) or is_review_section(tag):
                tag.decompose()

        texts = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p']):
            text = tag.get_text(separator=' ').strip()
            if text:
                texts.append(text)

        clean_text = ' '.join(texts)
        return clean_text, site_name
    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping {url}: {e}")
        return '', ''

def scrape_competitors(urls):
    all_text = []
    site_names = set()
    failed_urls = []
    for url in urls:
        print(f"Scraping: {url}")
        content, site_name = scrape_website(url)
        if content:
            all_text.append(content)
        else:
            failed_urls.append(url)
        if site_name:
            site_names.add(site_name)
    if failed_urls:
        print(f"\nFailed to scrape content from the following URLs:")
        for url in failed_urls:
            print(f"- {url}")
    return all_text, site_names

def preprocess_text(text):
    # Normalize apostrophes and quotation marks
    text = text.replace("’", "'")  # Replace Unicode apostrophes with ASCII apostrophe
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    # Expand contractions
    text = contractions.fix(text)
    # Process text with spaCy
    doc = nlp(text.lower())
    lemmatized_tokens = []
    for token in doc:
        if not token.is_punct and not token.is_space:
            if len(token.text.strip()) > 1:
                lemmatized_tokens.append(token.lemma_)
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

def extract_keywords_tfidf(documents, max_features=5000):
    def custom_analyzer(text):
        tokens = text.split()
        ngrams = []
        for n in range(1, 5):  # Include 4-word phrases
            ngrams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
        return ngrams

    vectorizer = TfidfVectorizer(
        analyzer=custom_analyzer,
        max_features=max_features,
        stop_words=None,
        lowercase=False
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    tfidf_pairs = list(zip(feature_names, tfidf_scores))
    sorted_tfidf = sorted(tfidf_pairs, key=lambda x: x[1], reverse=True)
    return sorted_tfidf, vectorizer

def compute_total_counts(keywords, documents):
    keyword_counts = {}
    for keyword in keywords:
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        counts_per_page = [len(pattern.findall(doc)) for doc in documents]
        total = sum(counts_per_page)
        max_count = max(counts_per_page) if counts_per_page else 0
        min_count = min(counts_per_page) if counts_per_page else 0
        avg_count = math.ceil(total / len(documents)) if documents else 0
        range_count = max_count - min_count
        keyword_counts[keyword.lower()] = {
            'total': total,
            'max': max_count,
            'min': min_count,
            'avg': avg_count,
            'range': range_count,
            'counts_per_page': counts_per_page
        }
    return keyword_counts

def is_relevant(term, counts_info, site_names, site_name_phrases, seed_variations, ngram_inclusion_limits, ngram_inclusion_limits_default):
    term_lower = term.lower()

    # Always include seed keyword and its variations
    if term_lower in seed_variations:
        logging.debug(f"Term '{term}' is a seed variation and is automatically included.")
        return term_lower

    # Additional boundary stopwords
    additional_boundary_stopwords = set([
        'you', 'your', 'they', 'their', 'can', 'could', 'will', 'would', 'shall', 'should',
        'may', 'might', 'must', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'if', 'for', 'in', 'on', 'with', 'as', 'by',
        'at', 'from', 'into', 'about', 'than', 'so', 'that', 'the', 'a', 'an', 'it', 'i',
        'we', 'he', 'she', 'him', 'her', 'us', 'our', 'ours', 'my', 'mine', 'need', 'get',
        'apply', 'provide', 'include', 'make', 'use', 'take', 'go', 'come', 'say', 'see',
        'know', 'give', 'find', 'think', 'tell', 'become', 'show', 'leave', 'feel', 'put',
        'bring', 'mean', 'keep', 'let', 'begin', 'seem', 'help', 'talk', 'turn', 'start',
        'might', 'must', 'could', 'would', 'should', 'do', 'does', 'did', 'done', 'doing',
        'll', 're', 've', 's', 'd', 'm', 't', 'cookie', 'cookies', 'consent', 'settings', 'privacy'
    ])
    boundary_stopwords = set(stopwords.words('english')).union(additional_boundary_stopwords)
    boundary_stopwords = boundary_stopwords - ESSENTIAL_STOPWORDS
    boundary_stopwords = {word.lower() for word in boundary_stopwords}

    words = [word.lower() for word in term.split()]

    # Strip boundary stopwords from start and end of the phrase, regardless of length
    while words and words[0] in boundary_stopwords:
        words.pop(0)
    while words and words[-1] in boundary_stopwords:
        words.pop()
    if not words:
        logging.debug(f"Term '{term}' excluded: no words left after stripping stopwords.")
        return None
    stripped_term = ' '.join(words)

    # Remove terms with numbers
    if re.search(r'\d', stripped_term):
        logging.debug(f"Term '{term}' excluded: contains numbers.")
        return None

    # Exclude site names and their phrases
    if stripped_term in site_name_phrases:
        logging.debug(f"Term '{term}' excluded: matches site name phrases.")
        return None
    if stripped_term in site_names:
        logging.debug(f"Term '{term}' excluded: matches site names.")
        return None

    num_words = len(stripped_term.split())

    # Exclude single-word verbs that are stopwords
    if num_words == 1:
        doc = nlp(stripped_term)
        if doc[0].pos_ == 'VERB' and stripped_term in boundary_stopwords:
            logging.debug(f"Term '{term}' excluded: single-word verb stopword.")
            return None

    # Apply inclusion limits based on n-gram length
    ngram_limit = ngram_inclusion_limits.get(num_words, ngram_inclusion_limits_default)
    ngram_rank = counts_info.get(term_lower, {}).get('ngram_rank', None)
    if ngram_rank is None:
        logging.debug(f"Term '{term}' excluded: ngram_rank not found.")
        return None
    if ngram_rank > ngram_limit:
        logging.debug(f"Term '{term}' excluded: ngram_rank {ngram_rank} exceeds limit {ngram_limit}.")
        return None

    return stripped_term

def generate_seed_variations(seed_keyword):
    words = seed_keyword.lower().split()
    variations = set()
    n = len(words)
    max_ngram = n  # Use n as the maximum n-gram length for variations
    for ngram in range(1, max_ngram +1):
        for i in range(len(words) - ngram +1):
            variation = ' '.join(words[i:i+ngram])
            variations.add(variation)
    return variations

def deduplicate_terms(filtered_keywords, seed_variations, seed_keyword, final_limit=100):
    final_keywords = []
    existing_terms = {}

    # 1. Add seed keyword first
    seed_keyword_lower = seed_keyword.lower()
    for term, score in filtered_keywords:
        if term.lower() == seed_keyword_lower:
            final_keywords.append((term, score))
            existing_terms[term.lower()] = (term, score)
            break

    # 2. Add other seed variations
    for variation in seed_variations:
        variation_lower = variation.lower()
        if variation_lower == seed_keyword_lower:
            continue
        for term, score in filtered_keywords:
            if term.lower() == variation_lower and term.lower() not in existing_terms:
                final_keywords.append((term, score))
                existing_terms[term.lower()] = (term, score)
                break

    # 3. Add remaining keywords, ensuring no exact duplicates and applying deduplication rules
    remaining_keywords = [item for item in filtered_keywords if item[0].lower() not in existing_terms]

    # Sort remaining keywords by TF-IDF score (descending)
    remaining_sorted = sorted(remaining_keywords, key=lambda x: -x[1])

    for term, score in remaining_sorted:
        term_lower = term.lower()
        term_words = term_lower.split()
        is_duplicate = False

        # Compare with existing terms
        for existing_term_lower, (existing_term, existing_score) in list(existing_terms.items()):
            existing_term_words = existing_term_lower.split()

            # Exclude exact duplicates
            if term_lower == existing_term_lower:
                is_duplicate = True
                break

            # Rule 1: Conflict between 2-word and 3-word terms
            if len(term_words) == 2 and len(existing_term_words) == 3:
                if all(word in existing_term_words for word in term_words):
                    # Both words of the 2-word term exist in the 3-word term
                    if existing_score > 0.08 or existing_score > score:
                        # The 3-word term takes priority
                        is_duplicate = True
                        break
                    else:
                        # The 2-word term takes priority
                        # Remove the existing 3-word term
                        final_keywords = [item for item in final_keywords if item[0].lower() != existing_term_lower]
                        existing_terms.pop(existing_term_lower)
                        break
            elif len(term_words) == 3 and len(existing_term_words) == 2:
                if all(word in term_words for word in existing_term_words):
                    # Both words of the 2-word term exist in the 3-word term
                    if score > 0.08 or score > existing_score:
                        # The 3-word term takes priority
                        # Remove the existing 2-word term
                        final_keywords = [item for item in final_keywords if item[0].lower() != existing_term_lower]
                        existing_terms.pop(existing_term_lower)
                        break
                    else:
                        # The 2-word term takes priority
                        is_duplicate = True
                        break

            # Rule 2: Conflict between 1-word and 3+ word terms
            if len(term_words) >= 3 and len(existing_term_words) == 1:
                if existing_term_words[0] in term_words:
                    if score > 0.1 and existing_score <= 0.1:
                        # 3+ word term takes priority
                        # Remove the existing 1-word term
                        final_keywords = [item for item in final_keywords if item[0].lower() != existing_term_lower]
                        existing_terms.pop(existing_term_lower)
                        break
                    elif score > 0.1 and existing_score > 0.1:
                        # Both terms can show up
                        pass
                    else:
                        # Existing 1-word term takes priority
                        is_duplicate = True
                        break
            elif len(term_words) == 1 and len(existing_term_words) >= 3:
                if term_words[0] in existing_term_words:
                    if existing_score > 0.1 and score <= 0.1:
                        # Existing 3+ word term takes priority
                        is_duplicate = True
                        break
                    elif existing_score > 0.1 and score > 0.1:
                        # Both terms can show up
                        pass
                    else:
                        # 1-word term takes priority
                        # Remove existing 3+ word term
                        final_keywords = [item for item in final_keywords if item[0].lower() != existing_term_lower]
                        existing_terms.pop(existing_term_lower)
                        break

            # Rule 3: Conflict between 3-word and 4-word terms
            if len(term_words) == 3 and len(existing_term_words) == 4:
                if all(word in existing_term_words for word in term_words):
                    # The 3-word term exists fully in the 4-word term
                    if existing_score > 0.08 or existing_score > score:
                        # The 4-word term takes priority
                        is_duplicate = True
                        break
                    else:
                        # The 3-word term takes priority
                        # Remove the existing 4-word term
                        final_keywords = [item for item in final_keywords if item[0].lower() != existing_term_lower]
                        existing_terms.pop(existing_term_lower)
                        break
            elif len(term_words) == 4 and len(existing_term_words) == 3:
                if all(word in term_words for word in existing_term_words):
                    # The existing 3-word term exists fully in the 4-word term
                    if score > 0.08 or score > existing_score:
                        # The 4-word term takes priority
                        # Remove the existing 3-word term
                        final_keywords = [item for item in final_keywords if item[0].lower() != existing_term_lower]
                        existing_terms.pop(existing_term_lower)
                        break
                    else:
                        # The 3-word term takes priority
                        is_duplicate = True
                        break

        if not is_duplicate:
            final_keywords.append((term, score))
            existing_terms[term_lower] = (term, score)

        if len(final_keywords) >= final_limit:
            break

    return final_keywords

def filter_keywords(tfidf_keywords, counts_info, site_names, site_name_phrases, seed_variations, ngram_inclusion_limits, ngram_inclusion_limits_default):
    filtered_keywords = []
    for term, score in tfidf_keywords:
        stripped_term = is_relevant(
            term, counts_info, site_names, site_name_phrases, seed_variations, ngram_inclusion_limits, ngram_inclusion_limits_default
        )
        if stripped_term:
            filtered_keywords.append((stripped_term, score))
    return filtered_keywords


@app.route('/extract_keywords', methods=['POST'])
def extract_keywords_api():
    data = request.json
    urls = data.get("urls", [])
    seed_keyword = data.get("seed_keyword", "")
    
    if not urls:
        return jsonify({"error": "URLs: not found"}), 400
    if not seed_keyword:
        return jsonify({"error":"seed keyword: not found"}), 400
    
    # Step 1: Download NLTK resources
    download_nltk_resources()
    
    # Step 2: Scrape competitor contents
    competitor_texts, site_names = scrape_competitors(urls)
    if not competitor_texts:
        return jsonify({"error": "No content scraped from provided URLs"}), 400
    
    # Step 3: Preprocess each competitor's text
    lemmatized_texts = [preprocess_text(text) for text in competitor_texts]
    
    # Step 4: Extract keywords using TF-IDF
    tfidf_keywords, vectorizer = extract_keywords_tfidf(lemmatized_texts, max_features=5000)
    
    # Step 5: Compute total counts for all potential keywords
    potential_keywords = [term for term, score in tfidf_keywords]
    counts_info = compute_total_counts(potential_keywords, lemmatized_texts)
    
    # Step 6: Rank terms within their n-gram groups
    ngram_groups = defaultdict(list)
    for term, score in tfidf_keywords:
        num_words = len(term.split())
        ngram_groups[num_words].append((term.lower(), score))
    
    for num_words, terms in ngram_groups.items():
        terms_sorted = sorted(terms, key=lambda x: -x[1])
        for rank, (term_lower, score) in enumerate(terms_sorted, start=1):
            counts_info[term_lower]['ngram_rank'] = rank
    
    # Step 7: Generate site name sequential phrases
    site_name_phrases = {" ".join(site_name.split()[i:i+2]).lower() for site_name in site_names for i in range(len(site_name.split()) - 1)}
    
    # Step 8: Generate seed keyword variations
    seed_variations = generate_seed_variations(seed_keyword)
    
    # Step 9: Define inclusion limits based on n-gram length
    ngram_inclusion_limits = {1: 50, 2: 150, 3: 150, 4: 150}
    
    # Step 10: Filter keywords based on relevancy
    filtered_keywords = filter_keywords(tfidf_keywords, counts_info, site_names, site_name_phrases, seed_variations, ngram_inclusion_limits, 50)
    
    # Step 11: Ensure seed keyword and its variations are included
    for variation in seed_variations:
        if variation not in [term.lower() for term, _ in filtered_keywords]:
            score = next((score for term, score in tfidf_keywords if term.lower() == variation), 0)
            if score > 0:
                filtered_keywords.append((variation, score))
    
    # Step 12: Limit multi-word phrases (3+ words) to top N
    multi_word_phrases_sorted = sorted([item for item in filtered_keywords if len(item[0].split()) >= 3], key=lambda x: -x[1])[:150]
    filtered_keywords = [item for item in filtered_keywords if len(item[0].split()) < 3] + multi_word_phrases_sorted
    
    # Step 13: Deduplicate terms
    final_keywords = deduplicate_terms(filtered_keywords, seed_variations, seed_keyword, 100)
    
    # Step 14: Prepare response format
    keyword_counts_display = {term.lower(): f"{counts_info.get(term.lower(), {}).get('avg', 0)}-{counts_info.get(term.lower(), {}).get('range', 0)}" for term, _ in final_keywords}
    
    # Combine all keywords into a single list
    all_keywords = [
        {"term": term, "score": score, "suggested_count": keyword_counts_display.get(term.lower(), '0-0')}
        for term, score in final_keywords
    ]
    
    # Sort the combined list by score in descending order
    all_keywords_sorted = sorted(all_keywords, key=lambda x: -x["score"])
    
    # Return the single list in the response
    response = {"keywords": all_keywords_sorted}
    
    return jsonify(response)

# Load the pre-trained sentence transformer model
sentence_model = SentenceTransformer('all-mpnet-base-v2')


@app.route('/cluster', methods=['POST'])
def cluster_headings():
    try:
        warnings.filterwarnings('ignore')

        # Parse request JSON
        data = request.get_json()
        headings = data.get("headings", [])
        distance_threshold = data.get("distance_threshold", 0.36)

        if not headings:
            return jsonify({"error": "Headings list cannot be empty"}), 400

        # Generate embeddings
        embeddings = sentence_model.encode(headings)

        # Compute the cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Convert similarity to distance (distance = 1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # Perform clustering
        clustering = AgglomerativeClustering(
            metric='precomputed',
            linkage='average',
            distance_threshold=distance_threshold,
            n_clusters=None
        )

        labels = clustering.fit_predict(distance_matrix)

        # Group headings into clusters
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(headings[idx])

        # Select representative headings
        representative_headings = {}
        for label, cluster_headings in clusters.items():
            if len(cluster_headings) == 1:
                representative = cluster_headings[0]
            else:
                indices = [headings.index(h) for h in cluster_headings]
                cluster_distances = distance_matrix[np.ix_(indices, indices)]
                avg_distances = cluster_distances.mean(axis=1)
                representative = cluster_headings[np.argmin(avg_distances)]
            representative_headings[label] = representative

        # Maintain order of first occurrence
        ordered_representatives = OrderedDict()
        for idx, heading in enumerate(headings):
            label = labels[idx]
            representative = representative_headings[label]
            if representative not in ordered_representatives:
                ordered_representatives[representative] = None

        return jsonify({"representative_headings": list(ordered_representatives.keys())})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "Konvart helper app is up"

if __name__ == '__main__':
    app.run(debug=True)