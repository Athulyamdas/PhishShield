import requests
from bs4 import BeautifulSoup
import tldextract
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import Levenshtein
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


social_media_domains = ["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com"]
bank_terms = ["bank", "finance", "loan", "credit", "debit", "investment"]
pay_terms = ["pay", "payment", "wallet", "billing", "checkout"]
crypto_terms = ["crypto", "bitcoin", "blockchain", "ethereum", "coin", "token"]

tld_legitimacy = {'com': 0.95,'org': 0.90,'edu': 0.98,'gov': 0.99,'net': 0.85,'info': 0.75,'xyz': 0.40,'top': 0.30,'mil':0.99,'int':0.98,'biz':0.91,'co':0.90,
                  'us':0.96,'uk':0.96,'ca':0.97,'au':0.97,'in':0.95,'jp':0.97,'de':0.96,'fr':0.96,'it':0.96,'es':0.95,'nl':0.95,'ru':0.92,'cn':0.91,'br':0.95,
                  'za':0.94,'mx':0.93,'kr':0.94,'sg':0.94,'hk':0.93,'tw':0.93,'tr':0.92,'vn':0.92,'gr':0.94,'cz':0.94,'se':0.95,'pl':0.94,
                  'ch':0.95,'no':0.92  
}


df = pd.read_csv("phishing_url.csv")
urls_dataset = df['URL'].dropna().tolist()
    

# Function to check if URL is live
def is_url_live(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:           
            return True
    except requests.exceptions.RequestException as e:
        print(f"check 2: {e}  URL Live Check Error")
        return False


# Function to check if URL is responsive
def is_url_responsive(url):
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    
    chrome_driver_path = "C:/WebDriver/chromedriver.exe" 
    service = ChromeService(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        
        driver.get(url)
        time.sleep(2)  

        
        viewports = [320, 768, 1024]
        layouts = []

        for width in viewports:
            driver.set_window_size(width, 800)  
            time.sleep(2)
            
            layout = driver.find_element(By.TAG_NAME, 'body').get_attribute('innerHTML')
            layouts.append(layout)

        driver.quit()

        
        is_responsive = len(set(layouts)) > 1
        return 0 if is_responsive else 1  

    except Exception as e:
        print(f"Error checking responsiveness: {e}")
        driver.quit()
        return 1  


# Function to extract additional features from live website
def extract_additional_features(url, soup):
    features = {}
    
    
    features['Bank'] = 1 if any(term in url.lower() for term in bank_terms) else 0
    features['Pay'] = 1 if any(term in url.lower() for term in pay_terms) else 0
    features['Crypto'] = 1 if any(term in url.lower() for term in crypto_terms) else 0
    
    
    copyright_keywords = ["©", "copyright", "all rights reserved"]
    copyright_found = False
    for keyword in copyright_keywords:
        if soup.find_all(string=lambda text: text and keyword.lower() in text.lower()):
            copyright_found = True
            break
    features['HasCopyrightInfo'] = 1 if copyright_found else 0
    
    
    social_net_found = False
    for link in soup.find_all('a', href=True):
        href = link['href'].lower()
        if any(domain in href for domain in social_media_domains):
            social_net_found = True
            break
    features['HasSocialNet'] = 1 if social_net_found else 0
    
    return features

def url_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def check_obfuscation(url):
    obfuscated_patterns = [
        r"%[0-9A-Fa-f]{2}",   
        r"\\u[0-9A-Fa-f]{4}"  
    ]
    
    obfuscated_count = 0
    
    for pattern in obfuscated_patterns:
        matches = re.findall(pattern, url)
        obfuscated_count += len(matches)
    
    
    obfuscation_ratio = obfuscated_count / len(url) if len(url) > 0 else 0
    has_obfuscation = 1 if obfuscated_count > 0 else 0
    
    return has_obfuscation, obfuscated_count, obfuscation_ratio



def get_tld_legitimacy(url):
    domain = urlparse(url).netloc
    tld = domain.split('.')[-1] 
    return tld_legitimacy.get(tld, 0.50)


def get_url_char_prob(url):
    char_count = len(url)
    special_chars = re.findall(r'[^A-Za-z0-9]', url)
    special_count = len(special_chars)
    return (char_count - special_count) / char_count if char_count > 0 else 0


def check_robots_txt(url):
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=5)
        return 1 if response.status_code == 200 else 0
    except requests.exceptions.RequestException:
        return 0

def tokenize_url(url):
    parsed_url = urlparse(url)
    tokens = []
    tokens.append(parsed_url.netloc) 
    tokens.extend(parsed_url.path.split('/')) 
    tokens.append(parsed_url.query) 
    return [token for token in tokens if token]
tokenized_dataset = [tokenize_url(url) for url in urls_dataset]

def calculate_levenshtein_similarity(url, dataset):
    similarities = []
    for data_url in dataset:
        similarity = 1 - (Levenshtein.distance(url, data_url) / max(len(url), len(data_url)))
        similarities.append(similarity)
    return max(similarities) if similarities else 0


def calculate_cosine_similarity(url, dataset):
    vectorizer = CountVectorizer(analyzer='char')
    vectors = vectorizer.fit_transform([url] + dataset).toarray()
    similarities = cosine_similarity([vectors[0]], vectors[1:])[0]
    return max(similarities) if similarities.size > 0 else 0


def calculate_jaccard_similarity(url, dataset, ngram_range=(3, 3)):
    similarities = []

    def get_ngrams(text, n):
        return set([text[i:i + n] for i in range(len(text) - n + 1)])

    url_ngrams = get_ngrams(url, ngram_range[0])

    for data_url in dataset:
        data_url_ngrams = get_ngrams(data_url, ngram_range[0])

        # Calculate Jaccard Similarity
        intersection = len(url_ngrams & data_url_ngrams)
        union = len(url_ngrams | data_url_ngrams)
        similarity = intersection / union if union > 0 else 0

        similarities.append(similarity)


    return max(similarities) if similarities else 0


def calculate_url_similarity_index(url, dataset):
    lev_sim = calculate_levenshtein_similarity(url, dataset)
    cos_sim = calculate_cosine_similarity(url, dataset)
    jacc_sim = calculate_jaccard_similarity(url, dataset)
    similarity_index = (lev_sim + cos_sim + jacc_sim) / 3
    return similarity_index


# Function to extract features from live website

def extract_features_from_live_site(url):
    features = {}
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        response = session.get(url, headers=headers, timeout=10, allow_redirects=True, verify=True)
        
        
        if response.status_code == 200:
            print("Page loaded successfully")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            features['URLLength'] = len(url)
            features['IsHTTPS'] = 1 if url.startswith('https') else 0
            features['NoOfImage'] = len(soup.find_all('img'))
            features['NoOfCSS'] = len(soup.find_all('link', {'rel': 'stylesheet'}))
            features['NoOfJS'] = len(soup.find_all('script'))
            features['HasTitle'] = 1 if soup.title else 0
            features['HasFavicon'] = 1 if soup.find('link', {'rel': 'icon'}) else 0
            features['NoOfiFrame'] = len(soup.find_all('iframe'))
            features['HasSubmitButton'] = 1 if soup.find('input', {'type': 'submit'}) else 0
            features['HasPasswordField'] = 1 if soup.find('input', {'type': 'password'}) else 0
            features['NoOfEmptyRef'] = len([a for a in soup.find_all('a') if not a.get('href')])
            features['NoOfExternalRef'] = len([a for a in soup.find_all('a') if a.get('href') and a['href'].startswith('http')])
            
            parsed_url = tldextract.extract(url)
            domain = f"www.{parsed_url.domain}.{parsed_url.suffix}"
            features['Domain'] = domain
            
            features['DomainLength'] = len(parsed_url.domain)
            
            features['TLD'] = parsed_url.suffix
            features['TLDLength']=len(features['TLD'])
            features['IsDomainIP'] = 1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", parsed_url.domain) else 0
            features['NoOfSubDomain'] = len(parsed_url.subdomain.split('.')) if parsed_url.subdomain else 0
            features['NoOfLettersInURL'] = len(re.findall(r'[A-Za-z]', url))
            features['LetterRatioInURL'] = features['NoOfLettersInURL'] / len(url) if len(url) > 0 else 0
            features['NoOfDegitsInURL'] = len(re.findall(r'[0-9]', url))
            features['DigitRatioInURL'] = features['NoOfDegitsInURL'] / len(url) if len(url) > 0 else 0
            features['NoOfEqualsInURL'] = url.count('=')
            features['NoOfQMarkInURL'] = url.count('?')
            features['NoOfAmpersandInURL'] = url.count('&')
            features['NoOfOtherSpecialCharsInURL'] = len(re.findall(r'[@_!#$%^*()<>?/|}{~:]', url))
            
            features['IsResponsive'] = is_url_responsive(url)

            additional_features = extract_additional_features(url, soup)
            features.update(additional_features)

            features['NoOfURLRedirect'] = len(response.history)
            features['HasDescription'] = 1 if soup.find('meta', attrs={'name': 'description'}) else 0

            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_driver_path = "C:/WebDriver/chromedriver.exe" 
            service = ChromeService(executable_path=chrome_driver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get(url)
            time.sleep(3)  
            popups = driver.execute_script("return window.open.length;")
            features['NoOfPopup'] = popups
            driver.quit()

            forms = soup.find_all('form')
            external_form_submit = 0
            for form in forms:
                action = form.get('action')
                if action and not action.startswith(url):
                    external_form_submit = 1
                    break
            features['HasExternalFormSubmit'] = external_form_submit

            hidden_fields = len(soup.find_all('input', {'type': 'hidden'}))
            features['HasHiddenFields'] = 1 if hidden_fields > 0 else 0

            self_redirects = 0
            self_refs = 0
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if domain in href:
                    self_refs += 1
                    if href != url:
                        self_redirects += 1
            features['NoOfSelfRedirect'] = self_redirects
            features['NoOfSelfRef'] = self_refs

            title_tag = soup.title
            title = title_tag.string.strip() if title_tag else ""
            features['Title'] = title

            features['DomainTitleMatchScore'] = url_similarity(domain, title)

            url_words = re.findall(r'\b\w+\b', url)
            url_text = ' '.join(url_words)
            features['URLTitleMatchScore'] = url_similarity(url_text, title)

            has_obfuscation, obfuscated_count, obfuscation_ratio = check_obfuscation(url)
            features['HasObfuscation'] = has_obfuscation
            features['NoOfObfuscatedChar'] = obfuscated_count
            features['ObfuscationRatio'] = obfuscation_ratio


            page_content = str(soup.prettify())
            features['LineOfCode'] = len(page_content.splitlines())

    
            lines = page_content.splitlines()
            largest_line_length = max(len(line) for line in lines)
            features['LargestLineLength'] = largest_line_length
    
    
            continuation_chars = ['\\', '%']
            continuation_lines = sum(1 for line in lines if line and line[-1] in continuation_chars)
            features['CharContinuationRate'] = continuation_lines / features['LineOfCode'] if features['LineOfCode'] > 0 else 0
    
    
            special_chars = r"!@#$%^&*()_+-=[]{}|;:',.<>?/"
            special_char_count = sum(1 for char in url if char in special_chars)
            features['SpacialCharRatioInURL'] = special_char_count / len(url) if len(url) > 0 else 0

                        
            features['TLDLegitimateProb'] = get_tld_legitimacy(url)
            features['URLCharProb'] = get_url_char_prob(url)
            features['Robots'] = check_robots_txt(url)

            URLSimilarityIndex = calculate_url_similarity_index(url, urls_dataset)
            features['URLSimilarityIndex'] = float(URLSimilarityIndex)*100
            
            
        else:
            print(f"Failed to load page. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error extracting features: {e}")
    return features



# Main function
if __name__ == "__main__":
    input_url = input("Enter URL: ")
    if is_url_live(input_url):
        features = extract_features_from_live_site(input_url)
        print("Extracted Features:", features)
    else:
        print("URL is not live.")
    
