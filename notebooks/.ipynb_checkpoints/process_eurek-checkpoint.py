import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count


'''

def extract_related_journal_article_link(soup):
    related_journal_article_link = None
    related_journal_article_block = soup.find(class_='widget hidden-print')
    print(related_journal_article_block)
    if related_journal_article_block:
        related_journal_article_link_tag = related_journal_article_block.find('a', rel='nofollow')
        if related_journal_article_link_tag:
            related_journal_article_link = related_journal_article_link_tag['href']
    return related_journal_article_link
'''

def extract_links(soup):
    
    original_source_link = None
    related_journal_article = None
    
    original_source_tag = soup.find('h4', text='Original Source')
    if original_source_tag:
        original_source_link = original_source_tag.find_next('a')['href']
    
    related_journal_article_tag = soup.find('h4', text='Related Journal Article')
    if related_journal_article_tag:
        related_journal_article = related_journal_article_tag.find_next('a')['href']
    
    return original_source_link, related_journal_article

def extract_php_file_data(file_path):
    try:
        with open(file_path, 'r') as php_file:
            php_code = php_file.read()
            soup = BeautifulSoup(php_code, 'html.parser')
            
            release_date = soup.find(class_='release_date').text.strip()
            page_title = soup.find(class_='page_title').text.strip()
            meta_institute = soup.find(class_='meta_institute').text.strip()
            
            media_contact_div = soup.find(class_='contact-info')
            media_contact = extract_media_contact(media_contact_div)
            
            keywords_ul = soup.find(class_='tags')
            keywords = extract_keywords(keywords_ul)
            
            source_link, doi_link = extract_links(soup)
            
            full_text = clean_text(soup.get_text())
            
            return {
                'release_date': release_date,
                'page_title': page_title,
                'meta_institute': meta_institute,
                'media_contact': media_contact,
                'keywords': keywords,
                'full_text': full_text,
                'original_source': source_link,
                'related_journal_article_link': doi_link
            }
    except Exception as e:
        #print(f"An error occurred while parsing {file_path}: {str(e)}")
        return None

def extract_media_contact(media_contact_div):
    contact_info = {}
    if media_contact_div:
        strong_tag = media_contact_div.find('strong', string='Media Contact')
        
        if strong_tag:
            paragraphs = media_contact_div.find_all('p')
            if paragraphs:
                contact_info['name'] = paragraphs[1].get_text().strip().split('\n')[0]
                email_link = paragraphs[1].find('a', href=re.compile(r'mailto:'))
                if email_link:
                    contact_info['email'] = email_link['href'].replace('mailto:', '').strip()
                phone_match = re.match(r'(\d{2,}-\d{2,})', paragraphs[1].get_text().strip())
                if phone_match:
                    contact_info['phone'] = phone_match.group(1)
                twitter_link = paragraphs[1].find('a', href=re.compile(r'twitter\.com'))
                if twitter_link:
                    contact_info['twitter'] = twitter_link.get_text().replace('@', '').strip()
                website_link = paragraphs[-1].find('a', href=re.compile(r'http'))
                if website_link:
                    contact_info['website'] = website_link['href'].strip()
    return contact_info
    
#def extract_media_contact(media_contact_div):
#    if media_contact_div:
#        media_contact_text = media_contact_div.get_text()
#        return media_contact_text.strip()
#    return None

def extract_keywords(keywords_ul):
    if keywords_ul:
        keywords = [keyword.get_text() for keyword in keywords_ul.find_all('a')]
        return keywords
    return []

def clean_text(text):
    # Replace multiple consecutive newline characters with one newline character using regex
    cleaned_text = re.sub(r'\n+', '\n', text)
    return cleaned_text.strip()

def save_to_jsonl(parsed_data, output_file):
    with open(output_file, 'w') as jsonl_file:
        for data in parsed_data:
            jsonl_file.write(json.dumps(data) + '\n')

            
def process_directory(subdir_path):
    parsed_data = []
    for file in os.listdir(subdir_path):
        if file.endswith('.php'):
            file_path = os.path.join(subdir_path, file)
            data = extract_php_file_data(file_path)
            if data:
                parsed_data.append(data)
    
    output_file = output_file = subdir_path.replace('pub_releases', 'extracted4match') + '.jsonl'
    save_to_jsonl(parsed_data, output_file)
    print(len(parsed_data), 'lines saved to', output_file)

def process(base_directory):
    with Pool(processes=12) as pool:
        pool.map(process_directory, [os.path.join(base_directory, subdir) for subdir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, subdir)) and subdir[:4] >= '2013'])
            
directory_path = '/shared/3/projects/jiaxin/datasets/eurekalert/wget/archive.eurekalert.org/pub_releases/'
process(directory_path)