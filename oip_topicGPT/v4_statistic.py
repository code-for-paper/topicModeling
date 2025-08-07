import json

from transformers.data import data_collator
from util import get_config, load_corpus
from tqdm import tqdm
from collections import Counter
"""
Access files based on keywords output dictionary and perform statistics
Filter: remove keywords that appear less than 5 times in the text
Statistics: proportion of word count that certain keywords under a topic occupy in the form

Input:
    keywords.json path
    corpus path
    keywords.json format:
    {
    "0001768224": {
        "2025-03-06": [
            {
                "topic": "External Partnerships",
                "keywords": [
                    "partner",
                    "collaborat",
                    "develop",
                    "commercial",
                    "progress",
                    "trial",
                    "dosing"
                ]
            },
        ]
    }
}
Output:
    A dictionary containing statistical information for each company's report
    Statistics are scores for each topic. A topic score is the sum of word count proportions for all keywords of that topic
    {
        'cik':cik,
        'forms':{
            'date':{
                'topics':[
                    {
                        'topic':topic,
                        'score':score
                    }
                ]
            }
        }
    }
"""

def clean_duplicate(words: set[str]) -> set[str]:
    # Some words in words may be prefixes of other words, keep only the shortest prefix, delete others
    words_list = sorted(words, key=len)
    result = set()
    
    for word in words_list:
        # Check if current word starts with any already added word as prefix
        is_duplicate = False
        for existing in result:
            if word.startswith(existing):  # Current word has existing as prefix
                is_duplicate = True
                break
        
        if not is_duplicate:
            result.add(word)
    
    return result

def stat(corpus_path, keywords_path, stat_path):

    with open(keywords_path, 'r') as f:
        keywords_dict: dict = json.load(f)

    # Delete original stat file
    with open(stat_path, 'w') as f:
        f.truncate()
    # return

    corpus = load_corpus(corpus_path)

    for i, company in enumerate(tqdm(corpus, desc='Companies')):
        cik = company['cik']
        # print(f'cik: {cik}')
        if cik not in keywords_dict:
            continue
        date_topic_dict = keywords_dict[cik]
        # print(f'date_topic_dict:{date_topic_dict}')
        tenks = company['10-k']

        scores = {
            'cik': cik,
            'forms': {}
        }

        for tenk in tenks:
            date = str(tenk['date']).split(' ')[0]
            if date not in date_topic_dict:
                continue
            # print(f'{cik} {date} ','-'*50)
            # One date corresponds to one document and statistics
            result = []
            topics = date_topic_dict[date]

            business_text = tenk['form'][:5000]  # Truncate to max 5000 characters
            # Preprocess text into word list
            words = business_text.split()
            # Count each word
            word_counts = Counter(words)
            # Calculate total word count
            total_words = len(words)
            if type(topics) != list:
                continue
            
            all_keywords = set()
            
            for topic in topics:
                try:
                    topic_score = {
                        'topic': topic['topic'],
                        'score': 0,
                        'keyword_stat': {}
                    }
                except:
                    print(f'error topic:{topic},cik:{cik}')
                keywords = clean_duplicate(set(topic['keywords']))
                
                all_keywords.update(keywords)
                
                # Keywords are generally prefixes
                keyword_counter = Counter()
                for keyword in keywords:
                    for word in word_counts:
                        word: str = word.lower()
                        if word.startswith(keyword):
                            keyword_counter[keyword] += word_counts[word]

                # Normalize
                for key in keyword_counter:
                    # Words appearing 5 times or less don't count towards keyword contribution
                    if keyword_counter[key] < 5:
                        # print(f'key {key} occurrence count:{keyword_counter[key]}')
                        pass
                    else:
                        topic_score['score'] += keyword_counter[key]
                        topic_score['keyword_stat'][key] = keyword_counter[key]

                topic_score['score'] /= total_words
                if topic_score['score'] > 0:
                    result.append(topic_score)
            
            # Calculate OIP score. Since different topics may have same keywords, calculate independently to avoid duplication
            all_keywords = clean_duplicate(all_keywords)
            oip_score = 0
            for keyword in all_keywords:
                for word in word_counts:
                    if word.startswith(keyword):
                        oip_score += word_counts[word]
            oip_score /= total_words
            
            # Add results to scores
            if len(result) > 0:
                scores['forms'][date] = {
                    'score': result,
                    'refer_date': str(tenk['refer_date']).split()[0],
                    'refer_tobins_q': tenk['refer_tobins_q'],
                    'oip_score': oip_score
                }

        # Save scores to file
        if len(scores['forms']) > 0:
            with open(stat_path, 'a') as f:
                json.dump(scores, f)
                f.write('\n')
            # break


if __name__ == '__main__':
    config = get_config()
    stat(
        config['generation_raw']['input'],
        config['assignment']['keywords_output'],
        config['stat']['output']
    )
