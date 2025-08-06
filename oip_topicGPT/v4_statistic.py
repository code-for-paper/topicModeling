import json

from transformers.data import data_collator
from util import get_config, load_corpus
from tqdm import tqdm
from collections import Counter
"""
根据keywords output的字典 来访问文件 并进行统计
过滤: 去除掉在文中出现次数少于5次的keyword
统计信息有: 某个topic下的某个keyword的占据form的单词数量比例 

输入:
    keywords.json路径
    corpus路径
    keywords.json的格式为:
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
输出:
    一个字典 包含各个公司的一个报表的统计信息
    统计信息是各个topic的分数 某个topic分数由这个topic的所有keyword的单词数量比例 求和得到
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
    # words中有的单词可能是其他单词的前缀，只保留最短的前缀，删掉其他的
    words_list = sorted(words, key=len)
    result = set()
    
    for word in words_list:
        # 检查当前单词是否以已添加的任何单词为前缀
        is_duplicate = False
        for existing in result:
            if word.startswith(existing):  # 当前word以existing为前缀
                is_duplicate = True
                break
        
        if not is_duplicate:
            result.add(word)
    
    return result

def stat(corpus_path, keywords_path, stat_path):

    with open(keywords_path, 'r') as f:
        keywords_dict: dict = json.load(f)

    # 删除原来的stat文件
    with open(stat_path, 'w') as f:
        f.truncate()
    # return

    corpus = load_corpus(corpus_path)

    for i, company in enumerate(tqdm(corpus, desc='公司')):
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
            # 一个日期对应一个文档 以及统计信息
            result = []
            topics = date_topic_dict[date]

            business_text = tenk['form'][:5000]  # 截断最长为5000字符
            # 把文本预处理为单词列表
            words = business_text.split()
            # 对每个单词进行计数
            word_counts = Counter(words)
            # 计算总单词数
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
                
                # 关键词 一般是前缀
                keyword_counter = Counter()
                for keyword in keywords:
                    for word in word_counts:
                        word: str = word.lower()
                        if word.startswith(keyword):
                            keyword_counter[keyword] += word_counts[word]

                # 归一化
                for key in keyword_counter:
                    # 少于等于5次的单词不算入关键词贡献
                    if keyword_counter[key] < 5:
                        # print(f'key {key} 出现次数:{keyword_counter[key]}')
                        pass
                    else:
                        topic_score['score'] += keyword_counter[key]
                        topic_score['keyword_stat'][key] = keyword_counter[key]

                topic_score['score'] /= total_words
                if topic_score['score'] > 0:
                    result.append(topic_score)
            
            # 计算OIP score 因为可能不同的topic有同一个keyword 为了避免重复 独立计算
            all_keywords = clean_duplicate(all_keywords)
            oip_score = 0
            for keyword in all_keywords:
                for word in word_counts:
                    if word.startswith(keyword):
                        oip_score += word_counts[word]
            oip_score /= total_words
            
            # 把结果加入到scores中
            if len(result) > 0:
                scores['forms'][date] = {
                    'score': result,
                    'refer_date': str(tenk['refer_date']).split()[0],
                    'refer_tobins_q': tenk['refer_tobins_q'],
                    'oip_score': oip_score
                }

        # scores存到文件中
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
