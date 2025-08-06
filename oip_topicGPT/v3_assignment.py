from util import ApiClient, TopicTree, get_config, load_corpus, load_prompt
from tqdm import tqdm
import json
import regex as re

""" 
document有一系列主题 对于每个主题 要从文中找出符合该主题的关键字 
输入: document, topics:[
    [1] Customer Engagement  
    [2] University Collaboration 
    ],
输出:
[1] Customer Engagement  
    [2] customers  
    [2] co-creation  
    [2] workshops  
[1] University Collaboration  
    [2] universities  
    [2] joint_research  
    [2] collaborative_experiments  
    [2] labs
    
keywords_output的格式为
{
    'cik':{
        'date':{
            [{
                'topic':'',
                'keywords':[]
            }]
        }
    }
}

flag_dict的格式为
{
    'cik':{
        'date':{
            'topic':[] # 这里topic没啥用 实际还是得等到分配的时候给主题
        }
    }
}

prompt_template 需要输入 一级topics document 
"""

# api_client = ApiClient('./tokenizer/deepseek-r1')
# corpus_path = ''  # 10-k数据集
# prompt_file = ''  # assignment和keywords生成的prompt模版
# topic_file = ''  # 之前阶段得到的topic列表
# flag_file = ''  # jsonl文件 这里要先加载然后处理为一个dict
# topic_output_file = ''  # 输出的一级+二级topic的地址
# keywords_output_file = ''  # 保存:cik的年报的一级topic和二级keywords的字典


def assign_and_keywords_generate(
    api_client: ApiClient,
    corpus_path: str,
    prompt_file: str,
    topic_file: str,
    flag_file: str,
    topic_output_file: str,
    keywords_output_file: str,
    verbose: bool = False
):
    """
    给定10-k数据集 一级topic 二级topic的prompt 一级topic的flag 生成每个公司每个日期的二级topic和keywords

    Args:
        api_client: ApiClient
        corpus_path: 10-k数据集地址
        prompt_file: assignment和keywords生成的prompt模版
        topic_file: 之前阶段得到的topic列表
        flag_file: jsonl文件 这里要先加载然后处理为一个dict
        topic_output_file: 输出的一级+二级topic的地址
        keywords_output_file: 保存:cik的年报的一级topic和二级keywords的字典
        verbose: 是否打印详细信息
    """

    keywords_output = {}  # return

    # 加载一级topic文件
    tree = TopicTree().from_file(topic_file)  # return
    print(f'topic from file :{tree.to_prompt()}')
    # return 
    flag_dict = {}
    with open(flag_file, 'r') as f:
        for line in f:
            cik_dict = json.loads(line)
            cik = cik_dict['cik']
            indexes = cik_dict['indexes']
            
            # 修复：初始化cik字典
            if cik not in flag_dict:
                flag_dict[cik] = {}
            
            for date in indexes:
                topics = indexes[date]
                flag_dict[cik][date] = topics

    # 加载语料库
    corpus = load_corpus(corpus_path)  # 修复：函数名从load_dataset改为load_corpus
    # 加载prompt
    prompt_template = load_prompt(prompt_file)
    # 加载一级topic
    top_topic_str = tree.top_topic_to_prompt()

    for i, company in enumerate(tqdm(corpus, desc='公司')):
        if (i+1) % 5 == 0:
            tree.to_file(topic_output_file)
            with open(keywords_output_file, 'w') as f:
                json.dump(keywords_output, f, indent=4, ensure_ascii=False)
                
        # 先过滤没有topic的公司
        cik = company['cik']
        if cik not in flag_dict:
            continue

        tenks = company['10-k']
        for item in tenks:
            date = str(item['date']).split()[0]
            # 修复：检查date是否在flag_dict[cik]中
            if date not in flag_dict[cik]:
                continue
            # 现在文档是有topic的，现在先准备好prompt
            business_text = item['form'][-10000:]
            prompt = prompt_template.format(
                topics=top_topic_str, document=business_text)
            # 现在有prompt了 调用api
            response = api_client.generate(prompt)

            if response == 'None':
                continue

            # 解析response文本 用正则表达式提取主题和关键词
            def parse_response(response_text):
                """
                解析API响应文本,将层级结构转换为字典列表
                输入格式:
                [1] Customer Engagement  
                    [2] customers  
                    [2] co-creation  
                [1] University Collaboration  
                    [2] universities  
                    [2] joint_research  

                输出格式:
                [
                    {
                        'topic':'Customer Engagement',
                        'keywords':['customers','co-creation']
                    }
                ]
                """
                try:
                    result = []
                    lines = response_text.strip().split('\n')
                    current_topic = None
                    current_keywords = []

                    # 正则表达式模式
                    topic_pattern = re.compile(r'\[1\]\s*([^\[\]]+)\s*$')  # 匹配一级主题
                    keyword_pattern = re.compile(
                        r'\[2\]\s*([^\[\]]+)\s*$')  # 匹配二级关键词

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        # 匹配一级主题
                        topic_match = topic_pattern.match(line)
                        if topic_match:
                            # 如果之前有主题，先保存
                            if current_topic is not None:
                                result.append({
                                    'topic': current_topic,
                                    'keywords': current_keywords
                                })

                            # 开始新主题
                            current_topic = topic_match.group(1).strip()
                            current_keywords = []
                            continue

                        # 匹配二级关键词
                        keyword_match = keyword_pattern.match(line)
                        if keyword_match and current_topic is not None:
                            keyword = keyword_match.group(1).strip()
                            current_keywords.append(keyword)

                    # 保存最后一个主题
                    if current_topic is not None:
                        result.append({
                            'topic': current_topic,
                            'keywords': current_keywords
                        })

                    # 检查解析结果是否有效
                    if not result or any(not item['keywords'] for item in result):
                        return None

                    return result

                except Exception as e:
                    # 任何解析错误都返回None
                    return None
                    
            # 解析响应
            parsed_topics = parse_response(response)
            if parsed_topics is None:
                # 存一下错误信息 todo
                # 修复：确保字典结构存在再赋值
                if cik not in keywords_output:
                    keywords_output[cik] = {}
                if date not in keywords_output[cik]:
                    keywords_output[cik][date] = []
                keywords_output[cik][date] = 'error'
                continue

            # 遍历topics 把生成的二级标题添加到一级标题下面
            for item in parsed_topics:
                topic = item['topic']
                keywords = item['keywords']
                try:
                    parent_node = tree.find_top_parent(topic)[0]
                    if parent_node:
                        for keyword in keywords:
                            tree.add_node(2, keyword, 1, parent_node)
                except:
                    print(f'parent_node not found for {topic}')

            # 检查并初始化字典结构
            if cik not in keywords_output:
                keywords_output[cik] = {}
            if date not in keywords_output[cik]:
                keywords_output[cik][date] = []
            keywords_output[cik][date].extend(parsed_topics)
            
            # break
        print(f'current cik:{keywords_output[cik] if cik in keywords_output else "not found"}')

        # break
    
    tree.to_file(topic_output_file)
    with open(keywords_output_file, 'w') as f:
        json.dump(keywords_output, f, indent=4, ensure_ascii=False)

    return tree,keywords_output

if __name__ == '__main__':
    config = get_config()
    api_client = ApiClient("qwen3")
    assign_and_keywords_generate(
        api_client,
        config["generation_raw"]["input"],
        config["assignment"]["prompt"],
        config['generation_raw']['output'],
        config["generation_raw"]["output_index"],
        config["assignment"]["topic_output"],
        config["assignment"]["keywords_output"],
        # config['verbose']
    )