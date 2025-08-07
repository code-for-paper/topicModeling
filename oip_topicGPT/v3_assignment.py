from util import ApiClient, TopicTree, get_config, load_corpus, load_prompt
from tqdm import tqdm
import json
import regex as re

""" 
Document has a series of topics. For each topic, find keywords from the text that match the topic.
Input: document, topics:[
    [1] Customer Engagement  
    [2] University Collaboration 
    ],
Output:
[1] Customer Engagement  
    [2] customers  
    [2] co-creation  
    [2] workshops  
[1] University Collaboration  
    [2] universities  
    [2] joint_research  
    [2] collaborative_experiments  
    [2] labs
    
keywords_output format:
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

flag_dict format:
{
    'cik':{
        'date':{
            'topic':[] # topic is not useful here, still need to assign topics during assignment
        }
    }
}

prompt_template needs input: level 1 topics, document 
"""


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
    Given 10-k dataset, level 1 topics, level 2 topic prompts, level 1 topic flags,
    generate level 2 topics and keywords for each company and date

    Args:
        api_client: ApiClient
        corpus_path: 10-k dataset path
        prompt_file: assignment and keywords generation prompt template
        topic_file: topic list from previous stage
        flag_file: jsonl file, need to load and process into a dict
        topic_output_file: output path for level 1 + level 2 topics
        keywords_output_file: save: cik annual report level 1 topics and level 2 keywords dictionary
        verbose: whether to print detailed information
    """

    keywords_output = {}  # return

    # Load level 1 topic file
    tree = TopicTree().from_file(topic_file)  # return
    print(f'topic from file :{tree.to_prompt()}')
    # return 
    flag_dict = {}
    with open(flag_file, 'r') as f:
        for line in f:
            cik_dict = json.loads(line)
            cik = cik_dict['cik']
            indexes = cik_dict['indexes']
            
            # Fix: initialize cik dictionary
            if cik not in flag_dict:
                flag_dict[cik] = {}
            
            for date in indexes:
                topics = indexes[date]
                flag_dict[cik][date] = topics

    # Load corpus
    corpus = load_corpus(corpus_path)  # Fix: function name from load_dataset to load_corpus
    # Load prompt
    prompt_template = load_prompt(prompt_file)
    # Load level 1 topics
    top_topic_str = tree.top_topic_to_prompt()

    for i, company in enumerate(tqdm(corpus, desc='Companies')):
        if (i+1) % 5 == 0:
            tree.to_file(topic_output_file)
            with open(keywords_output_file, 'w') as f:
                json.dump(keywords_output, f, indent=4, ensure_ascii=False)
                
        # Filter companies without topics first
        cik = company['cik']
        if cik not in flag_dict:
            continue

        tenks = company['10-k']
        for item in tenks:
            date = str(item['date']).split()[0]
            # Fix: check if date is in flag_dict[cik]
            if date not in flag_dict[cik]:
                continue
            # Now document has topics, prepare prompt
            business_text = item['form'][-10000:]
            prompt = prompt_template.format(
                topics=top_topic_str, document=business_text)
            # Now have prompt, call api
            response = api_client.generate(prompt)

            if response == 'None':
                continue

            # Parse response text using regex to extract topics and keywords
            def parse_response(response_text):
                """
                Parse API response text, convert hierarchical structure to dictionary list
                Input format:
                [1] Customer Engagement  
                    [2] customers  
                    [2] co-creation  
                [1] University Collaboration  
                    [2] universities  
                    [2] joint_research  

                Output format:
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

                    # Regex patterns
                    topic_pattern = re.compile(r'\[1\]\s*([^\[\]]+)\s*$')  # Match level 1 topics
                    keyword_pattern = re.compile(
                        r'\[2\]\s*([^\[\]]+)\s*$')  # Match level 2 keywords

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        # Match level 1 topics
                        topic_match = topic_pattern.match(line)
                        if topic_match:
                            # Save previous topic if exists
                            if current_topic is not None:
                                result.append({
                                    'topic': current_topic,
                                    'keywords': current_keywords
                                })

                            # Start new topic
                            current_topic = topic_match.group(1).strip()
                            current_keywords = []
                            continue

                        # Match level 2 keywords
                        keyword_match = keyword_pattern.match(line)
                        if keyword_match and current_topic is not None:
                            keyword = keyword_match.group(1).strip()
                            current_keywords.append(keyword)

                    # Save last topic
                    if current_topic is not None:
                        result.append({
                            'topic': current_topic,
                            'keywords': current_keywords
                        })

                    # Check if parsing result is valid
                    if not result or any(not item['keywords'] for item in result):
                        return None

                    return result

                except Exception as e:
                    # Return None for any parsing errors
                    return None
                    
            # Parse response
            parsed_topics = parse_response(response)
            if parsed_topics is None:
                # Save error info todo
                # Fix: ensure dictionary structure exists before assignment
                if cik not in keywords_output:
                    keywords_output[cik] = {}
                if date not in keywords_output[cik]:
                    keywords_output[cik][date] = []
                keywords_output[cik][date] = 'error'
                continue

            # Iterate through topics, add generated level 2 titles under level 1 titles
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

            # Check and initialize dictionary structure
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