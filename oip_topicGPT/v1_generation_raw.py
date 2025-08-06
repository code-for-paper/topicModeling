from huggingface_hub.hf_api import api
import traceback
from util import *
from tqdm import tqdm
from datasets import load_dataset
import time
import random
import regex
import json


def load_corpus(data_dir):
    dataset = load_dataset("json", data_files=data_dir, split="train")
    # dataset = dataset.shuffle()
    return dataset


def format_topics(topics):
    # [1] Trade: Mentions the exchange of capital, goods, and services.
    # 只保留topic的标题，不保留描述
    topics = [topic.split(":")[0].strip() for topic in topics]


def generate_topics(api_client: ApiClient, data_dir, prompt_file, seed_file, output_file,output_index_file, verbose=False):

    corpus = load_corpus(data_dir)
    prompt_template = load_prompt(prompt_file)

    early_stop = 1500
    dup_count = 0
    responses = []
    topic_list = []
    root = TopicTree("root")
    token_count = 0

    topic_format = regex.compile(r"^\[(\d+)\] ([^\[\]]+)$")
    

    for i, company in enumerate(tqdm(corpus, desc='公司')):
        
        # if i < 183:
        #     continue
        
        # print(corpus[i]["ABSTRACT"])
        cik = company['cik']
        ten_k = company['10-k']
        
        output_index = {
            'cik':cik,
            'indexes':{}
        }
        
        for item in tqdm(ten_k, desc='10-K文件', leave=False):
            date = str(item['date']).split(' ')[0]
            file_id = item['file_id']
            content = item['form'][-10000:]

            prompt = prompt_template.format(
                Document=content,
                Topics=root.to_prompt()
            )
            token_count += api_client.estimate_token_count(prompt)
            try:
                response = api_client.generate(prompt)
                topics = [t.strip() for t in response.split("\n")]
                # 检查topic是否合法，提取topic
                # 检查响应是否合法
                for t in topics:
                    if not regex.match(topic_format, t):
                        if verbose:
                            print("Invalid response: ", t)
                            print("Response: ", response)
                        continue
                    groups = regex.match(topic_format, t)
                    lvl, name = int(groups[1]), groups[2].strip()

                    if lvl != 1 or name=='None' or name=='none': 
                        if verbose:
                            print(f"raw topics生成阶段只允许一级topic,跳过{t}...")
                        continue
                    # 到这里检查就合法了 可以给当前document 添加标记  在date
                    if date not in output_index['indexes']:
                        output_index['indexes'][date] = [name]
                    else :
                        output_index['indexes'][date].append(name)
                    
                    # 检查是否和已有topic重复
                    dups = root.find_duplicates(1, name)
                    if dups:
                        dups[0].count += 1
                        dup_count += 1
                        if dup_count > early_stop:
                            print(f"重复次数超过阈值:{early_stop}，提前停止...")
                            return responses, topic_list, root
                    else:
                        dup_count = 0
                        root.add_node(lvl, name, 1, root.root)
                        topic_list = root.to_topic_list(count=False)

                
                if verbose:
                    print("="*100)
                    # print("Prompt: ", prompt)
                    print("document:", corpus[i]["ABSTRACT"])
                    print("Response: ", response)

                responses.append(response)
            except Exception as e:
                traceback.print_exc()
                responses.append("Error")
                break
        
        # 保存索引 以便后续快速查找document的是否有topic
        if len(output_index['indexes']) > 0:
            with open(output_index_file, 'a') as f:
                f.write(json.dumps(output_index, ensure_ascii=False) + '\n')
        print(f'\n{i+1}th iteration topic:\n{root.to_prompt()}')
        print(f'response:{responses[-len(ten_k):]}')
        time.sleep(random.uniform(1.0, 1.5))
        # if i == 100:
        #     break
        # pass

        if (i+1) % 10 == 0:
            root.to_file(output_file)
        # if i==100:
        #     break

    print("total token count: ", token_count)
    return responses, topic_list, root


def main(api_client: ApiClient, data_dir, prompt_file, seed_file, output_file,output_index_file, verbose=False):

    resps, topic_list, root = generate_topics(
        api_client, data_dir, prompt_file, seed_file, output_file,output_index_file, verbose)
    # 把得到的结果(topic和responses)保存起来
    root.to_file(output_file)

    # resps和topic_list是一一对应的，resps可以作为新的一列加入到dataset中
    # 也可以直接保存为csv文件，然后在refinement阶段读取
    print("="*100)
    print("responses: ", resps)
    print("="*100)
    print("topic_list: ", topic_list)
    print("="*100)
    print("root: ", root.to_prompt())


if __name__ == "__main__":
    api_client = ApiClient("qwen3")
    config = get_config()
    main(api_client,
         config["generation_raw"]["input"],
         config["generation_raw"]["prompt"],
         config["generation_raw"]["seed"],
         config["generation_raw"]["output"],
         config["generation_raw"]["output_index"],
         # config["verbose"]
         )

    # print(load_corpus(config["generation_raw"]["input"]))
