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
    # Keep only topic titles, not descriptions
    topics = [topic.split(":")[0].strip() for topic in topics]


def generate_topics(api_client: ApiClient, data_dir, prompt_file, seed_file, output_file,output_index_file, verbose=False):

    corpus = load_corpus(data_dir)
    prompt_template = load_prompt(prompt_file)

    early_stop = 1500
    dup_count = 0
    responses = []
    topic_list = []
    root = TopicTree("root")

    topic_format = regex.compile(r"^\[(\d+)\] ([^\[\]]+)$")
    

    for i, company in enumerate(tqdm(corpus, desc='Companies')):
        
        # if i < 183:
        #     continue
        
        # print(corpus[i]["ABSTRACT"])
        cik = company['cik']
        ten_k = company['10-k']
        
        output_index = {
            'cik':cik,
            'indexes':{}
        }
        
        for item in tqdm(ten_k, desc='10-K files', leave=False):
            date = str(item['date']).split(' ')[0]
            file_id = item['file_id']
            content = item['form'][-10000:]

            prompt = prompt_template.format(
                Document=content,
                Topics=root.to_prompt()
            )
            try:
                response = api_client.generate(prompt)
                topics = [t.strip() for t in response.split("\n")]
                # Check if topic is valid, extract topic
                # Check if response is valid
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
                            print(f"Raw topics generation phase only allows level 1 topics, skipping {t}...")
                        continue
                    # Valid check passed, can add marker to current document at date
                    if date not in output_index['indexes']:
                        output_index['indexes'][date] = [name]
                    else :
                        output_index['indexes'][date].append(name)
                    
                    # Check if duplicate with existing topics
                    dups = root.find_duplicates(1, name)
                    if dups:
                        dups[0].count += 1
                        dup_count += 1
                        if dup_count > early_stop:
                            print(f"Duplicate count exceeded threshold: {early_stop}, stopping early...")
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
        
        # Save index for quick lookup of whether document has topics
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

    return responses, topic_list, root


def main(api_client: ApiClient, data_dir, prompt_file, seed_file, output_file,output_index_file, verbose=False):

    resps, topic_list, root = generate_topics(
        api_client, data_dir, prompt_file, seed_file, output_file,output_index_file, verbose)
    # Save the results (topics and responses)
    root.to_file(output_file)

    # resps and topic_list are one-to-one correspondence, resps can be added as a new column to dataset
    # Can also be saved directly as csv file, then read in refinement phase
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
