from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
import torch
from anytree import Node
import yaml
from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI
import os
from ast import List
from platform import node
import warnings

from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")


def get_config():
    with open('./config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_model():
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"), # Use custom GPT model provider's baseurl and api_key
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1",

    )
    return model


def get_tokenizer(path):
    """
    Get Qwen model tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    return tokenizer


def get_corpus(data_path, shuffle=True):
    dataset = load_dataset(data_path, split="train")
    return DataLoader(dataset, batch_size=1, shuffle=shuffle)


def read_and_format_generation_raw_prompt(topics=None, document=None):
    """
    Read generation_raw.txt file and format it into a usable prompt

    Args:
        topics (str, optional): Top-level topic list to insert
        document (str, optional): Document content to analyze

    Returns:
        str: Formatted prompt string
    """
    # Read template file
    template_path = './prompt/generation_raw.txt'
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except FileNotFoundError as e:
        # If relative path doesn't work, try absolute path
        print(e)
        return None

    # Format template
    formatted_prompt = template.format(
        Topics=topics if topics is not None else "{Topics}",
        Document=document if document is not None else "{Document}"
    )

    return formatted_prompt


def load_prompt(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            template = f.read()
    except FileNotFoundError as e:
        print(e)
        return None
    return template

def load_corpus(data_dir,type:str="json"):
    dataset = load_dataset(type, data_files=data_dir, split="train")
    return dataset


def load_seed(path) -> list:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            seed = f.read()
            seed = seed.split("\n")
            return seed
    except FileNotFoundError as e:
        print(e)
        return []


def load_sentence_transformer():
    # todo

    model_path = os.path.expanduser("~/models/stella_en_1.5B_v5")
    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"

    # Build SentenceTransformer model
    word_embedding_model = models.Transformer(model_path)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension())
    sentence_model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model], device=device)

    return sentence_model


class ApiClient:
    def __init__(self, model_name):
        self.model_name = model_name
        config = get_config()
        self.model = get_model()
        self.tokenizer = get_tokenizer(config['tokenizer'][model_name])
        self.max_length = self.tokenizer.model_max_length


    def estimate_token_count(self, text):
        return len(self.tokenizer(text)["input_ids"])

    def generate(self, prompt):
        try:
            output = self.model.invoke(prompt)
            return output.content
        except Exception as e:
            print(e)
            return ""


class TopicTree:
    def __init__(self, title="Topic Root"):
        self.root = Node(name=title, lvl=0, count=1)
        # self.level_nodes = {0:self.root}

    def find_duplicates(self, lvl, name):
        """
        Find nodes by name and level
        """
        return [
            node
            for node in self.root.descendants
            if node.name.lower() == name.lower() and node.lvl == lvl
        ]
    
    def find_top_parent(self,name):
        """
        Find first-level topic node
        """
        return self.find_duplicates(1,name)

    def add_node(self, lvl, name, count=1, parent_node: Node=None):
        """
        Add a new node to the tree, if it already exists, merge and update count
        """
        if parent_node:
            existing = next(
                (n for n in parent_node.children if n.name.lower() == name.lower()), None)
            if existing:
                existing.count += count
            else:
                new_node = Node(
                    name=name, lvl=lvl, count=count, parent=parent_node
                )

    def to_prompt(self):
        """
        Convert tree to prompt
        """
        def traverse(node: Node, result=""):
            if node.lvl > 0:
                result += (
                    "\t" * (node.lvl - 1) +
                    f"[{node.lvl}] {node.name}" + "\n"
                )
            for child in node.children:
                result = traverse(child, result)

            return result

        return traverse(self.root)

    def to_topic_list(self, count=True):
        return [self.node_to_str(node, count) for node in self.root.descendants]

    def to_file(self, path):
        with open(path, "w") as f:
            for node in self.root.descendants:
                indentation = "\t" * (node.lvl - 1)
                f.write(indentation + self.node_to_str(node) + "\n")

    @staticmethod
    def node_to_str(node, count=True):
        """
        Convert a node to a string representation.

        Parameters:
        - node: Node to convert
        - count: Include count in the string

        Returns:
        - str: String representation of the node
        """
        if not count:
            return f"[{node.lvl}] {node.name}"
        else:
            return f"[{node.lvl}] {node.name} (Count: {node.count})"
    
    def from_file(self, path):
        """Load topic tree structure from file"""
        tree = TopicTree()
        current_parent = {0: tree.root}  # Track parent nodes for each level
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse line content
                try:
                    lvl = int(line[1])
                    if '(' in line:
                        name = line[3:line.index('(')-1].strip()
                        count = int(line[line.index('Count:')+6:line.index(')')].strip())
                    else:
                        name = line[3:].strip()
                        count = 1
                        
                    # Find correct parent node and add new node
                    parent = current_parent[lvl-1] if lvl > 0 else tree.root
                    new_node = Node(name=name, lvl=lvl, count=count, parent=parent)
                    current_parent[lvl] = new_node
                    
                except (ValueError, IndexError) as e:
                    print(f"Failed to parse line: {line}, error: {str(e)}")
                    continue
                    
        return tree

    def top_topic_to_prompt(self):
        # Extract only level 1 node topics and join with \n
        topics = [node.name for node in self.root.children if node.lvl == 1]
        return "\n".join(topics)

if __name__ == "__main__":
    # Test case 1: Create a simple topic tree
    tree1 = TopicTree()
    tree1.add_node(1, "Technology", 1, tree1.root)
    tree1.add_node(2, "AI", 1, tree1.root.children[0])
    print("Test case 1 - Simple topic tree:")
    print(tree1.to_prompt())

    # Test case 2: Test duplicate node merging
    tree2 = TopicTree()
    tree2.add_node(1, "Sports", 1, tree2.root)
    tree2.add_node(2, "Football", 1, tree2.root.children[0])
    # Add duplicate node, count should increase
    tree2.add_node(2, "Football", 1, tree2.root.children[0])
    print("\nTest case 2 - Duplicate node merging:")
    print(tree2.to_prompt())

    # Test case 3: Test finding duplicate nodes
    tree3 = TopicTree()
    tree3.add_node(1, "Music", 1, tree3.root)
    tree3.add_node(2, "Rock", 1, tree3.root.children[0])
    tree3.add_node(2, "Jazz", 1, tree3.root.children[0])
    duplicates = tree3.find_duplicates(2, "Rock")
    print("\nTest case 3 - Find duplicate nodes:")
    print(f"Number of duplicate nodes found: {len(duplicates)}")
    print(tree3.to_prompt())
    tree3.to_file("data/output/test_raw_topics.md")

    # Empty tree
    tree4 = TopicTree()
    print("\nTest case 4 - Empty tree:")
    print(tree4.to_prompt())
    
    tree4 = TopicTree().from_file("data/output/raw_topics.md")
    print("\nTest case 5 - Load from file:")
    print(tree4.to_prompt())
