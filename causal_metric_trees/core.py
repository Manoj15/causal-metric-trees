import pandas as pd
from transformers import pipeline

class DataLoader:
    @staticmethod
    def load_dataset(file_paths):
        datasets = {}
        for file_path in file_paths:
            datasets[file_path] = pd.read_csv(file_path)
        return datasets

class CausalMetricNode:
    def __init__(self, name, value=None, children=None):
        self.name = name
        self.value = value
        self.children = children if children else []

    def add_child(self, child):
        self.children.append(child)

class CausalMetricTree:
    def __init__(self, root):
        self.root = root

    def query(self, name):
        return self._search(self.root, name)

    def _search(self, node, name):
        if node.name == name:
            return node
        for child in node.children:
            result = self._search(child, name)
            if result:
                return result
        return None

    def display(self, node=None, level=0):
        if node is None:
            node = self.root
        print('  ' * level + f'{node.name}: {node.value}')
        for child in node.children:
            self.display(child, level + 1)

class AnalysisAssistant:
    def __init__(self, model_name="mistral"):
        self.model = pipeline("text-generation", model=model_name)

    def ask(self, query):
        response = self.model(query, max_length=200, num_return_sequences=1)[0]['generated_text']
        return response
