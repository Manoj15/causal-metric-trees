import pytest
from causal_metric_trees.core import CausalMetricNode, CausalMetricTree, DataLoader, AnalysisAssistant
import pandas as pd

def test_metric_tree():
    root = CausalMetricNode('Root', 100)
    sales = CausalMetricNode('Sales', 60)
    marketing = CausalMetricNode('Marketing', 40)
    online = CausalMetricNode('Online', 30)
    offline = CausalMetricNode('Offline', 30)

    sales.add_child(online)
    sales.add_child(offline)
    root.add_child(sales)
    root.add_child(marketing)

    tree = CausalMetricTree(root)
    assert tree.query('Online').value == 30

def test_data_loader():
    file_paths = ["sample_file.csv"]
    datasets = DataLoader.load_dataset(file_paths)
    assert isinstance(datasets, dict)

def test_analysis_assistant():
    assistant = AnalysisAssistant()
    response = assistant.ask("What is a Causal Metric Tree?")
    assert isinstance(response, str) and len(response) > 0
