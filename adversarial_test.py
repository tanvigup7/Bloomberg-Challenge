# adversial_test.py
import pandas as pd
from filters import apply_filters
from utils import visualize_word_cloud, visualize_word_frequency, visualize_sentence_length

def test_adversarial_examples(data, column="text"):
    filtered_data = apply_filters(data)
    print("Running adversarial tests...")
    visualize_word_cloud(filtered_data, column=column)
    visualize_word_frequency(filtered_data, column=column)
    visualize_sentence_length(filtered_data, column=column)

if __name__ == "__main__":
    from config import DATASET_PATH
    data = pd.read_csv(DATASET_PATH)
    test_adversarial_examples(data)
