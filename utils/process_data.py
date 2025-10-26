import jieba
import nltk
import numpy as np
import os
import json

def load_data(data_dir):
    data_format = ['train.json', 'val.json', 'test.json']
    data_paths  = [os.path.join(data_dir, filename) for filename in data_format]

    def load_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return map(load_json, data_paths)

def load_stop_words(file_path):
    stopwords = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()  # 去除首尾空白字符
                if word:  # pip install hanlp
                    stopwords.add(word)
    except FileNotFoundError:
        print(f"警告：停用词文件 {file_path} 未找到，使用空停用词列表")
    return stopwords


def split_dataset(dataset, split_rate, random_seed=None):
    """
    Split the dataset into training set, validation set, and test set according to a proportion
    Parameters:
    dataset (list/np.ndarray): Input dataset
    split_rate (list): Split proportion, e.g., [8,1,1] represents the ratio 8:1:1
    random_seed (int): Random seed to ensure reproducibility
    Returns:
    train, val, test: The three split datasets, with the same type as the input
    """

    # random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    total = sum(split_rate)
    train_ratio, val_ratio, _ = [r / total for r in split_rate]

    n = len(dataset)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    def get_data(indices):
        if isinstance(dataset, np.ndarray):
            return dataset[indices]
        return [dataset[i] for i in indices]

    return (
        get_data(indices[:train_end]),
        get_data(indices[train_end:val_end]),
        get_data(indices[val_end:])
    )

class DataProcessor:
    def __init__(self, stop_word, language='chinese'):
        self.stop_words = stop_word
        self.language = language

    def preprocess_text(self, text):
        if self.language == 'chinese':
            return self._preprocess_cn_text(text)
        else:
            return self._preprocess_en_text(text)

    def _preprocess_cn_text(self, text):
        words = jieba.cut(text)
        clean_words = [
            word.strip() for word in words
            if (word not in self.stop_words) and (not word.isspace())
        ]
        return ' '.join(clean_words)

    def _preprocess_en_text(self, text):
        text = text.lower()
        words = nltk.word_tokenize(text)
        clean_words = [
            word.strip() for word in words
            if (word not in self.stop_words) and (not word.isspace())
        ]
        return ' '.join(clean_words)