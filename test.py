from Model.Config import Config
import os.path
from Model.MyModel import MyModel
from utils.process_data import load_data

MODEL_DIR = 'saved'

DATA_DIR = 'data_with_emotion'
SAVE_DIR = 'saved'
datasets = [
    {"name":'weibo',
     'language':'chinese',
     'embedding_path':'resource/embedding/sgns.weibo.word/sgns.weibo.word',
     'embedding_dim': 300,
     'class_num':2
    },
    {"name":'r-19',
     'language':'english',
     'embedding_path':'resource/embedding/glove.42B.300d.txt',
     'embedding_dim': 300,
     'class_num':3
    }
]
if __name__ == '__main__':

    for dataset in datasets:
        name = dataset['name']
        print('#'*60)
        print('#'*60)
        print(f'dataset {name} is testing')
        config = Config(dataset['class_num'], dataset['embedding_dim'])

        dataset_dir = os.path.join(DATA_DIR, dataset['name'])
        model_dir = os.path.join(MODEL_DIR, dataset['name'])

        train_data, val_data, test_data = load_data(dataset_dir)
        model = MyModel(config)
        model.load_model(model_path=os.path.join(model_dir,'best_model.pth_3_0.05'))
        model.test(test_data)