import json
import os
from tqdm import tqdm
from utils.save_as_json import save_as_json
from utils.process_data import DataProcessor, load_stop_words, split_dataset

out_dir = 'data_with_emotion'
datasets = [
    {"name":'weibo',
     'data_path':'data_with_emotion/weibo/weibo_emotion_results.json',
     'language':'chinese',
     'data_split': [7,2,1]
    },
    {"name":'r-19',
     'data_path':'data_with_emotion/r-19/r_19_emotion_results.json',
     'language':'english',
     'data_split': [7,2,1]
    }
]
stop_words = {
    'chinese':load_stop_words('resource/stop_word/cn_stop_word.txt'),
    'english':load_stop_words('resource/stop_word/en_stop_word.txt')
}
if __name__ == "__main__":
    for dataset in datasets:
        print('*'*30)
        print(f"processing dataset {dataset['name']}")
        save_data = []
        data_processor = DataProcessor(stop_words[dataset['language']], dataset['language'])

        with open(dataset['data_path'], 'r') as f:
            data = json.load(f)

            for key in tqdm(data.keys()):
                value = data[key]
                comments = value['comments']
                if dataset['name'] == 'weibo':
                    label = value['lable']
                else:
                    label = value['label']
                save_news_content = data_processor.preprocess_text(value['news_content'])
                save_comments = []
                save_emotions = []
                for comment in comments:
                    if dataset['language'] == 'chinese':
                        emotion =  comment['emotion']["confidence"]['positive']
                    else:
                        emotion = comment['emotion']['compound']

                    save_emotions.append(emotion)
                    save_comments.append(data_processor.preprocess_text(comment['text']))

                save_data.append({
                    'news':save_news_content,
                    "comments":save_comments,
                    'emotions': save_emotions,
                    'label': label
                })

            train_data, val_data, test_data = split_dataset(save_data,dataset['data_split'],11)
            dataset_save_dir = os.path.join(out_dir,dataset['name'])
            if not os.path.exists(dataset_save_dir):
                os.mkdir(dataset_save_dir)
            save_as_json(train_data,os.path.join(dataset_save_dir,'train.json'))
            save_as_json(val_data, os.path.join(dataset_save_dir, 'val.json'))
            save_as_json(test_data, os.path.join(dataset_save_dir, 'test.json'))
