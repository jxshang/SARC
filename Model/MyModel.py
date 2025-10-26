import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Model.Cluster import ClusterLoss
from utils.NewsCommentDataset import NewsCommentDataset, collate_fn
from utils.calculate import calculate_macro_f1, calculate_macro_precision, calculate_macro_recall, print_eval, \
    calculate_rmse
from utils.load_embedding import load_embedding
from utils.Tokenizer import MyTokenizer, pad_sequences
from Model.FakeNewsDetect import FakeNewsDetector


class MyModel:

    def __init__(self, config,device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.tokenizer = MyTokenizer(max_vocab_size= self.config.MAX_TOKENIZER_LEN)

        self.data_processor = None
        self.analyzer = None
        self.model = None


    def test(self,test_data):
        news, comments, labels, comments_emotion = self.decode_text_and_emotion(test_data)
        test_dataset =  NewsCommentDataset(news, comments, comments_emotion, labels)
        test_loader = DataLoader(test_dataset ,collate_fn=collate_fn, batch_size=1)
        test_metrics = self.evaluate(test_loader)
        print_eval(test_metrics)

    def train(self, train_data, val_data, batch_size, lr, epochs ,embedding_path, saved_dir):
        self._word2index_on_news_and_comment(train_data, val_data)
        train_dataset = NewsCommentDataset(self.train_news, self.train_comments, self.train_emotions, self.train_labels)
        val_dataset = NewsCommentDataset(self.val_news, self.val_comments, self.val_emotions, self.val_labels)

        # DataLoader
        train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)

        self.model = self._build_model(embedding_path, self.tokenizer.word2idx)
        self.model.to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        cluster_loss_fn = ClusterLoss(alpha=self.config.alpha)
        best_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            cls_loss = 0.0
            cluster_loss = 0.0
            for news, comments, comments_mask, emotions, labels in train_loader:
                assert not torch.isnan(news).any(), "NaN in news data"
                assert not torch.isinf(news).any(), "Inf in news data"

                assert not torch.isnan(comments).any(), "NaN in comments data"
                assert not torch.isinf(comments).any(), "Inf in comments data"

                assert not torch.isnan(emotions).any(), "NaN in emotions data"
                assert not torch.isinf(emotions).any(), "Inf in emotions data"

                assert not torch.isnan(labels).any(), "NaN in labels data"
                assert not torch.isinf(labels).any(), "Inf in labels data"

                news = news.to(self.device)
                comments = comments.to(self.device)
                emotions = emotions.to(self.device)
                comments_mask = comments_mask.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs, combined_comment_feat, cluster_assign, cluster_centers = self.model(news, comments, comments_mask,emotions)


                # loss function
                cls_loss = criterion(outputs, labels)
                cluster_loss = cluster_loss_fn(combined_comment_feat, cluster_assign,cluster_centers)
                total_loss = cls_loss + cluster_loss
                total_loss.backward()
                optimizer.step()

            # val stage
            val_metrics = self.evaluate(val_loader)
            print(30*"*")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {total_loss:.4f} | cls_loss: {cls_loss:.4f} | cluster_loss: {cluster_loss:.4f}")
            print_eval(val_metrics)

            # save best model
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                self.save_model(self.model, os.path.join(saved_dir, f"best_model.pth_{self.config.num_clusters}_{self.config.alpha}"))

        # save final model
        self.save_model(self.model, os.path.join(saved_dir,f"final_model.pth_{self.config.num_clusters}_{self.config.alpha}"))
        return

    def _build_model(self, embedding_path, word_index):
        embedding_tensor = load_embedding(embedding_path, self.config.embedding_dim, word_index)
        if embedding_tensor.device.type != 'cpu':
            embedding_tensor = embedding_tensor.cpu()
        model = FakeNewsDetector(
            embedding_matrix=embedding_tensor,
            news_hidden_dim=self.config.news_hidden_dim,
            comments_hidden_dim=self.config.comments_hidden_dim,
            combined_comment_mlp_dim=self.config.combined_comment_mlp_dim,
            comments_emotion_dim=self.config.comments_emotion_dim,
            num_clusters=self.config.num_clusters,
            class_num=self.config.class_num
        )
        return model

    def save_model(self, model, path):
        """save model and tokenizer"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': {
                'word2idx': self.tokenizer.word2idx,
                'max_vocab_size': self.tokenizer.max_vocab_size
            }
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, model_path, emotion_analyzer=None, data_processor=None, device='cuda'):
        """load model and tokenizer"""
        checkpoint = torch.load(model_path, map_location=device)
        # build tokenizer
        tokenizer = MyTokenizer(max_vocab_size=checkpoint['tokenizer']['max_vocab_size'])
        tokenizer.word2idx = checkpoint['tokenizer']['word2idx']
        tokenizer.idx2word = {v: k for k, v in tokenizer.word2idx.items()}
        self.model = FakeNewsDetector(
            embedding_matrix=torch.zeros((len(tokenizer.word2idx), self.config.embedding_dim)),
            news_hidden_dim=self.config.news_hidden_dim,
            comments_hidden_dim=self.config.comments_hidden_dim,
            combined_comment_mlp_dim=self.config.combined_comment_mlp_dim,
            comments_emotion_dim=self.config.comments_emotion_dim,
            num_clusters=self.config.num_clusters,
            class_num=self.config.class_num
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.analyzer = emotion_analyzer # for predict
        self.data_processor = data_processor # for predict

    def predict(self, news_text, comments):
        """
        predict fake news
        :param news_text: news text
        :param comments: comment list
        :return: label and probability
        """
        # 确保模型和tokenizer已加载
        if self.model is None or self.tokenizer is None or self.analyzer is None:
            raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
        # extract emotion feature
        emotions = []
        for comment in comments:
            emotion = self.analyzer.predict(comment)
            emotions.append(emotion["confidence"]["positive"])

        # 将输入数据处理为模型需要的格式
        news_text = self.data_processor.preprocess_text(news_text)
        process_comments = [self.data_processor.preprocess_text(comment) for comment in comments]
        self.model.eval()
        with torch.no_grad():
            news_seq = self.texts_to_padded_sequences([news_text], self.config.MAX_NEWS_LENGTH)[0]
            comment_seqs = self.texts_to_padded_sequences(process_comments, self.config.MAX_COMMENT_LENGTH)

            news_tensor = torch.LongTensor(news_seq).unsqueeze(0).to(self.device)  # shape: [1, max_news_len]
            comments_tensor = torch.LongTensor(comment_seqs).unsqueeze(0).to(
                self.device)  # shape: [1, comment_count, max_comment_len]
            emotions_tensor = torch.FloatTensor(emotions).unsqueeze(0).to(
                self.device)  # shape: [1, comment_count, emotion_dim]


            outputs, _, cluster_assign, _ = self.model(news_tensor, comments_tensor, comment_emotion=emotions_tensor)
            assign_np = cluster_assign.squeeze(0).cpu().numpy()  # [B, N_C, K]


            valid_id = np.argmax(assign_np, axis=1)
            reason_comment = []

            for i in range(len(valid_id)):
                if valid_id[i] == 2:
                    reason_comment.append(i)


            probs = torch.softmax(outputs, dim=1)[0, 1].item()

            pred_label = 1 if probs >= 0.5 else 0

        return pred_label, probs, reason_comment

    def evaluate(self, data_loader):
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for news, comments, comments_mask, emotions, labels in data_loader:
                news = news.to(self.device)
                comments = comments.to(self.device)
                emotions = emotions.to(self.device)
                comments_mask = comments_mask.to(self.device)
                # 修正标签处理为长整型类别索引
                labels = labels.to(self.device).long().flatten()  # 确保形状为(batch,)

                outputs, *_ = self.model(news, comments, comments_mask, emotions)

                # 获取类别概率
                probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs)
                all_labels.append(labels)

        probs_tensor = torch.cat(all_probs)
        labels_tensor = torch.cat(all_labels)


        preds_tensor = torch.argmax(probs_tensor, dim=1)
        labels_tensor = labels_tensor.long()

        accuracy = (preds_tensor == labels_tensor).float().mean().item()
        macro_f1 = calculate_macro_f1(labels_tensor, preds_tensor, self.config.class_num)
        macro_pre = calculate_macro_precision(labels_tensor, preds_tensor, self.config.class_num)
        macro_rec = calculate_macro_recall(labels_tensor, preds_tensor, self.config.class_num)
        rmse = calculate_rmse(labels_tensor, preds_tensor)

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_precision": macro_pre,
            "macro_recall": macro_rec,
            "rmse": rmse
        }



    def _word2index_on_news_and_comment(self, train_data, val_data):
        all_texts = []
        for data in train_data:
            all_texts.append(data['news'])
            all_texts.extend(data['comments'])

        for data in val_data:
            all_texts.append(data['news'])
            all_texts.extend(data['comments'])
        self.tokenizer.fit_on_texts(all_texts)
        self._reverse_word_index()

        self.train_news, self.train_comments, self.train_labels, self.train_emotions = self.decode_text_and_emotion(train_data)
        self.val_news, self.val_comments, self.val_labels, self.val_emotions= self.decode_text_and_emotion(val_data)


    def decode_text_and_emotion(self,datas):
        news = []
        comments = []
        comments_emotion = []
        labels = []
        for data in datas:
            news_seq = self.texts_to_padded_sequences([data['news']], self.config.MAX_NEWS_LENGTH)[0]
            news.append(news_seq)

            comment_seqs = self.texts_to_padded_sequences(data['comments'], self.config.MAX_COMMENT_LENGTH)
            comments.append(comment_seqs)
            comments_emotion.append(torch.tensor(data['emotions']))
            labels.append(data['label'])

        return  news, comments, labels, comments_emotion

    def texts_to_padded_sequences(self,texts, max_length):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=max_length,
            padding='post',
            truncating='post'
        )
        return padded

    def _reverse_word_index(self):
        self.word_index = {value: key for key, value in self.tokenizer.word2idx.items()}
