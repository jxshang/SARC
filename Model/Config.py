class Config(object):
    """Configuration Parameters"""

    def __init__(self,class_num, embedding_dim):
        self.class_num = class_num
        self.embedding_dim = embedding_dim

        self.MAX_NEWS_LENGTH = 600
        self.MAX_COMMENT_LENGTH = 150

        self.MAX_TOKENIZER_LEN = 45000

        self.news_hidden_dim = 256
        self.comments_hidden_dim = 128
        self.combined_comment_mlp_dim = 256
        self.comments_emotion_dim = 1
        self.num_clusters = 3
        self.alpha = 0.05
