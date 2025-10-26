import torch
import torch.nn as nn
import torch.nn.functional as F

from .Cluster import ClusterLayer


class FakeNewsDetector(nn.Module):

    def __init__(self, embedding_matrix, news_hidden_dim=256, comments_hidden_dim=128, combined_comment_mlp_dim=256, comments_emotion_dim=1,num_clusters=4, class_num=2):
        super().__init__()
        self.embed_dim = embedding_matrix.shape[1]
        self.news_hidden_dim = news_hidden_dim
        self.comments_hidden_dim = comments_hidden_dim
        self.comments_emotion_dim = comments_emotion_dim
        self.class_num = class_num

        # news
        self.news_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix)
        )
        self.news_text_att = TextAttention(self.embed_dim, news_hidden_dim)

        # comment
        self.comment_hidden_dim = comments_hidden_dim
        self.comment_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix)
        )
        self.comment_text_att = TextAttention(self.embed_dim, self.comment_hidden_dim)
        self.comments_mlp = nn.Linear(2*comments_hidden_dim + comments_emotion_dim, combined_comment_mlp_dim)
        self.comments_means = CommentsMeans(mlp_dim=combined_comment_mlp_dim, num_clusters=num_clusters)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(2*news_hidden_dim + num_clusters * combined_comment_mlp_dim, 512),  # news_feature + cluster_feature
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, class_num)
        )

    def forward(self, news_input=None, comment_input=None, comments_mask_input=None, comment_emotion=None):
        """
        input dim:
        - news_input:       [batch_size, news_len]
        - comment_input:    [batch_size, num_comments, comment_len]
        - comment_emotion:  [batch_size, num_comments, emotion_dim]
        """

        # news_encode
        news_emb = self.news_embedding(news_input)  # [B, N_L, E]
        news_out, news_att_wei = self.news_text_att(news_emb)  # [B, 2H]
        # comment_encode
        B, N_C, C_L = comment_input.shape
        comments_flat = comment_input.view(B * N_C, C_L)  # [B*N_C, C_L]
        comments_emb = self.comment_embedding(comments_flat) # [B*N_C, C_L, E]
        comments_out, comments_att = self.comment_text_att(comments_emb) # [B*N_C, 2H]
        comments_out = comments_out.view(B, N_C, 2 * self.comment_hidden_dim)

        combined_comment_feat = torch.cat([
            comments_out,  # [B, N_C, 2H]
            comment_emotion.unsqueeze(2)  # [B, N_C, emotion_dim]
        ], dim=2)  # [B, N_C, 2H+emotion_dim]
        # cluster_feature
        combined_comment_feat = self.comments_mlp(combined_comment_feat)
        cluster_assign, cluster_feat = self.comments_means(combined_comment_feat, comments_mask_input) # [B, K, mlp_dim]

        cluster_feat = cluster_feat.view(cluster_feat.size(0), -1)  # [B, 5 * 256]
        # combine feature
        combined = torch.cat([news_out, cluster_feat], dim=1)
        return self.classifier(combined), combined_comment_feat, cluster_assign, self.comments_means.cluster.cluster_centers

class CommentsMeans(nn.Module):
    def __init__(self, mlp_dim, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster = ClusterLayer(num_clusters=num_clusters, mlp_dim=mlp_dim)

    def forward(self, combined_comment_feat, comments_mask_input=None):
        """

        :param comments_mask_input:
        :param combined_comment_feat:  [B, N_C, mlp_dim]
        """
        cluster_assign, cluster_feat = self.cluster(combined_comment_feat, comments_mask_input) # [B, K, mlp_dim]
        return cluster_assign, cluster_feat


class TextAttention(nn.Module):
    """
        BiGRU + self_attention + mean_pool
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.BiGru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.attention = SelfAttention(2 * hidden_dim)

    def forward(self, x):
        """
        input dim:
        :param x: [batch_size, len, embedding_dim]
        """
        x, _ = self.BiGru(x) # [B, L, 2H]
        x, attn_weights = self.attention(x) # [B, L, 2H]
        x = x.mean(dim=1) # [B, 2H]
        return x, attn_weights



class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, hidden_states):
        """
        input and output dim:
        - hidden_states: [batch, seq_len, hidden_dim]
        - return:        [batch, hidden_dim]
        """
        Q = self.query(hidden_states)  # [B, L, H]
        K = self.key(hidden_states)  # [B, L, H]
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, hidden_states)
        context += hidden_states
        return context,attn_weights