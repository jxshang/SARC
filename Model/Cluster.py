import torch
import torch.nn as nn
import torch.nn.functional as F
class ClusterLayer(nn.Module):
    def __init__(self, num_clusters, mlp_dim):
        super().__init__()
        self.num_clusters = num_clusters

        # cluster center
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, mlp_dim),
            requires_grad=True
        )


        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None):
        """
        input:
            x: [B, N_C, mlp_dim] comment_feature
        output:
            cluster_assign: [B, N_C, K] cluster_weight
            cluster_feat:   [B, K, mlp_dim] cluster_feature
        """
        B, N_C, _ = x.shape
        self.temperature.data.clamp_(min=0.1, max=5.0)

        x_norm = F.normalize(x, p=2, dim=-1)  # [B, N_C, D]
        centers_norm = F.normalize(self.cluster_centers, p=2, dim=-1)  # [K, D]

        # similarity
        similarity = torch.matmul(x_norm, centers_norm.transpose(0, 1))  # [B, N_C, K]

        if mask is not None:
            mask = mask.unsqueeze(-1)
            similarity = similarity.masked_fill(mask == 0, -1e9)

        # soft cluster
        cluster_assign = F.softmax(similarity * self.temperature, dim=-1) # [B, N_C, K]


        cluster_feat = torch.einsum('bnk,bnd->bkd', cluster_assign, x)  # [B, K, D]

        return cluster_assign, cluster_feat



class ClusterLoss(nn.Module):
    def __init__(self, alpha=0.05, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, combined_comment_feat, cluster_assign, cluster_centers):
        """
        input:
            combined_comment_feat: [B, N_C, D]
            cluster_assign:        [B, N_C, K]
            cluster_centers:      [K, D]
        output:
            loss: cluster loss
        """
        B, N_C, D = combined_comment_feat.shape
        K = cluster_centers.shape[0]

        # --- Intra Loss: Weighted Distance from Features to Center ---
        x_norm = F.normalize(combined_comment_feat, p=2, dim=-1)  # [B, N_C, D]
        centers_norm = F.normalize(cluster_centers, p=2, dim=-1)  # [K, D]
        # Calculate the cosine similarity between each feature and all centers
        distance_matrix = torch.matmul(x_norm, centers_norm.transpose(0, 1))
        # Weighted Average
        embedding_loss = - torch.sum(cluster_assign * distance_matrix) / (B * N_C)



        # --- Inter-class Loss: Reciprocal of Distances Between Centers ---
        # Calculate the distance matrix between all centers [K, K]
        distance_center = torch.matmul(centers_norm, centers_norm.transpose(0,1))
        mask = torch.eye(K, dtype=torch.bool, device=distance_center.device)
        valid_center_distances = distance_center[~mask].view(K, K - 1)  # [K, K-1]
        center_loss = valid_center_distances.mean()

        # total loss
        total_loss = self.alpha * (center_loss + embedding_loss)
        return total_loss