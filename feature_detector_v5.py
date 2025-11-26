#!/usr/bin/env python
"""
Feature Detector V5: Enhanced Architecture with Transformer Decoder

Key improvements from V4:
1. Deeper GNN encoder (6 layers instead of 4)
2. Transformer decoder instead of LSTM for better long-range dependencies
3. Larger hidden dimension (384 instead of 256)
4. Multi-head attention mechanism
5. Better positional encoding for sequences
6. Improved feature representation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import math


class GNNEncoderV5(nn.Module):
    """Enhanced Graph encoder using deeper GAT layers."""

    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, num_layers=6):
        super().__init__()

        self.initial = GATv2Conv(
            node_feature_dim,
            hidden_dim,
            heads=4,
            edge_dim=edge_feature_dim,
            concat=False
        )

        self.layers = nn.ModuleList([
            GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=4,
                edge_dim=edge_feature_dim,
                concat=False
            )
            for _ in range(num_layers - 1)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.initial(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.norms[0](x)
        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            identity = x
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.norms[i + 1](x)
            x = self.dropout(x)
            x = x + identity  # Residual connection

        return x, batch


class AttentionPooling(nn.Module):
    """Attention-based graph pooling."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, batch):
        # x: [num_nodes, hidden_dim]
        # batch: [num_nodes] indicating which graph each node belongs to

        # Compute attention scores
        scores = self.attention(x)  # [num_nodes, 1]

        # Apply softmax per graph
        scores = torch.exp(scores - scores.max())

        # Group by batch and normalize
        batch_size = batch.max().item() + 1
        pooled = []

        for i in range(batch_size):
            mask = (batch == i)
            node_scores = scores[mask]
            node_features = x[mask]

            # Normalize scores
            node_scores = node_scores / (node_scores.sum() + 1e-8)

            # Weighted sum
            graph_emb = (node_features * node_scores).sum(dim=0)
            pooled.append(graph_emb)

        return torch.stack(pooled)


class CountHead(nn.Module):
    """Enhanced count prediction head."""

    def __init__(self, hidden_dim, max_count=30):
        super().__init__()
        self.max_count = max_count

        # Deeper network with more capacity
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, max_count + 1)
        )

        # Initialize with small weights to encourage exploration
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to encourage uniform predictions initially."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, graph_emb):
        """
        Args:
            graph_emb: [batch_size, hidden_dim]

        Returns:
            count_logits: [batch_size, max_count + 1]
        """
        return self.fc(graph_emb)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, hidden_dim, max_len=50):
        super().__init__()

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for feature sequence generation.

    Replaces LSTM with Transformer for better long-range dependencies.
    """

    def __init__(self, hidden_dim, num_layers, num_feature_types, num_params, nhead=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_feature_types = num_feature_types
        self.num_params = num_params

        # Input projection: previous feature (types + params) -> hidden_dim
        self.input_proj = nn.Linear(num_feature_types + num_params, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Graph embedding projection
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output heads
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_feature_types)
        )

        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_params)
        )

    def _generate_square_subsequent_mask(self, sz, device):
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _forward_teacher_forcing(self, graph_emb, features, counts):
        """
        Teacher forcing for training.

        Args:
            graph_emb: [batch_size, hidden_dim]
            features: [batch_size, max_features, feature_dim] (9 types + 10 params)
            counts: [batch_size] ground truth feature counts

        Returns:
            type_logits: [batch_size, max_features, num_types]
            param_preds: [batch_size, max_features, num_params]
        """
        batch_size = graph_emb.size(0)
        max_features = features.size(1)
        device = graph_emb.device

        # Shift features by 1 (teacher forcing)
        # First step gets zero input
        prev_features = torch.cat([
            torch.zeros(batch_size, 1, features.size(-1), device=device),
            features[:, :-1, :]
        ], dim=1)

        # Project previous features to hidden dimension
        tgt = self.input_proj(prev_features)  # [batch_size, max_features, hidden_dim]

        # Add positional encoding
        tgt = self.pos_encoder(tgt)

        # Project graph embedding as memory for cross-attention
        memory = self.graph_proj(graph_emb).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Generate causal mask
        tgt_mask = self._generate_square_subsequent_mask(max_features, device)

        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [batch_size, max_features, hidden_dim]

        # Predict type and parameters
        type_logits = self.type_head(output)
        param_preds = self.param_head(output)

        return type_logits, param_preds

    def _forward_autoregressive(self, graph_emb, counts):
        """
        Autoregressive generation for inference.

        Args:
            graph_emb: [batch_size, hidden_dim]
            counts: [batch_size] predicted feature counts

        Returns:
            type_logits: [batch_size, max_count, num_types]
            param_preds: [batch_size, max_count, num_params]
        """
        batch_size = graph_emb.size(0)
        device = graph_emb.device
        max_count = counts.max().item()

        if max_count == 0:
            # No features to generate
            return (
                torch.zeros(batch_size, 1, self.num_feature_types, device=device),
                torch.zeros(batch_size, 1, self.num_params, device=device)
            )

        # Initialize with zeros
        prev_features = torch.zeros(batch_size, 1, self.num_feature_types + self.num_params, device=device)

        # Project graph embedding as memory
        memory = self.graph_proj(graph_emb).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        type_logits_list = []
        param_preds_list = []

        for t in range(max_count):
            # Project previous features
            tgt = self.input_proj(prev_features)  # [batch_size, t+1, hidden_dim]
            tgt = self.pos_encoder(tgt)

            # Generate causal mask
            tgt_mask = self._generate_square_subsequent_mask(t + 1, device)

            # Transformer decoder
            output = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask
            )  # [batch_size, t+1, hidden_dim]

            # Get last timestep output
            last_output = output[:, -1, :]  # [batch_size, hidden_dim]

            # Predict type and parameters
            type_logits = self.type_head(last_output)
            param_preds = self.param_head(last_output)

            type_logits_list.append(type_logits)
            param_preds_list.append(param_preds)

            # Use predicted features as input for next step
            pred_type = torch.argmax(type_logits, dim=-1)
            type_onehot = F.one_hot(pred_type, num_classes=self.num_feature_types).float()

            next_feature = torch.cat([type_onehot, param_preds], dim=-1).unsqueeze(1)
            prev_features = torch.cat([prev_features, next_feature], dim=1)

        type_logits = torch.stack(type_logits_list, dim=1)
        param_preds = torch.stack(param_preds_list, dim=1)

        return type_logits, param_preds

    def forward(self, graph_emb, features=None, counts=None):
        """
        Args:
            graph_emb: [batch_size, hidden_dim]
            features: [batch_size, max_features, 19] for teacher forcing (optional)
            counts: [batch_size] for teacher forcing or autoregressive (required)

        Returns:
            type_logits: [batch_size, max_features, num_types]
            param_preds: [batch_size, max_features, num_params]
        """
        if features is not None:
            # Teacher forcing (training)
            return self._forward_teacher_forcing(graph_emb, features, counts)
        else:
            # Autoregressive (inference)
            return self._forward_autoregressive(graph_emb, counts)


class FeatureDetectorV5(nn.Module):
    """
    Feature Detector V5: Enhanced Architecture with Transformer Decoder.

    Architecture improvements from V4:
    1. Deeper GNN Encoder: 6 layers with stronger residual connections
    2. Transformer Decoder: Replaces LSTM for better long-range dependencies
    3. Larger Capacity: 384 hidden dimension
    4. Multi-Head Attention: Better feature interaction modeling
    5. Positional Encoding: Improved sequence awareness
    """

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        hidden_dim=384,
        num_gnn_layers=6,
        num_transformer_layers=4,
        num_feature_types=9,
        num_params=10,
        max_count=30,
        nhead=8
    ):
        super().__init__()

        self.encoder = GNNEncoderV5(node_feature_dim, edge_feature_dim, hidden_dim, num_gnn_layers)
        self.pooling = AttentionPooling(hidden_dim)
        self.count_head = CountHead(hidden_dim, max_count)
        self.decoder = TransformerDecoder(
            hidden_dim,
            num_transformer_layers,
            num_feature_types,
            num_params,
            nhead
        )

    def forward(self, graph, features=None, counts=None):
        """
        Args:
            graph: PyG Data object
            features: [batch_size, max_features, 19] for training (optional)
            counts: [batch_size] for training or inference

        Returns:
            count_logits: [batch_size, max_count + 1]
            type_logits: [batch_size, max_features, num_types]
            param_preds: [batch_size, max_features, num_params]
        """
        # Encode graph
        node_emb, batch = self.encoder(graph)

        # Pool to graph embedding
        graph_emb = self.pooling(node_emb, batch)

        # Predict count
        count_logits = self.count_head(graph_emb)

        # If counts not provided, use predicted counts
        if counts is None:
            counts = torch.argmax(count_logits, dim=-1)

        # Decode features
        type_logits, param_preds = self.decoder(graph_emb, features, counts)

        return count_logits, type_logits, param_preds
