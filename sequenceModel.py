import torch
import torch.nn as nn
import pandas as pd
from utils.model import Model


class SequenceModel(Model):
    ALPHABET = ['A', 'C', 'G', 'T']

    def __init__(
        self,
        n_chars=4,
        seq_len=10,
        bidirectional=True,
        batch_size=32,
        hidden_layers=1,
        hidden_size=32,
        lin_dim=16,
        emb_dim=10,
        dropout=0,
    ):
        super(SequenceModel, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.lin_dim = lin_dim
        self.batch_size = batch_size

        # The encoder
        self.emb_lstm = torch.nn.LSTM(
            input_size=n_chars,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Self-attention for encoder
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional doubles hidden size
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        self.latent_linear = torch.nn.Sequential(
            nn.Linear(hidden_size * seq_len * 2, lin_dim),
            nn.ReLU()
        )

        self.latent_mean = nn.Linear(lin_dim, emb_dim)
        self.latent_log_std = nn.Linear(lin_dim, emb_dim)

        # Analyzer
        self.analyzer = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single property value
        )

        # Self-attention for decoder
        self.decoder_attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        self.dec_lin = torch.nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

        # Decoder LSTM
        self.dec_lstm = torch.nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Final layer for decoding
        self.dec_final = torch.nn.Linear(hidden_size * 2 * self.seq_len, n_chars * seq_len)

        self.xavier_initialization()

    def xavier_initialization(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode(self, x):
        hidden, _ = self.emb_lstm(x.float())  # LSTM output
        attn_output, _ = self.encoder_attention(hidden, hidden, hidden)  # Self-attention
        hidden = self.latent_linear(torch.flatten(attn_output, 1))  # Linear transformation
        z_mean = self.latent_mean(hidden)
        z_log_std = self.latent_log_std(hidden)
        return torch.distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))

    def decode(self, z):
        hidden = self.dec_lin(z)

        # Self-attention 
        hidden, _ = self.decoder_attention(
            hidden.unsqueeze(1).repeat(1, self.seq_len, 1),
            hidden.unsqueeze(1).repeat(1, self.seq_len, 1),
            hidden.unsqueeze(1).repeat(1, self.seq_len, 1),
        )
        hidden, _ = self.dec_lstm(hidden)
        out = self.dec_final(torch.flatten(hidden, 1))
        return out.view(-1, self.seq_len, self.n_chars)

    def reparametrize(self, dist):
        sample = dist.rsample()
        prior = torch.distributions.Normal(
            torch.zeros_like(dist.loc), torch.ones_like(dist.scale)
        )
        prior_sample = prior.sample()
        return sample, prior_sample, prior

    def forward(self, x):
        latent_dist = self.encode(x)
        latent_sample, prior_sample, prior = self.reparametrize(latent_dist)
        output = self.decode(latent_sample).view(-1, self.seq_len, self.n_chars)
        property_prediction = self.analyzer(latent_sample)  # Analyzer prediction
        return output, latent_dist, prior, latent_sample, prior_sample, latent_dist.loc, torch.log(latent_dist.scale), property_prediction

    def __repr__(self):
        return 'SequenceVAE' + self.trainer_config
