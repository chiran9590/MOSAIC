
import torch
import torch.nn as nn


class MOSAICTransformer(nn.Module):
    def __init__(
        self,
        input_dim=144,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=3,
        dropout=0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, 100, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position_embedding[:, :x.size(1), :]

        x = self.encoder(x)

        # Use the representation of the final observation.
        x = x[:, -1, :]

        return self.classifier(x)
