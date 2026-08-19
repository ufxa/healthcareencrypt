"""3-layer MLP (256-128-64) + output layer, matching Table III / Sec. VI-B."""
import torch
import torch.nn as nn


class ClinicalMLP(nn.Module):
    """Two hidden layers (256, 128) + a third hidden layer (64) before the
    output logit, ReLU activations. Binary mortality output as a single
    logit (sigmoid applied at loss/eval time), matching the HE inference
    path which evaluates one logit homomorphically rather than a full
    softmax (softmax/sigmoid is computed in cleartext after decryption by
    the Data Owner, consistent with the threat model in Sec. IV)."""

    def __init__(self, in_dim=48, hidden=(256, 128, 64)):
        super().__init__()
        h1, h2, h3 = hidden
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.out(x).squeeze(-1)  # raw logit

    def layer_weights(self):
        """Returns [(W, b), ...] as numpy arrays, layer order fc1->fc2->fc3->out,
        for use by the homomorphic inference path (he_inference.py)."""
        layers = [self.fc1, self.fc2, self.fc3, self.out]
        return [(l.weight.detach().cpu().numpy(), l.bias.detach().cpu().numpy())
                for l in layers]
