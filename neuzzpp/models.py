from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence
from neuzzpp.utils import pad_sequences


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            learning_rate: float,
            hidden_dim: int=4096,
            output_bias: Optional[float] = None,
            fast: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # set up one hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    
def predict_coverage(model: MLP, inputs: List[np.ndarray]) -> np.ndarray:
    """
    Get binary labels from model for non-normalized input data.

    The input data is first normalized and preprocessed to the length required by the model.

    Args:
        model: Pytorch model predicting coverage bitmap from program input.
        inputs: List or equivalent of non-normalized inputs.
    """
    
    # truncate
    inputs_preproc = [input[-model.input_dim:] for input in inputs]

    # pad and normalize
    for input in inputs_preproc:
        pad_length = model.input_dim - len(input)
        if pad_length > 0:
            input = np.pad(input, pad_length)
        input = input.astype("float32") / 255.0

    model.eval()
    # TODO: inputs need to be tensors
    with torch.no_grad():
        preds = model(inputs_preproc).cpu().numpy()
    model.train()
    return preds > 0.5



    