from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequences


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
        x = torch.sigmoid(self.fc2(x))
        return x

    def forward_logits(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Unlike keras version, we don't have a create_logits_model() function
# the idea is that we'd do something like model.forward_logits(x) to get the output of that
    
def predict_coverage(model: MLP, inputs: List[np.ndarray]) -> np.ndarray:
    """
    Get binary labels from model for non-normalized input data.

    The input data is first normalized and preprocessed to the length required by the model.

    Args:
        model: Keras model predicting coverage bitmap from program input.
        inputs: List or equivalent of non-normalized inputs.
    """
    

    # need to mimic keras' pad_sequences
    inputs.preproc = pad_sequences(inputs, padding='post', maxlen=model.input_dim)
    inputs_preproc = inputs_preproc.to(torch.float32) / 255.0

    model.eval()
    with torch.no_grad():
        preds = model(inputs_preproc).cpu().numpy()
    model.train()
    return preds > 0.5



    