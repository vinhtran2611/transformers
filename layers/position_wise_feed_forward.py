import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    The dimensionality of input and output is  dmodel=512 , and the inner-layer has dimensionality  dff=2048 .
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))