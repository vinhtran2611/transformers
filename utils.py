import torch.nn as nn
import torch
import copy
from layers.layer_norm import LayerNorm
from models import make_tranformers_model
import torch.nn.functional as F

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

# Batches and Masking
class Batch:
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, tgt=None, pad=2):
        """
        Args:
            src: (batch_size, src_vocab)
            tgt: (batch_size, tgt_vocab)
            2 = <blank>
        """
        self.src = src
        # (batch_size, 1, src_vocab)
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        """
        
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    """
    index = 0 for "de", index = 1 for "en"
    """
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())