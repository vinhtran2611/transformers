import torch
import tqdm
from utils import subsequent_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1, :])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    
    return ys

import torch

# Function to perform beam search decoding
def beam_search(model, src, src_mask, max_len, start_symbol, beam_size):
    """
    Reference begin slide 34/79.
    Link: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf
    """
    # Encode the source sentence
    memory = model.encode(src, src_mask)
    
    # Initialize the beam with a single hypothesis containing only the start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    beam = [(ys, 0)]
    
    # Loop until either the maximum target sentence length is reached or 
    # all hypotheses in the beam have reached an end-of-sequence token
    for i in range(max_len - 1):
        candidates = []
        # Extend each hypothesis in the current beam with the beam_size most likely next words 
        # according to the model's output probability distribution
        for ys, score in beam:
            out = model.decode(
                memory, src_mask, ys,
                subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = model.generator(out[:, -1, :])
            topk_prob, topk_indices = torch.topk(prob, k=beam_size)
            for j in range(beam_size):
                # Concatenate the current hypothesis with each of the topk_indices to form a new candidate hypothesis
                candidate = torch.cat([ys, topk_indices[:, j].unsqueeze(1)], dim=1)
                # Compute the score of the new hypothesis as the sum of the log probabilities of its constituent words
                candidates.append((candidate, score + torch.log(topk_prob[:, j])))
                
        # Sort the candidate hypotheses by their scores
        candidates.sort(key=lambda x: x[1])
        # Select the top beam_size hypotheses to form the new beam
        beam = candidates[:beam_size]
        
    # Return the hypothesis with the highest score
    return beam[0][0]
