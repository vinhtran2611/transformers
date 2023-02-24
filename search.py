import torch
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
    
    # Return the target sequence without the start symbol
    return ys[:, 1:]

def beam_search(model, src, src_mask, max_len, start_symbol, k, end_symbol = 1):
    """
    Ref: https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24
    """
    
    # Encode the source sequence
    memory = model.encode(src, src_mask)
    
    # Initialize the list of current hypotheses with the start symbol
    hypotheses = [(torch.tensor([start_symbol]), 0)]
    
    # Repeat until the maximum length is reached
    for _ in range(max_len - 1):
        # Initialize a list to store the next set of hypotheses
        next_hypotheses = []
        
        # Expand each hypothesis in the current set
        for hypothesis in hypotheses:
            # Get the current target sequence and its log probability
            trg_seq, trg_log_prob = hypothesis
            
            # If the end-of-sequence token is generated, add the hypothesis to the final set
            # if trg_seq[-1] == end_symbol:
            #     next_hypotheses.append(hypothesis)
            #     continue
            
            # Get the top k predictions for the next token
            out = model.decode(
                memory, src_mask, trg_seq.unsqueeze(0), subsequent_mask(trg_seq.size(0)).type_as(src.data)
            )
            log_probs = model.generator(out[:, -1]).log_softmax(dim=-1).squeeze()
            top_k_probs, top_k_indices = torch.topk(log_probs, k)
            
            # Add each new hypothesis to the list of next hypotheses
            for prob, index in zip(top_k_probs, top_k_indices):
                next_hypothesis = (
                    torch.cat([trg_seq, index.unsqueeze(0)]),
                    trg_log_prob + prob.item()
                )
                next_hypotheses.append(next_hypothesis)
        
        # Sort the next set of hypotheses by their log probabilities
        next_hypotheses = sorted(next_hypotheses, key=lambda x: x[1], reverse=True)
        
        # Select the top k hypotheses to keep for the next iteration
        hypotheses = next_hypotheses[:k]
    
    # Return the target sequences of the top k hypotheses
    # return [hypothesis[0] for hypothesis in hypotheses]
    return hypothesis[0]     
