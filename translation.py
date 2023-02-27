import torch
from torchtext.data.metrics import bleu_score
from utils import subsequent_mask
from prepare import Batch, create_dataloaders
from search import greedy_decode, beam_search
from prepare import load_vocab, load_tokenizers
from models import make_tranformers_model

def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
): 
    bleu = []
    results = [()] * n_examples

    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt_list = []
        for x in model_out: 
            if eos_string == vocab_tgt.get_itos()[x]:
                model_txt_list.append(vocab_tgt.get_itos()[x])
                break
            elif x != pad_idx:
                model_txt_list.append(vocab_tgt.get_itos()[x])

        model_txt =  " ".join(model_txt_list)
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)

        bleu.append(bleu_score([model_txt_list], [tgt_tokens]))

    return results, sum(bleu) / len(bleu)


def validate(n_examples):
    print("Preparing Data ...")
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_tranformers_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data, bleu = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )

    print(f"Bleu score: {bleu}")
    return model, example_data, bleu

if __name__ == "__main__":
    validate(10)

