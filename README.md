# Introduction to the Transformer Model
The Transformer model is a neural network architecture that was introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). It was designed to address some of the limitations of previous sequence-to-sequence models such as recurrent neural networks (RNNs) and convolutional neural networks (CNNs).

The Transformer model uses a novel self-attention mechanism that allows it to process input sequences in parallel, rather than in a sequential manner like RNNs. This makes the model highly parallelizable and much faster than traditional sequence models.

Another key feature of the Transformer model is its ability to handle variable-length input sequences. Unlike RNNs, which require fixed-length sequences, the Transformer can handle sequences of any length by using positional encoding.

The Transformer has achieved state-of-the-art results on a wide range of natural language processing tasks such as machine translation, text summarization, and language modeling. Its success has led to its widespread adoption in the research community and industry.

In this repository, we provide an implementation of the Transformer model in PyTorch. **The implementation includes a class that simplifies the learning process, making it easy for users to understand**


## Table of Contents
  * [Model Architecture](#model-architecture)
  * [Machine translation](#machine-translation)
  * [Setup](#setup)
  * [Usage](#usage)


## Model Architecture
Most competitive neural sequence transduction models have an
encoder-decoder structure
[(cite)](https://arxiv.org/abs/1409.0473). Here, the encoder maps an
input sequence of symbol representations $(x_1, ..., x_n)$ to a
sequence of continuous representations $\mathbf{z} = (z_1, ...,
z_n)$. Given $\mathbf{z}$, the decoder then generates an output
sequence $(y_1,...,y_m)$ of symbols one element at a time. At each
step the model is auto-regressive
[(cite)](https://arxiv.org/abs/1308.0850), consuming the previously
generated symbols as additional input when generating the next.

<br><br>
![model](images/model.png)
<br><br>

``` python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


```

``` python
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
```

### Encoder and Decoder Stacks

#### Encoder

The encoder is composed of a stack of $N=6$ identical layers.
``` python 
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```