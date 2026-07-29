# GPT from Scratch - Project Notes

## Chapter 1: Project Goal

The objective is to build a small decoder-only GPT model from scratch and train it on a text corpus.

The core idea is:
- map tokens to vectors with an embedding layer,
- add positional information,
- pass the sequence through transformer blocks,
- predict the next token from the final hidden state.

The model learns by minimizing cross-entropy loss between predicted logits and the next-token targets.

---

## Chapter 2: Tokenization and Text Preparation

The text is loaded from [LLM/text.txt](LLM/text.txt) and tokenized with GPT-2 tokenizer from tiktoken.

Important steps:
- read the raw text,
- encode it into token IDs,
- split the text into training and validation parts,
- prepare batches using a sliding-window approach.

The data loader creates pairs:
- input chunk: tokens $x_{1:T}$
- target chunk: tokens $x_{2:T+1}$

This is the standard next-token prediction setup for language modeling.

---

## Chapter 3: Data Loader Logic

The implementation in [DataLoader.py](DataLoader.py) uses a sliding window over the tokenized text.

For a sequence of length $L$ and window size $T$ with stride $s$:
- each sample is a chunk of $T$ tokens,
- the target is the same chunk shifted by one position.

This gives many training examples while preserving local context.

Key idea:
- input batch shape: $(B, T)$
- target batch shape: $(B, T)$

The model learns to predict the next token for every position in the sequence.

---

## Chapter 4: Transformer Building Blocks

The model is built from three main components:
1. token embedding,
2. positional embedding,
3. transformer blocks.

The embedding step is:
$$
 h_i = E(x_i) + P(i)
$$
where $E(x_i)$ is the token embedding and $P(i)$ is the positional embedding.

Each transformer block contains:
- multi-head self-attention,
- feed-forward network,
- layer normalization,
- residual connections.

The residual form is:
$$
x_{out} = x + 	ext{Block}(x)
$$
This helps training stay stable and deep networks learn better.

---

## Chapter 5: Self-Attention Logic

The attention layer in [MultiHeadAttn.Py](MultiHeadAttn.Py) projects the input into queries, keys, and values.

For each token, attention scores are computed as:
$$
	ext{scores} = rac{QK^T}{
oot{d_k}}
$$

Then softmax is applied to get attention weights:
$$
	ext{weights} = 	ext{softmax}(	ext{scores})
$$

The context vector is:
$$
	ext{context} = 	ext{weights} 	imes V
$$

The causal mask ensures that a token can only attend to previous tokens and itself, which is important for autoregressive generation.

---

## Chapter 6: GPT Model Structure

The GPT model in [gptModel.py](gptModel.py) is a decoder-only transformer.

The forward pass is:
1. convert token IDs to embeddings,
2. add positional embeddings,
3. pass through stacked transformer blocks,
4. normalize the final hidden states,
5. project to vocabulary logits.

The output shape is:
$$
(B, T, V)
$$
where $B$ is batch size, $T$ is sequence length, and $V$ is vocabulary size.

---

## Chapter 7: Text Generation

Generation is done autoregressively.

Given a starting context, the model repeatedly:
1. takes the last $C$ tokens as context,
2. predicts logits for the next token,
3. selects the most likely token,
4. appends it to the sequence.

This is the core loop:
$$
x_{t+1} = 	ext{argmax}(	ext{softmax}(f(x_{1:t})))
$$

This is how the model generates new text one token at a time.

---

## Chapter 8: Loss Function

The training objective is next-token prediction.

For a batch, the loss is computed with cross-entropy:
$$
	ext{Loss} = -rac{1}{N} 	extstyle\sum_i 	ext{log} 	ext{softmax}(z_i)_{y_i}
$$

In practice, the implementation uses:
- logits from the model,
- flattened predictions,
- flattened targets,
- cross-entropy loss over the whole batch.

This is the standard training signal for language models.

---

## Chapter 9: What Was Left Off

The notebook already contains the major building blocks:
- tokenizer and data loading,
- transformer architecture,
- GPT model,
- text generation.

The next unfinished part is the training loop and evaluation stage:
- define optimizer,
- run training over epochs,
- compute training and validation loss,
- log progress,
- generate sample text during training.

This is the natural next chapter to continue with.

---

## Chapter 10: Next Chapter Plan

The next step should be:
1. finish the training loop,
2. add evaluation on validation data,
3. track training loss over steps,
4. generate sample text periodically,
5. save checkpoints and inspect improvements.

The logic is simple:
- feed batches,
- compute loss,
- backpropagate,
- update weights,
- repeat until convergence.
