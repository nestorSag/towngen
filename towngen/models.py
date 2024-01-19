import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow.keras.layers as layers
import keras_nlp

import numpy as np

class MaskedPositionEncoding(keras_nlp.layers.SinePositionEncoding):

  """Add masking capabilities to keras-nlp's `SinePositionEncoding` layer. Also returns aggregate token and position embeddings.
  """

  def compute_mask(self, inputs, mask=None):
    return mask

  def call(self, inputs, *args, **kwargs):

    return inputs + super().call(inputs)

def stacked_decoder_model(
  embedding_dim: int,
  n_tokens: int,
  n_att_heads: int,
  dense_dim: int,
  n_decoders: int):
  """
  Returns a model architecture with stacked transformer decoders, similar to GPT2. 

  Args:
      embedding_dim (int): Dimensionality of token embeddings
      n_tokens (int): Number of tokens
      n_att_heads (int): Number of self attention heads
      dense_dim (int): Dimensionality of feed forward network
      n_decoders (int): Number of stacked decoders
  """

  embedding = layers.Embedding(
    input_dim = n_tokens,
    output_dim = embedding_dim,
    mask_zero = True)

  pos_encoding = MaskedPositionEncoding()
  
  decoders = []
  for k in range(n_decoders):
    decoder = keras_nlp.layers.TransformerDecoder(
      intermediate_dim = dense_dim,
      num_heads = n_att_heads)
    decoders.append(decoder)

  output_layer = layers.Dense(units = n_tokens, activation="softmax")

  model = keras.Sequential(
    [embedding, pos_encoding] + 
    decoders + 
    [output_layer])

  return model

def sample(
  model: keras.Model,
  tokenizer: Tokenizer,
  sos_char: str = "^",
  eos_char: str = "$",
  max_length: int = 100) -> str:
  """Sample words or sentences from a trained language model of stacked transformer decoders.
  
  Args:
      model (keras.Model): Trained model
      tokenizer (Tokenizer): fitted tokenizer
      sos_char (str, optional): start-of-speech token
      eos_char (str, optional): end-of-speech token
      max_length (int, optional): maximum allowed sentence length
  
  Returns:
      str: sampled sentence or word
  """
  sampled_word = ""
  # set up token to char mapping
  n_tokens = len(tokenizer.word_index)
  token_to_char = {token: char for char, token in tokenizer.word_index.items()}
  # define starting sequence
  current_char = sos_char
  current_sequence = tokenizer.texts_to_sequences(np.array([current_char]))
  while current_char != eos_char and len(sampled_word) < max_length:
    # sample predictive distribution from model
    current_dist = model.predict(np.array(current_sequence), verbose=0)[0][-1].reshape(-1)
    # sample token from distribution
    current_token = int(np.random.choice(np.arange(n_tokens+1), size=1, p=current_dist))
    # map token to character/word
    current_char = token_to_char[current_token]
    # update sampled word
    sampled_word += current_char
    current_sequence[0] += [current_token]
  return sampled_word.replace(eos_char, "")