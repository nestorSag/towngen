import logging
import argparse
import sys
from pathlib import Path
import hashlib
import time

import numpy as np

import tensorflow as tf
from tensorflow import keras

from towngen import loaders
from towngen.models import stacked_decoder_model, sample
import towngen.preprocessing as preprocessing

MODEL_CACHE = Path(".models")
MODEL_CACHE.mkdir(exist_ok=True)

TOKENIZER_CACHE = Path(".tokenizers")
TOKENIZER_CACHE.mkdir(exist_ok=True)

SOS_TOKEN = "^"
EOS_TOKEN = "$"

logging.basicConfig(
  level=logging.INFO, 
  format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s"
)


def fetch_data(country_code, query=None):
  """Fetches city names data from geonames.org
  
  Args:
      country_code (str): Alpha-2 country code, e.g. GB for Great Britain, FR for France.
      query (str, optional): data-filtering query. Defaults to None.
  
  Returns:
      np.ndarray: city names
  """
  logging.info(f"Fetching data for country code {country_code}...")
  data = loaders.CityNames.load(
    country_code=country_code, 
    query=query
  )
  logging.info(f"Retrieved {len(data)} city names")
  return data

def preprocess_data(data, SOS_TOKEN, EOS_TOKEN):
  """Preprocesses city names data
  
  Args:
      data (np.ndarray): city names
      SOS_TOKEN (str): start-of-speech token
      EOS_TOKEN (str): end-of-speech token
  
  Returns:
      t.Tuple[np.ndarray, np.ndarray]: Triplet with numpy arrays (features and targets) and the fitted Tensorflow tokenizer.
  """
  logging.info("Preprocessing data...")
  replacement_orders = [
    ("-"," "),
    ("&", "and"),
    (r'\(.+\)|#|\.|[0-9]|,', "")
  ]

  data = preprocessing.text_replacer(
    data, 
    replacement_orders
  )
  data = preprocessing.add_delimiter_tokens(data, SOS_TOKEN, EOS_TOKEN)
  features, targets, tokenizer = preprocessing.get_char_prediction_features(data, SOS_TOKEN, EOS_TOKEN)
  return features, targets, tokenizer


def train_model(
  features,
  targets,
  n_tokens,
  embedding_dim,
  dense_nn_dim,
  num_heads,
  n_decoders,
  optimizer,
  batch_size,
  epochs):
  """Trains a stacked transformer decoder model
  
  Args:
      features (np.ndarray): Features
      targets (np.ndarray): Targets
      n_tokens (int): Number of tokens, including start-of-speech, end-of-speech tokens and mask token
      embedding_dim (int): Dimensionality of token embeddings
      dense_nn_dim (int): Dimensionality of feed forward network
      num_heads (int): Number of self attention heads
      n_decoders (int): Number of stacked decoders
      optimizer (str): Optimizer
      batch_size (int): Batch size
      epochs (int): Number of epochs
  
  Returns:
      keras.Model: trained model
  """
  logging.info("Training model...")
  model = stacked_decoder_model(
    sequence_length=features.shape[1],
    embedding_dim=embedding_dim,
    n_tokens = n_tokens,
    n_att_heads = num_heads,
    dense_dim = dense_nn_dim,
    n_decoders = n_decoders)

  model.compile(
    optimizer=optimizer,
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
  )

  hist = model.fit(
    features,
    targets,
    batch_size=batch_size,
    epochs=epochs)
  
  return model


def main(
  seed: int,
  country_code: str,
  query: str,
  embedding_dim: int,
  dense_nn_dim: int,
  num_heads: int,
  n_decoders: int,
  optimizer: str,
  batch_size: int,
  epochs: int,
  # n_samples: int
):
  """Main function
  """
  tf.random.set_seed(seed)
  np.random.seed(seed)

  # load data
  data = fetch_data(country_code, query)

  # data preprocessing
  features, targets, tokenizer = preprocess_data(data, SOS_TOKEN, EOS_TOKEN)

  dict_length = len(tokenizer.word_index)
  # model training
  model = train_model(
    features,
    targets,
    dict_length+1, #account for mask token
    embedding_dim,
    dense_nn_dim,
    num_heads,
    n_decoders,
    optimizer,
    batch_size,
    epochs)
  
  # save model
  run_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()
  model.save(MODEL_CACHE / f"model-{run_hash}")
  logging.info(f"Model saved in {MODEL_CACHE / f'model-{run_hash}'}")

  # save tokenizer
  tokenizer_json = tokenizer.to_json()
  with open(TOKENIZER_CACHE / f"tokenizer-{run_hash}.json", "w") as f:
    f.write(tokenizer_json)
  logging.info(f"Tokenizer saved in {TOKENIZER_CACHE / f'tokenizer-{run_hash}.json'}")

  while True:
    user_input = input("Press enter to produce a sample, any other key to exit: ")
    if user_input != "":
      break
    else:
      print(sample(model, tokenizer))


def entrypoint():
    # seed, country code, query, model parameters, training parameters
    parser = argparse.ArgumentParser(description="Trains a transformer model to generate city names")
    parser.add_argument("--seed", type=int, default=0, help="random seed. Defaults to 0")
    parser.add_argument("--country_code", type=str, default="GB", help="country code. Defaults to GB")
    parser.add_argument("--query", type=str, default=None, help="data-filtering query. Defaults to None")
    parser.add_argument("--embedding_dim", type=int, default=32, help="embedding dimension. Defaults to 32")
    parser.add_argument("--dense_nn_dim", type=int, default=512, help="dense neural network dimension. Defaults to 512")
    parser.add_argument("--num_heads", type=int, default=3, help="number of attention heads. Defaults to 2")
    parser.add_argument("--n_decoders", type=int, default=3, help="number of stacked decoders. Defaults to 1")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer. Defaults to adam")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size. Defaults to 32")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs. Defaults to 150")
    # parser.add_argument("--n_samples", type=int, default=10, help="number of samples to generate. Defaults to 10")
    args = parser.parse_args(sys.argv[1:])
    main(
      args.seed,
      args.country_code,
      args.query,
      args.embedding_dim,
      args.dense_nn_dim,
      args.num_heads,
      args.n_decoders,
      args.optimizer,
      args.batch_size,
      args.epochs,
      # args.n_samples
    )