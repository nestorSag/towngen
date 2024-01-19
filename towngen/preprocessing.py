
import re
import typing as t
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def text_replacer(
  data: np.ndarray,
  replacements: t.List[t.Tuple[str, str]]) -> np.ndarray:
  """
  Makes a series of substring replacements in each entry of the passed text array
  
  Args:
      data (np.ndarray): text array
      replacements (t.List[t.Tuple[str, str]]): List of tuples where the first entry is the regex pattern to be replaced, and the second entry is the replacement.
  
  """
  for replacement in replacements:
    before, after = replacement
    data = np.array(list(map(lambda v: re.sub(before,after, v) ,data)))

  return data


def add_delimiter_tokens(
  data: np.ndarray,
  sos_token: str = "^",
  eos_token: str = "$") -> np.ndarray:
  """Adds start-of-speech and end-of-speech tokens to text
  
  Args:
      data (np.ndarray): text corpus
      sos_token (str, optional): start-of-speech token
      eos_token (str, optional): end-of-speech token
  
  Returns:
      np.ndarray: updated corpus
  """
  return np.array(list(map(lambda v: sos_token + v + eos_token ,data)))

def get_char_prediction_features(
  data: np.ndarray,
  sos_token: str,
  eos_token: str) -> t.Tuple[np.ndarray, np.ndarray]:
  """Map corpus to arrays ready for model consumption for the task of character prediction.
  
  Args:
      data (np.ndarray): Text corpus
      sos_token (str): Start-of-speech token
      eos_token (str): end-of-speech token
  
  Returns:
      t.Tuple[np.ndarray, np.ndarray]: Triplet with numpy arrays (features and targets) and the fitted Tensorflow tokenizer.
  """
  char_tk = Tokenizer(
    num_words=None, 
    char_level=True
  )

  char_tk.fit_on_texts(data)

  sequences = char_tk.texts_to_sequences(data)

  sequences = pad_sequences(sequences, padding="post")

  m, n = sequences.shape

  # split sequences into features and target
  features, targets = np.copy(sequences)[:,0:n-1], np.copy(sequences)[:,1:n]

  # in input features, replace EOS id by 0 (i.e. mask token) since predicting what follows after EOS is moot
  features[features == char_tk.word_index[eos_token]] = 0

  return features, targets, char_tk
