import torch
import train

def predict(text, model, text_field, label_field):
  assert isinstance(text, str)
  # ????
  model.eval()

  text  = text_field.preprocess(text)


def save():
  print()

if __name__=="__main__":
  print()