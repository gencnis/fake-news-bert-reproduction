import torch
import transformers
import datasets
import evaluate
import sklearn
import pandas
import matplotlib

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("evaluate:", evaluate.__version__)
print("sklearn:", sklearn.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")