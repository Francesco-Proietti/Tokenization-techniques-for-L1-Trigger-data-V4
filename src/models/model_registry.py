# Useful file used by train.py in order to pick the correct model specified in the config file 

from src.models.mlp_vqvae import MLPVQVAE
from src.models.transformer_vqvae import TransformerVQVAE

MODEL_REGISTRY = {
    "mlp": MLPVQVAE,
    "transformer": TransformerVQVAE,
}