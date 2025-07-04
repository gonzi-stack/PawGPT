# FILE: src/model/__init__.py

"""
Modulo principal del modelo PawGPT.

Este paquete contiene la implementacion de la arquitectura GPT
desde cero, incluyendo las capas de atencion, feed-forward,
embedding, y el modelo completo para causal language modeling.
"""

# Importar las clases y funciones clave de los sub-modulos
# Esto permite importarlos directamente desde el paquete src.model
# Ejemplo: from src.model import GPTConfig, GPTForCausalLM

# Asumimos que config.py contendra la clase GPTConfig
# Si config.py esta vacio, esta importacion fallara hasta que se implemente.
try:
    from .config import GPTConfig
except ImportError:
    print("Advertencia: No se pudo importar GPTConfig desde .config. Asegurese de que src/model/config.py la contenga.")
    # Define una dummy class si no se puede importar, para que el __init__ no falle
    # En un proyecto real, esto indica un problema de dependencias no resueltas.
    class GPTConfig:
         def __init__(self, **kwargs):
             print("Usando GPTConfig dummy.")
             # Rellena con atributos minimos si es estrictamente necesario para otras imports dummy
             self.d_model = kwargs.get("d_model", 512)
             self.n_heads = kwargs.get("n_heads", 8)
             self.dropout = kwargs.get("dropout", 0.1)
             # ... otros atributos necesarios para inicializacion dummy de otras clases si aplica


# Asumimos que attention.py contendra la clase MultiHeadAttention
try:
    from .attention import MultiHeadAttention
except ImportError:
    print("Advertencia: No se pudo importar MultiHeadAttention desde .attention. Asegurese de que src/model/attention.py la contenga.")
    # Define una dummy class si no se puede importar
    class MultiHeadAttention:
         def __init__(self, config):
              print("Usando MultiHeadAttention dummy.")
         def forward(self, *args, **kwargs):
              raise NotImplementedError("MultiHeadAttention dummy no implementada")

# Asumimos que layers.py contendra FeedForward, LayerNorm, PositionalEmbedding
try:
    from .layers import FeedForward, LayerNorm, PositionalEmbedding
except ImportError:
    print("Advertencia: No se pudo importar FeedForward, LayerNorm, PositionalEmbedding desde .layers. Asegurese de que src/model/layers.py las contenga.")
    # Define dummy classes si no se pueden importar
    class FeedForward:
         def __init__(self, config):
              print("Usando FeedForward dummy.")
         def forward(self, *args, **kwargs):
              raise NotImplementedError("FeedForward dummy no implementada")
    class LayerNorm:
         def __init__(self, normalized_shape, eps):
              print("Usando LayerNorm dummy.")
         def forward(self, *args, **kwargs):
              # Retornar el input, ya que LayerNorm dummy no hace nada real
              if args: return args[0]
              raise NotImplementedError("LayerNorm dummy no implementada")
    class PositionalEmbedding:
         def __init__(self, config):
              print("Usando PositionalEmbedding dummy.")
         def forward(self, input_ids, position_ids=None):
              # Retornar ceros con la forma de embedding, para que gpt.py no falle inmediatamente
              # Esto no es correcto, pero evita errores de shape si se usa la dummy
              batch_size, seq_len = input_ids.shape
              return torch.zeros(batch_size, seq_len, config.d_model, device=input_ids.device)


# Importar las clases principales del modelo desde gpt.py
from .gpt import GPTBlock, GPTModel, GPTForCausalLM, create_model, GPTOutput


# __all__ define la interfaz publica del paquete.
# Especifica que nombres se importaran cuando alguien use 'from src.model import *'.
# Es una buena practica definirlo para claridad.
__all__ = [
    "GPTConfig",
    "MultiHeadAttention",
    "FeedForward",
    "LayerNorm",
    "PositionalEmbedding",
    "GPTBlock", # Aunque GPTBlock es interna, podria ser util para debug o extension
    "GPTModel",
    "GPTForCausalLM",
    "create_model",
    "GPTOutput"
]

# Opcionalmente, podrias incluir alguna inicializacion o verificacion aqui
# print("Paquete src.model inicializado.")

