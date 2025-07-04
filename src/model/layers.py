# FILE: src/model/layers.py

"""
Implementacion de las capas modulares para el modelo GPT.

Contiene la red Feed-Forward, Layer Normalization,
y la capa de Positional Embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Asumimos que GPTConfig se define en src.model.config
# Si config.py esta vacio, deberas definir la clase GPTConfig alli
# con los atributos necesarios (d_model, d_ff, dropout, activation_function,
# max_seq_length, vocab_size, layer_norm_eps, position_embedding_type, etc.)
try:
    from .config import GPTConfig
except ImportError:
    print("Advertencia: No se pudo importar GPTConfig desde .config. Asegurese de que src/model/config.py la contenga.")
    # Define una dummy class si no se puede importar (para que este archivo sea parseable)
    # En un proyecto real, esto indica un problema de dependencias no resueltas.
    class GPTConfig:
         def __init__(self, d_model=512, d_ff=2048, dropout=0.1, activation_function="gelu",
                      max_seq_length=1024, vocab_size=32000, layer_norm_eps=1e-5,
                      position_embedding_type="absolute", **kwargs):
             print("Usando GPTConfig dummy en layers.py.")
             self.d_model = d_model
             self.d_ff = d_ff
             self.dropout = dropout
             self.activation_function = activation_function
             self.max_seq_length = max_seq_length
             self.vocab_size = vocab_size
             self.layer_norm_eps = layer_norm_eps
             self.position_embedding_type = position_embedding_type
             # Otros atributos...


class LayerNorm(nn.Module):
    """
    Implementacion simplificada de Layer Normalization.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        # Parametros learnables de escala (weight) y desplazamiento (bias)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps # Epsilon para evitar division por cero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica Layer Normalization.

        Args:
            x: Tensor de entrada [..., last_dim].
               La normalizacion se aplica sobre la ultima dimension.

        Returns:
            Tensor normalizado.
        """
        # Calcular media y varianza a lo largo de la ultima dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # Variancia de poblacion

        # Normalizar y aplicar parametros learnables
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class FeedForward(nn.Module):
    """
    Implementacion de la red Feed-Forward (MLP) usada en bloques Transformer.

    Consiste en dos capas lineales con una activacion intermedia.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.dropout = config.dropout

        # Primera capa lineal (expansion)
        self.fc1 = nn.Linear(self.d_model, self.d_ff)
        # Segunda capa lineal (contraccion)
        self.fc2 = nn.Linear(self.d_ff, self.d_model)

        # Funcion de activacion
        # Soporte basico para GELU, ReLU, Swish/SiLU
        if config.activation_function == "gelu":
            self.activation = nn.GELU()
        elif config.activation_function == "relu":
            self.activation = nn.ReLU()
        elif config.activation_function == "swish" or config.activation_function == "silu":
            self.activation = nn.SiLU()
        else:
            # Usar GELU por defecto si la funcion no es reconocida
            warnings.warn(f"Funcion de activacion '{config.activation_function}' no reconocida. Usando GELU por defecto.")
            self.activation = nn.GELU()

        # Dropout despues de la segunda capa lineal
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red Feed-Forward.

        Args:
            x: Tensor de entrada [batch_size, seq_len, d_model].

        Returns:
            Tensor de salida [batch_size, seq_len, d_model].
        """
        # x -> fc1 -> activation -> fc2 -> dropout
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout_layer(x) # Aplicar dropout aqui es comun en Transformers (despues de la MLP)
        return x


class PositionalEmbedding(nn.Module):
    """
    Implementacion de Positional Embedding.

    Soporta embeddings posicionales absolutos aprendidos.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.max_seq_length = config.max_seq_length
        self.d_model = config.d_model
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')

        if self.position_embedding_type == "absolute":
            # Crear una tabla de embeddings para cada posicion hasta max_seq_length
            self.pos_embeds = nn.Embedding(self.max_seq_length, self.d_model)
        else:
            # Podrias anadir soporte para otros tipos (ej. sinusoidal) aqui
            raise NotImplementedError(f"Tipo de embedding posicional '{self.position_embedding_type}' no soportado todavia.")

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aplica el embedding posicional.

        Args:
            input_ids: Tensor de IDs de tokens [batch_size, seq_len].
                       Usado solo para obtener la forma y el dispositivo.
            position_ids: Tensor de IDs de posicion [batch_size, seq_len].
                          Si es None, se asume que son [0, 1, ..., seq_len-1].

        Returns:
            Tensor con los embeddings posicionales [batch_size, seq_len, d_model].
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if self.position_embedding_type == "absolute":
            if position_ids is None:
                # Crear IDs de posicion si no se proporcionan
                # Esto es el caso tipico para secuencias de entrada
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len) # [batch_size, seq_len]

            # Asegurarse de que los IDs de posicion esten dentro del rango maximo
            if torch.max(position_ids) >= self.max_seq_length:
                 raise ValueError(f"position_ids exceden max_seq_length ({self.max_seq_length}).")

            # Obtener embeddings de la tabla
            position_embeddings = self.pos_embeds(position_ids)
            return position_embeddings
        else:
             # No implementado (ya manejado en __init__)
             pass # Este path no deberia ser alcanzado si NotImplementedError se lanza en __init__


# Nota: La clase GPTConfig es necesaria para la inicializacion de estas capas.
# Debe estar definida en src/model/config.py y contener los atributos
# d_model, d_ff, dropout, activation_function, max_seq_length, vocab_size,
# layer_norm_eps, position_embedding_type.
# El codigo dummy de GPTConfig arriba es solo para que este archivo sea parseable.
