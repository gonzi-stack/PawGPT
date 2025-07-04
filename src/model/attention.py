# FILE: src/model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings # Para warnings sobre mask shape, si es necesario
from typing import Optional, Tuple, Dict, Any

# Asumimos que GPTConfig se define en src.model.config
# Si config.py esta vacio, deberas definir la clase GPTConfig alli
# con los atributos necesarios (d_model, n_heads, attention_dropout, dropout, etc.)
try:
    from .config import GPTConfig
except ImportError:
    # Define una dummy class si no se puede importar (para que este archivo sea parseable)
    # En un proyecto real, esto deberia estar correctamente importado.
    print("Advertencia: No se pudo importar GPTConfig. Usando una clase dummy.")
    class GPTConfig:
         def __init__(self, d_model=512, n_heads=8, attention_dropout=0.1, dropout=0.1, layer_norm_eps=1e-5, initializer_range=0.02, **kwargs):
             self.d_model = d_model
             self.n_heads = n_heads
             self.attention_dropout = attention_dropout
             self.dropout = dropout
             self.layer_norm_eps = layer_norm_eps
             self.initializer_range = initializer_range
             # Asumimos use_bias_in_attention por defecto si no esta en la config
             self.use_bias_in_attention = kwargs.get("use_bias_in_attention", True)
             # Otros parametros...


class MultiHeadAttention(nn.Module):
    """
    Implementacion de Atencion Multi-Cabeza (Multi-Head Self-Attention)
    para un modelo tipo GPT (Transformer Decoder).

    Soporta enmascaramiento causal, enmascaramiento de padding y caching
    para una inferencia eficiente.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Validaciones basicas
        # Asegurarse de que la dimension del modelo sea divisible por el numero de cabezas
        assert config.d_model % config.n_heads == 0, "d_model debe ser divisible por n_heads"

        self.d_model = config.d_model # Dimension del modelo
        self.n_heads = config.n_heads # Numero de cabezas de atencion
        # Usar el dropout especifico para la atencion (si existe, sino el general)
        self.dropout_prob = getattr(config, 'attention_dropout', config.dropout)
        self.head_dim = self.d_model // self.n_heads # Dimension de cada cabeza
        # Factor de escala para los scores de atencion
        self.scale = self.head_dim**-0.5

        # Proyecciones lineales para Q, K, V y la salida
        # Combinamos Q, K, V para eficiencia (un solo tensor de salida 3*d_model)
        # Asumimos que config tiene un parametro use_bias_in_attention o usamos True por defecto
        use_bias = getattr(config, 'use_bias_in_attention', True)
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=use_bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=use_bias)

        # Dropout para los pesos de atencion (softmax)
        self.attn_dropout = nn.Dropout(self.dropout_prob)
        # Dropout para la salida final (despues de la proyeccion de salida)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None # Cache (key, value) de la capa anterior
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass del mecanismo de atencion multi-cabeza.

        Args:
            x: Tensor de entrada [batch_size, seq_len_q, d_model].
               seq_len_q es 1 durante la generacion token a token con cache.
            attention_mask: Mascara de atencion. [batch_size, 1, seq_len_q, total_seq_len_kv].
                            Combina mascara causal y de padding. Un valor de 0 indica
                            posiciones a enmascarar (-inf), 1 indica posiciones a atender (0).
            use_cache: Si usar cache para la generacion.
            past_key_values: Una tupla de tensores (key_cache, value_cache)
                             de pasos de generacion anteriores.

        Returns:
            Tuple:
                - attn_output: Tensor de salida [batch_size, seq_len_q, d_model]
                - cache: Una tupla con los tensores key y value actualizados para caching.
                         Sera None si use_cache es False.
        """
        batch_size, seq_len_q, d_model = x.size()

        # 1. Proyecciones lineales y split en cabezas
        # x: [batch_size, seq_len_q, d_model] -> qkv: [batch_size, seq_len_q, 3 * d_model]
        qkv = self.qkv_proj(x)

        # Split en Q, K, V y reshape para multi-cabeza
        # qkv: [batch_size, seq_len_q, 3 * d_model] -> [batch_size, seq_len_q, 3, n_heads, head_dim]
        # Transpose: [batch_size, n_heads, seq_len_q, head_dim] para Q
        # [batch_size, n_heads, seq_len_k, head_dim] para K y V
        qkv = qkv.reshape(batch_size, seq_len_q, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]

        # 2. Manejo del cache
        # Concatena K y V del paso actual con el cache de pasos anteriores
        if use_cache and past_key_values is not None:
            # past_key_values es (past_key, past_value)
            past_key, past_value = past_key_values
            k = torch.cat([past_key, k], dim=2) # Concatena K a lo largo de la dimension de secuencia
            v = torch.cat([past_value, v], dim=2) # Concatena V a lo largo de la dimension de secuencia

        # La cache para la proxima iteracion es simplemente el K y V calculados *hasta este paso*
        cache = (k, v) if use_cache else None

        # La longitud total de K y V despues de concatenar con el cache
        total_seq_len_kv = k.size(2)
        # La longitud de Q siempre es la del input actual
        current_seq_len_q = q.size(2)


        # 3. Calcular scores de atencion
        # Q [batch_size, n_heads, seq_len_q, head_dim]
        # K transpose [batch_size, n_heads, head_dim, total_seq_len_kv]
        # (Q @ K.transpose) -> [batch_size, n_heads, seq_len_q, total_seq_len_kv]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 4. Aplicar mascara de atencion
        # La mascara esperada es [batch_size, 1, seq_len_q, total_seq_len_kv]
        # Un valor de 0 en la mascara se convierte en -inf en los scores
        if attention_mask is not None:
             # Asegurarse de que la mascara tiene las dimensiones correctas para broadcasting
             # Esperamos [batch, 1, current_seq_len_q, total_seq_len_kv]
             # Si la mascara viene con otra forma (ej. de gpt.py), puede ser necesario adaptarla aqui.
             # Asumimos que gpt.py la prepara correctamente [batch, 1, total_seq_len_kv, total_seq_len_kv]
             # y que solo necesitamos el slice para seq_len_q
             if attention_mask.dim() == 4:
                 # Si la mascara viene como [batch, 1, total_seq_len_kv, total_seq_len_kv]
                 # Slice para coincidir con la longitud de Q (current_seq_len_q)
                 # Esto es crucial cuando use_cache es True y seq_len_q es 1
                 if attention_mask.size(2) != current_seq_len_q:
                      # Tomar el ultimo 'current_seq_len_q' de la dimension 2
                      mask_slice = attention_mask[:, :, -current_seq_len_q:, :]
                 else:
                      # Si ya coincide, usar la mascara completa (caso entrenamiento o seq_len_q == total_seq_len_kv)
                      mask_slice = attention_mask

                 # Ahora mask_slice deberia ser [batch, 1, current_seq_len_q, total_seq_len_kv]
                 # Donde mask_slice es 0, ponemos -inf en attn_scores
                 attn_scores = attn_scores.masked_fill(mask_slice == 0, float('-inf'))

             elif attention_mask.dim() == 2:
                 # Si la mascara original era solo de padding [batch_size, total_seq_len_kv]
                 # Esto requeriria re-generar la mascara causal aqui y combinarla.
                 # Es mejor que gpt.py lo maneje y pase la mascara 4D combinada.
                 # Emitimos un warning para indicar que la forma esperada es 4D.
                  warnings.warn(f"Unexpected attention_mask shape: {attention_mask.shape}. Expected 4 dimensions after combining causal/padding.")
                  # Intentar un manejo basico asumiendo mascara de padding [batch_size, total_seq_len_kv]
                  # Expandir a [batch_size, 1, 1, total_seq_len_kv]
                  mask_2d = attention_mask.unsqueeze(1).unsqueeze(1)
                  # Crear mascara causal para la longitud de Q
                  causal_mask = torch.tril(torch.ones((current_seq_len_q, total_seq_len_kv), device=x.device)) # [current_seq_len_q, total_seq_len_kv]
                  # Combinar mascara de padding con mascara causal
                  combined_mask = mask_2d * causal_mask.unsqueeze(0).unsqueeze(0) # [batch, 1, current_seq_len_q, total_seq_len_kv]
                  attn_scores = attn_scores.masked_fill(combined_mask == 0, float('-inf'))
             else:
                 warnings.warn(f"Unhandled attention_mask shape: {attention_mask.shape}")


        # 5. Softmax para obtener pesos de atencion
        # Aplicar softmax a lo largo de la dimension total_seq_len_kv
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Aplicar dropout a los pesos de atencion
        attn_weights = self.attn_dropout(attn_weights)

        # 6. Multiplicar pesos de atencion por V
        # attn_weights [batch_size, n_heads, seq_len_q, total_seq_len_kv]
        # V [batch_size, n_heads, total_seq_len_kv, head_dim]
        # (attn_weights @ V) -> [batch_size, n_heads, seq_len_q, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # 7. Concatenar cabezas y proyeccion final
        # attn_output: [batch_size, n_heads, seq_len_q, head_dim]
        # Transponer para tener [batch_size, seq_len_q, n_heads, head_dim]
        # Reshape para concatenar en la dimension d_model -> [batch_size, seq_len_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, current_seq_len_q, self.d_model)

        # Proyeccion final de la salida
        attn_output = self.out_proj(attn_output)

        # Aplicar dropout residual
        attn_output = self.resid_dropout(attn_output)

        return attn_output, cache

# Nota: La clase GPTConfig es necesaria para la inicializacion de MultiHeadAttention.
# Debe estar definida en src/model/config.py y contener atributos como
# d_model, n_heads, attention_dropout, dropout, y potencialmente use_bias_in_attention.
# Si usas el codigo dummy de arriba, es solo para evitar errores de importacion
# en este archivo especifico si config.py esta realmente vacio, pero el modelo GPT
# no funcionara correctamente sin una configuracion valida.
