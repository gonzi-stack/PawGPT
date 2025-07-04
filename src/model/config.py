# FILE: src/model/config.py

"""
Configuracion para el modelo GPT de PawGPT.

Este modulo define la clase GPTConfig que encapsula todos
los hiperparametros necesarios para construir la arquitectura
del modelo GPT.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class GPTConfig:
    """
    Representa la configuracion del modelo GPT.
    """
    Atributos:
        vocab_size (int): Tamano del vocabulario del tokenizador.
        d_model (int): Dimension de los embeddings y los estados ocultos
                       en las capas Transformer.
        n_layers (int): Numero de capas Transformer (GPTBlocks).
        n_heads (int): Numero de cabezas de atencion en Multi-Head Attention.
        d_ff (int): Dimension de la capa oculta en la red Feed-Forward.
        max_seq_length (int): Longitud maxima de secuencia que el modelo puede manejar.
        dropout (float): Tasa de dropout aplicada a varias capas (embeddings, output de blocks).
        attention_dropout (float): Tasa de dropout especifica para los pesos de atencion.
        activation_function (str): Funcion de activacion usada en la red Feed-Forward
                                   (ej. 'gelu', 'relu').
        initializer_range (float): Rango para la inicializacion normal de pesos.
        layer_norm_eps (float): Epsilon para la estabilidad numerica en Layer Normalization.
        position_embedding_type (str): Tipo de embedding posicional ('absolute').
        use_cache (bool): Si habilitar el cache de K/V para optimizar la inferencia
                          paso a paso.
        use_bias_in_attention (bool): Si usar bias en las proyecciones lineales de atencion.
                                      (Este no estaba explicitamente en model_config.yaml,
                                       pero es comun y util). Por defecto, True.
        # Parametros relacionados con el idioma (opcional, para documentacion/contexto)
        language (str): Idioma objetivo del modelo (ej. 'es').
        case_sensitive (bool): Si el tokenizador es sensible a mayusculas/minusculas.

        # Otros parametros que puedan venir de la configuracion general
        # y que necesiten ser accesibles desde la configuracion del modelo.
        # Se pueden anadir mas campos explicitamente si se vuelven criticos,
        # o manejar via **kwargs si son menos frecuentes.
        # Ejemplo:
        # pad_token_id: int = 0
        # eos_token_id: int = 2

        # Permite aceptar otros parametros via **kwargs sin fallar
        # (util cuando la config YAML tiene campos adicionales)
        # Note: dataclass automaticamente maneja **kwargs si no hay un __init__ personalizado,
        # pero podemos anadir un campo catch-all si es necesario, aunque suele no serlo
        # si los campos se definen explicitamente.
        pass

    # Nota: Dataclasses generan __init__, __repr__, __eq__ automaticamente.
    # No necesitamos definirlos a menos que necesitemos logica especial.

# Ejemplo de como se usaria (esto no va en el archivo config.py, es solo ilustracion):
#
# from src.model.config import GPTConfig
# import yaml
#
# # Suponiendo que tienes un archivo config.yaml
# with open('configs/model_config.yaml', 'r') as f:
#     config_data = yaml.safe_load(f)
#
# # Extraer solo la parte del modelo de la configuracion total
# model_params = config_data.get('model', {})
#
# # Crear una instancia de GPTConfig
# # Pasamos los parametros del YAML. dataclass los mapeara automaticamente.
# # Si el YAML tiene mas campos de los definidos en GPTConfig, seran ignorados por defecto,
# # a menos que el __init__ de GPTConfig se personalice para capturarlos con **kwargs.
# # Sin personalizar __init__, simplemente no se pasaran a los atributos de la dataclass.
# # Una forma comun de manejar esto es:
#
# # Filtrar solo los argumentos que la dataclass acepta
# valid_params = {k: v for k, v in model_params.items() if k in GPTConfig.__annotations__}
#
# # Crear la instancia
# config = GPTConfig(**valid_params)
#
# print(config)
# print(f"Dimension del modelo: {config.d_model}")
