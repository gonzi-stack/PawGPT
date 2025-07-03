"""
PawGPT Inference Module

Este módulo proporciona la funcionalidad de inferencia para PawGPT,
un modelo de lenguaje diseñado para generar respuestas coherentes y contextualmente relevantes.
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Versión del módulo
__version__ = "1.0.0"
__author__ = "Gonzalo (gonzi-stack)"

# Configuración por defecto
DEFAULT_CONFIG = {
    "model_path": "models/pawgpt_model.pt",
    "vocab_path": "models/vocab.json",
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "pad_token_id": 0,
    "eos_token_id": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

@dataclass
class InferenceConfig:
    """Configuración para la inferencia del modelo."""
    model_path: str = DEFAULT_CONFIG["model_path"]
    vocab_path: str = DEFAULT_CONFIG["vocab_path"]
    max_length: int = DEFAULT_CONFIG["max_length"]
    temperature: float = DEFAULT_CONFIG["temperature"]
    top_p: float = DEFAULT_CONFIG["top_p"]
    top_k: int = DEFAULT_CONFIG["top_k"]
    do_sample: bool = DEFAULT_CONFIG["do_sample"]
    pad_token_id: int = DEFAULT_CONFIG["pad_token_id"]
    eos_token_id: int = DEFAULT_CONFIG["eos_token_id"]
    device: str = DEFAULT_CONFIG["device"]

class TokenizerManager:
    """Gestor del tokenizador para PawGPT."""
    
    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self.vocab = self._load_vocab()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def _load_vocab(self) -> Dict[str, int]:
        """Carga el vocabulario desde el archivo JSON."""
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            logger.info(f"Vocabulario cargado exitosamente desde {self.vocab_path}")
            return vocab
        except FileNotFoundError:
            logger.error(f"Archivo de vocabulario no encontrado: {self.vocab_path}")
            # Vocabulario básico por defecto
            return {
                "<pad>": 0, "<unk>": 1, "<eos>": 2, "<bos>": 3,
                "hola": 4, "como": 5, "estas": 6, "que": 7, "tal": 8
            }
        except json.JSONDecodeError:
            logger.error(f"Error al decodificar el archivo de vocabulario: {self.vocab_path}")
            return {}
    
    def encode(self, text: str) -> List[int]:
        """Convierte texto a tokens."""
        tokens = text.lower().split()
        return [self.vocab.get(token, self.vocab.get("<unk>", 1)) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convierte tokens a texto."""
        tokens = [self.inverse_vocab.get(token_id, "<unk>") for token_id in token_ids]
        return " ".join(tokens)
    
    def get_vocab_size(self) -> int:
        """Retorna el tamaño del vocabulario."""
        return len(self.vocab)

class ModelManager:
    """Gestor del modelo PawGPT."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = TokenizerManager(config.vocab_path)
        self.device = torch.device(config.device)
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo desde el archivo."""
        try:
            if os.path.exists(self.config.model_path):
                self.model = torch.load(self.config.model_path, map_location=self.device)
                self.model.eval()
                logger.info(f"Modelo cargado exitosamente desde {self.config.model_path}")
            else:
                logger.warning(f"Archivo de modelo no encontrado: {self.config.model_path}")
                self.model = self._create_dummy_model()
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            self.model = self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Crea un modelo dummy para testing."""
        logger.info("Creando modelo dummy para testing...")
        
        class DummyModel(torch.nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=512):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
                self.transformer = torch.nn.TransformerEncoder(
                    torch.nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=8,
                        batch_first=True
                    ),
                    num_layers=6
                )
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
            
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.transformer(x)
                logits = self.lm_head(x)
                return {"logits": logits}
        
        model = DummyModel(vocab_size=self.tokenizer.get_vocab_size())
        model.to(self.device)
        model.eval()
        return model
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Genera texto basado en el prompt."""
        # Combinar configuración por defecto con kwargs
        generation_config = {
            "max_length": self.config.max_length,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "do_sample": self.config.do_sample,
            **kwargs
        }
        
        # Tokenizar el prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Generar respuesta
        with torch.no_grad():
            generated_ids = self._generate_tokens(input_tensor, generation_config)
        
        # Decodificar la respuesta
        response = self.tokenizer.decode(generated_ids[0].tolist())
        
        # Limpiar la respuesta (remover el prompt original)
        response = response.replace(prompt.lower(), "").strip()
        
        return response
    
    def _generate_tokens(self, input_ids: torch.Tensor, config: Dict) -> torch.Tensor:
        """Genera tokens usando el modelo."""
        batch_size = input_ids.shape[0]
        max_length = config["max_length"]
        temperature = config["temperature"]
        top_p = config["top_p"]
        top_k = config["top_k"]
        do_sample = config["do_sample"]
        
        # Preparar para la generación
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.model(generated)
            logits = outputs["logits"][:, -1, :]  # Obtener logits del último token
            
            # Aplicar temperatura
            logits = logits / temperature
            
            # Aplicar top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Aplicar softmax para obtener probabilidades
            probabilities = torch.softmax(logits, dim=-1)
            
            # Aplicar top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Encontrar el índice de corte
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Aplicar el filtro
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probabilities[indices_to_remove] = 0
                probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            
            # Muestrear el siguiente token
            if do_sample:
                next_token = torch.multinomial(probabilities, 1)
            else:
                next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
            
            # Añadir el token generado
            generated = torch.cat([generated, next_token], dim=1)
            
            # Verificar si se alcanzó el token de fin
            if next_token.item() == self.config.eos_token_id:
                break
        
        return generated

class PawGPTInference:
    """Clase principal para la inferencia de PawGPT."""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.model_manager = ModelManager(self.config)
        self.conversation_history = []
        
        logger.info("PawGPT Inference inicializado correctamente")
    
    def chat(self, message: str, system_prompt: str = None) -> str:
        """Función principal para chat."""
        try:
            # Construir el prompt con contexto
            prompt = self._build_prompt(message, system_prompt)
            
            # Generar respuesta
            response = self.model_manager.generate(prompt)
            
            # Añadir a la historia de conversación
            self.conversation_history.append({
                "user": message,
                "assistant": response
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error en chat: {e}")
            return "Lo siento, ocurrió un error al procesar tu mensaje."
    
    def _build_prompt(self, message: str, system_prompt: str = None) -> str:
        """Construye el prompt con contexto."""
        prompt_parts = []
        
        # Añadir prompt del sistema si existe
        if system_prompt:
            prompt_parts.append(f"Sistema: {system_prompt}")
        
        # Añadir historia de conversación reciente (últimos 3 intercambios)
        recent_history = self.conversation_history[-3:]
        for exchange in recent_history:
            prompt_parts.append(f"Usuario: {exchange['user']}")
            prompt_parts.append(f"Asistente: {exchange['assistant']}")
        
        # Añadir el mensaje actual
        prompt_parts.append(f"Usuario: {message}")
        prompt_parts.append("Asistente:")
        
        return "\n".join(prompt_parts)
    
    def clear_history(self):
        """Limpia la historia de conversación."""
        self.conversation_history.clear()
        logger.info("Historia de conversación limpiada")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo."""
        return {
            "model_path": self.config.model_path,
            "vocab_size": self.model_manager.tokenizer.get_vocab_size(),
            "device": str(self.model_manager.device),
            "config": self.config.__dict__
        }

# Funciones de utilidad para fácil uso
def create_inference_engine(config_path: str = None) -> PawGPTInference:
    """Crea una instancia de PawGPTInference."""
    config = InferenceConfig()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
    
    return PawGPTInference(config)

def quick_chat(message: str, model_path: str = None) -> str:
    """Función rápida para chat sin mantener estado."""
    config = InferenceConfig()
    if model_path:
        config.model_path = model_path
    
    engine = PawGPTInference(config)
    return engine.chat(message)

# Exportar las clases y funciones principales
__all__ = [
    "PawGPTInference",
    "InferenceConfig", 
    "TokenizerManager",
    "ModelManager",
    "create_inference_engine",
    "quick_chat",
    "DEFAULT_CONFIG"
]

# Mensaje de inicio
logger.info(f"PawGPT Inference Module v{__version__} cargado exitosamente")