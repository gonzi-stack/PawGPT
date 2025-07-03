"""
PawGPT Generator Module

Este módulo implementa algoritmos avanzados de generación de texto para PawGPT,
incluyendo diferentes estrategias de sampling, control de repetición y optimizaciones.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Configurar logging
logger = logging.getLogger(__name__)

class SamplingStrategy(Enum):
    """Estrategias de sampling disponibles."""
    GREEDY = "greedy"
    RANDOM = "random"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TEMPERATURE = "temperature"
    NUCLEUS = "nucleus"
    TYPICAL = "typical"
    CONTRASTIVE = "contrastive"
    BEAM_SEARCH = "beam_search"

class StoppingCriteria(Enum):
    """Criterios de parada para la generación."""
    MAX_LENGTH = "max_length"
    EOS_TOKEN = "eos_token"
    CUSTOM_TOKENS = "custom_tokens"
    PERPLEXITY_THRESHOLD = "perplexity_threshold"
    REPETITION_PENALTY = "repetition_penalty"

@dataclass
class GenerationConfig:
    """Configuración para la generación de texto."""
    # Configuración básica
    max_length: int = 512
    min_length: int = 10
    do_sample: bool = True
    early_stopping: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
    
    # Configuración de sampling
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    
    # Control de repetición
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    encoder_repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    
    # Tokens especiales
    bos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = 0
    eos_token_id: Optional[int] = 2
    decoder_start_token_id: Optional[int] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    
    # Configuración avanzada
    diversity_penalty: float = 0.0
    remove_invalid_values: bool = False
    exponential_decay_length_penalty: Optional[Tuple[int, float]] = None
    suppress_tokens: Optional[List[int]] = None
    begin_suppress_tokens: Optional[List[int]] = None
    forced_decoder_ids: Optional[List[List[int]]] = None
    
    # Configuración de control
    guidance_scale: float = 1.0
    low_memory: bool = False
    num_return_sequences: int = 1
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False
    return_dict_in_generate: bool = False
    
    # Configuración personalizada
    custom_stopping_criteria: Optional[List[Callable]] = None
    logits_processors: Optional[List[Callable]] = None
    prefix_allowed_tokens_fn: Optional[Callable] = None

@dataclass
class GenerationOutput:
    """Salida de la generación de texto."""
    sequences: torch.Tensor
    scores: Optional[torch.Tensor] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    beam_indices: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la salida a diccionario."""
        return {
            "sequences": self.sequences.tolist() if self.sequences is not None else None,
            "scores": self.scores.tolist() if self.scores is not None else None,
            "num_sequences": self.sequences.shape[0] if self.sequences is not None else 0,
            "sequence_length": self.sequences.shape[1] if self.sequences is not None else 0
        }

class LogitsProcessor(ABC):
    """Clase base para procesadores de logits."""
    
    @abstractmethod
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Procesa los logits."""
        pass

class TemperatureLogitsProcessor(LogitsProcessor):
    """Procesador de logits con temperatura."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.temperature != 1.0:
            scores = scores / self.temperature
        return scores

class TopKLogitsProcessor(LogitsProcessor):
    """Procesador de logits con top-k sampling."""
    
    def __init__(self, top_k: int = 50):
        self.top_k = top_k
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.top_k <= 0:
            return scores
        
        top_k = min(self.top_k, scores.size(-1))
        top_k_scores, _ = torch.topk(scores, top_k)
        indices_to_remove = scores < top_k_scores[..., -1, None]
        scores = scores.masked_fill(indices_to_remove, -float('Inf'))
        return scores

class TopPLogitsProcessor(LogitsProcessor):
    """Procesador de logits con top-p (nucleus) sampling."""
    
    def __init__(self, top_p: float = 0.9):
        self.top_p = top_p
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.top_p >= 1.0:
            return scores
        
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Eliminar tokens con probabilidad acumulada > top_p
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, -float('Inf'))
        return scores

class TypicalLogitsProcessor(LogitsProcessor):
    """Procesador de logits con typical sampling."""
    
    def __init__(self, typical_p: float = 0.9):
        self.typical_p = typical_p
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.typical_p >= 1.0:
            return scores
        
        # Calcular probabilidades y entropía
        probs = F.softmax(scores, dim=-1)
        log_probs = F.log_softmax(scores, dim=-1)
        
        # Calcular diferencia con entropía típica
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        deviation = torch.abs(log_probs + entropy)
        
        # Ordenar por desviación típica
        sorted_deviations, sorted_indices = torch.sort(deviation)
        sorted_probs = probs.gather(-1, sorted_indices)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Encontrar cutoff
        cutoff = torch.searchsorted(cumulative_probs, self.typical_p)
        cutoff = torch.clamp(cutoff, min=1, max=scores.size(-1))
        
        # Aplicar máscara
        indices_to_remove = sorted_indices >= cutoff.unsqueeze(-1)
        scores = scores.masked_fill(indices_to_remove, -float('Inf'))
        return scores

class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """Procesador de logits con penalización de repetición."""
    
    def __init__(self, repetition_penalty: float = 1.0):
        self.repetition_penalty = repetition_penalty
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.repetition_penalty == 1.0:
            return scores
        
        # Aplicar penalización a tokens ya generados
        score = torch.gather(scores, 1, input_ids)
        
        # Penalizar tokens repetidos
        score = torch.where(score < 0, score * self.repetition_penalty, score / self.repetition_penalty)
        scores.scatter_(1, input_ids, score)
        
        return scores

class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Procesador que previene repetición de n-gramas."""
    
    def __init__(self, ngram_size: int = 2):
        self.ngram_size = ngram_size
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.ngram_size <= 0 or input_ids.size(1) < self.ngram_size:
            return scores
        
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)
        
        for batch_idx in range(batch_size):
            # Extraer n-gramas prohibidos
            banned_batch_tokens = self._calc_banned_ngram_tokens(
                input_ids[batch_idx], cur_len, self.ngram_size
            )
            
            # Aplicar penalización
            for token in banned_batch_tokens:
                scores[batch_idx, token] = -float('inf')
        
        return scores
    
    def _calc_banned_ngram_tokens(self, prev_input_ids: torch.Tensor, num_hypos: int, ngram_size: int) -> List[int]:
        """Calcula tokens prohibidos basados en n-gramas anteriores."""
        if num_hypos + 1 < ngram_size:
            return []
        
        generated_ngrams = {}
        for i in range(num_hypos + 1 - ngram_size):
            ngram = tuple(prev_input_ids[i:i + ngram_size - 1].tolist())
            next_token = prev_input_ids[i + ngram_size - 1].item()
            generated_ngrams[ngram] = generated_ngrams.get(ngram, []) + [next_token]
        
        # Encontrar n-grama actual
        current_ngram = tuple(prev_input_ids[-ngram_size + 1:].tolist())
        return generated_ngrams.get(current_ngram, [])

class BeamSearchScorer:
    """Scorer para beam search."""
    
    def __init__(self, batch_size: int, num_beams: int, device: torch.device, length_penalty: float = 1.0):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        
        # Inicializar beams
        self.beams = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        self.beam_tokens = torch.zeros((batch_size, num_beams), dtype=torch.long, device=device)
        self.beam_indices = torch.zeros((batch_size, num_beams), dtype=torch.long, device=device)
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    def process(self, input_ids: torch.Tensor, next_scores: torch.Tensor, next_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Procesa un paso de beam search."""
        cur_len = input_ids.shape[-1]
        batch_size = len(self.beams)
        
        # Aplicar penalización de longitud
        if self.length_penalty != 1.0:
            length_penalty = ((5 + cur_len) / (5 + 1)) ** self.length_penalty
            next_scores = next_scores / length_penalty
        
        # Reshape para beam search
        next_scores = next_scores.view(batch_size, self.num_beams * next_scores.shape[-1])
        next_tokens = next_tokens.view(batch_size, self.num_beams * next_tokens.shape[-1])
        
        # Seleccionar top beams
        next_scores, next_tokens = torch.topk(next_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True)
        
        # Calcular índices de beam
        next_indices = torch.div(next_tokens, next_tokens.shape[-1] // self.num_beams, rounding_mode='floor')
        next_tokens = next_tokens % (next_tokens.shape[-1] // self.num_beams)
        
        return next_scores, next_tokens, next_indices
    
    def finalize(self, input_ids: torch.Tensor, final_beam_scores: torch.Tensor) -> torch.Tensor:
        """Finaliza beam search y retorna las mejores secuencias."""
        batch_size = input_ids.shape[0] // self.num_beams
        
        # Aplicar penalización de longitud final
        if self.length_penalty != 1.0:
            length_penalty = ((5 + input_ids.shape[-1]) / (5 + 1)) ** self.length_penalty
            final_beam_scores = final_beam_scores / length_penalty
        
        # Seleccionar mejores beams
        best_beam_indices = torch.argmax(final_beam_scores.view(batch_size, self.num_beams), dim=1)
        best_beam_indices = best_beam_indices + torch.arange(batch_size, device=self.device) * self.num_beams
        
        return input_ids[best_beam_indices]

class TextGenerator:
    """Generador de texto principal para PawGPT."""
    
    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"TextGenerator inicializado en dispositivo: {self.device}")
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 generation_config: Optional[GenerationConfig] = None,
                 **kwargs) -> GenerationOutput:
        """Genera texto usando el modelo."""
        
        # Configuración por defecto
        config = generation_config or GenerationConfig()
        
        # Actualizar configuración con kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Mover tensores al dispositivo
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Seleccionar estrategia de generación
        if config.num_beams > 1:
            return self._beam_search_generate(input_ids, attention_mask, config)
        else:
            return self._sample_generate(input_ids, attention_mask, config)
    
    def _sample_generate(self, 
                        input_ids: torch.Tensor,
                        attention_mask: Optional[torch.Tensor],
                        config: GenerationConfig) -> GenerationOutput:
        """Generación con sampling."""
        
        batch_size = input_ids.shape[0]
        max_length = config.max_length
        
        # Preparar procesadores de logits
        logits_processors = self._get_logits_processors(config)
        
        # Preparar criterios de parada
        stopping_criteria = self._get_stopping_criteria(config)
        
        # Inicializar variables
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        scores = None
        
        # Bucle de generación
        for cur_len in range(input_ids.shape[1], max_length):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
            
            # Aplicar procesadores de logits
            for processor in logits_processors:
                next_token_logits = processor(input_ids, next_token_logits)
            
            # Sampling
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Actualizar secuencias no terminadas
            next_tokens = next_tokens * unfinished_sequences + config.pad_token_id * (1 - unfinished_sequences)
            
            # Concatenar tokens
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Actualizar attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=self.device)], dim=-1)
            
            # Verificar criterios de parada
            if config.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(config.eos_token_id.shape[0], 1).ne(config.eos_token_id.unsqueeze(1)).prod(dim=0)
                    if isinstance(config.eos_token_id, list)
                    else next_tokens.ne(config.eos_token_id)
                )
            
            # Parar si todas las secuencias están terminadas
            if unfinished_sequences.max() == 0:
                break
        
        return GenerationOutput(sequences=input_ids, scores=scores)
    
    def _beam_search_generate(self,
                             input_ids: torch.Tensor,
                             attention_mask: Optional[torch.Tensor],
                             config: GenerationConfig) -> GenerationOutput:
        """Generación con beam search."""
        
        batch_size = input_ids.shape[0]
        num_beams = config.num_beams
        max_length = config.max_length
        
        # Expandir input_ids para beam search
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1).contiguous().view(batch_size * num_beams, -1)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1).contiguous().view(batch_size * num_beams, -1)
        
        # Inicializar beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=self.device,
            length_penalty=config.length_penalty
        )
        
        # Inicializar beam scores
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        
        # Bucle de generación
        for cur_len in range(input_ids.shape[1], max_length):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
            
            # Calcular scores
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            
            # Reshape para beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Seleccionar top tokens
            next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            
            # Calcular índices de beam
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % vocab_size
            
            # Reordenar beams
            beam_outputs = []
            for batch_idx in range(batch_size):
                beam_outputs.append([])
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_token_scores[batch_idx], next_indices[batch_idx])
                ):
                    batch_beam_idx = batch_idx * num_beams + next_index
                    beam_outputs[batch_idx].append((next_score, next_token, batch_beam_idx))
            
            # Seleccionar mejores beams
            beam_scores = []
            beam_tokens = []
            beam_indices = []
            
            for batch_idx in range(batch_size):
                beam_outputs[batch_idx] = sorted(beam_outputs[batch_idx], key=lambda x: x[0], reverse=True)
                for i in range(num_beams):
                    next_score, next_token, batch_beam_idx = beam_outputs[batch_idx][i]
                    beam_scores.append(next_score)
                    beam_tokens.append(next_token)
                    beam_indices.append(batch_beam_idx)
            
            beam_scores = torch.tensor(beam_scores, device=self.device)
            beam_tokens = torch.tensor(beam_tokens, device=self.device)
            beam_indices = torch.tensor(beam_indices, device=self.device)
            
            # Reordenar input_ids
            input_ids = input_ids[beam_indices]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(-1)], dim=-1)
            
            # Actualizar attention mask
            if attention_mask is not None:
                attention_mask = attention_mask[beam_indices]
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size * num_beams, 1), device=self.device)], dim=-1)
            
            # Verificar criterios de parada
            if config.eos_token_id is not None:
                eos_mask = beam_tokens.eq(config.eos_token_id)
                if eos_mask.any():
                    break
        
        # Finalizar beam search
        final_sequences = beam_scorer.finalize(input_ids, beam_scores)
        
        return GenerationOutput(sequences=final_sequences, scores=beam_scores)
    
    def _get_logits_processors(self, config: GenerationConfig) -> List[LogitsProcessor]:
        """Obtiene la lista de procesadores de logits."""
        processors = []
        
        if config.temperature != 1.0:
            processors.append(TemperatureLogitsProcessor(config.temperature))
        
        if config.top_k > 0:
            processors.append(TopKLogitsProcessor(config.top_k))
        
        if config.top_p < 1.0:
            processors.append(TopPLogitsProcessor(config.top_p))
        
        if config.typical_p < 1.0:
            processors.append(TypicalLogitsProcessor(config.typical_p))
        
        if config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(config.repetition_penalty))
        
        if config.no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(config.no_repeat_ngram_size))
        
        # Añadir procesadores personalizados
        if config.logits_processors:
            processors.extend(config.logits_processors)
        
        return processors
    
    def _get_stopping_criteria(self, config: GenerationConfig) -> List[Callable]:
        """Obtiene los criterios de parada."""
        criteria = []
        
        # Criterio de longitud máxima
        def max_length_criterion(input_ids, scores):
            return input_ids.shape[-1] >= config.max_length
        
        criteria.append(max_length_criterion)
        
        # Añadir criterios personalizados
        if config.custom_stopping_criteria:
            criteria.extend(config.custom_stopping_criteria)
        
        return criteria
    
    def generate_text(self, 
                     prompt: str, 
                     generation_config: Optional[GenerationConfig] = None,
                     **kwargs) -> str:
        """Genera texto a partir de un prompt."""
        
        # Tokenizar prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        # Generar
        output = self.generate(input_ids, generation_config=generation_config, **kwargs)
        
        # Decodificar
        generated_text = self.tokenizer.decode(output.sequences[0].tolist())
        
        # Limpiar prompt del resultado
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def batch_generate(self, 
                      prompts: List[str],
                      generation_config: Optional[GenerationConfig] = None,
                      **kwargs) -> List[str]:
        """Genera texto para múltiples prompts."""
        
        # Tokenizar prompts
        input_ids_list = [self.tokenizer.encode(prompt) for prompt in prompts]
        
        # Pad secuencias
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids = torch.zeros((len(prompts), max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(prompts), max_len), dtype=torch.long, device=self.device)
        
        for i, ids in enumerate(input_ids_list):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attention_mask[i, :len(ids)] = 1
        
        # Generar
        output = self.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs)
        
        # Decodificar
        results = []
        for i, sequence in enumerate(output.sequences):
            generated_text = self.tokenizer.decode(sequence.tolist())
            if generated_text.startswith(prompts[i]):
                generated_text = generated_text[len(prompts[i]):].strip()
            results.append(generated_text)
        
        return results

# Funciones de utilidad
def create_generation_config(strategy: str = "nucleus", **kwargs) -> GenerationConfig:
    """Crea una configuración de generación basada en estrategia."""
    
    if strategy == "greedy":
        config = GenerationConfig(
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            **kwargs
        )
    elif strategy == "beam_search":
        config = GenerationConfig(
            do_sample=False,
            num_beams=kwargs.get('num_beams', 4),
            early_stopping=True,
            **kwargs
        )
    elif strategy == "nucleus":
        config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            **kwargs
        )
    elif strategy == "top_k":
        config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=1.0,
            **kwargs
        )
    elif strategy == "creative":
        config = GenerationConfig(
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            **kwargs
        )
    elif strategy == "precise":
        config = GenerationConfig(
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.2,
            **kwargs
        )
    else:
        config = GenerationConfig(**kwargs)
    
    return config