"""
Arquitectura GPT completa implementada desde cero en PyTorch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .attention import MultiHeadAttention
from .layers import FeedForward, LayerNorm, PositionalEmbedding
from .config import GPTConfig


@dataclass
class GPTOutput:
    """Salida del modelo GPT con información adicional."""
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    loss: Optional[torch.Tensor] = None


class GPTBlock(nn.Module):
    """Bloque básico del transformer GPT."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Capas del bloque
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
        # Layer normalization
        self.ln1 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass del bloque GPT.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, d_model]
            attention_mask: Máscara de atención [batch_size, seq_len, seq_len]
            use_cache: Si usar caché para generación
            past_key_values: Valores pasados de key y value
            
        Returns:
            Tuple con el tensor de salida y los valores de caché
        """
        # Conexión residual 1: Multi-Head Attention
        residual = x
        x = self.ln1(x)
        
        attn_output, cache = self.attention(
            x, 
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        
        x = residual + self.dropout(attn_output)
        
        # Conexión residual 2: Feed Forward
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        
        return x, cache


class GPTModel(nn.Module):
    """Modelo GPT completo desde cero."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = PositionalEmbedding(config)
        
        # Layers del transformer
        self.layers = nn.ModuleList([
            GPTBlock(config) for _ in range(config.n_layers)
        ])
        
        # Layer normalization final
        self.ln_final = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Inicialización de pesos
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización de pesos siguiendo las mejores prácticas."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> GPTOutput:
        """
        Forward pass del modelo GPT.
        
        Args:
            input_ids: IDs de tokens [batch_size, seq_len]
            attention_mask: Máscara de atención [batch_size, seq_len]
            position_ids: IDs de posición [batch_size, seq_len]
            use_cache: Si usar caché para generación
            past_key_values: Valores pasados de key y value
            output_attentions: Si retornar pesos de atención
            output_hidden_states: Si retornar estados ocultos
            
        Returns:
            GPTOutput con logits y información adicional
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Crear máscara de atención si no se proporciona
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)
            
        # Crear máscara causal para GPT
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Combinar máscaras
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
        attention_mask = attention_mask * causal_mask  # Broadcasting
        
        # Aplicar embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(input_ids, position_ids)
        
        # Combinar embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Almacenar estados ocultos si se requiere
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cache = () if use_cache else None
        
        # Pasar por todas las capas
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # Obtener caché de la capa actual
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Forward pass de la capa
            hidden_states, cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=layer_past
            )
            
            if use_cache:
                all_cache += (cache,)
                
        # Layer normalization final
        hidden_states = self.ln_final(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        return GPTOutput(
            logits=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class GPTForCausalLM(nn.Module):
    """Modelo GPT para generación de lenguaje causal."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Modelo base
        self.gpt = GPTModel(config)
        
        # Cabeza de salida para predicción de tokens
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Compartir pesos entre embedding y lm_head
        self.lm_head.weight = self.gpt.token_embedding.weight
        
        # Inicializar pesos
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización específica para el modelo de lenguaje."""
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> GPTOutput:
        """
        Forward pass del modelo GPT para causal LM.
        
        Args:
            input_ids: IDs de tokens [batch_size, seq_len]
            attention_mask: Máscara de atención [batch_size, seq_len]
            position_ids: IDs de posición [batch_size, seq_len]
            labels: Etiquetas para calcular loss [batch_size, seq_len]
            use_cache: Si usar caché para generación
            past_key_values: Valores pasados de key y value
            output_attentions: Si retornar pesos de atención
            output_hidden_states: Si retornar estados ocultos
            
        Returns:
            GPTOutput con logits, loss y información adicional
        """
        # Forward pass del modelo base
        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Obtener logits
        logits = self.lm_head(outputs.logits)
        
        # Calcular loss si se proporcionan labels
        loss = None
        if labels is not None:
            # Desplazar tokens para predicción
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calcular cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
        return GPTOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss=loss
        )
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generar texto usando el modelo.
        
        Args:
            input_ids: IDs de tokens de entrada [batch_size, seq_len]
            max_length: Longitud máxima de generación
            temperature: Temperatura para sampling
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Si usar sampling o greedy
            pad_token_id: ID del token de padding
            eos_token_id: ID del token de fin de secuencia
            
        Returns:
            Tensor con los tokens generados
        """
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Inicializar caché
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                
                # Obtener logits del último token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Aplicar temperatura
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                    
                # Aplicar top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                # Aplicar top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sampling o greedy
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Agregar tokens generados
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                
                # Verificar si se generó EOS
                if next_tokens.item() == eos_token_id:
                    break
                    
                # Actualizar caché para próxima iteración
                past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
                
        return input_ids
        
    def get_num_params(self) -> int:
        """Obtener número de parámetros del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Obtener información de uso de memoria."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2  # MB
            }
        return {'message': 'CUDA not available'}


def create_model(config: GPTConfig) -> GPTForCausalLM:
    """
    Crear modelo GPT con configuración dada.
    
    Args:
        config: Configuración del modelo
        
    Returns:
        Modelo GPT inicializado
    """
    model = GPTForCausalLM(config)
    
    # Imprimir información del modelo
    num_params = model.get_num_params()
    print(f"Modelo creado con {num_params:,} parámetros")
    
    return model