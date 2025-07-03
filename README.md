# 🤖 GPT desde Cero - Modelo de Lenguaje en Español

Un modelo GPT completo implementado desde cero en PyTorch, optimizado para español con capacidades de chat y system prompts personalizables.

## 🎯 Características Principales

- **Arquitectura GPT**: Transformer decoder completo desde cero
- **Tokenizador BPE**: Optimizado para español
- **Sistema de Chat**: Historial de conversaciones y context management
- **System Prompts**: Personalización del comportamiento del modelo
- **Entrenamiento Flexible**: Configuración completa y modular
- **Inferencia Rápida**: Optimizado para generación en tiempo real

## 🚀 Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/gonzi-stack/PawGPT.git
cd PawGPT

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete
pip install -e .
```

## 📋 Dependencias Principales

```txt
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
numpy>=1.21.0
pandas>=1.3.0
pyyaml>=6.0
tqdm>=4.64.0
matplotlib>=3.5.0
tensorboard>=2.10.0
datasets>=2.12.0
```

## 🛠️ Configuración Inicial

### 1. Preparar Datos de Entrenamiento

```bash
# Colocar tus datos en data/raw/
# Formatos soportados: .txt, .json, .jsonl

# Ejemplo de estructura de datos:
# data/raw/
#   ├── corpus_español.txt
#   ├── conversaciones.jsonl
#   └── libros_español.txt

# Preparar y tokenizar datos
python scripts/prepare_data.py --input_dir data/raw --output_dir data/processed
```

### 2. Configurar el Modelo

Editar `configs/model_config.yaml`:

```yaml
model:
  vocab_size: 32000
  d_model: 512
  n_layers: 12
  n_heads: 8
  d_ff: 2048
  max_seq_length: 1024
  dropout: 0.1
  
tokenizer:
  vocab_size: 32000
  min_frequency: 2
  special_tokens:
    - "<PAD>"
    - "<UNK>"
    - "<BOS>"
    - "<EOS>"
    - "<SYS>"
    - "<USER>"
    - "<ASSISTANT>"
```

### 3. Configurar Entrenamiento

Editar `configs/training_config.yaml`:

```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 1e-4
  weight_decay: 0.01
  max_epochs: 10
  warmup_steps: 1000
  save_steps: 1000
  eval_steps: 500
  
optimizer:
  type: "adamw"
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  type: "cosine"
  warmup_ratio: 0.1
```

## 🎓 Entrenamiento del Modelo

### Entrenamiento Básico

```bash
# Entrenamiento estándar
python scripts/train.py \
  --model_config configs/model_config.yaml \
  --training_config configs/training_config.yaml \
  --data_path data/processed \
  --output_dir checkpoints/modelo_base

# Entrenamiento con múltiples GPUs
python -m torch.distributed.launch --nproc_per_node=2 scripts/train.py \
  --model_config configs/model_config.yaml \
  --training_config configs/training_config.yaml \
  --data_path data/processed \
  --output_dir checkpoints/modelo_base \
  --distributed
```

### Monitoreo del Entrenamiento

```bash
# Iniciar TensorBoard
tensorboard --logdir logs/

# Ver métricas en tiempo real
# Ir a http://localhost:6006
```

### Fine-tuning Personalizado

```bash
# Fine-tuning desde checkpoint
python scripts/train.py \
  --model_config configs/model_config.yaml \
  --training_config configs/training_config.yaml \
  --data_path data/processed \
  --output_dir checkpoints/modelo_personalizado \
  --resume_from checkpoints/modelo_base/best_model.pt \
  --learning_rate 5e-5
```

## 💬 Uso del Modelo

### Generación de Texto Simple

```python
from src.inference.generator import TextGenerator

# Cargar modelo entrenado
generator = TextGenerator(
    model_path="checkpoints/modelo_base/best_model.pt",
    config_path="configs/model_config.yaml"
)

# Generar texto
texto = generator.generate(
    prompt="El día de hoy aprendimos sobre",
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
print(texto)
```

### Sistema de Chat Interactivo

```python
from src.inference.chat import ChatSystem

# Inicializar sistema de chat
chat = ChatSystem(
    model_path="checkpoints/modelo_base/best_model.pt",
    config_path="configs/model_config.yaml"
)

# Configurar system prompt
chat.set_system_prompt(
    "Eres un asistente útil que responde en español de manera amigable y precisa."
)

# Conversación interactiva
while True:
    user_input = input("Usuario: ")
    if user_input.lower() in ['salir', 'exit']:
        break
    
    response = chat.chat(user_input)
    print(f"Asistente: {response}")
```

### Demo de Chat desde Terminal

```bash
# Ejecutar demo interactivo
python scripts/chat_demo.py \
  --model_path checkpoints/modelo_base/best_model.pt \
  --config_path configs/model_config.yaml \
  --system_prompt "Eres un asistente experto en inteligencia artificial."
```

## 🔧 Personalización Avanzada

### System Prompts Personalizados

```python
# Diferentes personalidades
prompts = {
    "profesional": "Eres un asistente profesional que responde de manera formal y precisa.",
    "amigable": "Eres un asistente amigable que usa un tono casual y cercano.",
    "educativo": "Eres un tutor que explica conceptos de manera clara y didáctica.",
    "creativo": "Eres un asistente creativo que proporciona ideas originales e innovadoras."
}

chat.set_system_prompt(prompts["profesional"])
```

### Configuración de Generación

```python
# Configuraciones de generación
configs = {
    "conservador": {"temperature": 0.3, "top_k": 20, "top_p": 0.8},
    "equilibrado": {"temperature": 0.7, "top_k": 40, "top_p": 0.9},
    "creativo": {"temperature": 1.0, "top_k": 50, "top_p": 0.95}
}

response = generator.generate(
    prompt="Escribe una historia sobre",
    **configs["creativo"]
)
```

## 📊 Evaluación del Modelo

### Métricas de Entrenamiento

```bash
# Evaluar modelo en dataset de validación
python scripts/evaluate.py \
  --model_path checkpoints/modelo_base/best_model.pt \
  --config_path configs/model_config.yaml \
  --data_path data/processed/val.jsonl
```

### Pruebas de Generación

```python
# Ejemplos de pruebas
test_prompts = [
    "La inteligencia artificial es",
    "En el futuro, la tecnología",
    "Explícame qué es",
    "¿Cómo funciona"
]

for prompt in test_prompts:
    response = generator.generate(prompt, max_length=50)
    print(f"Prompt: {prompt}")
    print(f"Respuesta: {response}")
    print("-" * 50)
```

## 🛡️ Mejores Prácticas

### Datos de Entrenamiento

1. **Calidad**: Usar textos bien escritos en español
2. **Diversidad**: Incluir diferentes dominios y estilos
3. **Limpieza**: Eliminar contenido duplicado o de baja calidad
4. **Formato**: Mantener formato consistente en conversaciones

### Entrenamiento

1. **Checkpoints**: Guardar modelo cada 1000 pasos
2. **Validación**: Evaluar regularmente en dataset de validación
3. **Learning Rate**: Usar warmup y decay apropiados
4. **Gradient Clipping**: Evitar exploding gradients

### Inferencia

1. **Temperatura**: Ajustar según el nivel de creatividad deseado
2. **Top-k/Top-p**: Balancear diversidad y coherencia
3. **Context Length**: Mantener contexto relevante pero manejable
4. **Memory**: Limpiar caché regularmente en sesiones largas

## 🔍 Resolución de Problemas

### Problemas Comunes

**Out of Memory (OOM)**
```bash
# Reducir batch size
# Usar gradient accumulation
# Activar gradient checkpointing
```

**Loss no converge**
```bash
# Verificar learning rate
# Revisar calidad de datos
# Ajustar warmup steps
```

**Generación repetitiva**
```bash
# Aumentar temperature
# Usar top-k/top-p sampling
# Verificar vocabulario
```

### Logs y Debugging

```python
# Activar logging detallado
import logging
logging.basicConfig(level=logging.DEBUG)

# Verificar shapes de tensores
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
```

## 📈 Escalabilidad y Optimización

### Entrenamiento Distribuido

```bash
# Múltiples nodos
python -m torch.distributed.launch \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr="192.168.1.100" \
  --master_port=12345 \
  scripts/train.py
```

### Optimizaciones de Memoria

```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(input_ids, labels=labels)
```

## 🤝 Contribución

1. Fork el repositorio
2. Crear branch para feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push al branch (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Reconocimientos

- Inspirado en la arquitectura GPT de OpenAI
- Basado en las mejores prácticas de Hugging Face
- Comunidad de PyTorch por las herramientas excepcionales

---
