"""
PawGPT System Prompt Module

Este módulo maneja el sistema de prompts para PawGPT, incluyendo templates,
roles, contexto dinámico y personalización de la personalidad del modelo.
"""

import json
import os
import re
import yaml
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import random

# Configurar logging
logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Tipos de prompts disponibles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CONTEXT = "context"
    INSTRUCTION = "instruction"
    EXAMPLE = "example"
    CONSTRAINT = "constraint"

class PersonalityType(Enum):
    """Tipos de personalidad disponibles."""
    HELPFUL = "helpful"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EXPERT = "expert"
    TEACHER = "teacher"
    COMPANION = "companion"
    CUSTOM = "custom"

class ContextType(Enum):
    """Tipos de contexto."""
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    CODE = "code"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    PERSONAL = "personal"
    EDUCATIONAL = "educational"

@dataclass
class PromptTemplate:
    """Template para prompts."""
    name: str
    template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    author: str = "PawGPT"
    
    def render(self, **kwargs) -> str:
        """Renderiza el template con las variables proporcionadas."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            logger.error(f"Variable faltante en template '{self.name}': {missing_var}")
            raise ValueError(f"Variable requerida no encontrada: {missing_var}")

@dataclass
class ConversationContext:
    """Contexto de conversación."""
    user_name: str = "Usuario"
    assistant_name: str = "PawGPT"
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    context_summary: str = ""
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str):
        """Añade un mensaje al historial."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_messages(self, limit: int = 5) -> List[Dict[str, str]]:
        """Obtiene los mensajes más recientes."""
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def clear_history(self):
        """Limpia el historial de conversación."""
        self.conversation_history.clear()

@dataclass
class PersonalityConfig:
    """Configuración de personalidad."""
    personality_type: PersonalityType = PersonalityType.HELPFUL
    tone: str = "friendly"
    formality: str = "casual"  # casual, formal, professional
    verbosity: str = "balanced"  # concise, balanced, verbose
    creativity: float = 0.7  # 0.0 - 1.0
    empathy: float = 0.8  # 0.0 - 1.0
    humor: float = 0.5  # 0.0 - 1.0
    technical_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    language_style: str = "natural"  # natural, formal, technical, casual
    cultural_context: str = "universal"  # universal, specific culture
    
    def to_prompt_additions(self) -> List[str]:
        """Convierte la configuración en adiciones al prompt."""
        additions = []
        
        # Tono y formalidad
        if self.formality == "formal":
            additions.append("Mantén un tono formal y profesional en tus respuestas.")
        elif self.formality == "casual":
            additions.append("Usa un tono casual y amigable en tus respuestas.")
        
        # Verbosidad
        if self.verbosity == "concise":
            additions.append("Sé conciso y directo en tus respuestas.")
        elif self.verbosity == "verbose":
            additions.append("Proporciona respuestas detalladas y completas.")
        
        # Creatividad
        if self.creativity > 0.8:
            additions.append("Sé creativo e innovador en tus respuestas y soluciones.")
        elif self.creativity < 0.3:
            additions.append("Enfócate en respuestas prácticas y directas.")
        
        # Empatía
        if self.empathy > 0.7:
            additions.append("Muestra empatía y comprensión hacia las necesidades del usuario.")
        
        # Humor
        if self.humor > 0.6:
            additions.append("Incluye humor apropiado cuando sea relevante.")
        
        # Nivel técnico
        tech_levels = {
            "beginner": "Explica conceptos técnicos de manera simple y accesible.",
            "intermediate": "Asume conocimiento básico y proporciona explicaciones balanceadas.",
            "advanced": "Usa terminología técnica apropiada y profundiza en detalles.",
            "expert": "Comunícate a nivel experto con terminología especializada."
        }
        if self.technical_level in tech_levels:
            additions.append(tech_levels[self.technical_level])
        
        return additions

class PromptBuilder(ABC):
    """Clase base para constructores de prompts."""
    
    @abstractmethod
    def build(self, context: ConversationContext, **kwargs) -> str:
        """Construye el prompt final."""
        pass

class SystemPromptBuilder(PromptBuilder):
    """Constructor de prompts del sistema."""
    
    def __init__(self, 
                 personality_config: PersonalityConfig = None,
                 base_instructions: str = None,
                 custom_instructions: List[str] = None):
        self.personality_config = personality_config or PersonalityConfig()
        self.base_instructions = base_instructions or self._get_default_instructions()
        self.custom_instructions = custom_instructions or []
    
    def _get_default_instructions(self) -> str:
        """Obtiene las instrucciones por defecto."""
        return """Eres PawGPT, un asistente de IA avanzado y útil. Tu objetivo es ayudar a los usuarios de manera efectiva, precisa y amigable."""
    
    def build(self, context: ConversationContext, **kwargs) -> str:
        """Construye el prompt del sistema."""
        sections = []
        
        # Instrucciones base
        sections.append(self.base_instructions)
        
        # Personalidad
        personality_additions = self.personality_config.to_prompt_additions()
        if personality_additions:
            sections.append("Características de personalidad:")
            sections.extend([f"- {addition}" for addition in personality_additions])
        
        # Contexto de usuario
        if context.user_name and context.user_name != "Usuario":
            sections.append(f"El usuario se llama {context.user_name}.")
        
        # Preferencias del usuario
        if context.user_preferences:
            prefs = []
            for key, value in context.user_preferences.items():
                prefs.append(f"- {key}: {value}")
            if prefs:
                sections.append("Preferencias del usuario:")
                sections.extend(prefs)
        
        # Instrucciones personalizadas
        if self.custom_instructions:
            sections.append("Instrucciones adicionales:")
            sections.extend([f"- {instruction}" for instruction in self.custom_instructions])
        
        # Contexto de la conversación
        if context.context_summary:
            sections.append(f"Contexto de la conversación: {context.context_summary}")
        
        return "\n\n".join(sections)

class ContextualPromptBuilder(PromptBuilder):
    """Constructor de prompts contextuales."""
    
    def __init__(self, context_type: ContextType = ContextType.CONVERSATION):
        self.context_type = context_type
        self.context_templates = self._load_context_templates()
    
    def _load_context_templates(self) -> Dict[ContextType, str]:
        """Carga los templates de contexto."""
        return {
            ContextType.CONVERSATION: "Basándote en nuestra conversación anterior:",
            ContextType.DOCUMENT: "Considerando el documento proporcionado:",
            ContextType.CODE: "Analizando el código proporcionado:",
            ContextType.CREATIVE: "Para esta tarea creativa:",
            ContextType.TECHNICAL: "Para este problema técnico:",
            ContextType.PERSONAL: "Considerando tu situación personal:",
            ContextType.EDUCATIONAL: "Para este objetivo de aprendizaje:"
        }
    
    def build(self, context: ConversationContext, **kwargs) -> str:
        """Construye el prompt contextual."""
        sections = []
        
        # Template de contexto
        if self.context_type in self.context_templates:
            sections.append(self.context_templates[self.context_type])
        
        # Historial reciente
        recent_messages = context.get_recent_messages(kwargs.get('history_limit', 3))
        if recent_messages:
            sections.append("Mensajes recientes:")
            for msg in recent_messages:
                role_name = "Usuario" if msg["role"] == "user" else "Asistente"
                sections.append(f"{role_name}: {msg['content']}")
        
        # Contexto adicional
        additional_context = kwargs.get('additional_context', '')
        if additional_context:
            sections.append(f"Contexto adicional: {additional_context}")
        
        return "\n\n".join(sections)

class PromptTemplateManager:
    """Gestor de templates de prompts."""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = templates_dir
        self.templates: Dict[str, PromptTemplate] = {}
        self.load_templates()
    
    def load_templates(self):
        """Carga todos los templates disponibles."""
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)
            self._create_default_templates()
        
        # Cargar templates desde archivos
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                self._load_template_file(os.path.join(self.templates_dir, filename))
    
    def _create_default_templates(self):
        """Crea templates por defecto."""
        default_templates = {
            "helpful_assistant": PromptTemplate(
                name="helpful_assistant",
                template="Eres un asistente útil y amigable. Responde a las preguntas del usuario de manera clara y concisa. {additional_instructions}",
                variables=["additional_instructions"],
                description="Template básico para asistente útil",
                category="general"
            ),
            "creative_writer": PromptTemplate(
                name="creative_writer",
                template="Eres un escritor creativo especializado en {genre}. Tu estilo es {style} y tu audiencia objetivo es {audience}. {task_description}",
                variables=["genre", "style", "audience", "task_description"],
                description="Template para escritura creativa",
                category="creative"
            ),
            "code_assistant": PromptTemplate(
                name="code_assistant",
                template="Eres un asistente de programación experto en {language}. Tu experiencia incluye {expertise_areas}. Ayuda con: {task_type}. {code_context}",
                variables=["language", "expertise_areas", "task_type", "code_context"],
                description="Template para asistencia en programación",
                category="technical"
            ),
            "teacher": PromptTemplate(
                name="teacher",
                template="Eres un profesor experto en {subject} enseñando a estudiantes de nivel {level}. Tu estilo de enseñanza es {teaching_style}. {lesson_context}",
                variables=["subject", "level", "teaching_style", "lesson_context"],
                description="Template para enseñanza",
                category="educational"
            )
        }
        
        # Guardar templates por defecto
        for name, template in default_templates.items():
            self.save_template(template)
    
    def _load_template_file(self, filepath: str):
        """Carga un template desde archivo."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                template = PromptTemplate(**data)
                self.templates[template.name] = template
        except Exception as e:
            logger.error(f"Error cargando template {filepath}: {e}")
    
    def save_template(self, template: PromptTemplate):
        """Guarda un template."""
        self.templates[template.name] = template
        
        # Guardar a archivo
        filepath = os.path.join(self.templates_dir, f"{template.name}.yaml")
        template_data = {
            'name': template.name,
            'template': template.template,
            'variables': template.variables,
            'description': template.description,
            'category': template.category,
            'tags': template.tags,
            'version': template.version,
            'author': template.author
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(template_data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"Error guardando template {template.name}: {e}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Obtiene un template por nombre."""
        return self.templates.get(name)
    
    def list_templates(self, category: str = None) -> List[PromptTemplate]:
        """Lista todos los templates, opcionalmente filtrados por categoría."""
        templates = list(self.templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates
    
    def search_templates(self, query: str) -> List[PromptTemplate]:
        """Busca templates por nombre, descripción o tags."""
        query = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query in template.name.lower() or 
                query in template.description.lower() or
                any(query in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results

class PromptProcessor:
    """Procesador de prompts con funciones avanzadas."""
    
    def __init__(self):
        self.variables_pattern = re.compile(r'\{([^}]+)\}')
        self.processors: Dict[str, Callable] = {
            'uppercase': str.upper,
            'lowercase': str.lower,
            'title': str.title,
            'capitalize': str.capitalize,
            'strip': str.strip,
            'length': len,
            'reverse': lambda x: x[::-1],
            'truncate': lambda x, n=100: x[:n] + '...' if len(x) > n else x
        }
    
    def process_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Procesa variables en el texto."""
        def replace_variable(match):
            var_expr = match.group(1)
            
            # Verificar si hay procesadores
            if '|' in var_expr:
                var_name, processors = var_expr.split('|', 1)
                var_name = var_name.strip()
                
                if var_name in variables:
                    value = variables[var_name]
                    
                    # Aplicar procesadores
                    for processor in processors.split('|'):
                        processor = processor.strip()
                        if ':' in processor:
                            proc_name, proc_arg = processor.split(':', 1)
                            if proc_name in self.processors:
                                value = self.processors[proc_name](value, int(proc_arg))
                        else:
                            if processor in self.processors:
                                value = self.processors[processor](value)
                    
                    return str(value)
            else:
                var_name = var_expr.strip()
                if var_name in variables:
                    return str(variables[var_name])
            
            return match.group(0)  # Devolver original si no se encuentra
        
        return self.variables_pattern.sub(replace_variable, text)
    
    def add_processor(self, name: str, func: Callable):
        """Añade un procesador personalizado."""
        self.processors[name] = func
    
    def validate_template(self, template: str, required_vars: List[str]) -> List[str]:
        """Valida que un template tenga todas las variables requeridas."""
        found_vars = set(self.variables_pattern.findall(template))
        missing_vars = []
        
        for var in required_vars:
            if var not in found_vars:
                missing_vars.append(var)
        
        return missing_vars

class SystemPromptManager:
    """Gestor principal del sistema de prompts."""
    
    def __init__(self, 
                 templates_dir: str = "templates",
                 default_personality: PersonalityConfig = None):
        self.template_manager = PromptTemplateManager(templates_dir)
        self.prompt_processor = PromptProcessor()
        self.default_personality = default_personality or PersonalityConfig()
        self.active_contexts: Dict[str, ConversationContext] = {}
    
    def create_context(self, 
                      session_id: str,
                      user_name: str = "Usuario",
                      user_preferences: Dict[str, Any] = None) -> ConversationContext:
        """Crea un nuevo contexto de conversación."""
        context = ConversationContext(
            user_name=user_name,
            session_id=session_id,
            user_preferences=user_preferences or {}
        )
        self.active_contexts[session_id] = context
        return context
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Obtiene un contexto existente."""
        return self.active_contexts.get(session_id)
    
    def build_system_prompt(self, 
                           session_id: str,
                           personality_config: PersonalityConfig = None,
                           template_name: str = None,
                           custom_instructions: List[str] = None,
                           **template_vars) -> str:
        """Construye el prompt del sistema completo."""
        context = self.get_context(session_id)
        if not context:
            raise ValueError(f"Contexto no encontrado para sesión: {session_id}")
        
        personality = personality_config or self.default_personality
        
        # Usar template si se especifica
        if template_name:
            template = self.template_manager.get_template(template_name)
            if not template:
                raise ValueError(f"Template no encontrado: {template_name}")
            
            base_instructions = template.render(**template_vars)
        else:
            base_instructions = None
        
        # Construir prompt del sistema
        system_builder = SystemPromptBuilder(
            personality_config=personality,
            base_instructions=base_instructions,
            custom_instructions=custom_instructions
        )
        
        return system_builder.build(context)
    
    def build_contextual_prompt(self,
                               session_id: str,
                               context_type: ContextType = ContextType.CONVERSATION,
                               **kwargs) -> str:
        """Construye un prompt contextual."""
        context = self.get_context(session_id)
        if not context:
            raise ValueError(f"Contexto no encontrado para sesión: {session_id}")
        
        contextual_builder = ContextualPromptBuilder(context_type)
        return contextual_builder.build(context, **kwargs)
    
    def update_context(self, 
                      session_id: str,
                      role: str,
                      content: str,
                      context_summary: str = None,
                      user_preferences: Dict[str, Any] = None):
        """Actualiza el contexto de conversación."""
        context = self.get_context(session_id)
        if context:
            context.add_message(role, content)
            
            if context_summary:
                context.context_summary = context_summary
            
            if user_preferences:
                context.user_preferences.update(user_preferences)
    
    def clear_context(self, session_id: str):
        """Limpia el contexto de una sesión."""
        if session_id in self.active_contexts:
            self.active_contexts[session_id].clear_history()
    
    def remove_context(self, session_id: str):
        """Elimina completamente un contexto."""
        if session_id in self.active_contexts:
            del self.active_contexts[session_id]
    
    def get_random_personality(self) -> PersonalityConfig:
        """Obtiene una configuración de personalidad aleatoria."""
        personalities = list(PersonalityType)
        personality_type = random.choice(personalities)
        
        return PersonalityConfig(
            personality_type=personality_type,
            tone=random.choice(["friendly", "professional", "casual", "warm"]),
            formality=random.choice(["casual", "formal", "professional"]),
            verbosity=random.choice(["concise", "balanced", "verbose"]),
            creativity=random.uniform(0.3, 0.9),
            empathy=random.uniform(0.5, 0.9),
            humor=random.uniform(0.2, 0.8),
            technical_level=random.choice(["beginner", "intermediate", "advanced"]),
            language_style=random.choice(["natural", "formal", "casual"])
        )
    
    def export_context(self, session_id: str) -> Optional[Dict]:
        """Exporta un contexto a diccionario."""
        context = self.get_context(session_id)
        if not context:
            return None
        
        return {
            "user_name": context.user_name,
            "assistant_name": context.assistant_name,
            "conversation_history": context.conversation_history,
            "context_summary": context.context_summary,
            "user_preferences": context.user_preferences,
            "session_id": context.session_id,
            "timestamp": context.timestamp.isoformat()
        }
    
    def import_context(self, context_data: Dict) -> str:
        """Importa un contexto desde diccionario."""
        session_id = context_data.get("session_id", f"imported_{datetime.now().timestamp()}")
        
        context = ConversationContext(
            user_name=context_data.get("user_name", "Usuario"),
            assistant_name=context_data.get("assistant_name", "PawGPT"),
            conversation_history=context_data.get("conversation_history", []),
            context_summary=context_data.get("context_summary", ""),
            user_preferences=context_data.get("user_preferences", {}),
            session_id=session_id,
            timestamp=datetime.fromisoformat(context_data.get("timestamp", datetime.now().isoformat()))
        )
        
        self.active_contexts[session_id] = context
        return session_id

# Funciones de utilidad
def create_personality_preset(name: str, **kwargs) -> PersonalityConfig:
    """Crea un preset de personalidad."""
    presets = {
        "helpful_assistant": PersonalityConfig(
            personality_type=PersonalityType.HELPFUL,
            tone="friendly",
            formality="casual",
            verbosity="balanced",
            creativity=0.6,
            empathy=0.8,
            humor=0.4,
            technical_level="intermediate"
        ),
        "creative_writer": PersonalityConfig(
            personality_type=PersonalityType.CREATIVE,
            tone="inspiring",
            formality="casual",
            verbosity="verbose",
            creativity=0.9,
            empathy=0.7,
            humor=0.6,
            technical_level="intermediate"
        ),
        "technical_expert": PersonalityConfig(
            personality_type=PersonalityType.EXPERT,
            tone="professional",
            formality="formal",
            verbosity="balanced",
            creativity=0.4,
            empathy=0.5,
            humor=0.2,
            technical_level="expert"
        ),
        "friendly_companion": PersonalityConfig(
            personality_type=PersonalityType.COMPANION,
            tone="warm",
            formality="casual",
            verbosity="balanced",
            creativity=0.7,
            empathy=0.9,
            humor=0.7,
            technical_level="beginner"
        )
    }
    
    if name in presets:
        preset = presets[name]
        # Actualizar con kwargs
        for key, value in kwargs.items():
            if hasattr(preset, key):
                setattr(preset, key, value)
        return preset
    else:
        return PersonalityConfig(**kwargs)

def quick_prompt(template_name: str, **variables) -> str:
    """Función rápida para crear un prompt desde template."""
    manager = PromptTemplateManager()
    template = manager.get_template(template_name)
    
    if not template:
        raise ValueError(f"Template no encontrado: {template_name}")
    
    return template.render(**variables)

# Ejemplo de uso
if __name__ == "__main__":
    # Crear gestor de prompts
    prompt_manager = SystemPromptManager()
    
    # Crear contexto
    session_id = "test_session"
    context = prompt_manager.create_context(
        session_id=session_id,
        user_name="Juan",
        user_preferences={"idioma": "español", "estilo": "casual"}
    )
    
    # Crear personalidad
    personality = create_personality_preset("helpful_assistant")
    
    # Construir prompt del sistema
    system_prompt = prompt_manager.build_system_prompt(
        session_id=session_id,
        personality_config=personality,
        custom_instructions=["Siempre incluye ejemplos cuando sea posible"]
    )
    
    print("Sistema de Prompts para PawGPT:")
    print("=" * 50)
    print(system_prompt)
    
    # Simular conversación
    prompt_manager.update_context(session_id, "user", "Hola, ¿cómo estás?")
    prompt_manager.update_context(session_id, "assistant", "¡Hola Juan! Estoy muy bien, gracias por preguntar.")
    
    # Construir prompt contextual
    contextual_prompt = prompt_manager.build_contextual_prompt(
        session_id=session_id,
        context_type=ContextType.CONVERSATION,
        additional_context="El usuario parece estar iniciando una conversación casual."
    )
    
    print("\nPrompt Contextual:")
    print("=" * 50)
    print(contextual_prompt)
