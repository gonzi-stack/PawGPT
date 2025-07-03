"""
PawGPT Chat Module

Este módulo proporciona una interfaz de chat interactiva para PawGPT,
incluyendo manejo de conversaciones, personalidades, y funcionalidades avanzadas.
"""

import os
import json
import time
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Imports locales
from . import PawGPTInference, InferenceConfig, create_inference_engine

# Configurar logging
logger = logging.getLogger(__name__)

class ChatState(Enum):
    """Estados posibles del chat."""
    IDLE = "idle"
    THINKING = "thinking"
    RESPONDING = "responding"
    ERROR = "error"
    WAITING_INPUT = "waiting_input"

class MessageType(Enum):
    """Tipos de mensajes en el chat."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"
    INFO = "info"

@dataclass
class ChatMessage:
    """Estructura para un mensaje de chat."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    type: MessageType = MessageType.USER
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el mensaje a diccionario."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Crea un mensaje desde un diccionario."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            type=MessageType(data.get("type", "user")),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            metadata=data.get("metadata", {})
        )

@dataclass
class ChatPersonality:
    """Define la personalidad del chatbot."""
    name: str = "PawGPT"
    description: str = "Un asistente amigable y útil"
    system_prompt: str = "Eres PawGPT, un asistente de IA amigable y útil."
    response_style: str = "conversational"  # conversational, formal, playful, etc.
    max_response_length: int = 200
    temperature: float = 0.7
    personality_traits: List[str] = field(default_factory=list)
    
    def get_system_prompt(self) -> str:
        """Genera el prompt del sistema basado en la personalidad."""
        base_prompt = self.system_prompt
        
        if self.personality_traits:
            traits = ", ".join(self.personality_traits)
            base_prompt += f" Tus características principales son: {traits}."
        
        if self.response_style == "playful":
            base_prompt += " Responde de manera juguetona y divertida."
        elif self.response_style == "formal":
            base_prompt += " Mantén un tono formal y profesional."
        elif self.response_style == "conversational":
            base_prompt += " Mantén un tono conversacional y natural."
        
        return base_prompt

class ChatSession:
    """Maneja una sesión de chat individual."""
    
    def __init__(self, session_id: str = None, personality: ChatPersonality = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.personality = personality or ChatPersonality()
        self.messages: List[ChatMessage] = []
        self.state = ChatState.IDLE
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.metadata = {}
        
        # Añadir mensaje inicial del sistema
        self.add_message(
            content=self.personality.get_system_prompt(),
            message_type=MessageType.SYSTEM
        )
    
    def add_message(self, content: str, message_type: MessageType = MessageType.USER, metadata: Dict = None) -> ChatMessage:
        """Añade un mensaje a la sesión."""
        message = ChatMessage(
            content=content,
            type=message_type,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_activity = datetime.now()
        return message
    
    def get_conversation_history(self, limit: int = 10) -> List[ChatMessage]:
        """Obtiene el historial de conversación reciente."""
        # Filtrar mensajes del sistema para el historial
        conversation_messages = [
            msg for msg in self.messages 
            if msg.type in [MessageType.USER, MessageType.ASSISTANT]
        ]
        return conversation_messages[-limit:]
    
    def get_context_for_inference(self) -> str:
        """Construye el contexto para la inferencia."""
        context_parts = []
        
        # Añadir prompt del sistema
        system_messages = [msg for msg in self.messages if msg.type == MessageType.SYSTEM]
        if system_messages:
            context_parts.append(f"Sistema: {system_messages[-1].content}")
        
        # Añadir historial reciente
        history = self.get_conversation_history(6)  # Últimos 6 mensajes
        for msg in history:
            if msg.type == MessageType.USER:
                context_parts.append(f"Usuario: {msg.content}")
            elif msg.type == MessageType.ASSISTANT:
                context_parts.append(f"Asistente: {msg.content}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la sesión a diccionario."""
        return {
            "session_id": self.session_id,
            "personality": {
                "name": self.personality.name,
                "description": self.personality.description,
                "system_prompt": self.personality.system_prompt,
                "response_style": self.personality.response_style,
                "personality_traits": self.personality.personality_traits
            },
            "messages": [msg.to_dict() for msg in self.messages],
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Crea una sesión desde un diccionario."""
        personality_data = data.get("personality", {})
        personality = ChatPersonality(
            name=personality_data.get("name", "PawGPT"),
            description=personality_data.get("description", "Un asistente amigable y útil"),
            system_prompt=personality_data.get("system_prompt", "Eres PawGPT, un asistente de IA amigable y útil."),
            response_style=personality_data.get("response_style", "conversational"),
            personality_traits=personality_data.get("personality_traits", [])
        )
        
        session = cls(
            session_id=data.get("session_id"),
            personality=personality
        )
        
        # Cargar mensajes
        session.messages = [
            ChatMessage.from_dict(msg_data) 
            for msg_data in data.get("messages", [])
        ]
        
        session.state = ChatState(data.get("state", "idle"))
        session.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        session.last_activity = datetime.fromisoformat(data.get("last_activity", datetime.now().isoformat()))
        session.metadata = data.get("metadata", {})
        
        return session

class ChatManager:
    """Gestor principal para múltiples sesiones de chat."""
    
    def __init__(self, inference_engine: PawGPTInference = None, config: InferenceConfig = None):
        self.inference_engine = inference_engine or create_inference_engine()
        self.config = config or InferenceConfig()
        self.sessions: Dict[str, ChatSession] = {}
        self.active_session_id: Optional[str] = None
        self.message_handlers: List[Callable] = []
        
        # Personalidades predefinidas
        self.personalities = {
            "default": ChatPersonality(
                name="PawGPT",
                description="Asistente general amigable y útil",
                system_prompt="Eres PawGPT, un asistente de IA amigable y útil. Responde de manera clara y concisa.",
                response_style="conversational",
                personality_traits=["amigable", "útil", "claro"]
            ),
            "formal": ChatPersonality(
                name="PawGPT Profesional",
                description="Asistente formal para contextos profesionales",
                system_prompt="Eres PawGPT, un asistente profesional. Mantén un tono formal y preciso.",
                response_style="formal",
                personality_traits=["profesional", "preciso", "formal"]
            ),
            "creative": ChatPersonality(
                name="PawGPT Creativo",
                description="Asistente creativo para tareas artísticas",
                system_prompt="Eres PawGPT, un asistente creativo. Sé imaginativo y expresivo.",
                response_style="playful",
                personality_traits=["creativo", "imaginativo", "expresivo"]
            )
        }
        
        logger.info("ChatManager inicializado correctamente")
    
    def create_session(self, personality_name: str = "default") -> ChatSession:
        """Crea una nueva sesión de chat."""
        personality = self.personalities.get(personality_name, self.personalities["default"])
        session = ChatSession(personality=personality)
        self.sessions[session.session_id] = session
        self.active_session_id = session.session_id
        
        logger.info(f"Nueva sesión creada: {session.session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Obtiene una sesión por ID."""
        return self.sessions.get(session_id)
    
    def get_active_session(self) -> Optional[ChatSession]:
        """Obtiene la sesión activa."""
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None
    
    def set_active_session(self, session_id: str) -> bool:
        """Establece la sesión activa."""
        if session_id in self.sessions:
            self.active_session_id = session_id
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """Elimina una sesión."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.active_session_id == session_id:
                self.active_session_id = None
            logger.info(f"Sesión eliminada: {session_id}")
            return True
        return False
    
    def add_message_handler(self, handler: Callable):
        """Añade un manejador de mensajes."""
        self.message_handlers.append(handler)
    
    def _trigger_message_handlers(self, session: ChatSession, message: ChatMessage):
        """Dispara los manejadores de mensajes."""
        for handler in self.message_handlers:
            try:
                handler(session, message)
            except Exception as e:
                logger.error(f"Error en manejador de mensaje: {e}")
    
    async def send_message_async(self, content: str, session_id: str = None) -> Tuple[str, ChatMessage]:
        """Envía un mensaje de forma asíncrona."""
        session = self.get_session(session_id) if session_id else self.get_active_session()
        
        if not session:
            session = self.create_session()
        
        # Cambiar estado a procesando
        session.state = ChatState.THINKING
        
        # Añadir mensaje del usuario
        user_message = session.add_message(content, MessageType.USER)
        self._trigger_message_handlers(session, user_message)
        
        try:
            # Simular tiempo de procesamiento
            await asyncio.sleep(0.1)
            
            # Cambiar estado a respondiendo
            session.state = ChatState.RESPONDING
            
            # Generar respuesta usando el contexto
            context = session.get_context_for_inference()
            response = await self._generate_response_async(context, session.personality)
            
            # Añadir respuesta del asistente
            assistant_message = session.add_message(response, MessageType.ASSISTANT)
            self._trigger_message_handlers(session, assistant_message)
            
            # Cambiar estado a inactivo
            session.state = ChatState.IDLE
            
            return response, assistant_message
            
        except Exception as e:
            session.state = ChatState.ERROR
            error_message = f"Error al procesar mensaje: {str(e)}"
            logger.error(error_message)
            
            error_msg = session.add_message(error_message, MessageType.ERROR)
            return error_message, error_msg
    
    def send_message(self, content: str, session_id: str = None) -> Tuple[str, ChatMessage]:
        """Envía un mensaje de forma síncrona."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.send_message_async(content, session_id))
        except RuntimeError:
            # Si no hay loop, crear uno nuevo
            return asyncio.run(self.send_message_async(content, session_id))
    
    async def _generate_response_async(self, context: str, personality: ChatPersonality) -> str:
        """Genera una respuesta usando el motor de inferencia."""
        # Ejecutar la inferencia en un hilo separado para no bloquear
        loop = asyncio.get_event_loop()
        
        def generate():
            return self.inference_engine.model_manager.generate(
                context + "\nAsistente:",
                max_length=personality.max_response_length,
                temperature=personality.temperature
            )
        
        response = await loop.run_in_executor(None, generate)
        
        # Limpiar y procesar la respuesta
        response = response.strip()
        if not response:
            response = "Lo siento, no pude generar una respuesta adecuada."
        
        return response
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de las sesiones."""
        total_sessions = len(self.sessions)
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        
        active_sessions = sum(
            1 for session in self.sessions.values() 
            if (datetime.now() - session.last_activity).seconds < 3600  # Activa en la última hora
        )
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "average_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0
        }
    
    def save_session(self, session_id: str, filepath: str):
        """Guarda una sesión en un archivo."""
        session = self.get_session(session_id)
        if session:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"Sesión guardada en: {filepath}")
    
    def load_session(self, filepath: str) -> Optional[ChatSession]:
        """Carga una sesión desde un archivo."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = ChatSession.from_dict(data)
            self.sessions[session.session_id] = session
            logger.info(f"Sesión cargada desde: {filepath}")
            return session
            
        except Exception as e:
            logger.error(f"Error al cargar sesión: {e}")
            return None

class InteractiveChatInterface:
    """Interfaz de chat interactiva para línea de comandos."""
    
    def __init__(self, chat_manager: ChatManager = None):
        self.chat_manager = chat_manager or ChatManager()
        self.running = False
        self.session = None
    
    def start(self, personality: str = "default"):
        """Inicia la interfaz de chat interactiva."""
        self.running = True
        self.session = self.chat_manager.create_session(personality)
        
        print(f"🐾 ¡Bienvenido a PawGPT! 🐾")
        print(f"Personalidad: {self.session.personality.name}")
        print(f"Descripción: {self.session.personality.description}")
        print("Escribe 'quit' para salir, 'help' para ayuda")
        print("-" * 50)
        
        while self.running:
            try:
                user_input = input("\n👤 Tú: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    self.stop()
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                elif user_input.lower() == 'clear':
                    self._clear_history()
                    continue
                
                # Procesar mensaje
                print("🤔 PawGPT está pensando...")
                response, _ = self.chat_manager.send_message(user_input)
                print(f"🐾 PawGPT: {response}")
                
            except KeyboardInterrupt:
                self.stop()
                break
            except EOFError:
                self.stop()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def stop(self):
        """Detiene la interfaz de chat."""
        self.running = False
        print("\n👋 ¡Hasta luego! Gracias por usar PawGPT.")
    
    def _show_help(self):
        """Muestra ayuda."""
        print("\n🆘 Comandos disponibles:")
        print("  quit  - Salir del chat")
        print("  help  - Mostrar esta ayuda")
        print("  stats - Mostrar estadísticas")
        print("  clear - Limpiar historial de conversación")
    
    def _show_stats(self):
        """Muestra estadísticas."""
        stats = self.chat_manager.get_session_stats()
        print(f"\n📊 Estadísticas:")
        print(f"  Sesiones totales: {stats['total_sessions']}")
        print(f"  Sesiones activas: {stats['active_sessions']}")
        print(f"  Mensajes totales: {stats['total_messages']}")
        print(f"  Promedio por sesión: {stats['average_messages_per_session']:.1f}")
    
    def _clear_history(self):
        """Limpia el historial de conversación."""
        if self.session:
            # Mantener solo el mensaje del sistema
            system_messages = [msg for msg in self.session.messages if msg.type == MessageType.SYSTEM]
            self.session.messages = system_messages
            print("✅ Historial limpiado")

# Funciones de utilidad
def create_chat_manager(config_path: str = None) -> ChatManager:
    """Crea un gestor de chat con configuración personalizada."""
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
    
    inference_engine = create_inference_engine()
    return ChatManager(inference_engine, config)

def start_interactive_chat(personality: str = "default"):
    """Inicia una sesión de chat interactiva."""
    chat_manager = create_chat_manager()
    interface = InteractiveChatInterface(chat_manager)
    interface.start(personality)

# Ejemplo de uso como script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PawGPT Chat Interface")
    parser.add_argument("--personality", default="default", 
                       choices=["default", "formal", "creative"],
                       help="Personalidad del chatbot")
    parser.add_argument("--config", help="Ruta al archivo de configuración")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Iniciar chat interactivo
    start_interactive_chat(args.personality)