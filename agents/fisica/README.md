# Physics Assistant Agent

Este agente utiliza el Agent Development Kit (ADK) para crear un asistente de física basado en documentos PDF. El agente indexa documentos PDF, los almacena en una base de datos vectorial (Qdrant) y responde a consultas sobre física utilizando los documentos como fuente de conocimiento.

## Características

- Indexa automáticamente documentos PDF sobre física
- Utiliza embeddings para búsqueda semántica en los documentos
- Mantiene memoria de conversación por sesión
- Proporciona respuestas basadas en el contexto relevante de los documentos
- Implementado como un servidor A2A (Agent-to-Agent)

## Requisitos previos

- Python 3.9 o superior
- [UV](https://docs.astral.sh/uv/)
- Acceso a un LLM (Gemini) con API Key
- Instancia de Qdrant (local o en la nube)

## Variables de entorno

Crea un archivo `.env` con las siguientes variables:

```
GOOGLE_API_KEY=tu_api_key_aqu