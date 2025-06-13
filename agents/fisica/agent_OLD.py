import os
import asyncio
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
import json
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
import numpy as np
from collections import deque
import logging

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_KEY")
COLLECTION_NAME = "documentos_pdf"
PDF_DIR = os.getenv("PDF_DIR", "/home/cetec/Downloads/apuntes_fisica")

# ----------------------------
# Utilidades para embeddings y chunks
# ----------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def split_into_chunks(text, chunk_size=2000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def generate_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        if len(batch) == 1:
            batch_emb = [batch_emb]
        embeddings.extend(batch_emb)
    return embeddings

# ----------------------------
# Agente Indexador
# ----------------------------
class QdrantIndexerAgent:
    def __init__(self):
        self.client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
        self.chunkid_to_text = {}  # Memoria local de chunks
        self.chunks_file = "chunks_memory.json"
        self._load_chunks()  # Cargar chunks al inicio

    def _load_chunks(self):
        """Carga los chunks desde el archivo JSON si existe."""
        try:
            if os.path.exists(self.chunks_file):
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    self.chunkid_to_text = json.load(f)
                print(f"✓ {len(self.chunkid_to_text)} chunks cargados de {self.chunks_file}")
        except Exception as e:
            print(f"Error cargando chunks: {e}")
            self.chunkid_to_text = {}

    def _save_chunks(self):
        """Guarda los chunks en un archivo JSON."""
        try:
            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunkid_to_text, f, ensure_ascii=False, indent=2)
            print(f"✓ {len(self.chunkid_to_text)} chunks guardados en {self.chunks_file}")
        except Exception as e:
            print(f"Error guardando chunks: {e}")

    async def collection_exists(self):
        """Verifica si la colección existe en Qdrant."""
        try:
            await self.client.get_collection(COLLECTION_NAME)
            return True
        except Exception:
            return False

    async def ensure_collection(self, vector_size):
        """Asegura que la colección existe, creándola si es necesario."""
        from qdrant_client.models import VectorParams, Distance
        
        if await self.collection_exists():
            print(f"✓ Colección {COLLECTION_NAME} ya existe")
            return False  # Ya existía
        
        print(f"Creando colección {COLLECTION_NAME}...")
        try:
            await self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✓ Colección {COLLECTION_NAME} creada exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error creando colección: {e}")
            raise

    async def index_pdfs(self):
        """Indexa los PDFs en Qdrant solo si la colección no existe."""
        # Verificar si la colección existe
        if await self.collection_exists():
            print(f"✓ Colección {COLLECTION_NAME} ya existe. No es necesario indexar.")
            return "Colección ya existe. No se requiere indexación."

        # Si no existe, proceder con la indexación
        pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            return "No se encontraron PDFs para indexar."

        # Procesar primer PDF para obtener tamaño del vector
        first_pdf = pdf_files[0]
        text = read_pdf(first_pdf)
        chunks = split_into_chunks(text)
        if not chunks:
            return "Error: No se pudieron extraer chunks del PDF"
            
        embeddings = generate_embeddings([chunks[0]])  # Solo necesitamos uno para el tamaño
        vector_size = len(embeddings[0])
        
        # Crear la colección
        try:
            created = await self.ensure_collection(vector_size)
        except Exception as e:
            return f"Error creando colección: {e}"

        # Continuar con el procesamiento de todos los PDFs
        all_chunks = []
        pdf_metadata = []
        self.chunkid_to_text.clear()
        
        for pdf_file in pdf_files:
            pdf_name = os.path.basename(pdf_file)
            print(f"Procesando {pdf_name}...")
            
            text = read_pdf(pdf_file)
            chunks = split_into_chunks(text)
            embeddings = generate_embeddings(chunks)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{pdf_name}_chunk_{i}"
                all_chunks.append(chunk)
                pdf_metadata.append({
                    "pdf_name": pdf_name,
                    "chunk_id": chunk_id,
                    "embedding": embedding
                })
                self.chunkid_to_text[chunk_id] = chunk

        if not pdf_metadata:
            return "No se encontró contenido para indexar en los PDFs."

        points = [
            PointStruct(
                id=idx,  # Usa un entero como ID
                vector=meta["embedding"].tolist(),
                payload={
                    "pdf_name": meta["pdf_name"],
                    "chunk_id": meta["chunk_id"]
                }
            )
            for idx, meta in enumerate(pdf_metadata)
        ]

        try:
            await self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            self._save_chunks()
            return f"Colección creada y {len(points)} chunks indexados en Qdrant y guardados en memoria."
        except Exception as e:
            return f"Error indexando puntos en Qdrant: {e}"

# ----------------------------
# Agente Consultor
# ----------------------------
class QdrantConsultAgent:
    def __init__(self, indexer_agent):
        self.client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)
        self.indexer_agent = indexer_agent
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    async def search(self, query: str, top_k: int = 5) -> str:
        # Verificar si la colección existe
        if not await self.indexer_agent.collection_exists():
            print("⚠️ Colección no existe. Iniciando indexación...")
            index_msg = await self.indexer_agent.index_pdfs()
            print(index_msg)
            if "Error" in index_msg:
                return f"Error: {index_msg}"

        # Verificar si tenemos chunks cargados
        if not self.indexer_agent.chunkid_to_text:
            print("⚠️ No hay chunks cargados en memoria. Intentando cargar...")
            self.indexer_agent._load_chunks()
            if not self.indexer_agent.chunkid_to_text:
                print("❌ No se pudieron cargar chunks. Iniciando indexación...")
                index_msg = await self.indexer_agent.index_pdfs()
                print(index_msg)
                if "Error" in index_msg:
                    return f"Error: {index_msg}"

        inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        try:
            results = await self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True
            )
        except Exception as e:
            print(f"Error en búsqueda: {e}")
            return "Error al buscar en Qdrant."

        if not results:
            return "No se encontraron resultados relevantes en Qdrant."

        # Manejo de diferentes formatos de respuesta con debug
        contextos = []
        for r in results:
            try:
                if hasattr(r, 'payload'):
                    chunk_id = r.payload.get('chunk_id', 'unknown')
                    pdf_name = r.payload.get('pdf_name', 'PDF desconocido')
                    score = getattr(r, 'score', 0.0)
                elif isinstance(r, (tuple, list)):
                    chunk_id = r[0]
                    pdf_name = r[1].get('pdf_name', 'PDF desconocido') if isinstance(r[1], dict) else 'PDF desconocido'
                    score = r[2] if len(r) > 2 else 0.0
                else:
                    continue

                # Debug de chunk_id y texto
                self.logger.debug(f"Buscando chunk_id: {chunk_id}")
                chunk_text = self.indexer_agent.chunkid_to_text.get(chunk_id)
                if chunk_text is None:
                    print(f"⚠️ No se encontró texto para chunk_id: {chunk_id}")
                    # Intentar obtener el chunk_id sin extensión del PDF
                    base_chunk_id = chunk_id.replace('.pdf', '') if chunk_id.endswith('.pdf') else chunk_id
                    chunk_text = self.indexer_agent.chunkid_to_text.get(base_chunk_id, "[Texto no disponible]")
                
                contextos.append(
                    f"--- ChunkID {chunk_id} ({pdf_name}) ---\n{chunk_text} (Similitud: {round(score, 4)})"
                )
            except Exception as e:
                print(f"Error procesando resultado: {e}")
                continue

        if not contextos:
            return "No se pudieron procesar los resultados de la búsqueda."
        
        return "\n".join(contextos)

# ----------------------------
# Memoria de Conversación
# ----------------------------
class ConversationMemory:
    def __init__(self, max_length=10):
        self.memories = {}  # Diccionario para almacenar conversaciones por sesión
        self.max_length = max_length

    def add(self, session_id, user, context, response):
        if session_id not in self.memories:
            self.memories[session_id] = deque(maxlen=self.max_length)
        
        self.memories[session_id].append({
            "user": user,
            "context": context[:1000] if context else "",  # Limitar contexto para evitar tokens excesivos
            "response": response
        })

    def get_context(self, session_id):
        if session_id not in self.memories:
            return ""
        
        return "\n".join([
            f"Usuario: {h['user']}\nContexto: {h['context']}\nAsistente: {h['response']}" 
            for h in self.memories[session_id]
        ])

# ----------------------------
# Agente Principal (Adaptado para ADK)
# ----------------------------
class AgentFisic:
    SUPPORTED_CONTENT_TYPES = ["text/plain"]
    
    def __init__(self):
        self.indexer_agent = QdrantIndexerAgent()
        self.consult_agent = QdrantConsultAgent(self.indexer_agent)
        self.memory = ConversationMemory(max_length=10)
        
        # Import LLM inside __init__ to avoid global import issues
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.schema import SystemMessage, HumanMessage
        
        # Eliminada la opción convert_system_message_to_human
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
            top_p=0.95
        )
        self.SystemMessage = SystemMessage
        self.HumanMessage = HumanMessage
        
        # Asegurar que el bucle de eventos esté activo antes de crear la tarea
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self._ensure_indexed())
        else:
            loop.run_until_complete(self._ensure_indexed())
    
    async def _ensure_indexed(self):
        """Asegura que los PDFs estén indexados al iniciar el agente."""
        # Solo indexar si la colección no existe
        if not await self.indexer_agent.collection_exists():
            print("🔄 Colección no existe. Iniciando indexación de PDFs...")
            await self.indexer_agent.index_pdfs()
        else:
            print("✓ Colección ya existe. Cargando chunks desde archivo...")
            # Cargar chunks desde archivo si la colección existe pero no están en memoria
            if not self.indexer_agent.chunkid_to_text:
                self.indexer_agent._load_chunks()
    
    async def _get_answer(self, query, session_id):
        # 1. Buscar contexto relevante en Qdrant
        context = await self.consult_agent.search(query)
        
        # 2. Construir prompt con memoria
        system_prompt = """Eres un profesor de física experto. Responde de forma clara, conversacional y accesible.
        Utiliza el contexto de documentos proporcionado para ofrecer respuestas precisas y educativas.
        Si la información no está en el contexto, indica honestamente que no tienes esa información específica.
        """
        
        # Obtener historial de conversación para esta sesión
        conversation_history = self.memory.get_context(session_id)
        
        user_prompt = f"""
CONSULTA: {query}

CONVERSACIÓN PREVIA:
{conversation_history}

DOCUMENTOS RELEVANTES:
{context}
        """
        
        # 3. Llamar al LLM
        respuesta = await asyncio.to_thread(
            lambda: self.llm.invoke([
                self.SystemMessage(content=system_prompt),
                self.HumanMessage(content=user_prompt)
            ])
        )
        
        # 4. Guardar en memoria
        self.memory.add(session_id, query, context, respuesta.content)
        
        return {"content": respuesta.content, "context": context}
    
    def invoke(self, query, session_id):
        """Método síncrono para invocar el agente (requerido por ADK)"""
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._get_answer(query, session_id))
        return result["content"]
    
    async def stream(self, query, session_id):
        """Método asíncrono para streaming de respuestas"""
        # Primero enviamos una actualización inicial
        yield {
            "is_task_complete": False,
            "updates": "Buscando información relevante en los documentos de física..."
        }
        
        # Obtenemos la respuesta completa
        result = await self._get_answer(query, session_id)
        
        # Enviamos la respuesta final
        yield {
            "is_task_complete": True,
            "content": result["content"]
        }

