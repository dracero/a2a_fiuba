import os
import asyncio
import json
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from collections import deque
import logging
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

load_dotenv()

class PDFQAModel:
    """Clase para procesar PDFs y generar respuestas basadas en su contenido."""
    
    def __init__(self, pdf_files):
        """
        Inicializa el modelo con los archivos PDF
        Args:
            pdf_files: Lista de rutas de archivos PDF
        """
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
        
        # Procesar PDFs
        self.contenido_completo = self._process_pdfs(pdf_files)

    def _process_pdfs(self, pdf_files):
        """Procesa todos los archivos PDF y devuelve su contenido concatenado"""
        contenido = ""
        for archivo in pdf_files:
            try:
                contenido += f"\n--- Contenido de {archivo} ---\n"
                contenido += self._leer_pdf(archivo)
            except Exception as e:
                print(f"Error al leer {archivo}: {e}")
        return contenido

    def _leer_pdf(self, nombre_archivo):
        """Lee y extrae texto de un archivo PDF"""
        reader = PdfReader(nombre_archivo)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text()
        return texto

    def generate_system_message(self):
        """Genera el mensaje del sistema con el contenido de los PDFs"""
        return f"""
        Eres un experto profesor de Física I de la Universidad de Buenos Aires.
        Tu tarea es responder preguntas sobre el temario que tiene en los archivos que lees, 
        proporcionando explicaciones claras, detalladas y ejemplos relevantes.
        Responde solo con el contenido, si no está en el contenido di que no tienes eso en tu base de datos.
        Utiliza el siguiente contenido como referencia para tus respuestas:
        ---
        {self.contenido_completo}
        ---
        """

class QdrantProcessor:
    """Clase para procesar PDFs y almacenarlos en Qdrant."""
    
    def __init__(self):
        self.client = AsyncQdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.chunks_file = "chunks_memory.json"
        self.chunkid_to_text = {}
        self._load_chunks()

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
            await self.client.get_collection("documentos_pdf")
            return True
        except Exception:
            return False

    async def ensure_collection(self, vector_size):
        """Asegura que la colección existe, creándola si es necesario."""
        if await self.collection_exists():
            print("✓ Colección documentos_pdf ya existe")
            return False
        
        print("Creando colección documentos_pdf...")
        try:
            await self.client.create_collection(
                collection_name="documentos_pdf",
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print("✓ Colección documentos_pdf creada exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error creando colección: {e}")
            raise

    def split_text(self, text, chunk_size=2000):
        """Divide el texto en chunks del tamaño especificado"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def generate_embeddings(self, chunks, batch_size=32):
        """Genera embeddings para los chunks de texto"""
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            if len(batch) == 1:
                batch_emb = [batch_emb]
            embeddings.extend(batch_emb)
        return embeddings

    async def process_and_store_pdfs(self, pdf_files):
        """Procesa los PDFs y los almacena en Qdrant"""
        # Procesar primer PDF para obtener tamaño del vector
        first_pdf = pdf_files[0]
        text = self._read_pdf(first_pdf)
        chunks = self.split_text(text)
        if not chunks:
            return "Error: No se pudieron extraer chunks del PDF"
            
        embeddings = self.generate_embeddings([chunks[0]])
        vector_size = len(embeddings[0])
        
        # Crear la colección si no existe
        try:
            await self.ensure_collection(vector_size)
        except Exception as e:
            return f"Error creando colección: {e}"

        # Procesar todos los PDFs
        all_chunks = []
        pdf_metadata = []
        self.chunkid_to_text.clear()
        
        for pdf_file in pdf_files:
            pdf_name = os.path.basename(pdf_file)
            print(f"Procesando {pdf_name}...")
            
            text = self._read_pdf(pdf_file)
            chunks = self.split_text(text)
            embeddings = self.generate_embeddings(chunks)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{pdf_name}_chunk_{i}"
                all_chunks.append(chunk)
                pdf_metadata.append({
                    "pdf_name": pdf_name,
                    "chunk_id": chunk_id,
                    "embedding": embedding
                })
                self.chunkid_to_text[chunk_id] = chunk

        # Crear puntos para Qdrant
        points = [
            PointStruct(
                id=idx,
                vector=meta["embedding"].tolist(),
                payload={
                    "pdf_name": meta["pdf_name"],
                    "chunk_id": meta["chunk_id"]
                }
            )
            for idx, meta in enumerate(pdf_metadata)
        ]

        # Almacenar en Qdrant
        try:
            await self.client.upsert(collection_name="documentos_pdf", points=points)
            self._save_chunks()
            return f"✓ {len(points)} chunks indexados en Qdrant y guardados en memoria."
        except Exception as e:
            return f"Error indexando puntos en Qdrant: {e}"

    def _read_pdf(self, file_path):
        """Lee y extrae texto de un archivo PDF"""
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    async def search(self, query: str, top_k: int = 5) -> str:
        """Realiza una búsqueda en la base de datos Qdrant y devuelve los resultados relevantes."""
        try:
            # Generar embedding de la consulta
            inputs = self.tokenizer([query], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            # Realizar la búsqueda en Qdrant
            results = await self.client.search(
                collection_name="documentos_pdf",
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True
            )

            if not results:
                return "No se encontraron resultados relevantes."

            # Procesar resultados
            contextos = []
            for r in results:
                chunk_id = r.payload.get('chunk_id', 'unknown')
                pdf_name = r.payload.get('pdf_name', 'PDF desconocido')
                score = getattr(r, 'score', 0.0)
                chunk_text = self.chunkid_to_text.get(chunk_id, "[Texto no disponible]")
                contextos.append(f"--- ChunkID {chunk_id} ({pdf_name}) ---\n{chunk_text} (Similitud: {round(score, 4)})")

            return "\n".join(contextos)
        except Exception as e:
            print(f"Error en búsqueda: {e}")
            return "Error al buscar en Qdrant."

    async def calculate_relevance(self, query: str, context: str) -> float:
        """Calcula la relevancia del contexto para la consulta."""
        try:
            if not context.strip():
                return 0.0

            # Generar embedding de la consulta
            inputs = self.tokenizer([query], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            # Generar embedding del contexto
            inputs = self.tokenizer([context], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            context_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            # Calcular similitud coseno
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(context_embedding).unsqueeze(0)
            ).item()

            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"Error calculando relevancia: {e}")
            return 0.0

    async def get_indexed_topics(self):
        """Obtiene los temas principales de los documentos indexados."""
        if not await self.collection_exists():
            return []

        try:
            points = await self.client.scroll(
                collection_name="documentos_pdf",
                with_payload=True,
                limit=1000
            )

            topics = set()
            for point in points[0]:
                pdf_name = point.payload.get("pdf_name", "")
                if pdf_name:
                    topics.add(pdf_name)

            return list(topics)
        except Exception as e:
            print(f"Error obteniendo temas indexados: {e}")
            return []

class ConversationMemory:
    """Clase para manejar el historial de conversación."""
    
    def __init__(self, max_length=10):
        self.memories = {}
        self.max_length = max_length

    def add(self, session_id, user, context, response):
        if session_id not in self.memories:
            self.memories[session_id] = deque(maxlen=self.max_length)
        
        self.memories[session_id].append({
            "user": user,
            "context": context[:1000] if context else "",
            "response": response
        })

    def get_context(self, session_id):
        if session_id not in self.memories:
            return ""
        
        return "\n".join([
            f"Usuario: {h['user']}\nContexto: {h['context']}\nAsistente: {h['response']}" 
            for h in self.memories[session_id]
        ])

    def clear_session(self, session_id):
        if session_id in self.memories:
            del self.memories[session_id]

class PhysicsAgent:
    """Agente principal que combina PDFQAModel y QdrantProcessor con agentes AutoGen."""
    
    SUPPORTED_CONTENT_TYPES = ["text/plain"]

    def __init__(self):
        """Inicializa el agente y sus componentes."""
        self.processor = QdrantProcessor()
        self.pdf_qa = None  # Se inicializará en el primer uso
        self.memory = ConversationMemory(max_length=10)
        self.topics = []  # Lista de tópicos indexados
        
        # Configuración del modelo LLM
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
            top_p=0.95
        )

        # Configuración de AutoGen
        self.config_list = [{
            "model": "gemini-2.0-flash",
            "api_type": "google",
            "api_key": os.getenv("GOOGLE_API_KEY")
        }]
        
        self.llm_config = {
            "config_list": self.config_list,
            "timeout": 120,
            "temperature": 0,
            "seed": 42
        }

        # Inicializar agentes AutoGen
        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            system_message="Eres un asistente que recibe consultas de física y las transmite al clasificador.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        self.classifier_agent = AssistantAgent(
            name="Clasificador",
            system_message="""Eres un agente especializado en clasificar consultas de física según el temario.
Debes proporcionar:
1. El número y título del tema principal
2. Los subtemas relevantes
3. Palabras clave para búsqueda
Formato:
TEMA: [número y título]
SUBTEMAS: [lista]
KEYWORDS: [palabras clave]""",
            llm_config=self.llm_config
        )
        
        self.search_agent = AssistantAgent(
            name="BuscadorQdrant",
            system_message="""Eres un agente de búsqueda especializado en física. 
Convierte clasificaciones en consultas efectivas para documentos.""",
            llm_config=self.llm_config
        )
        
        self.response_agent = AssistantAgent(
            name="RespondeConsulta",
            system_message="""Eres un profesor de física que responde consultas. 
Usa información del clasificador y resultados de búsqueda para crear respuestas claras y estructuradas.""",
            llm_config=self.llm_config
        )

        # Asegurar inicialización asíncrona
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self._ensure_indexed())
        else:
            loop.run_until_complete(self._ensure_indexed())

    async def _ensure_indexed(self):
        """Asegura que los PDFs estén indexados y los tópicos cargados."""
        print("\n🔄 Iniciando ensure_indexed...")
        try:
            print(f"📋 Estado inicial de tópicos: {self.topics}")
            
            # 1. Verificar si la colección existe y tiene contenido
            collection_exists = await self.processor.collection_exists()
            print(f"✓ Estado de colección: {'Existe' if collection_exists else 'No existe'}")
            
            if not collection_exists:
                print("🔄 Colección no existe. Iniciando indexación de PDFs...")
                pdf_dir = os.getenv("PDF_DIR")
                if not pdf_dir:
                    print("❌ Error: PDF_DIR no está configurado en las variables de entorno")
                    return

                # 2. Verificar PDF_DIR y archivos
                if not os.path.exists(pdf_dir):
                    print(f"❌ Error: El directorio {pdf_dir} no existe")
                    return

                pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
                pdf_paths = [os.path.join(pdf_dir, f) for f in pdf_files]
                
                print(f"📚 PDFs encontrados: {len(pdf_files)}")
                if not pdf_paths:
                    print("❌ No se encontraron archivos PDF para indexar")
                    return
                
                print(f"📄 Lista de PDFs a indexar:")
                for path in pdf_paths:
                    if not os.path.exists(path):
                        print(f"  ❌ No existe: {path}")
                        continue
                    print(f"  ✓ {os.path.basename(path)} ({os.path.getsize(path)} bytes)")
                
                # 3. Inicializar PDFQAModel y procesar PDFs
                print("🔄 Inicializando PDFQAModel...")
                try:
                    self.pdf_qa = PDFQAModel(pdf_paths)
                    print("✓ PDFQAModel inicializado correctamente")
                except Exception as e:
                    print(f"❌ Error inicializando PDFQAModel: {str(e)}")
                    return
                
                # 4. Procesar e indexar PDFs
                print("🔄 Iniciando procesamiento e indexación de PDFs...")
                try:
                    result = await self.processor.process_and_store_pdfs(pdf_paths)
                    print(f"✓ Resultado de indexación: {result}")
                except Exception as e:
                    print(f"❌ Error en procesamiento e indexación: {str(e)}")
                    return
            else:
                print("✓ Colección ya existe")
                # 5. Verificar y cargar chunks
                if not self.processor.chunkid_to_text:
                    print("🔄 Cargando chunks desde archivo...")
                    self.processor._load_chunks()
                    print(f"✓ Chunks cargados: {len(self.processor.chunkid_to_text)}")
                    if not self.processor.chunkid_to_text:
                        print("⚠️ Advertencia: No se cargaron chunks del archivo")
            
            # 6. Cargar y verificar tópicos
            print("\n📥 Intentando cargar tópicos desde Qdrant...")
            try:
                self.topics = await self.processor.get_indexed_topics()
                if self.topics:
                    print(f"✓ {len(self.topics)} tópicos cargados: {', '.join(self.topics)}")
                else:
                    print("❌ No se encontraron tópicos indexados")
                    print("📋 La lista de tópicos está vacía")
                    
                    # 7. Verificación detallada de Qdrant
                    try:
                        points = await self.processor.client.scroll(
                            collection_name="documentos_pdf",
                            with_payload=True,
                            limit=1000
                        )
                        print(f"\n🔍 Verificación de estado Qdrant:")
                        print(f"  - Puntos totales: {len(points[0]) if points and points[0] else 0}")
                        if points and points[0]:
                            print("  - Ejemplo de payload:", points[0][0].payload if points[0] else "No hay datos")
                            print("  - Número de chunks en memoria:", len(self.processor.chunkid_to_text))
                            print("  - IDs de chunks disponibles:", list(self.processor.chunkid_to_text.keys())[:5], "...")
                        
                        # 8. Verificar integridad de datos
                        if points and points[0]:
                            chunk_ids = set()
                            pdf_names = set()
                            for point in points[0]:
                                if 'chunk_id' in point.payload:
                                    chunk_ids.add(point.payload['chunk_id'])
                                if 'pdf_name' in point.payload:
                                    pdf_names.add(point.payload['pdf_name'])
                            print(f"  - PDFs únicos en Qdrant: {len(pdf_names)}")
                            print(f"  - Chunks únicos en Qdrant: {len(chunk_ids)}")
                    except Exception as e:
                        print(f"❌ Error verificando estado Qdrant: {str(e)}")
                
            except Exception as e:
                print(f"❌ Error cargando tópicos: {str(e)}")
                self.topics = []
            
            print(f"\n📋 Estado final de tópicos: {self.topics}")
                
        except Exception as e:
            print(f"❌ Error en _ensure_indexed: {str(e)}")
            import traceback
            print("Stack trace:")
            print(traceback.format_exc())
            self.topics = []

    async def handle_query(self, query, session_id):
        """Maneja una consulta, asegurándose de responder solo con información indexada."""
        # Verificar inicialización del sistema
        if not await self.processor.collection_exists():
            return {
                "content": "El sistema aún no está inicializado correctamente. No hay documentos indexados.",
                "context": "",
                "metadata": {
                    "has_context": False,
                    "source": None,
                    "confidence": 0.0
                }
            }

        # Si pregunta sobre el temario disponible
        print(f"\n🔍 Estado actual de tópicos: {self.topics}")
        if any(keyword in query.lower() for keyword in [
            "sobre qué", "de qué", "qué temas", "qué contenidos", "que temas", "que contenidos",
            "cuales son los temas", "cuáles son los temas", "qué me puedes explicar", "que me puedes explicar"
        ]):
            print(f"📚 Consultando tópicos disponibles...")
            if not self.topics:
                await self._ensure_indexed()  # Recargar tópicos si no están disponibles

            if self.topics:
                response = "Puedo responder preguntas sobre los siguientes documentos:\n\n"
                for topic in self.topics:
                    response += f"- {topic}\n"
                response += "\nPor favor, asegúrate de hacer preguntas relacionadas con estos temas."
            else:
                response = "Lo siento, no hay documentos indexados en este momento."

            return {
                "content": response, 
                "context": "",
                "metadata": {
                    "has_context": bool(self.topics),
                    "source": "topic_list",
                    "confidence": 1.0
                }
            }

        # Obtener clasificación de la consulta
        clasificacion_respuesta = self.classifier_agent.generate_reply(messages=[{
            "role": "user",
            "content": f"""Consulta: {query}
Contexto de la conversación: {self.memory.get_context(session_id)}
Tópicos disponibles: {', '.join(self.topics)}

IMPORTANTE: Clasifica la consulta según los tópicos disponibles y proporciona palabras clave relevantes."""
        }])
        clasificacion = clasificacion_respuesta.get("content", "")

        # Generar consulta de búsqueda
        consulta_busqueda_respuesta = self.search_agent.generate_reply(messages=[{
            "role": "user",
            "content": f"""Clasificación: {clasificacion}
Consulta original: {query}
Tópicos disponibles: {', '.join(self.topics)}

IMPORTANTE: 
1. Formula una consulta de búsqueda que use palabras clave tanto de la pregunta como de los tópicos relevantes
2. Verifica que los tópicos mencionados en la clasificación existan en la lista de tópicos disponibles
3. Si la consulta no coincide con ningún tópico disponible, indica que se debe buscar en todos los documentos"""
        }])
        consulta_busqueda = consulta_busqueda_respuesta.get("content", "")

        # Buscar contexto relevante en Qdrant
        context = await self.processor.search(consulta_busqueda)

        if not context.strip():
            if not self.topics:
                await self._ensure_indexed()  # Recargar tópicos si no están disponibles

            response = "Lo siento, no encontré información sobre ese tema en mis documentos. "
            if self.topics:
                response += "Los documentos disponibles son:\n\n"
                for topic in self.topics:
                    response += f"- {topic}\n"
            else:
                response += "No hay documentos indexados en este momento."
            return {
                "content": response, 
                "context": context,
                "metadata": {
                    "has_context": False,
                    "source": None,
                    "confidence": 0.0
                }
            }

        # Generar respuesta usando el agente de respuesta
        response = await asyncio.to_thread(
            lambda: self.response_agent.generate_reply(messages=[{
                "role": "user",
                "content": f"""
Consulta original: {query}
Clasificación: {clasificacion}
Contexto: {context}
Historial: {self.memory.get_context(session_id)}

IMPORTANTE:
1. Responde ÚNICAMENTE utilizando la información que aparece explícitamente en el contexto proporcionado
2. Si la pregunta no se puede responder completamente con el contexto disponible, di específicamente: "Lo siento, no tengo información suficiente en los documentos indexados para responder a esa pregunta"
3. NO USES conocimiento general ni información que no esté explícitamente en el contexto, incluso si sabes que es correcta
4. Si el contexto solo permite responder parcialmente la pregunta, indica claramente qué partes de la pregunta puedes responder con la información disponible
5. Si se te pide elaborar o dar más detalles, usa SOLAMENTE la información del contexto
6. NO HAGAS INFERENCIAS ni conexiones que no estén explícitamente establecidas en el contexto

Por favor, proporciona una respuesta clara y estructurada basada SOLAMENTE en la información disponible en el contexto.
"""
            }])
        )

        self.memory.add(session_id, query, context, response.get("content", ""))

        return {"content": response.get("content", ""), "context": context}

    def invoke(self, query, session_id):
        """Método síncrono para invocar el agente."""
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.handle_query(query, session_id))
        return result["content"]

    async def stream(self, query, session_id):
        """Método asíncrono para streaming de respuestas."""
        # Verificar inicialización
        if not await self.processor.collection_exists():
            yield {
                "is_task_complete": True,
                "content": "El sistema aún no está inicializado correctamente. No hay documentos indexados.",
                "metadata": {
                    "has_context": False,
                    "source": None,
                    "confidence": 0.0
                }
            }
            return

        # Informar sobre la búsqueda de documentos
        yield {
            "is_task_complete": False,
            "updates": "Buscando información relevante en documentos de física...",
            "metadata": {
                "stage": "searching",
                "progress": 0.25
            }
        }

        result = await self.handle_query(query, session_id)

        # Si no hay contexto, informar sin intentar responder
        if not result.get("context", "").strip():
            yield {
                "is_task_complete": True,
                "content": result["content"],
                "metadata": {
                    "has_context": False,
                    "source": None,
                    "confidence": 0.0,
                    "stage": "completed"
                }
            }
            return

        # Informar que estamos procesando el contexto encontrado
        yield {
            "is_task_complete": False,
            "updates": "Analizando información encontrada...",
            "metadata": {
                "stage": "processing",
                "progress": 0.75,
                "has_context": True
            }
        }

        # Enviar respuesta final con toda la metadata
        yield {
            "is_task_complete": True,
            "content": result["content"],
            "metadata": {
                "has_context": True,
                "source": result.get("metadata", {}).get("source"),
                "confidence": result.get("metadata", {}).get("confidence", 1.0),
                "stage": "completed"
            }
        }

