from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from task_manager import AgentTaskManager
from agent import PhysicsAgent
import click
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10003)
@click.option("--pdf-dir", default=None, help="Directorio donde se encuentran los archivos PDF de física")
def main(host, port, pdf_dir):
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")
        
        # Si se proporciona un directorio de PDFs, configurarlo
        if pdf_dir:
            os.environ["PDF_DIR"] = pdf_dir
        
        # Verificamos la configuración de Qdrant
        if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_KEY"):
            logger.warning("QDRANT_URL o QDRANT_KEY no configuradas. Asegúrate de configurar estas variables para el funcionamiento correcto.")
        
        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id="physics_assistant",
            name="Asistente de Física",
            description="Proporciona respuestas e información sobre temas de física basados en documentos PDF.",
            tags=["física", "educación", "ciencia"],
            examples=[
                "¿Puedes explicarme el concepto de momento angular?",
                "¿Cómo se calcula la fuerza de un cuerpo en movimiento?",
                "Explícame la tercera ley de Newton",
            ],
        )
        agent_card = AgentCard(
            name="Asistente de Física",
            description="Este agente proporciona respuestas e información sobre temas de física basados en documentos PDF.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=PhysicsAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=PhysicsAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=PhysicsAgent()),
            host=host,
            port=port,
        )
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)
    
if __name__ == "__main__":
    main()