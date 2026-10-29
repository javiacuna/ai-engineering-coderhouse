import logging
import re
import tiktoken
from typing import List, NamedTuple, Pattern
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("document_processor")


class ReglaLimpieza(NamedTuple):
    """Una acción de limpieza: qué busca (regex compilada) y por qué la reemplaza."""
    nombre: str
    patron: Pattern[str]
    reemplazo: str


# Pipeline de limpieza declarativo: las reglas se aplican EN ESTE ORDEN.
# Para agregar una limpieza nueva alcanza con sumar una línea a la lista.
REGLAS_LIMPIEZA: List[ReglaLimpieza] = [
    ReglaLimpieza("(a) encabezados repetidos", re.compile(r"ACME S\.A\. — Manual interno — Pág\. \d+"), ""),
    ReglaLimpieza("(d) caracteres basura", re.compile(r"[■□▪�\x0c]"), ""),
    ReglaLimpieza("(c) espacios múltiples", re.compile(r"[ \t]+"), " "),
    ReglaLimpieza("espacios en bordes de línea", re.compile(r" *\n *"), "\n"),
    ReglaLimpieza("(b) saltos de línea de más", re.compile(r"\n{3,}"), "\n\n"),
]


class DocumentProcessor:
    def __init__(self, model_encoding: str = "cl100k_base", chunk_size: int = 500, chunk_overlap: int = 50):
        # TODO: Inicializar el encoding de tiktoken
        self.tokenizer = tiktoken.get_encoding(model_encoding)

        # TODO: Configurar RecursiveCharacterTextSplitter
        # Debe usar una función personalizada para medir longitud por TOKENS, no caracteres.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,                     # máximo de TOKENS por chunk
            chunk_overlap=chunk_overlap,               # tokens que se repiten entre chunks seguidos
            length_function=self.calculate_tokens,     # mide en tokens, no en caracteres
            separators=["\n\n", "\n", ". ", " ", ""],  # párrafo > línea > oración > palabra > carácter
            keep_separator="end",                      # el punto queda al final de su oración
        )

    def clean_text(self, text: str) -> str:
        """Limpia el texto eliminando espacios duplicados y saltos de línea innecesarios."""
        if text is None:
            logger.warning("clean_text recibió None: se devuelve un texto vacío")
            return ""
        if not isinstance(text, str):
            raise TypeError(f"clean_text espera un str y recibió {type(text).__name__}")

        for regla in REGLAS_LIMPIEZA:
            try:
                text = regla.patron.sub(regla.reemplazo, text)
            except Exception as e:  # una regla rota no debe frenar el resto de la limpieza
                logger.error("Falló la regla '%s': %s. Se continúa con las demás.", regla.nombre, e)
        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        """Calcula la cantidad de tokens usando el tokenizer de tiktoken."""
        return len(self.tokenizer.encode(text))

    def process_document(self, raw_text: str) -> List[str]:
        """Pipeline: Limpieza -> Fragmentación."""
        # TODO: 1. Limpiar el texto
        try:
            texto_limpio = self.clean_text(raw_text)
        except TypeError as e:
            logger.error("Documento inválido: %s", e)
            return []
        if not texto_limpio:
            logger.warning("El documento quedó vacío después de la limpieza: no hay chunks para generar")
            return []

        # TODO: 2. Aplicar el splitter
        chunks = self.splitter.split_text(texto_limpio)

        # TODO: 3. Retornar los chunks procesados
        for i, chunk in enumerate(chunks, 1):
            print(f"[chunk {i}] {self.calculate_tokens(chunk):>3} tokens | {chunk!r}")
        return chunks


if __name__ == "__main__":
    # Texto de prueba: ruido típico de un PDF convertido
    text = """ACME S.A. — Manual interno — Pág. 1


Política de devoluciones■■

Los clientes pueden devolver un producto dentro de los 30 días    corridos desde la compra, siempre que conserven el ticket.



ACME S.A. — Manual interno — Pág. 2

El reintegro se hace por el mismo medio de pago. Si la compra fue en efectivo, se entrega una nota de crédito.

Envíos

Los envíos dentro de la ciudad demoran 48 horas hábiles. Para el interior del país, el plazo es de 5 a 7 días hábiles y el costo depende del peso del paquete.



ACME S.A. — Manual interno — Pág. 3

Garantía

Todos los productos electrónicos tienen 12 meses de garantía. La garantía no cubre daños por mal uso ni golpes."""

    # 1) Chunks chicos para ver el corte
    print("=== chunk_size=40, chunk_overlap=10 ===")
    processor = DocumentProcessor(chunk_size=40, chunk_overlap=10)
    print("Texto limpio:\n" + processor.clean_text(text) + "\n")
    chunks = processor.process_document(text)
    print(f"Chunks generados: {len(chunks)}\n")

    # 2) Configuración pedida: 500 / 50
    print("=== chunk_size=500, chunk_overlap=50 ===")
    chunks = DocumentProcessor().process_document(text)
    print(f"Chunks generados: {len(chunks)}\n")

    # 3) Extra: con 25 tokens las oraciones ya no entran enteras y el splitter corta por palabras.
    #    Ahí se ve el overlap: el final de un chunk se repite al principio del siguiente.
    print("=== Extra: chunk_size=25, chunk_overlap=10 (para ver el overlap) ===")
    chunks = DocumentProcessor(chunk_size=25, chunk_overlap=10).process_document(text)
    print(f"Chunks generados: {len(chunks)}")

    # 4) Entradas inesperadas: no rompen el pipeline
    print("\n=== Entradas inesperadas ===")
    for entrada in [None, 123, "", "ACME S.A. — Manual interno — Pág. 9 ■■"]:
        print(f"{entrada!r:45} -> {DocumentProcessor(chunk_size=40, chunk_overlap=10).process_document(entrada)}")
