import logging
import re
import tiktoken
from typing import List, NamedTuple, Pattern
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("document_processor")


class CleaningRule(NamedTuple):
    """One cleaning action: what it looks for (compiled regex) and what it replaces it with."""
    name: str
    pattern: Pattern[str]
    replacement: str


# Declarative cleaning pipeline: rules are applied IN THIS ORDER.
# Adding a new cleaning step only takes one more line in the list.
CLEANING_RULES: List[CleaningRule] = [
    CleaningRule("(a) repeated page headers", re.compile(r"ACME S\.A\. — Manual interno — Pág\. \d+"), ""),
    CleaningRule("(d) junk characters", re.compile(r"[■□▪�\x0c]"), ""),
    CleaningRule("(c) multiple spaces", re.compile(r"[ \t]+"), " "),
    CleaningRule("spaces at line edges", re.compile(r" *\n *"), "\n"),
    CleaningRule("(b) extra line breaks", re.compile(r"\n{3,}"), "\n\n"),
]


class DocumentProcessor:
    def __init__(self, model_encoding: str = "cl100k_base", chunk_size: int = 500, chunk_overlap: int = 50):
        # TODO: Inicializar el encoding de tiktoken
        self.tokenizer = tiktoken.get_encoding(model_encoding)

        # TODO: Configurar RecursiveCharacterTextSplitter
        # Debe usar una función personalizada para medir longitud por TOKENS, no caracteres.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,                     # max TOKENS per chunk
            chunk_overlap=chunk_overlap,               # tokens repeated between consecutive chunks
            length_function=self.calculate_tokens,     # measures tokens, not characters
            separators=["\n\n", "\n", ". ", " ", ""],  # paragraph > line > sentence > word > character
            keep_separator="end",                      # the period stays at the end of its sentence
        )

    def clean_text(self, text: str) -> str:
        """Limpia el texto eliminando espacios duplicados y saltos de línea innecesarios."""
        if text is None:
            logger.warning("clean_text received None: returning an empty string")
            return ""
        if not isinstance(text, str):
            raise TypeError(f"clean_text expects a str, got {type(text).__name__}")

        for rule in CLEANING_RULES:
            try:
                text = rule.pattern.sub(rule.replacement, text)
            except Exception as e:  # a broken rule must not stop the rest of the cleaning
                logger.error("Rule '%s' failed: %s. Continuing with the others.", rule.name, e)
        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        """Calcula la cantidad de tokens usando el tokenizer de tiktoken."""
        return len(self.tokenizer.encode(text))

    def process_document(self, raw_text: str) -> List[str]:
        """Pipeline: Limpieza -> Fragmentación."""
        # TODO: 1. Limpiar el texto
        try:
            clean = self.clean_text(raw_text)
        except TypeError as e:
            logger.error("Invalid document: %s", e)
            return []
        if not clean:
            logger.warning("The document is empty after cleaning: no chunks to generate")
            return []

        # TODO: 2. Aplicar el splitter
        chunks = self.splitter.split_text(clean)

        # TODO: 3. Retornar los chunks procesados
        for i, chunk in enumerate(chunks, 1):
            print(f"[chunk {i}] {self.calculate_tokens(chunk):>3} tokens | {chunk!r}")
        return chunks


if __name__ == "__main__":
    # Test input: typical noise of a PDF converted to text
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

    # 1) Small chunks to see where the splitter cuts
    print("=== chunk_size=40, chunk_overlap=10 ===")
    processor = DocumentProcessor(chunk_size=40, chunk_overlap=10)
    print("Clean text:\n" + processor.clean_text(text) + "\n")
    chunks = processor.process_document(text)
    print(f"Chunks generated: {len(chunks)}\n")

    # 2) Required configuration: 500 / 50
    print("=== chunk_size=500, chunk_overlap=50 ===")
    chunks = DocumentProcessor().process_document(text)
    print(f"Chunks generated: {len(chunks)}\n")

    # 3) Extra: with 25 tokens sentences no longer fit whole, so the splitter cuts by words.
    #    That is where the overlap shows: the end of a chunk is repeated at the start of the next one.
    print("=== Extra: chunk_size=25, chunk_overlap=10 (to see the overlap) ===")
    chunks = DocumentProcessor(chunk_size=25, chunk_overlap=10).process_document(text)
    print(f"Chunks generated: {len(chunks)}")

    # 4) Unexpected inputs: they do not break the pipeline
    print("\n=== Unexpected inputs ===")
    for item in [None, 123, "", "ACME S.A. — Manual interno — Pág. 9 ■■"]:
        print(f"{item!r:45} -> {DocumentProcessor(chunk_size=40, chunk_overlap=10).process_document(item)}")
