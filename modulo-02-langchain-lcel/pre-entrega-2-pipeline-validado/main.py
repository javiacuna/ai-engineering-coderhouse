"""Async test script: runs the pipeline on a few sample texts, including an ambiguous one (stress test).

Usage:
  python main.py
"""

import asyncio
import logging

from dotenv import load_dotenv

from chain import ExtractionError, PipelineConfig, build_chain, process_text

SAMPLES = {
    "Architecture with a bottleneck": (
        "Nuestra API está hecha con FastAPI y usa Redis como caché delante de PostgreSQL. "
        "Desde el lunes, en horario pico, las respuestas pasan de 80 ms a 4 segundos y el pool "
        "de conexiones de Postgres se agota; varios clientes reportan timeouts al pagar."
    ),
    "Error log": (
        "2026-10-30T03:12:44Z ERROR consumer-group=billing topic=invoices lag=184233 "
        "KafkaConsumer poll timeout. Pod billing-worker-7d9f restarted (OOMKilled) in namespace prod, "
        "Kubernetes node pool gke-n2-standard-4. Java 21 heap usage 98%."
    ),
    "Ambiguous text (stress test)": (
        "Anda raro desde ayer. Creo que es la base o algo del server, a veces carga y a veces no. "
        "Nadie tocó nada, o eso dicen."
    ),
}


async def main() -> None:
    try:
        config = PipelineConfig.from_env()
    except ValueError as e:
        print(f"Invalid configuration: {e}")
        return
    print(f"Provider: {config.provider.value} · model: {config.model} · fallback: {config.fallback_model} · "
          f"attempts: {config.max_attempts}\n")
    chain = build_chain(config)

    for title, text in SAMPLES.items():
        print(f"=== {title} ===")
        try:
            result = await process_text(text, chain)
            print(result.to_contract_json(), "\n")
        except ExtractionError as e:
            print(f"Controlled error: {e}\n")


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "google_genai", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main())
