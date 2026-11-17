"""Simulated external sources: product reviews (store, forums, blogs) and prices in online stores.

In a real system these would be a search API (Tavily) or the vector DB of the previous milestones.
"""

import asyncio
import unicodedata
from typing import TypedDict

LATENCY_SECONDS = 0.05


class RawReview(TypedDict):
    rating: int        # 1 to 5 stars
    text: str
    source: str
    date: str


class RawPrice(TypedDict):
    store: str
    price: int         # ARS


PRODUCTS: dict[str, dict] = {
    "SoundMax Pro": {
        "aliases": ["soundmax", "sound max", "soundmax pro"],
        "category": "auriculares inalámbricos",
        "reviews": [
            {"rating": 5, "text": "Excelente sonido y la batería dura muchísimo. Muy cómodos para usar todo el día.", "source": "tienda", "date": "2026-09-02"},
            {"rating": 4, "text": "Muy buena cancelación de ruido, aunque el estuche es grande.", "source": "foro", "date": "2026-09-15"},
            {"rating": 5, "text": "Increíble calidad de audio, los recomiendo. Conexión rápida y estable.", "source": "blog", "date": "2026-09-21"},
            {"rating": 2, "text": "Se desconectan del celular cada tanto, bastante molesto.", "source": "tienda", "date": "2026-10-01"},
            {"rating": 5, "text": "Cómodos, livianos y con excelente batería.", "source": "tienda", "date": "2026-10-09"},
            {"rating": 4, "text": "Buen sonido, aunque un poco caros para lo que ofrecen.", "source": "foro", "date": "2026-10-18"},
        ],
        "prices": [{"store": "TecnoShop", "price": 189000}, {"store": "AudioWorld", "price": 175000},
                   {"store": "MegaStore", "price": 182000}],
    },
    "AudioZen X": {
        "aliases": ["audiozen", "audio zen", "audiozen x"],
        "category": "auriculares inalámbricos",
        "reviews": [
            {"rating": 3, "text": "El sonido es correcto pero la batería dura poco.", "source": "tienda", "date": "2026-08-28"},
            {"rating": 2, "text": "Incómodos después de una hora, me duelen los oídos.", "source": "foro", "date": "2026-09-05"},
            {"rating": 4, "text": "Buena relación precio calidad, son baratos y suenan bien.", "source": "blog", "date": "2026-09-19"},
            {"rating": 1, "text": "Se rompió la bisagra al mes. Mala calidad de materiales.", "source": "tienda", "date": "2026-10-03"},
            {"rating": 4, "text": "Livianos y económicos, ideales para el gimnasio.", "source": "tienda", "date": "2026-10-12"},
            {"rating": 3, "text": "No están mal, pero la cancelación de ruido es débil.", "source": "foro", "date": "2026-10-20"},
        ],
        "prices": [{"store": "TecnoShop", "price": 89000}, {"store": "AudioWorld", "price": 92000},
                   {"store": "MegaStore", "price": 85000}],
    },
    "BassBoom 3": {
        "aliases": ["bassboom", "bass boom", "bassboom 3"],
        "category": "parlante bluetooth",
        "reviews": [
            {"rating": 5, "text": "Potente y resistente al agua, excelente para la playa.", "source": "tienda", "date": "2026-09-10"},
            {"rating": 4, "text": "Buen volumen, los graves son increíbles.", "source": "blog", "date": "2026-09-30"},
            {"rating": 3, "text": "Suena bien pero es pesado para llevar.", "source": "foro", "date": "2026-10-14"},
        ],
        "prices": [{"store": "TecnoShop", "price": 120000}, {"store": "MegaStore", "price": 115000}],
    },
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().strip())
    return " ".join("".join(ch for ch in text if not unicodedata.combining(ch)).split())


def resolve_product(name: str) -> str | None:
    """Official name of a product, accepting aliases and accents/case differences."""
    wanted = normalize(name)
    for official, info in PRODUCTS.items():
        if wanted == normalize(official) or wanted in info["aliases"]:
            return official
    return None


def suggestions(name: str) -> list[str]:
    """Catalog products that share a word with the requested name."""
    words = set(normalize(name).split())
    return [p for p in PRODUCTS if words & set(normalize(p).split())]


def products_mentioned(text: str) -> list[str]:
    """Catalog products mentioned in a free text (official name or alias)."""
    norm = normalize(text)
    return [p for p, info in PRODUCTS.items() if normalize(p) in norm or any(a in norm for a in info["aliases"])]


async def fetch_reviews(product: str) -> list[RawReview]:
    await asyncio.sleep(LATENCY_SECONDS)
    return list(PRODUCTS[product]["reviews"])


async def fetch_prices(product: str) -> list[RawPrice]:
    await asyncio.sleep(LATENCY_SECONDS)
    return list(PRODUCTS[product]["prices"])
