"""Simulated orders database (in memory). The async functions stand in for real queries."""

import asyncio
import unicodedata
from datetime import date
from typing import TypedDict

QUERY_LATENCY_SECONDS = 0.05   # simulated network round trip


class Customer(TypedDict):
    id: int
    name: str
    city: str


class OrderItem(TypedDict):
    product: str
    quantity: int
    unit_price: int


class Order(TypedDict):
    id: int
    customer_id: int
    date: date
    status: str          # "entregado", "en_camino" or "cancelado"
    items: list[OrderItem]


CUSTOMERS: list[Customer] = [
    {"id": 101, "name": "Martín García", "city": "Rosario"},
    {"id": 102, "name": "Lucía Fernández", "city": "Córdoba"},
    {"id": 103, "name": "Sofía García", "city": "Mendoza"},
    {"id": 104, "name": "Diego Pereyra", "city": "Córdoba"},
    {"id": 105, "name": "Ana Ruiz", "city": "Buenos Aires"},
]

ORDERS: list[Order] = [
    {"id": 5001, "customer_id": 102, "date": date(2026, 8, 14), "status": "entregado",
     "items": [{"product": "Teclado mecánico", "quantity": 1, "unit_price": 6000}]},
    {"id": 5007, "customer_id": 102, "date": date(2026, 9, 30), "status": "entregado",
     "items": [{"product": "Mouse inalámbrico", "quantity": 2, "unit_price": 1750}]},
    {"id": 5012, "customer_id": 102, "date": date(2026, 10, 22), "status": "en_camino",
     "items": [{"product": "Soporte para notebook", "quantity": 1, "unit_price": 2500},
               {"product": "Hub USB-C", "quantity": 1, "unit_price": 2500}]},
    {"id": 5002, "customer_id": 101, "date": date(2026, 7, 2), "status": "entregado",
     "items": [{"product": "Monitor 27\"", "quantity": 1, "unit_price": 98000}]},
    {"id": 5009, "customer_id": 101, "date": date(2026, 10, 5), "status": "cancelado",
     "items": [{"product": "Silla ergonómica", "quantity": 1, "unit_price": 120000}]},
    {"id": 5004, "customer_id": 103, "date": date(2026, 9, 11), "status": "entregado",
     "items": [{"product": "Auriculares", "quantity": 1, "unit_price": 18500},
               {"product": "Webcam HD", "quantity": 1, "unit_price": 21000}]},
    {"id": 5010, "customer_id": 104, "date": date(2026, 10, 18), "status": "entregado",
     "items": [{"product": "Notebook 14\"", "quantity": 1, "unit_price": 850000}]},
]


def normalize(text: str) -> str:
    """Lowercase without accents, so "garcia" matches "García"."""
    text = unicodedata.normalize("NFKD", text.lower().strip())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def order_total(order: Order) -> int:
    return sum(item["quantity"] * item["unit_price"] for item in order["items"])


async def find_customers(name: str) -> list[Customer]:
    await asyncio.sleep(QUERY_LATENCY_SECONDS)
    words = normalize(name).split()
    return [c for c in CUSTOMERS if all(w in normalize(c["name"]) for w in words)]


async def get_customer(customer_id: int) -> Customer | None:
    await asyncio.sleep(QUERY_LATENCY_SECONDS)
    return next((c for c in CUSTOMERS if c["id"] == customer_id), None)


async def find_orders(customer_id: int) -> list[Order]:
    await asyncio.sleep(QUERY_LATENCY_SECONDS)
    return sorted((o for o in ORDERS if o["customer_id"] == customer_id), key=lambda o: o["date"])


async def get_order(order_id: int) -> Order | None:
    await asyncio.sleep(QUERY_LATENCY_SECONDS)
    return next((o for o in ORDERS if o["id"] == order_id), None)
