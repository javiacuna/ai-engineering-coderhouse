"""Agent tools: queries to the (simulated) orders database.

Every tool has a Pydantic input contract and a detailed docstring: the LLM chooses a tool ONLY from
its name, description and argument schema. Errors are returned as data ({"error", "sugerencia"})
instead of exceptions, so the model can read them and retry with other arguments or ask the user.
"""

from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

import database as db

ToolResult = dict[str, Any]


def error(message: str, suggestion: str) -> ToolResult:
    return {"error": message, "sugerencia": suggestion}


# ---------------------------------------------------------------------------
# 1. search_customers
# ---------------------------------------------------------------------------
class SearchCustomersInput(BaseModel):
    name: str = Field(..., min_length=2, max_length=100,
                      description="Nombre, apellido o nombre completo del cliente (por ejemplo 'García' o 'Lucía Fernández').")


@tool(args_schema=SearchCustomersInput)
async def search_customers(name: str) -> ToolResult:
    """Busca clientes por nombre y devuelve su ID, nombre y ciudad.

    Usala cuando el usuario mencione a un cliente por su nombre o apellido y necesites su ID para
    consultar sus pedidos. No hace falta si el usuario ya dio el ID numérico del cliente.

    Si hay VARIOS clientes que coinciden, no elijas uno al azar: preguntale al usuario cuál es,
    mostrando la ciudad de cada uno para que pueda distinguirlos.
    """
    matches = await db.find_customers(name)
    if not matches:
        return error(f"No hay clientes que coincidan con '{name}'.",
                     "Verificá cómo está escrito el nombre o pedile al usuario el ID del cliente.")
    result: ToolResult = {"coincidencias": len(matches), "clientes": matches}
    if len(matches) > 1:
        result["aviso"] = "Hay más de un cliente con ese nombre: pedile al usuario que aclare cuál."
    return result


# ---------------------------------------------------------------------------
# 2. get_customer_orders
# ---------------------------------------------------------------------------
class CustomerOrdersInput(BaseModel):
    customer_id: int = Field(..., ge=1, description="ID numérico del cliente (por ejemplo 102).")


@tool(args_schema=CustomerOrdersInput)
async def get_customer_orders(customer_id: int) -> ToolResult:
    """Devuelve el resumen de pedidos de un cliente a partir de su ID: cantidad de pedidos, monto
    total gastado (en pesos, sin contar los pedidos cancelados) y la lista de pedidos con su ID,
    fecha, estado y monto, ordenados del más viejo al más nuevo (el último de la lista es el más reciente).

    Usala para preguntas sobre cuántos pedidos hizo un cliente, cuánto gastó o cuál fue su último
    pedido. Necesita el ID: si solo tenés el nombre, primero usá search_customers. Para ver qué
    productos tiene un pedido, usá después get_order_detail con el ID del pedido.
    """
    customer = await db.get_customer(customer_id)
    if customer is None:
        return error(f"El cliente {customer_id} no existe.",
                     "Si el usuario mencionó un nombre, buscá el ID con search_customers; si no, pedile el ID correcto.")
    orders = await db.find_orders(customer_id)
    active = [o for o in orders if o["status"] != "cancelado"]
    return {
        "cliente": customer,
        "pedidos": len(orders),
        "pedidos_cancelados": len(orders) - len(active),
        "total": sum(db.order_total(o) for o in active),
        "moneda": "ARS",
        "detalle": [{"pedido_id": o["id"], "fecha": o["date"].isoformat(), "estado": o["status"],
                     "monto": db.order_total(o)} for o in orders],
    }


# ---------------------------------------------------------------------------
# 3. get_order_detail
# ---------------------------------------------------------------------------
class OrderDetailInput(BaseModel):
    order_id: int = Field(..., ge=1, description="ID numérico del pedido (por ejemplo 5012).")


@tool(args_schema=OrderDetailInput)
async def get_order_detail(order_id: int) -> ToolResult:
    """Devuelve el detalle de un pedido a partir de su ID: fecha, estado (entregado, en_camino o
    cancelado), productos con cantidad y precio unitario, y el monto total.

    Usala cuando el usuario quiera saber qué compró en un pedido o en qué estado está. Los IDs de
    pedido se obtienen con get_customer_orders. No confundas el ID de pedido (por ejemplo 5012)
    con el ID de cliente (por ejemplo 102).
    """
    order = await db.get_order(order_id)
    if order is None:
        return error(f"El pedido {order_id} no existe.",
                     "Obtené los IDs de pedido válidos con get_customer_orders.")
    return {"pedido_id": order["id"], "cliente_id": order["customer_id"], "fecha": order["date"].isoformat(),
            "estado": order["status"], "productos": order["items"], "monto": db.order_total(order), "moneda": "ARS"}


TOOLS: list[BaseTool] = [search_customers, get_customer_orders, get_order_detail]
