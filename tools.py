from dataset import orders


def check_order_status(record_id: str) -> dict:
    """Return status, order value, and escalation score for a SwiftKart order."""
    order = next((o for o in orders if o["record_id"] == record_id), None)

    if order is None:
        return {"error": f"Order {record_id} not found"}

    recency_score = order["days_since_created"] / 30
    delay_score = 1.0 if order["delayed_shipment"] else 0.0
    escalation_score = (0.6 * delay_score) + (0.4 * recency_score)

    return {
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "escalation_score": round(escalation_score, 2),
    }
