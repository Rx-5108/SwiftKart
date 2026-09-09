import random

random.seed(8)

categories = [
    "Apparel", "Electronics", "Appliances", "Home", "Beauty", "Footwear",
    "Food & Health", "Sports", "Furniture", "Books", "Toys & Babycare",
]

statuses = [
    "Placed", "Shipped", "Delivered", "Returned", "Refunded", "Cancelled",
    "Pending Payment",
]

orders = []

for i in range(45):
    orders.append({
        "category": random.choice(categories),
        "record_id": f"ORD-{1000 + i}",
        "status": random.choice(statuses),
        "order_value_inr": random.randint(200, 20_000),
        "days_since_created": random.randint(0, 30),
        "delayed_shipment": random.random() < 0.2,
    })


def summarize_dataset():
    print("Total orders:", len(orders))
    print("First order:", orders[0])

    category_counts = {}
    for order in orders:
        category = order["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    print("Category counts:", category_counts)

    status_counts = {}
    for order in orders:
        status = order["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    print("Status counts:", status_counts)

    delayed_count = sum(order["delayed_shipment"] for order in orders)
    delayed_pct = (delayed_count / len(orders)) * 100
    print(f"Delayed %: {delayed_pct:.1f}%")


if __name__ == "__main__":
    summarize_dataset()
