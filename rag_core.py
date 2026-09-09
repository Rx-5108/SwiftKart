import re
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

documents = [
    {
        "id": "doc1",
        "text": "SwiftKart allows returns or exchanges within 7 days for apparel, footwear, sports & fitness, and toys, provided items are unused and unwashed with original tags or packaging intact. Electronics and appliances are eligible only for replacement (not return) within 7 days, valid for hardware defects and requiring all original parts and cords. Furniture allows replacement only within 5 days if reported for transit damage or manufacturing defects, while books allow return or exchange within 5 days if unmarked and intact. Beauty, skincare, and food & health items are non-returnable due to hygiene reasons but qualify for replacement if damaged or expired.",
        "metadata": {"topic": "return_window"}
    },
    {
        "id": "doc2",
        "text": "For SwiftKart's Cash on Delivery (COD) orders, refunds are processed via direct bank transfer or SwiftWallet credit to the account details provided during the return request, only if the product passes the required return checks. Bank transfers (COD) take 5-7 business days, UPI/card/net-banking refunds take 3-5 business days, and SwiftWallet credits are processed immediately or within 24 hours. If the customer does not receive the refund even after the applicable window, they must raise a new support ticket or contact their bank for further assistance.",
        "metadata": {"topic": "cod_refund_timelines"}
    },
    {
        "id": "doc3",
        "text": "SwiftKart delivery times vary depending on the product ordered and the customer's subscription tier. Standard delivery typically takes 7-10 business days from the date of purchase, while SwiftPro subscribers receive their orders within 3 days. Delivery timelines also depend on the customer's location, as remote or rural areas may have longer delivery windows or limited serviceability compared to urban areas. If an order exceeds its applicable delivery window, customers can contact support to check the delay and request assistance.",
        "metadata": {"topic": "delivery_sla"}
    },
    {
        "id": "doc4",
        "text": "SwiftKart's reverse pickup service is available only for orders that have an approved return request, meeting the return-window and condition checks outlined in the return policy. This is a paid service, charged separately from the return process. The courier partner will make up to 2 pickup attempts to collect the returned item from the customer's address. Customers are responsible for securely packing the item themselves before handover to ensure it is not damaged during transit.",
        "metadata": {"topic": "reverse_pickup"}
    },
    {
        "id": "doc5",
        "text": "SwiftKart offers a 1-year manufacturer warranty on electronics, 3 years on appliances, 6 months on toys, and 5 years on furniture, covering manufacturing defects only, applicable after the initial 7-day replacement window is over. Warranty claims must be raised through customer support along with proof of purchase. Physical damage and water damage are explicitly excluded from all warranty coverage, regardless of product category.",
        "metadata": {"topic": "warranty_terms"}
    },
    {
        "id": "doc6",
        "text": "Customers can cancel their SwiftKart order at any time before it is delivered, free of any cancellation charges. Refunds for cancelled orders follow the same tiered timeline as returns: 5-7 business days for COD/bank transfer, 3-5 business days for UPI/card/net-banking, and immediate to 24 hours for SwiftWallet credits. If the refund is not received within the applicable window, customers must raise a support ticket or contact their bank for assistance.",
        "metadata": {"topic": "order_cancellation"}
    },
    {
        "id": "doc7",
        "text": "SwiftKart customers earn 5 loyalty points for every ₹100 spent on eligible purchases. Each point holds a fixed value of ₹1 and can be redeemed directly as a discount during checkout. Loyalty points are valid for 1 year from the date they are earned, after which unused points expire automatically.",
        "metadata": {"topic": "loyalty_points"}
    },
    {
        "id": "doc8",
        "text": "If a payment fails during checkout, SwiftKart requires customers to manually retry the payment — there is no automatic retry system. Customers can attempt the payment as many times as needed until it succeeds. If an amount is debited or deducted but the order fails to confirm, SwiftKart refunds the amount to the original payment method within 5-7 business days. If the refund is not received within this window, customers must raise a support ticket or contact their bank.  ",
        "metadata": {"topic": "payment_failure_retry"}
    },
    {
        "id": "doc9",
        "text": "SwiftKart allows size exchanges within 7 days of delivery for eligible apparel, footwear, and sports & fitness products. Items must be unused, unwashed, unworn, and returned with their original tags and packaging intact. Customers may request an exchange for an available size of the same product, subject to stock availability. If the requested size is unavailable, the customer may choose an eligible return or replacement according to the applicable category policy.",
        "metadata": {"topic": "size_exchange"}
    },
    {
        "id": "doc10",
        "text": "If a SwiftKart customer receives a damaged, defective, expired, or incorrect item, the issue must be reported within 48 hours of delivery. Customers must provide clear photographs of the product and, when available, an unboxing video as evidence for the claim. After the claim is submitted, SwiftKart reviews the evidence and determines whether the customer qualifies for a replacement, return, or refund according to the applicable category policy. Approved claims are processed after the required quality checks are completed.",
        "metadata": {"topic": "damaged_item_claim"}
    },
    {
        "id": "doc11",
        "text": "SwiftKart provides international shipping only to countries and regions where its delivery partners support serviceability. Certain products, including restricted cosmetics, hazardous materials, perishable food and health products, and items subject to local import regulations, may not be eligible for international delivery. International orders that are eligible for returns or refunds follow the applicable SwiftKart return and refund policies, while customs duties, taxes, and international return-shipping charges are non-refundable unless SwiftKart explicitly approves otherwise. Customers are responsible for providing accurate delivery and customs information, and an international order may be delayed or cancelled if the destination does not permit the product to be imported.",
        "metadata": {"topic": "international_shipping"}
    },
    {
        "id": "doc12",
        "text": "SwiftKart support requests are prioritized based on urgency and the potential impact on the customer. Issues involving payment deductions without order confirmation, damaged or incorrect products, repeated delivery failures, and unresolved refunds should be escalated to a senior support team for further investigation. General policy questions, order-status requests, and routine return or exchange queries can be handled by the standard support team. If an issue remains unresolved after the initial support review, customers may request further escalation through a support ticket.",
        "metadata": {"topic": "support_escalation"}
    }
]


def chunk_fixed_size(text, chunk_size=200, overlap=50):
    if not text or not isinstance(text, str):
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_by_sentences(text, sentences_per_chunk=2, overlap=1):
    if not text or not isinstance(text, str):
        return []
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be greater than 0")
    if overlap < 0 or overlap >= sentences_per_chunk:
        raise ValueError("Overlap must be smaller than sentences_per_chunk")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + sentences_per_chunk
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += sentences_per_chunk - overlap
    return chunks


#  Build chunk lists 
fixed_chunks = []
fixed_chunk_ids = []
fixed_chunk_metadatas = []

for doc in documents:
    doc_chunks = chunk_fixed_size(doc["text"])
    for idx, chunk in enumerate(doc_chunks):
        fixed_chunks.append(chunk)
        fixed_chunk_ids.append(f"{doc['id']}_fixed_{idx}")
        fixed_chunk_metadatas.append({"parent_doc": doc["id"], "topic": doc["metadata"]["topic"]})

sentence_chunks = []
sentence_chunk_ids = []
sentence_chunk_metadatas = []

for doc in documents:
    doc_chunks = chunk_by_sentences(doc["text"])
    for idx, chunk in enumerate(doc_chunks):
        sentence_chunks.append(chunk)
        sentence_chunk_ids.append(f"{doc['id']}_sent_{idx}")
        sentence_chunk_metadatas.append({"parent_doc": doc["id"], "topic": doc["metadata"]["topic"]})


#  Model + ChromaDB setup (top-level)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.30

model = SentenceTransformer(EMBEDDING_MODEL)

fixed_embeddings = model.encode(fixed_chunks, convert_to_numpy=True).tolist()
sentence_embeddings = model.encode(sentence_chunks, convert_to_numpy=True).tolist()

client = chromadb.PersistentClient(path="./chroma_db")

fixed_collection = client.get_or_create_collection(
    name="fixed_size_collection",
    metadata={"hnsw:space": "cosine"}
)

sentence_collection = client.get_or_create_collection(
    name="sentence_based_collection",
    metadata={"hnsw:space": "cosine"}
)

fixed_collection.upsert(
    ids=fixed_chunk_ids,
    documents=fixed_chunks,
    metadatas=fixed_chunk_metadatas,
    embeddings=fixed_embeddings
)

sentence_collection.upsert(
    ids=sentence_chunk_ids,
    documents=sentence_chunks,
    metadatas=sentence_chunk_metadatas,
    embeddings=sentence_embeddings
)


def retrieve_chunks(query: str, collection, top_k: int = DEFAULT_TOP_K):
    """Retrieve up to ``top_k`` relevant chunks from a Chroma collection."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    count = collection.count()
    if count == 0:
        return None

    top_k = min(top_k, count)
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()
    return collection.query(query_embeddings=query_embedding, n_results=top_k)


def generate_grounded_answer(
    query: str,
    collection,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[str, float]:
    """Generate an answer only when retrieval similarity meets the threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    results = retrieve_chunks(query, collection, top_k)

    if results is None or not results["distances"] or not results["distances"][0]:
        return "I don't have enough information in the provided policy documents.", 0.0

    top_distance = results["distances"][0][0]
    top_similarity = 1 - top_distance

    if top_similarity < threshold:
        return "I don't have enough information in the provided policy documents.", top_similarity

    context = "\n".join(results["documents"][0])
    answer = f"Based on SwiftKart policy: {context}"

    return answer, top_similarity


def evaluate_retrieval(eval_queries: list[dict[str, str]], collection, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    results_log = []

    for item in eval_queries:
        query = item["query"]
        correct_doc = item["correct_doc"]

        results = retrieve_chunks(query, collection, top_k)
        if results is None or not results.get("metadatas") or not results["metadatas"][0]:
            retrieved_docs = set()
        else:
            retrieved_docs = {meta["parent_doc"] for meta in results["metadatas"][0]}

        retrieval_hit = 1 if correct_doc in retrieved_docs else 0
        precision = retrieval_hit / len(retrieved_docs) if retrieved_docs else 0.0
        recall = float(retrieval_hit)

        results_log.append({
            "query": query,
            "retrieved_docs": retrieved_docs,
            "precision": precision,
            "recall": recall,
        })

    return results_log


# 

if __name__ == "__main__":
    print("Total fixed-size chunks:", len(fixed_chunks))
    print("Total sentence-based chunks:", len(sentence_chunks))
    print("Both collections created and stored successfully!")

    query = "What is the return window for electronics?"

    print("\n--- FIXED-SIZE RETRIEVAL ---")
    results = retrieve_chunks(query, fixed_collection, top_k=3)
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1}:")
        print(doc)

    print("\n--- SENTENCE-BASED RETRIEVAL ---")
    results = retrieve_chunks(query, sentence_collection, top_k=3)
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1}:")
        print(doc)

    in_scope_queries = [
        "What is the return window for electronics?",
        "How long does COD refund take?",
        "Can I cancel my order after placing it?"
    ]

    for q in in_scope_queries:
        results = retrieve_chunks(q, fixed_collection, top_k=1)
        similarity = 1 - results["distances"][0][0]
        print(f"Query: {q}")
        print(f"Similarity: {similarity:.3f}\n")

    out_of_scope_queries = [
        "What is the capital of France?",
        "How do I bake a chocolate cake?"
    ]

    for q in out_of_scope_queries:
        results = retrieve_chunks(q, fixed_collection, top_k=1)
        similarity = 1 - results["distances"][0][0]
        print(f"Query: {q}")
        print(f"Similarity: {similarity:.3f}\n")

    print("\n=== FINAL GROUNDED GENERATION DEMO ===\n")

    demo_queries = [
        "What is the return window for electronics?",
        "How long does COD refund take?",
        "Can I cancel my order after placing it?",
        "How many loyalty points do I earn per purchase?",
        "What happens if my payment fails?",
        "What is the capital of France?"
    ]

    for q in demo_queries:
        answer, similarity = generate_grounded_answer(q, fixed_collection)
        print(f"Query: {q}")
        print(f"Similarity: {similarity:.3f}")
        print(f"Answer: {answer}\n")

    eval_queries = [
        {"query": "What is the return window for electronics?", "correct_doc": "doc1"},
        {"query": "How long does COD refund take?", "correct_doc": "doc2"},
        {"query": "Can I cancel my order after placing it?", "correct_doc": "doc6"},
        {"query": "How many loyalty points do I earn per purchase?", "correct_doc": "doc7"},
        {"query": "What happens if my payment fails?", "correct_doc": "doc8"}
    ]

    print("=== FIXED-SIZE COLLECTION ===")
    fixed_results = evaluate_retrieval(eval_queries, fixed_collection)
    for r in fixed_results:
        print(f"Query: {r['query']}")
        print(f"Retrieved docs: {r['retrieved_docs']}")
        print(f"Precision: {r['precision']:.2f}, Recall: {r['recall']:.2f}\n")

    print("=== SENTENCE-BASED COLLECTION ===")
    sentence_results = evaluate_retrieval(eval_queries, sentence_collection)
    for r in sentence_results:
        print(f"Query: {r['query']}")
        print(f"Retrieved docs: {r['retrieved_docs']}")
        print(f"Precision: {r['precision']:.2f}, Recall: {r['recall']:.2f}\n")