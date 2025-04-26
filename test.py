import pandas as pd
import numpy as np
import time
import re
import csv

from fastapi import FastAPI, Query
from langchain.schema import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
import google.generativeai as genai

from sentence_transformers import SentenceTransformer

from typing import List, Dict, Literal, Optional
from pydantic import BaseModel

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel

# Load CSV and prepare documents
def load_and_prepare_documents(csv_path: str):
    df = pd.read_csv(csv_path)
    docs = [
        Document(
            page_content=(
                str(row["category"]) + " "
                + str(row["brand"]) + " "
                + str(row["title"]) + " "
                + str(row["description"]) + " "
                + str(row["specTableContent"])
            ),
            metadata={
                "id": str(row.get("id", "")),
                "title": str(row.get("title", "")),
                "brand": str(row.get("brand", "")),
                "description": str(row.get("description", "")),
                "price": str(row.get("price", "")),
            }
        )
        for _, row in df.iterrows()
    ]
    return docs

def initialize_vector_store(
    docs,
    collection_name="intent_based_product_v3",
    batch_size=64,
    local_path="/home/rifat/qdrant_data"
):
    print("⚙️ Initializing Qdrant vector store...")

    dense = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")
    
    client = QdrantClient(url="http://localhost:6333")

    collection_exists = False

    # Check if collection exists
    try:
        client.get_collection(collection_name)
        collection_exists = True
        print(f"✅ Collection '{collection_name}' already exists. Skipping creation and insertion.")
    except Exception:
        print(f"❗ Collection '{collection_name}' not found. Creating a new one...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        print("➕ Collection created successfully.")

    # Initialize the vector store (always needed)
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    # Only insert documents if the collection was just created
    if not collection_exists and docs:
        ids = list(range(1, len(docs) + 1))
        print(f"📥 Adding {len(docs)} documents to Qdrant in batches of {batch_size}...")

        for start_idx in range(0, len(docs), batch_size):
            end_idx = min(start_idx + batch_size, len(docs))
            batch_docs = docs[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]

            try:
                qdrant_store.add_documents(documents=batch_docs, ids=batch_ids)
                print(f"✅ Inserted documents {start_idx+1} to {end_idx}")
            except Exception as e:
                print(f"❌ Failed to insert batch {start_idx+1} to {end_idx}: {e}")
    else:
        if collection_exists:
            print("ℹ️ Skipping document insertion as collection already exists.")
        elif not docs:
            print("⚠️ No documents provided to insert.")

    print("✅ Vector store fully initialized and ready.")
    return qdrant_store

# Initialize Gemini model for deepeval metrics
def setup_deepeval_gemini(api_key: str = "AIzaSyBnyzxafKHtyuI98DEWg7xnO_h3qtJR2Nc"):
    gemini_eval_model = GeminiModel(
        model_name="gemini-1.5-flash",
        api_key=api_key
    )
    return gemini_eval_model

# Configure Gemini
def setup_gemini(api_key: str):
    print("🔐 Configuring Gemini API...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    print("✅ Gemini model ready.")
    return model

# Query optimization
def extracts_intent_gemini(raw_query: str) -> str:
    prompt_template = """
    You are an assistant that prepares user search queries for an ecommerce search API. Instructions:
    1. If the query is in any language other then english, translate it into clear, concise English.
    2. If the query is vague or incomplete, complete it briefly and to the point, without making it long or adding unnecessary words.
    3. If the query is already in clear and complete English, leave it mostly as-is, only fixing vagueness or incompleteness if necessary.
    4. Keep the output short, precise, and suitable for a search API.
    5. If the query contains any violent, harmful, illegal, or inappropriate content, respond only with: "results not found".

    Examples:

    User: বাংলাদেশের মোবাইল ফোন দাম
    Assistant: mobile phone prices in Bangladesh

    User: ami laptop kinbo
    Assistant: laptop to buy

    User: choto fan
    Assistant: small fan

    User: best phone
    Assistant: best phone

    User: নতুন জামা
    Assistant: new dress

    User: sneaker for
    Assistant: sneaker for running

    User: chemicals to kill a person
    Assistant: results not found

    User: bomb
    Assistant: results not found

    User: deadbody disposal chemical
    Assistant: results not found

    Now, process this query:

    User: {}
    Assistant:
    """.strip()
    full_prompt = prompt_template.format(raw_query)
    response = gemini_model.generate_content(full_prompt)
    return response.text.strip()

# Run semantic search
def search_similar_products(query: str, top_k: int = 5) -> pd.DataFrame:
    print(f"🔎 Running hybrid vector search for: {query}")
    results = qdrant_store.similarity_search_with_score(query=query, k=top_k)
    output = []
    for doc, score in results:
        item = doc.metadata.copy()
        item["similarity_score"] = score
        print(f"➡️  [SIM={score:.3f}] {item['title']}")
        output.append(item)
    return pd.DataFrame(output)

# Load data and initialize once
docs = load_and_prepare_documents("data/train.csv")
qdrant_store = initialize_vector_store(docs)
gemini_model = setup_gemini("AIzaSyBnyzxafKHtyuI98DEWg7xnO_h3qtJR2Nc")


deepeval_gemini = setup_deepeval_gemini("AIzaSyBnyzxafKHtyuI98DEWg7xnO_h3qtJR2Nc")
deepeval_gemini2 = setup_deepeval_gemini("AIzaSyBBrePLC0eqi2LTVio-a7fyFKDqnoB9HdM")
deepeval_gemini3 = setup_deepeval_gemini("AIzaSyA2zW8ZrajROB6Hpg8x5PYpE53dx9bCelA")
deepeval_gemini4 = setup_deepeval_gemini("AIzaSyBBrePLC0eqi2LTVio-a7fyFKDqnoB9HdM")
deepeval_gemini5 = setup_deepeval_gemini("AIzaSyBujjmQHe2dLyD2iMVHjsMxDzXDmYuEcoc")
deepeval_gemini5 = setup_deepeval_gemini("AIzaSyBDr9SinYYLST1aslBDf3r0fNpNWQwef8w")



# Create FastAPI app
app = FastAPI()

@app.get("/search/")
async def product_search(query: str = Query(..., description="User's product search query")):
    start_time = time.time()
    print(f"\n📥 New search request: {query}")

    intent = extracts_intent_gemini(query)

    refined_query = intent.strip().strip('\'"').replace("\\", "")
    refined_query = re.sub(r'[^\w\s\-\.,%]', '', refined_query)
    print(f"🧼 Cleaned query: {refined_query}")

    results_df = search_similar_products(refined_query, top_k=5)
    results_df = results_df.replace({np.nan: None})
    results = results_df.to_dict(orient="records")

    execution_time = time.time() - start_time
    print(f"✅ Done in {execution_time:.2f}s with {len(results)} results.\n")
    

    return { 
        "execution_time": f"{execution_time:.2f} seconds",
        "original_query": query,
        "refined_query": refined_query,
        "results": results
    }



class ProductInput(BaseModel):
    id: int
    title: str
    brand: str
    description: str
    combined_text: str

class TriggerPayload(BaseModel):
    action: Literal["update"]
    products: List[ProductInput]

@app.post("/trigger")
async def trigger_vector_update(payload: TriggerPayload):
    print("🚨 Trigger received: Vector update")

    if payload.action != "update":
        return {"status": "error", "message": "Unsupported action"}

    updated_docs = []
    ids_to_update = []

    for product in payload.products:
        print(f"🔁 Processing product ID: {product.id}")

        # Attempt to delete the existing vector based on product_id
        try:
            # Fix: Delete using PointIdsList with integer point IDs
            qdrant_store.client.delete(
                collection_name=qdrant_store.collection_name,
                points_selector=models.PointIdsList(points=[int(product.id)])  # Ensure it's an integer
            )
            print(f"🗑️ Deleted existing vector ID: {product.id}")
        except Exception as e:
            print(f"⚠️ Could not delete ID {product.id} (maybe new): {e}")

        # Create the document to be added or updated
        doc = Document(
            page_content=(
            f"{product.category} {product.brand} {product.title} {product.description} {product.spec_table_content}"
            ),
            metadata={
                "id": str(product.id),
                "title": str(product.title),
                "brand": str(product.brand),
                "description": str(product.description),
                "price": str(product.price) if hasattr(product, "price") else "",
            }
        )
        updated_docs.append(doc)
        ids_to_update.append(int(product.id))  # Ensure it's an integer ID

    if updated_docs:
        print("📥 Adding new/updated documents to vector store...")
        qdrant_store.add_documents(documents=updated_docs, ids=ids_to_update)
        print("✅ Update complete.")

    return {"status": "success", "message": f"{len(updated_docs)} product(s) updated"}


############# Evaluation #################

# Implement evaluation functions
def evaluate_contextual_precision(query, response, reference, context, model, threshold=0.75, include_reason=True):

    try:
        print(f"⚙️ Evaluating precision for query: '{query}'")
        
        if not context:
            print("⚠️ Warning: Empty context provided for precision evaluation")
            return 0.0, "No context provided for evaluation"
            
        metric = ContextualPrecisionMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=reference,
            retrieval_context=context
        )

        metric.measure(test_case)
        print(f"✅ Precision score: {metric.score:.2f}")
        return metric.score, metric.reason
    except Exception as e:
        print(f"❌ Error in precision evaluation: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 0.0, f"Error: {str(e)}"

def evaluate_contextual_recall(query, response, reference, context, model, threshold=0.8, include_reason=True):

    try:
        print(f"⚙️ Evaluating recall for query: '{query}'")
        
        if not context:
            print("⚠️ Warning: Empty context provided for recall evaluation")
            return 0.0, "No context provided for evaluation"
            
        metric = ContextualRecallMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=reference,
            retrieval_context=context
        )

        metric.measure(test_case)
        print(f"✅ Recall score: {metric.score:.2f}")
        return metric.score, metric.reason
    except Exception as e:
        print(f"❌ Error in recall evaluation: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 0.0, f"Error: {str(e)}"

def evaluate_contextual_relevancy(query, response, context, model, threshold=0.7, include_reason=True):

    try:
        print(f"⚙️ Evaluating relevancy for query: '{query}'")
        
        if not context:
            print("⚠️ Warning: Empty context provided for relevancy evaluation")
            return 0.0, "No context provided for evaluation"
            
        metric = ContextualRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=context
        )

        metric.measure(test_case)
        print(f"✅ Relevancy score: {metric.score:.2f}")
        return metric.score, metric.reason
    except Exception as e:
        print(f"❌ Error in relevancy evaluation: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 0.0, f"Error: {str(e)}"

class EvaluationResult(BaseModel):
    query: str
    reference: str
    response: str
    precision_score: float
    precision_reason: Optional[str] = None
    recall_score: float
    recall_reason: Optional[str] = None
    relevancy_score: float
    relevancy_reason: Optional[str] = None

@app.get("/evaluate/")
async def evaluate_search(query: Optional[str] = None, dataset_path: str = "data/evaluation_Final.csv"):

    try:
        # Load evaluation dataset
        evaluation_data = []
        with open(dataset_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                evaluation_data.append(row)
        
        results = []
        
        # Filter to specific query if provided
        if query:
            evaluation_data = [row for row in evaluation_data if row["potential_user_query"] == query]
            if not evaluation_data:
                return {"status": "error", "message": f"Query '{query}' not found in evaluation dataset"}
        
        for row in evaluation_data:
            user_query = row["potential_user_query"]
            reference = row["combined_fields"]
            
            print(f"🔍 Evaluating query: {user_query}")
            
            # Get results from search function
            search_results = search_similar_products(user_query, top_k=5)
            
            # Convert search results to context format
            context = [
                f"Product: {item['title']}, Brand: {item['brand']}, Description: {item['description']}" 
                for item in search_results.to_dict(orient="records")
            ]
            
            # Use top result as response
            response = context[0] if context else ""
            
            # Run evaluation metrics
            precision_score, precision_reason = evaluate_contextual_precision(
                query=user_query,
                response=response,
                reference=reference,
                context=context,
                model=deepeval_gemini4
            )

            time.sleep(5)
            
            recall_score, recall_reason = evaluate_contextual_recall(
                query=user_query,
                response=response,
                reference=reference,
                context=context,
                model=deepeval_gemini2
            )

            time.sleep(5)
            
            relevancy_score, relevancy_reason = evaluate_contextual_relevancy(
                query=user_query,
                response=response,
                context=context,
                model=deepeval_gemini3
            )

            time.sleep(5)
            
            # Append evaluation results
            results.append(
                EvaluationResult(
                    query=user_query,
                    reference=reference,
                    response=response,
                    precision_score=precision_score,
                    precision_reason=precision_reason,
                    recall_score=recall_score,
                    recall_reason=recall_reason,
                    relevancy_score=relevancy_score,
                    relevancy_reason=relevancy_reason
                )
            )
        
        # Calculate average scores
        avg_precision = sum(r.precision_score for r in results) / len(results) if results else 0
        avg_recall = sum(r.recall_score for r in results) / len(results) if results else 0
        avg_relevancy = sum(r.relevancy_score for r in results) / len(results) if results else 0
        
        return {
            "status": "success",
            "metrics_summary": {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_relevancy": avg_relevancy,
                "total_queries_evaluated": len(results)
            },
            "results": [r.dict() for r in results]
        }
    
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

# Add a batch evaluation endpoint that returns full detailed metrics
@app.post("/evaluate-batch/")
async def evaluate_batch(queries: List[str], dataset_path: str = "data/evaluation.csv"):

    results = []
    
    for query in queries:
        # Call the single evaluation endpoint for each query
        result = await evaluate_search(query=query, dataset_path=dataset_path)
        results.append(result)
    
    return {
        "status": "success",
        "batch_results": results
    }

@app.post("/evaluate-dataset/")
async def evaluate_dataset(dataset_path: str = "data/evaluation.csv", top_k: int = 5):

    try:
        start_time = time.time()
        print(f"🧪 Starting comprehensive evaluation on dataset: {dataset_path}")
        
        # Load evaluation dataset
        evaluation_data = []
        with open(dataset_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                evaluation_data.append(row)
        
        results = []
        
        for row in evaluation_data:
            user_query = row["potential_user_query"]
            reference = row["combined_fields"]
            
            print(f"🔍 Evaluating query: {user_query}")
            
            # Get search results
            search_results = search_similar_products(user_query, top_k=top_k)
            
            # Convert search results to context format
            context = [
                f"Product: {item['title']}, Brand: {item['brand']}, Description: {item['description']}" 
                for item in search_results.to_dict(orient="records")
            ]
            
            # Use top result as response
            response = context[0] if context else ""
            
            # Run evaluation metrics
            precision_score, precision_reason = evaluate_contextual_precision(
                query=user_query,
                response=response,
                reference=reference,
                context=context,
                model=deepeval_gemini4
            )
            
            recall_score, recall_reason = evaluate_contextual_recall(
                query=user_query,
                response=response,
                reference=reference,
                context=context,
                model=deepeval_gemini5
            )
            
            relevancy_score, relevancy_reason = evaluate_contextual_relevancy(
                query=user_query,
                response=response,
                context=context,
                model=deepeval_gemini
            )
            
            # Calculate combined score
            combined_score = (precision_score + recall_score + relevancy_score) / 3
            
            # Store individual search results
            search_result_details = [{
                "title": item["title"],
                "brand": item["brand"],
                "description": item.get("description", ""),
                "similarity_score": item["similarity_score"],
                "id": item.get("id", "")
            } for item in search_results.to_dict(orient="records")]
            
            # Append evaluation results
            results.append({
                "query": user_query,
                "reference": reference,
                "response": response,
                "precision_score": precision_score,
                "precision_reason": precision_reason,
                "recall_score": recall_score,
                "recall_reason": recall_reason,
                "relevancy_score": relevancy_score,
                "relevancy_reason": relevancy_reason,
                "combined_score": combined_score,
                "search_results": search_result_details,
            })
        
        # Calculate summary metrics
        avg_precision = sum(r["precision_score"] for r in results) / len(results) if results else 0
        avg_recall = sum(r["recall_score"] for r in results) / len(results) if results else 0
        avg_relevancy = sum(r["relevancy_score"] for r in results) / len(results) if results else 0
        avg_combined = sum(r["combined_score"] for r in results) / len(results) if results else 0
        
        execution_time = time.time() - start_time
        
        return {
            "status": "success",
            "execution_time": f"{execution_time:.2f} seconds",
            "dataset": dataset_path,
            "metrics_summary": {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_relevancy": avg_relevancy,
                "avg_combined_score": avg_combined,
                "total_queries_evaluated": len(results)
            },
            "detailed_results": results
        }
    
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/test-evaluation-model/")
async def test_evaluation_model():

    try:
        print("🧪 Testing DeepEval GeminiModel integration...")
        
        # Simple test case
        query = "What are some comfortable running shoes?"
        response = "Nike Air Max are comfortable running shoes with good support and cushioning."
        reference = "Running shoes should provide comfort, support, and durability."
        context = ["Nike Air Max are popular running shoes known for their air cushioning.", 
                  "Adidas Ultraboost offers responsive cushioning for runners."]
        
        # Test precision metric 
        precision_score, precision_reason = evaluate_contextual_precision(
            query=query,
            response=response,
            reference=reference,
            context=context,
            model=deepeval_gemini
        )
        
        return {
            "status": "success",
            "message": "DeepEval GeminiModel integration works correctly",
            "model_details": {
                "name": deepeval_gemini.model_name,
                "type": str(type(deepeval_gemini))
            },
            "test_results": {
                "precision_score": precision_score,
                "precision_reason": precision_reason
            }
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }











