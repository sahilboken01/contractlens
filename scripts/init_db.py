from app.services.vector_store import initialize_schema


if __name__ == "__main__":
    initialize_schema()
    print("ContractLens pgvector schema initialized.")
