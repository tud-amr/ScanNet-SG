import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

batches = client.batches.list(limit=10)

for batch in batches:
    batch = client.batches.retrieve(batch.id)
    print(batch)
    print("="*50)

    # Cancel the batch
    try:
        client.batches.cancel(batch.id)
        print(f"Cancelled batch {batch.id}")
        print("="*50)
    except Exception as e:
        print(f"Error cancelling batch {batch.id}: {e}")
        print("="*50)




