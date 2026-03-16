from openai import OpenAI
import argparse
import os


def submit_batch_to_openai(input_file, client):

    ## Upload the input file to OpenAI
    batch_input_file = client.files.create(
        file=open(input_file, "rb"),
        purpose="batch",
    )

    batch_input_file_id = batch_input_file.id

    print(f"Batch file uploaded. Input file ID: {batch_input_file_id}")

    ## Create a batch
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(batch)

    # Save the batch id to a file
    with open("batch_id.txt", "w") as f:
        f.write(batch.id)

    print(f"Batch created. Batch ID: {batch.id}")

    return batch.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="batch_inputs.jsonl")
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    batch_id = submit_batch_to_openai(args.input_file, client)
    print(f"Batch ID: {batch_id}")

    