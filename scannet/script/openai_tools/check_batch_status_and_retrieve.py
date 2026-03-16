from openai import OpenAI
import os
import argparse
import time


def check_batch_status_and_retrieve(batch_id, client, if_retrieve_output=True, check_interval=120, save_file_name=None):
    batch = None
    while True:
        batch = client.batches.retrieve(batch_id)
        print(batch)
        # print("="*50)
        print(f"Batch ID: {batch_id}")
        print(f"Batch status: {batch.status}")
        # print(f"Batch request_counts: {batch.request_counts}")
        # print(f"Batch output_file_id: {batch.output_file_id}")

        if batch.status == "completed":
            break
        elif batch.status == "failed":
            print(f"Batch failed")
            return None

        time.sleep(check_interval)
    
    print("="*50)
    print(f"Batch Completed")
    print("current time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("="*50)
    
    if save_file_name is None:
        save_file_name = "batch_results_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + ".jsonl"
    
    if if_retrieve_output and batch is not None and batch.output_file_id is not None:  # once completed
        file_response = client.files.content(batch.output_file_id)
        
        with open(save_file_name, "w") as f:
            f.write(file_response.text)

        return save_file_name
    else:
        print(f"Batch failed or output file is not available. Batch ID: {batch_id}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id", type=str, default="batch_6878da0e69f48190861c0fd411b105f8")
    parser.add_argument("--if_retrieve_output", type=bool, default=True)
    parser.add_argument("--check_interval", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    check_batch_status_and_retrieve(args.batch_id, client, args.if_retrieve_output, args.check_interval)

