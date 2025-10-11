import numpy as np
import time
import os
import argparse
from cs336_basics.tokenizer import Tokenizer
from tqdm.auto import tqdm # Import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sample a text file and calculate the bytes/token ratio."
    )
    parser.add_argument("input_path", type=str, help="Path to the input text file.")
    parser.add_argument("vocab_filepath", type=str, help="Path to the vocabulary file.")

    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_filepath,
        merges_filepath="",
        special_tokens=special_tokens
    )
    
    # Get file size to set the total for the progress bar
    file_size_in_bytes = os.path.getsize(args.input_path)

    print ("Tokenizing...")
    all_ids = []
    start_time = time.time()
    
    # Use 'with' for both the progress bar and the file to ensure they are closed
    with tqdm(total=file_size_in_bytes, unit='B', unit_scale=True, desc=f"Processing {args.input_path}") as pbar:
        # It's good practice to specify encoding
        with open(args.input_path, 'r', encoding='utf-8') as f:
            for _id in tokenizer.encode_iterable(f, pbar=pbar):
                all_ids.append(_id)

    all_ids = np.uint16(all_ids)
    np.save(args.input_path.replace("txt", "npy"), all_ids)

    end_time = time.time()
    total_time_in_seconds = end_time - start_time

    # file_size_in_bytes is already calculated
    bytes_per_token =  file_size_in_bytes / len(all_ids)
    print (f"Number of bytes: {file_size_in_bytes}")
    print (f"Number of tokens: {len(all_ids)}")
    print(f"Bytes / token ratio: {bytes_per_token:.4f}")
    print(f"Throughput in bytes / sec: {file_size_in_bytes / total_time_in_seconds:.2f}")