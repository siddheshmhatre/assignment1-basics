import numpy as np
import time
import os
import argparse
from cs336_basics.tokenizer import Tokenizer
from tqdm.auto import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tokenize OWT data with optimized settings."
    )
    parser.add_argument("input_path", type=str, help="Path to the input text file.")
    parser.add_argument("vocab_filepath", type=str, help="Path to the vocabulary file.")
    parser.add_argument("--num_chunks", type=int, default=100, help="Number of parallel chunks.")

    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_filepath,
        merges_filepath="",
        special_tokens=special_tokens
    )

    file_size_in_bytes = os.path.getsize(args.input_path)

    print(f"Tokenizing {args.input_path} ({file_size_in_bytes / 1e9:.2f} GB) with {args.num_chunks} chunks...")
    all_ids = []
    start_time = time.time()

    with tqdm(total=file_size_in_bytes, unit='B', unit_scale=True, desc=f"Processing") as pbar:
        for _id in tokenizer.encode_parallel(args.input_path,
                                            desired_num_chunks=args.num_chunks,
                                            split_special_token=special_tokens[0],
                                            pbar=pbar):
            all_ids.append(_id)

    all_ids = np.uint16(all_ids)
    output_path = args.input_path.replace(".txt", ".npy")
    np.save(output_path, all_ids)

    end_time = time.time()
    total_time_in_seconds = end_time - start_time

    bytes_per_token = file_size_in_bytes / len(all_ids)
    print(f"Number of bytes: {file_size_in_bytes}")
    print(f"Number of tokens: {len(all_ids)}")
    print(f"Bytes / token ratio: {bytes_per_token:.4f}")
    print(f"Throughput in bytes / sec: {file_size_in_bytes / total_time_in_seconds:.2f}")
    print(f"Total time: {total_time_in_seconds / 60:.1f} minutes")
    print(f"Saved to: {output_path}")
