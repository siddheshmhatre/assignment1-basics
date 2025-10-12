import time
import os
import argparse
from cs336_basics.tokenizer import Tokenizer


def sample_documents(file_path: str, n: int, delimiter: str = "<|endoftext|>") -> str:
    """
    Takes the first n documents from a file and writes them to a new file.
    The new file will be named '{original_filename}_sample.txt'.
    Returns the path to the newly created sample file.
    """

    def read_documents_lazily(f, delimiter_str):
        """A generator that yields documents from a file without loading it all into memory."""
        buffer = ""
        while True:
            chunk = f.read(4096)  # Read in 4KB chunks
            if not chunk:
                if buffer.strip():
                    yield buffer.strip()
                return

            buffer += chunk
            while delimiter_str in buffer:
                document, _, buffer = buffer.partition(delimiter_str)
                if document.strip():
                    yield document.strip()

    first_n_docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        doc_iterator = read_documents_lazily(f, delimiter)

        # Take the first n documents
        for _ in range(n):
            try:
                first_n_docs.append(next(doc_iterator))
            except StopIteration:
                # Stop if the file has fewer than n documents
                break

    # Create the output filename
    base, ext = os.path.splitext(file_path)
    output_filename = f"{base}_sample{ext}"

    # Join the documents and write to file
    output_content = f"\n{delimiter}\n".join(first_n_docs) + f"\n{delimiter}\n"

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(output_content)

    return output_filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sample a text file and calculate the bytes/token ratio."
    )
    parser.add_argument("input_path", type=str, help="Path to the input text file.")
    parser.add_argument("vocab_filepath", type=str, help="Path to the vocabulary file.")
    parser.add_argument("-n", "--num_samples", type=int, default=10, help="Number of documents to sample.")

    args = parser.parse_args()

    print ("Sampling...")
    sample_filename = sample_documents(args.input_path, args.num_samples)

    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_filepath,
        merges_filepath="",
        special_tokens=special_tokens
    )

    print ("Tokenizing...")
    all_ids = []
    start_time = time.time()
    with open(sample_filename) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)

    end_time = time.time()
    total_time_in_seconds = end_time - start_time

    file_size_in_bytes = os.path.getsize(sample_filename)
    bytes_per_token =  file_size_in_bytes / len(all_ids)
    print (f"Number of bytes: {file_size_in_bytes}")
    print (f"Number of tokens: {len(all_ids)}")
    print(f"Bytes / token ratio: {bytes_per_token}")
    print(f"Throughput in bytes / sec: {file_size_in_bytes / total_time_in_seconds}")