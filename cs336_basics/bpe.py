import regex as re
import multiprocessing as mp
from collections import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries

def process_each_chunk(chunk: bytes, special_tokens: list[bytes])-> Counter:
    # Remove special tokens
    pattern = b'|'.join(re.escape(tok) for tok in special_tokens)
    chunk_no_special_tokens = b" ".join(re.split(pattern, chunk))
    counter = Counter()

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for pretoken in re.finditer(PAT, chunk_no_special_tokens.decode("utf-8")):
        pretoken = pretoken.group()
        pretoken = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        counter[tuple(pretoken)] += 1

    return counter

def pretokenize(input_path, desired_num_chunks, split_special_token, special_tokens):
    file = open(input_path, 'rb')
    chunk_boundaries = find_chunk_boundaries(file, desired_num_chunks, split_special_token)
    chunks = []

    for idx in range(len(chunk_boundaries) - 1):
        file.seek(chunk_boundaries[idx])
        chunks.append(file.read(chunk_boundaries[idx + 1] - chunk_boundaries[idx]))

    # Run above function on each chunk
    with mp.Pool() as pool:
        local_counters = pool.starmap(
            process_each_chunk,
            [(chunk, special_tokens) for chunk in chunks]
        )

    # Merge all local counters
    final_counter = Counter()
    for counter in local_counters:
        final_counter.update(counter)
    
    return final_counter

def get_all_max_keys(counter: Counter):
    """
    Returns a list of all keys in the Counter that have the maximum value.
    """
    if not counter:
        return []

    max_value = counter.most_common(1)[0][1]  # Get the maximum value
    max_keys = [key for key, value in counter.items() if value == max_value]
    return max_keys
       
def train_bpe(input_path, special_tokens, vocab_size, desired_num_chunks):
    # Initialize vocabulary
    max_token_idx = 0
    vocab = dict()

    # Set first indices to special tokens
    for token in special_tokens:
        vocab[max_token_idx] = token.encode("utf-8")
        max_token_idx += 1

    # One token per byte
    for i in range(256):
        vocab[max_token_idx] = bytes([i])
        max_token_idx += 1

    split_special_token = b'<|endoftext|>'
    special_tokens = [token.encode('utf-8') for token in special_tokens]

    pretoken_counter = pretokenize(input_path, desired_num_chunks, split_special_token, special_tokens)

    # Count all the pairs of bytes
    pairs_counter = Counter()
    for pretoken, count in pretoken_counter.items():
        for idx in range(len(pretoken) - 1):
            pairs_counter[(pretoken[idx], pretoken[idx+1])] += count

    merges = []

    # Loop while size of vocab is < vocab size
    while len(vocab) < vocab_size:
        # Get max pair
        max_pair = max(get_all_max_keys(pairs_counter))

        # Add to vocab + merges
        vocab[max_token_idx] = max_pair[0] + max_pair[1]
        max_token_idx += 1
        merges.append(max_pair)

        # Create new counter and repeat
        pairs_counter = Counter()
        new_pretoken_counter = Counter()
        for pretoken, count in pretoken_counter.items():
            idx = 0
            merged_pretoken = []
            while idx < len(pretoken) - 1:
                char_1 = pretoken[idx]
                char_2 = pretoken[idx + 1]

                if char_1 == max_pair[0] and char_2 == max_pair[1]:
                    merged_pretoken.append(char_1 + char_2)
                    idx += 2
                else:
                    merged_pretoken.append(char_1)
                    idx += 1

            if idx == len(pretoken) - 1:
                merged_pretoken.append(pretoken[idx])

            merged_pretoken = tuple(merged_pretoken)
            new_pretoken_counter[merged_pretoken] = count

            for idx in range(len(merged_pretoken) - 1):
                pairs_counter[(merged_pretoken[idx], merged_pretoken[idx+1])] += count
        pretoken_counter = new_pretoken_counter

    return vocab, merges