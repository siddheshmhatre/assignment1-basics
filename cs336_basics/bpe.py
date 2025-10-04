import regex as re
import heapq 
import multiprocessing as mp
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries

def process_each_chunk(chunk_boundary : tuple[int], input_path: str, special_tokens: list[bytes])-> Counter:
    # Remove special tokens
    pattern = b'|'.join(re.escape(tok) for tok in special_tokens)

    # Open file handle and only read in required bytes
    file = open(input_path, 'rb')
    file.seek(chunk_boundary[0])
    chunk = file.read(chunk_boundary[1] - chunk_boundary[0])

    # Remove special tokens from chunk
    chunk_no_special_tokens = b"".join(re.split(pattern, chunk))
    counter = Counter()

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for pretoken in re.finditer(PAT, chunk_no_special_tokens.decode("utf-8")):
        pretoken = pretoken.group()

        if re.fullmatch(r"\n+", pretoken):
            continue
        pretoken = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        counter[tuple(pretoken)] += 1

    return counter

def pretokenize(input_path, desired_num_chunks, split_special_token, special_tokens):
    file = open(input_path, 'rb')
    chunk_boundaries = find_chunk_boundaries(file, desired_num_chunks, split_special_token)

    # Run above function on each chunk
    with mp.Pool() as pool:
        local_counters = pool.starmap(
            process_each_chunk,
            [((chunk_boundaries[idx], chunk_boundaries[idx + 1]), input_path, special_tokens) 
             for idx in range(len(chunk_boundaries) - 1)]
        )

    # Merge all local counters
    final_counter = Counter()
    for counter in local_counters:
        final_counter.update(counter)
    
    return final_counter

def get_merge_pair(counter: Counter):
    """
    Returns a list of all keys in the Counter that have the maximum value.
    """
    if not counter:
        return []

    max_value = counter.most_common(1)[0][1]  # Get the maximum value
    merge_pairs = [key for key, value in counter.items() if value == max_value]
    # print (counter.most_common()[:15])
    # import pdb; pdb.set_trace()
    return max(merge_pairs)
       
def train_bpe(input_path, special_tokens, vocab_size, desired_num_chunks, split_special_token):
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

    special_tokens = [token.encode('utf-8') for token in special_tokens]

    print("Pretokenizing...")
    pretoken_counter = pretokenize(input_path, desired_num_chunks, split_special_token, special_tokens)

    # Count all the pairs of bytes
    pairs_counter = Counter()
    pairs_to_pretoken = defaultdict(set)
    for pretoken, count in pretoken_counter.items():
        for idx in range(len(pretoken) - 1):
            pair = (pretoken[idx], pretoken[idx+1])
            pairs_counter[pair] += count
            pairs_to_pretoken[pair].add(pretoken)

    merges = []

    print("Tokenizing...")
    
    # Loop while size of vocab is < vocab size
    while len(vocab) < vocab_size:

        # Log step number
        if len(vocab) % 100 == 0:
            print (f"Vocab size {len(vocab)} / {vocab_size}")

        # Get max pair
        max_pair = get_merge_pair(pairs_counter)

        # Add to vocab + merges
        new_token = max_pair[0] + max_pair[1]
        vocab[max_token_idx] = new_token
        max_token_idx += 1
        merges.append(max_pair)

        # Set max pair count to -1 so that it doesn't get picked again
        pretokens_to_update = list(pairs_to_pretoken[max_pair])

        for pretoken in pretokens_to_update:
            idx = 0
            merged_pretoken = []

            while idx < len(pretoken):
                # Check for merge at the current position
                if idx < len(pretoken) - 1 and (pretoken[idx], pretoken[idx + 1]) == max_pair:
                    merged_pretoken.append(new_token)
                    idx += 2
                else:
                    merged_pretoken.append(pretoken[idx])
                    idx += 1
            merged_pretoken = tuple(merged_pretoken)

            pretoken_count = pretoken_counter[pretoken]
            del pretoken_counter[pretoken]
            pretoken_counter[merged_pretoken] += pretoken_count

            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i+1])
                pairs_counter[pair] -= pretoken_count
                # Clean up the reverse map
                if pair in pairs_to_pretoken and pretoken in pairs_to_pretoken[pair]:
                    pairs_to_pretoken[pair].remove(pretoken)

            for i in range(len(merged_pretoken) - 1):
                pair = (merged_pretoken[i], merged_pretoken[i+1])
                pairs_counter[pair] += pretoken_count
                # Add the new pretoken to the reverse map for its pairs
                pairs_to_pretoken[pair].add(merged_pretoken)

            del pairs_counter[max_pair]

    return vocab, merges