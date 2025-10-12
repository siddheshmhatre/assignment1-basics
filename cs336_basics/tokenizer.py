import json
import regex as re
import pickle
from typing import Dict, List, Tuple, Optional, Iterable, Iterator
from functools import partial
import multiprocessing as mp

import json
import regex as re
import pickle
from typing import Dict, List, Tuple, Optional, Iterable, Iterator
from tqdm.auto import tqdm # Import tqdm
from cs336_basics.pretokenization_example import find_chunk_boundaries

class Tokenizer:
    """
    A BPE-based tokenizer class.
    """
    
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]], 
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab = vocab
        # Create a reverse mapping for encoding
        self.vocab_rev = {val: key for key, val in self.vocab.items()}

        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
    
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: Optional[List[str]] = None
    ):
        import pickle
        with open(vocab_filepath, 'rb') as f:
            loaded_vocab = pickle.load(f)

        return cls(loaded_vocab['vocab'], loaded_vocab['merges'], special_tokens)

    def _encode_no_special_tokens_text(self, tokenized_text: List[int], text:str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for pretoken in re.finditer(PAT, text):
            pretoken = pretoken.group()
            pretoken = tuple(bytes([b]) for b in pretoken.encode("utf-8"))

            # Apply merges
            for merge in self.merges:
                idx = 0
                merged_pretoken = []
                while idx < len(pretoken) - 1:
                    char_1 = pretoken[idx]
                    char_2 = pretoken[idx + 1]

                    if char_1 == merge[0] and char_2 == merge[1]:
                        merged_pretoken.append(char_1 + char_2)
                        idx += 2
                    else:
                        merged_pretoken.append(char_1)
                        idx += 1
                
                if idx == len(pretoken) - 1:
                    merged_pretoken.append(pretoken[-1])
                
                pretoken = merged_pretoken

            # Tokenize
            for token in pretoken:
                tokenized_text.append(self.vocab_rev[token])

    def encode(self, text: str) -> List[int]:
        matches = []
        for token in self.special_tokens:
            matches += list(re.finditer(re.escape(token), text))

        matches = sorted(matches, key=lambda x: x.span()[0])

        tokenized_text = []

        if len(matches) > 0:
            start_pos = 0
            processed_matches = [matches[0]]

            if len(matches) > 1:
                # Remove overlaps
                for match in matches[1:]:
                    last_match = processed_matches[-1]

                    if last_match.span()[1] <= match.span()[0]:
                        processed_matches.append(match)
                    else:
                        cond1 = match.span()[0] <= last_match.span()[0]
                        cond2 = match.span()[1] >= last_match.span()[1]
                        if cond1 and cond2:
                            processed_matches.pop(-1)
                            processed_matches.append(match)

            for match in processed_matches:
                end_pos = match.span()[0]
                self._encode_no_special_tokens_text(tokenized_text, text[start_pos:end_pos])
                special_token = match.group()
                tokenized_text.append(self.vocab_rev[special_token.encode('utf-8')])
                start_pos = match.span()[1]

            if len(text[start_pos:]) > 0:
                self._encode_no_special_tokens_text(tokenized_text, text[start_pos:])
        else:
            self._encode_no_special_tokens_text(tokenized_text, text)

        return tokenized_text

    def encode_iterable(self, iterable: Iterable[str], pbar: Optional[tqdm] = None) -> Iterator[int]:
        for line in iterable:
            if pbar:
                # Update progress bar by the number of bytes in the current line
                pbar.update(len(line.encode('utf-8')))
            yield from self.encode(line)

    def _encode_chunk(self, chunk_boundary: tuple[int], input_path: str) -> list[int]:
        # Read in text and return self.encode
        # Open file handle and only read in required bytes
        with open(input_path, 'rb') as file:
            file.seek(chunk_boundary[0])
            chunk = file.read(chunk_boundary[1] - chunk_boundary[0])

        return (self.encode(chunk.decode('utf-8')), len(chunk))

    def encode_parallel(self, input_path, desired_num_chunks, split_special_token, pbar: Optional[tqdm] = None) -> Iterator[int]:
        file = open(input_path, 'rb')

        chunk_boundaries = find_chunk_boundaries(file, desired_num_chunks, split_special_token.encode('utf-8'))

        encode_chunk = partial(self._encode_chunk, input_path=input_path)

        with mp.Pool() as pool:
            tokenized_chunks = pool.imap(
                encode_chunk,
                [(chunk_boundaries[idx], chunk_boundaries[idx + 1]) for idx in range(len(chunk_boundaries) - 1)]
                )

            for chunk, chunk_size in tokenized_chunks:
                if pbar:
                    pbar.update(chunk_size)
                yield from chunk

    def decode(self, ids: List[int]) -> str:
        decoded_text = b""
        for id in ids:
            decoded_text += self.vocab[id]
        return decoded_text.decode("utf-8", errors="replace")


if __name__ == "__main__":
    with open('/home/siddhesh/code/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt') as f:
        ts = f.read()

    sample = ts[:2500]

    print (f"Original text: \n {sample}")
    print("------------------------------------------------------------------------------")

    tokenizer = Tokenizer.from_files("data/TinyStoriesV2-GPT4-train.pkl", "", ["<|endoftext|>"])

    tokenized_seq = tokenizer.encode(sample)

    print (f"Encoded-decoded text: \n {tokenizer.decode(tokenized_seq)} ")
