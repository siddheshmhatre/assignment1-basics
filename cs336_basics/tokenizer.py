import json
import regex as re
import pickle
from typing import Dict, List, Tuple, Optional, Iterable, Iterator

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
        """
        Construct a tokenizer from a given vocabulary, list of merges, and 
        (optionally) a list of special tokens.
        """
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
    ) -> 'Tokenizer':
        """
        Class method that constructs and return a Tokenizer from a serialized 
        vocabulary and list of merges.
        """
        with open(vocab_filepath, 'r') as f:
            loaded_vocab = json.load(f)

        loaded_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    loaded_merges.append(tuple(cleaned_line.split(" ")))
        return cls(loaded_vocab, loaded_vocab, special_tokens)

    def _encode_no_special_tokens_text(self, tokenized_text: List[int], text:str):
        # For every pretoken -> apply merges in order
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
        """
        Encode an input text into a sequence of token IDs.
        """
        # Create mapping start position of special token 
        special_token_mapping = dict()
        matches = []
        for token in self.special_tokens:
            matches += list(re.finditer(re.escape(token), text))

        # Sort the matches based on start idx
        matches = sorted(matches, key=lambda x: x.span()[0])

        tokenized_text = []
        start_pos = 0

        for match in matches:
            end_pos = match.span()[0]

            self._encode_no_special_tokens_text(tokenized_text, text[start_pos:end_pos])

            special_token = match.group()
            tokenized_text.append(self.vocab_rev[special_token.encode('utf-8')])

            start_pos = match.span()[1]

        if matches[-1].span()[1] < len(text) - 1:
            self._encode_no_special_tokens_text(tokenized_text, text[start_pos:])

        return tokenized_text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        """
        # Placeholder for lazy encoding logic
        for item in iterable:
            yield from self.encode(item)

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
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
