from cs336_basics.bpe import train_bpe
import pickle
import time

if __name__ == "__main__":
    start_time = time.time()
    input_path = "data/owt_train.txt"
    special_tokens=["<|endoftext|>"]
    vocab_size = 32_000
    desired_num_chunks = 100
    split_special_token = b'<|endoftext|>'

    vocab, merges = train_bpe(input_path, special_tokens, vocab_size, desired_num_chunks, split_special_token)
    end_time = time.time()

    with open(f'{input_path.split('.')[0]}.pkl', 'wb') as f:
        pickle.dump({'vocab': vocab, 'merges' : merges}, f)
    print (f"Took total {end_time - start_time}")