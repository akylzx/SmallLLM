import os
import sentencepiece as spm
import json
import pandas as pd
import tempfile


input_files_json = [
    "/home/srp/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/en_data/train_en_books_cleaned_v2_splited.json",
    "/home/srp/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/en_data/train_en_fineweb_cleaned_v2_splited.json",
    "/home/srp/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/ru_data/russian/russian/train_ru_various_cleaned_v2.json",
    "/home/srp/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/ru_data/russian/russian/train_ru_wikipedia_cleaned_v2.json",
]

input_files_parquet = [
    "/home/srp/data/kazakh/snapshots/d62ba981d9ba825905753c27ee73ed5814ebb9ed/data/kk_various_cleaned_v2_split-00000-of-00001.parquet",
    "/home/srp/data/kazakh/snapshots/d62ba981d9ba825905753c27ee73ed5814ebb9ed/data/kk_wikipedia_cleaned_v2_split-00000-of-00001.parquet",
]

print("Reading input files...")
all_texts = []


for file_path in input_files_json:
    print(f"Processing JSON file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get("text", "").strip()
                    if text:
                        all_texts.append(text)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        continue

for file_path in input_files_parquet:
    print(f"Processing Parquet file: {file_path}")
    try:
        data = pd.read_parquet(file_path)
        for text in data["text"].astype(str).str.strip().dropna():
            if text and text != 'nan':  
                all_texts.append(text)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        continue

print(f"Total texts collected: {len(all_texts)}")

if not all_texts:
    print("Error: No texts were collected. Please check your input files.")
    exit(1)


output_dir = "./spm_unigram_tokenizer"
model_prefix = "smp_unigram_multilingual"

vocab_size = 10000
character_coverage = 1.0
model_type = "unigram"

pad_id = 0
eos_id = 1
bos_id = 2
unk_id = 3

normalization_rule = "nmt_nfkc_cf" 
user_defined_symbols = ['<pad>', '<eos>', '<bos>', '<unk>']  
shuffle_sentences = True
hard_vocab_limit = False
train_large_corpus = True


def train_tokenizer():
    os.makedirs(output_dir, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt')
    try:
        for text in all_texts:
            temp_file.write(text + '\n')
        temp_file.close()
        
        prefix = os.path.join(output_dir, model_prefix)
        
        print("Training SentencePiece model...")
        spm.SentencePieceTrainer.train(
            input=temp_file.name,  
            model_prefix=prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            normalization_rule_name=normalization_rule,
            user_defined_symbols=user_defined_symbols,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            unk_id=unk_id,
            shuffle_input_sentence=shuffle_sentences,
            hard_vocab_limit=hard_vocab_limit,
            train_extremely_large_corpus=train_large_corpus
        )
        
        print(f"Trained SentencePiece model: {prefix}.model")
        print(f"Vocabulary file: {prefix}.vocab")
        
        
        sp = spm.SentencePieceProcessor()
        sp.load(f"{prefix}.model")
        test_text = "Hello world! This is a test."
        encoded = sp.encode_as_pieces(test_text)
        print(f"Test encoding: {encoded}")
        
        
        try:
            from transformers import PreTrainedTokenizerFast
            
        
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=sp,  
                unk_token="<unk>",
                bos_token="<bos>",
                eos_token="<eos>",
                pad_token="<pad>",
                model_max_length=2048
            )
            tokenizer.save_pretrained(output_dir)
            print(f"Saved HuggingFace tokenizer to {output_dir}")
            
        except ImportError:
            print("transformers not installed; skipping HuggingFace tokenizer conversion.")
        except Exception as e:
            print(f"Error creating HuggingFace tokenizer: {e}")
            
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


if __name__ == "__main__":
    train_tokenizer()
