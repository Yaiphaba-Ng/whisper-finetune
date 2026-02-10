import os
import sys


def verify_tokenizer():
    # Path where we found the tokenizer files (tokenizer.model, etc.)
    tokenizer_path = "indic_voices_tokenizer"

    print(f"Attempting to load tokenizer from: {tokenizer_path}")

    if not os.path.exists(tokenizer_path):
        print(f"Error: Directory {tokenizer_path} does not exist.")
        return

    try:
        from transformers import AutoTokenizer

        print("Attempting to load with AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Success! Loaded {type(tokenizer).__name__}")
        print(f"Vocab size: {tokenizer.vocab_size}")

        # Test text
        text = "ꯃꯅꯤꯄꯨꯔꯤ ꯂꯣꯟ"
        print(f"\nTest Text: {text}")
        encoded = tokenizer(text)
        print(f"Encoded IDs: {encoded.input_ids}")
        decoded = tokenizer.decode(encoded.input_ids, skip_special_tokens=False)
        print(f"Decoded: '{decoded}'")

        # Custom Token Inspection
        print(f"\n--- detailed inspection ---")
        tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids)
        print(f"Tokens: {tokens}")
        print(f"Tokenizer class: {tokenizer.__class__}")

        # Check if it's splitting chars without merge markers
        # Try loading specific classes to force correct behavior
        from transformers import (
            BertTokenizer,
            T5Tokenizer,
            LlamaTokenizer,
            AlbertTokenizer,
            XLMRobertaTokenizer,
        )

        print("\n--- Trying specific classes ---")

        # 1. Try BERT (uses vocab.txt usually)
        if os.path.exists(os.path.join(tokenizer_path, "vocab.txt")):
            try:
                print("\nTrying BertTokenizer (WordPiece)...")
                bert_tok = BertTokenizer.from_pretrained(tokenizer_path)
                print(
                    f"Decoded (BERT): '{bert_tok.decode(bert_tok(text).input_ids, skip_special_tokens=True)}'"
                )
            except Exception as e:
                print(f"BertTokenizer failed: {e}")

        # 2. Try T5 (uses tokenizer.model usually)
        if os.path.exists(os.path.join(tokenizer_path, "tokenizer.model")):
            try:
                print("\nTrying T5Tokenizer (SentencePiece)...")
                t5_tok = T5Tokenizer.from_pretrained(tokenizer_path)
                print(
                    f"Decoded (T5): '{t5_tok.decode(t5_tok(text).input_ids, skip_special_tokens=True)}'"
                )
            except Exception as e:
                print(f"T5Tokenizer failed: {e}")

            try:
                print("\nTrying AlbertTokenizer (SentencePiece)...")
                albert_tok = AlbertTokenizer.from_pretrained(tokenizer_path)
                print(
                    f"Decoded (Albert): '{albert_tok.decode(albert_tok(text).input_ids, skip_special_tokens=True)}'"
                )
            except Exception as e:
                print(f"AlbertTokenizer failed: {e}")

            try:
                print("\nTrying XLMRobertaTokenizer (SentencePiece)...")
                xlm_tok = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
                print(
                    f"Decoded (XLMRoberta): '{xlm_tok.decode(xlm_tok(text).input_ids, skip_special_tokens=True)}'"
                )
            except Exception as e:
                print(f"XLMRobertaTokenizer failed: {e}")

        # 3. Try Llama (SentencePiece)
        try:
            print("\nTrying LlamaTokenizer (SentencePiece)...")
            llama_tok = LlamaTokenizer.from_pretrained(tokenizer_path)
            print(
                f"Decoded (Llama): '{llama_tok.decode(llama_tok(text).input_ids, skip_special_tokens=True)}'"
            )
        except Exception as e:
            print(f"LlamaTokenizer failed: {e}")

    except Exception as e:
        print(f"\nFailed to load or use tokenizer. Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_tokenizer()
