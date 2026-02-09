from transformers import WhisperTokenizer, PreTrainedTokenizerFast
import os

tokenizer_path = (
    r"d:\Edu\Internship - (IITG)\CODE\whisper-finetune\indic_voices_tokenizer"
)
print(f"Testing tokenizer load from: {tokenizer_path}")
print(f"Files in dir: {os.listdir(tokenizer_path)}")

try:
    print("Attempting to load as WhisperTokenizer...")
    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_path)
    print("Success loading as WhisperTokenizer!")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Tokenizer class: {type(tokenizer)}")
except Exception as e:
    print(f"Failed to load as WhisperTokenizer: {e}")

try:
    print("\nAttempting to load as PreTrainedTokenizerFast (using tokenizer.model)...")
    # T5 uses SentencePiece, maybe that works?
    # Or generically:
    from transformers import PreTrainedTokenizerFast

    # This might require a tokenizer.json or similar, but let's try.
    # If it's pure SP, maybe LlamaTokenizer or T5Tokenizer?
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    print("Success loading as T5Tokenizer!")
    print(f"Vocab size: {len(tokenizer)}")
except Exception as e:
    print(f"Failed to load as T5Tokenizer: {e}")

try:
    print("\nAttempting to load using sentencepiece directly...")
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(tokenizer_path, "tokenizer.model"))
    print("Success loading with sentencepiece!")
    print(f"Vocab size: {sp.get_piece_size()}")
except Exception as e:
    print(f"Failed to load with sentencepiece: {e}")
