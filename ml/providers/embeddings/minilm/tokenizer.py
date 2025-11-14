from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents

def load_minilm_tokenizer(vocab_path: str):
    tokenizer = Tokenizer(WordPiece(vocab=vocab_path, unk_token="[UNK]"))

    # Add normalization to match Hugging Face WordPiece
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 101), ("[SEP]", 102)],
    )
    return tokenizer
