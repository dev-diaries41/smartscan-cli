from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def load_clip_tokenizer(vocab_path: str, merges_path: str):
    tokenizer = Tokenizer(BPE(vocab_path, merges_path))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<start_of_text> $A <end_of_text>",
        pair="<start_of_text> $A <end_of_text> $B:1 <end_of_text>:1",
        special_tokens=[
            ("<start_of_text>", 49406),
            ("<end_of_text>", 49407),
        ],
    )
    return tokenizer
