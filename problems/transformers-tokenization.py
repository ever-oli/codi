import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Step 1: Add special tokens with their fixed IDs
        special_tokens = [self.pad_token, self.unk_token, 
                         self.bos_token, self.eos_token]
        
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        # Step 2: Find all unique words from texts
        unique_words = set()
        for text in texts:
            # Split text into words (simple whitespace splitting)
            words = text.split()
            unique_words.update(words)
        
        # Step 3: Add unique words to vocabulary
        current_id = len(special_tokens)  # Start from ID 4
        for word in sorted(unique_words):  # Sort for consistency
            if word not in self.word_to_id:  # Avoid duplicates
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        # Update vocabulary size
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # Split text into words
        words = text.split()
        
        # Convert each word to ID (use UNK if word not found)
        token_ids = []
        for word in words:
            # .get() returns the value if key exists, otherwise returns default (UNK)
            token_id = self.word_to_id.get(word, self.word_to_id[self.unk_token])
            token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for token_id in ids:
            # Get word from id_to_word dictionary
            word = self.id_to_word.get(token_id, self.unk_token)
            words.append(word)
        
        # Join words with spaces
        return ' '.join(words)