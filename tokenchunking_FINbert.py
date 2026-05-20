import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FinBERTInferenceEngine:
    def __init__(self, model_name="ProsusAI/finbert"):
        # Explicit initialization of tokenizer and sequence classification weights
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Turn off dropout/batchnorm for consistent inference execution
        
    def generate_document_sentiment(self, text, max_chunk_len=450, overlap=50):
        """
        Handles Step 2 (Chop text into token arrays) & Step 3 (Run FinBERT Softmax logit weights).
        """
        if not text or len(text.strip()) < 10:
            return 0.0 # Return perfectly neutral value if input data string is vacant
            
        # Raw tokenization without padding or max length clamps
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)
        
        chunk_scores = []
        stride = max_chunk_len - overlap
        
        # --- Step 2: Slicing the Token Windows ---
        for i in range(0, total_tokens, stride):
            token_slice = tokens[i : i + max_chunk_len]
            
            # Reconstruct structural requirements with mandatory BERT framing IDs
            processed_slice = [self.tokenizer.cls_token_id] + token_slice + [self.tokenizer.sep_token_id]
            
            # Final sequence array size clamp safety
            if len(processed_slice) > 512:
                processed_slice = processed_slice[:512]
                
            # Convert python integers directly to PyTorch Input Tensors
            input_ids = torch.tensor([processed_slice])
            attention_mask = torch.ones_like(input_ids)
            
            # --- Step 3: FinBERT Tensor Processing ---
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Apply Softmax activation across the logits matrix layer
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
            
            # Mapping positional float arrays -> [Positive, Negative, Neutral]
            prob_positive = probabilities[0]
            prob_negative = probabilities[1]
            
            # Compute final mathematical directional value
            slice_sentiment = prob_positive - prob_negative
            chunk_scores.append(slice_sentiment)
            
        # Average the weights across all analyzed document segments
        return np.mean(chunk_scores) if chunk_scores else 0.0