import os
import math
import torch
import torch.nn as nn
import sentencepiece as spm
from pathlib import Path
from config import (
    EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT, ENC_LAYERS, DEC_LAYERS,
    MAX_SEQ_LEN, TOKENIZER_PATH, MODEL_PATH, 
    DEFAULT_MAX_LENGTH, MAX_GENERATION_LENGTH
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def causal_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    return torch.triu(torch.ones(attn_shape), diagonal=1).bool()


def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = DROPOUT):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)

        def proj(x: torch.Tensor, lin: nn.Module) -> torch.Tensor:
            res = lin(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            return res

        q = proj(query, self.q_lin)
        k = proj(key, self.k_lin)
        v = proj(value, self.v_lin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        out = torch.matmul(p_attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.out_lin(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = DROPOUT):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.dropout(self.relu(self.lin1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = DROPOUT):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = DROPOUT):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))
        attn = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = EMBED_DIM, n_heads: int = NUM_HEADS,
                 enc_layers: int = ENC_LAYERS, dec_layers: int = DEC_LAYERS,
                 d_ff: int = FF_DIM, dropout: float = DROPOUT, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(enc_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(dec_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        src_emb = self.token_embed(src) * math.sqrt(self.d_model)
        src_emb = self.pos_enc(src_emb)

        if src_mask is None:
            src_mask = create_padding_mask(src, 0)

        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)

        tgt_emb = self.token_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_enc(tgt_emb)

        if tgt_mask is None:
            tgt_mask = create_padding_mask(tgt, 0) | causal_mask(tgt.size(1)).to(tgt.device)

        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)

        return self.fc_out(decoder_output)


class ChatbotInference:
    def __init__(self):
        self.device = DEVICE
        self.tokenizer = None
        self.model = None
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

    def load_tokenizer(self):
        """Load the SentencePiece tokenizer"""
        if not Path(TOKENIZER_PATH).exists():
            raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(TOKENIZER_PATH)
        self.pad_id = self.tokenizer.piece_to_id("<pad>")
        self.bos_id = self.tokenizer.piece_to_id("<s>")
        self.eos_id = self.tokenizer.piece_to_id("</s>")
        print(f"✅ Tokenizer loaded. Vocab size: {self.tokenizer.get_piece_size()}")

    def load_model(self):
        """Load the fine-tuned model"""
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        vocab_size = checkpoint.get('vocab_size', self.tokenizer.get_piece_size())
        
        self.model = EncoderDecoderTransformer(
            vocab_size=vocab_size,
            d_model=EMBED_DIM,
            n_heads=NUM_HEADS,
            enc_layers=ENC_LAYERS,
            dec_layers=DEC_LAYERS,
            d_ff=FF_DIM,
            dropout=DROPOUT,
            max_len=MAX_SEQ_LEN
        )
        
        self.model.load_state_dict(checkpoint.get('model_state', checkpoint))
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")

    def encode_text(self, text: str, add_bos_eos: bool = True, max_len: int = MAX_SEQ_LEN):
        """Encode text to token IDs"""
        ids = self.tokenizer.encode(text, out_type=int)
        if add_bos_eos:
            ids = [self.bos_id] + ids[:max_len - 2] + [self.eos_id]
        else:
            ids = ids[:max_len]
        return ids

    def decode_ids(self, ids):
        """Decode token IDs to text"""
        filtered = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.tokenizer.decode(filtered)

    def generate_response(self, question: str, max_length: int = None):
        """Generate a response to a question"""
        if max_length is None:
            max_length = DEFAULT_MAX_LENGTH
        
        # Clamp max_length to prevent overly long generations
        max_length = min(max_length, MAX_GENERATION_LENGTH)
        
        question_ids = self.encode_text(question, add_bos_eos=True)
        question_tensor = torch.tensor([question_ids], dtype=torch.long).to(self.device)

        answer_ids = [self.bos_id]

        for _ in range(max_length):
            answer_tensor = torch.tensor([answer_ids], dtype=torch.long).to(self.device)
            with torch.no_grad():
                logits = self.model(question_tensor, answer_tensor)
                next_token = logits[:, -1].argmax(-1)
                answer_ids.append(next_token.item())

                if next_token.item() == self.eos_id:
                    break

        response_text = self.decode_ids(answer_ids)
        return response_text

    def initialize(self):
        """Initialize tokenizer and model"""
        self.load_tokenizer()
        self.load_model()
        print("✅ Chatbot initialized successfully")


# Global chatbot instance
chatbot = ChatbotInference()

