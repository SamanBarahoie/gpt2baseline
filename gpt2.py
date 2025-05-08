import torch
from torch import Tensor, nn, optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizerFast
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
    dynamo.config.verbose = False
except ImportError:
    pass

@dataclass
class Config:
    vocab_size: int
    seq_len: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    dropout: float = 0.1
    grad_clip_norm: float = 1.0
    lr: float = 5e-5
    batch_size: int = 32
    epochs: int = 2
    steps_per_epoch: int = 1000

class TinyStoriesStreamDataset(IterableDataset):
    def __init__(self, token_file_path: str, seq_len: int):
        self.token_file_path = token_file_path
        self.seq_len = seq_len
        self._token_ids = None

    def _load_token_ids(self):
        if self._token_ids is None:
            self._token_ids = torch.load(self.token_file_path)
        return self._token_ids

    def __iter__(self):
        token_ids = self._load_token_ids()
        token_len = len(token_ids)
        while True:
            idx = torch.randint(0, token_len - self.seq_len - 1, (1,)).item()
            x = token_ids[idx:idx+self.seq_len]
            y = token_ids[idx+1:idx+self.seq_len+1]
            yield x, y

class TinyStoriesDataset(Dataset):
    def __init__(self, token_ids, seq_len: int):
        self.ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask: torch.Tensor):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        h = self.ln2(x)
        return x + self.ff(h)

class GPT2Simple(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            GPT2Block(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        bool_mask = torch.triu(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', bool_mask)

    def forward(self, input_ids: Tensor):
        bsz, seqlen = input_ids.size()
        x = self.tok_emb(input_ids) + self.pos_emb(
            torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        )
        mask = self.causal_mask[:seqlen, :seqlen]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

def train_epoch(model, loader, optimizer, scheduler, device, cfg, scaler):
    model.train()
    total_loss = 0.0
    it = iter(loader)
    for _ in tqdm(range(cfg.steps_per_epoch), desc='Train'):
        x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / cfg.steps_per_epoch
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Eval'):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    avg_loss = total_loss / len(loader)
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()

def generate_text(model, tokenizer, prompt, max_new, device, seq_len):
    seq = prompt.unsqueeze(0).to(device)
    for _ in range(max_new):
        context = seq[:, -seq_len:]
        logits = model(context)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)
    return tokenizer.decode(seq[0].tolist())

def main():
    scaler = GradScaler()
    cfg = Config(10000, 128, 256, 8, 16, 256, batch_size=32, epochs=2, steps_per_epoch=20000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    tokenizer = PreTrainedTokenizerFast(tokenizer_file='bpe-tokenizer_tinystories.json', pad_token='<|pad|>')
    valid_ids = torch.load('tokenized-valid-samples_vocab-10k.pt')

    train_loader = DataLoader(
        TinyStoriesStreamDataset('tokenized-train-samples_vocab-10k.pt', cfg.seq_len),
        batch_size=cfg.batch_size, num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        TinyStoriesDataset(valid_ids, cfg.seq_len),
        batch_size=cfg.batch_size, pin_memory=True
    )

    model = GPT2Simple(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    history = {'tokens': [], 'train_loss': [], 'valid_loss': []}

    for epoch in range(1, cfg.epochs + 1):
        print(f'=== Epoch {epoch}/{cfg.epochs} ===')
        tr_loss, tr_ppl = train_epoch(model, train_loader, optimizer, scheduler, device, cfg, scaler)
        batch_x, _ = next(iter(valid_loader))
        sample = generate_text(model, tokenizer, batch_x[0][:cfg.seq_len], max_new=64, device=device, seq_len=cfg.seq_len)
        print(f'--- Generated text after epoch {epoch} ---')
        print(sample)
        val_loss, val_ppl = eval_epoch(model, valid_loader, device)
        tokens = cfg.batch_size * cfg.seq_len * cfg.steps_per_epoch
        total = history['tokens'][-1] + tokens if history['tokens'] else tokens
        history['tokens'].append(total)
        history['train_loss'].append(tr_loss)
        history['valid_loss'].append(val_loss)
        print(f'Train: loss={tr_loss:.4f}, ppl={tr_ppl:.2f}')
        print(f'Valid: loss={val_loss:.4f}, ppl={val_ppl:.2f}')
        torch.save(model.state_dict(), f'ckpt_epoch{epoch}.pt')

    plt.figure()
    plt.plot(history['tokens'], history['train_loss'], label='Train')
    plt.plot(history['tokens'], history['valid_loss'], label='Valid')
    plt.xlabel('Tokens')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
