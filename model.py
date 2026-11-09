"""Small causal Transformer that emits one constrained navigation action token."""
import torch
from torch import nn
from data import ACTIONS, records, words

WINDOW = 48
WIDTH = 40
RANK = 4

def make_vocab():
    tokens=sorted({w for c in records("train") for w in words(c.text)})
    return {word:i+3 for i,word in enumerate(tokens)}

def encode(texts, vocab):
    rows=[]
    for text in texts:
        body=[vocab.get(w,1) for w in words(text)][-(WINDOW-1):]+[2]
        rows.append([0]*(WINDOW-len(body))+body)
    return torch.tensor(rows,dtype=torch.long)

class LoRAHead(nn.Module):
    def __init__(self, width, out):
        super().__init__()
        self.base=nn.Linear(width,out)
        self.a=nn.Parameter(torch.empty(RANK,width),requires_grad=False)
        self.b=nn.Parameter(torch.zeros(out,RANK),requires_grad=False)
        nn.init.normal_(self.a,std=.025)

    def forward(self,x):
        return self.base(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.a),self.b)*2

class Router(nn.Module):
    def __init__(self,n_vocab):
        super().__init__()
        self.embed=nn.Embedding(n_vocab,WIDTH,padding_idx=0)
        self.pos=nn.Embedding(WINDOW,WIDTH)
        self.block=nn.TransformerEncoderLayer(WIDTH,4,WIDTH*2,dropout=0,batch_first=True,norm_first=True)
        self.norm=nn.LayerNorm(WIDTH)
        self.head=LoRAHead(WIDTH,len(ACTIONS))
        self.register_buffer("causal",torch.ones(WINDOW,WINDOW,dtype=torch.bool).triu(1),persistent=False)

    def forward(self, x):
        h=self.embed(x)+self.pos(torch.arange(x.shape[1],device=x.device))
        # Left pads are fixed blank tokens; no all-masked query positions.
        h=self.block(h,src_mask=self.causal)
        return self.head(self.norm(h[:,-1]))

    def adapters_only(self):
        self.requires_grad_(False)
        self.head.a.requires_grad_(True)
        self.head.b.requires_grad_(True)

def load_router(path):
    ckpt=torch.load(path,map_location="cpu",weights_only=True)
    vocab=ckpt["vocab"]
    router=Router(len(vocab)+3)
    router.load_state_dict(ckpt["model"])
    return router.eval(),vocab
