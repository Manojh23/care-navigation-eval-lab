"""Train a routine-intent base, then LoRA SFT and reference-based DPO."""
import argparse, copy, hashlib, json
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from data import ACTIONS, SAFETY, audit_splits, records
from evaluate import gate, score
from model import Router, encode, make_vocab

def fit(out: Path, seed: int=23, quick: bool=False):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    audit_splits()
    cs=records("train")
    vocab=make_vocab()
    x=encode([c.text for c in cs],vocab)
    y=torch.tensor([ACTIONS.index(c.action) for c in cs])
    rejected=torch.tensor([ACTIONS.index(c.rejected) for c in cs])
    routine=torch.tensor([i for i,c in enumerate(cs) if c.action not in SAFETY])
    safety=torch.tensor([i for i,c in enumerate(cs) if c.action in SAFETY])
    m=Router(len(vocab)+3)
    trace=[]
    out.mkdir(parents=True,exist_ok=True)

    def save(name):
        torch.save({"model":m.state_dict(),"vocab":vocab},out/f"{name}.pt")

    def sft(stage, indices, steps, lr):
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr)
        for step in range(steps):
            batch=indices[torch.randint(len(indices),(24,))]
            loss=F.cross_entropy(m(x[batch]),y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step==0 or (step+1)%10==0 or step+1==steps:
                trace.append({"stage":stage,"step":step+1,"loss":round(loss.item(),6)})

    sft("base",routine,10 if quick else 65,.005)
    save("base")
    base={split:score(m,vocab,split) for split in ("validation","test")}
    m.adapters_only()
    sft("lora_sft",torch.arange(len(cs)),10 if quick else 120,.022)
    save("sft")
    sft_report={split:score(m,vocab,split) for split in ("validation","test")}
    ref=copy.deepcopy(m).eval().requires_grad_(False)
    with torch.no_grad():
        ref_log=ref(x).log_softmax(-1)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.005)
    for step in range(8 if quick else 55):
        batch=safety[torch.randint(len(safety),(20,))]
        logp=m(x[batch]).log_softmax(-1)
        ar=torch.arange(len(batch))
        yes,no=y[batch],rejected[batch]
        # DPO for one action-token completion; both scores condition on the same prompt.
        improvement=(logp[ar,yes]-logp[ar,no])-(ref_log[batch,yes]-ref_log[batch,no])
        loss=-F.logsigmoid(.3*improvement).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step==0 or (step+1)%10==0:
            trace.append({"stage":"dpo","step":step+1,"loss":round(loss.item(),6)})
    save("dpo")
    dpo={split:score(m,vocab,split) for split in ("validation","test")}
    release=gate(sft_report["validation"],dpo["validation"])
    audit=gate(sft_report["test"],dpo["test"])
    split_rows={s:[asdict(c) for c in records(s)] for s in ("train","validation","test")}
    payload=json.dumps(split_rows,sort_keys=True)
    report={
        "seed":seed,"quick":quick,"fixture_sha256":hashlib.sha256(payload.encode()).hexdigest(),
        "split_sizes":{s:len(cs) for s,cs in split_rows.items()},
        "parameters":sum(p.numel() for p in m.parameters()),
        "lora_parameters":sum(p.numel() for p in m.parameters() if p.requires_grad),
        "stages":{"base":base,"sft":sft_report,"dpo":dpo},
        "validation_gate":release,"validation_candidate":"dpo" if release["passed"] else "sft",
        "test_regression_audit":audit,
        "selected_stage":"dpo" if release["passed"] and audit["passed"] else "sft",
    }
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"trace.jsonl").write_text("".join(json.dumps(e)+"\n" for e in trace))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("runs/demo"))
    p.add_argument("--seed",type=int,default=23)
    p.add_argument("--quick",action="store_true")
    args=p.parse_args()
    r=fit(args.out,args.seed,args.quick)
    print(json.dumps({k:r[k] for k in ("split_sizes","parameters","lora_parameters","selected_stage","validation_gate","test_regression_audit")},indent=2))
    for stage,s in r["stages"].items():
        print(stage,{k:v for k,v in s["test"].items() if k!="errors"})
