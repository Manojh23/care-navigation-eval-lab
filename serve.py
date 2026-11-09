"""Local API for the synthetic benchmark only."""
import argparse, json
from pathlib import Path
from typing import Literal
import torch
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from data import ACTIONS, REPLIES
from model import encode, load_router
from evaluate import guard

class Request(BaseModel):
    model_config=ConfigDict(extra="forbid")
    mode: Literal["synthetic_demo"]
    conversation: str=Field(min_length=1,max_length=600)

def create_app(run: Path):
    torch.set_num_threads(2)
    report=json.loads((run/"report.json").read_text())
    stage=report["selected_stage"]
    if stage not in ("sft","dpo"):
        raise ValueError("Unknown stage")
    if stage=="dpo" and not (report["validation_gate"]["passed"] and report["test_regression_audit"]["passed"]):
        raise ValueError("Preference-tuned candidate did not pass regression checks")
    model,vocab=load_router(run/f"{stage}.pt")
    app=FastAPI(title="Care Navigation Eval Lab",description="Local demo using fictional scenarios only")

    @app.get("/health")
    def health():
        return {"status":"ready","stage":stage,"mode":"synthetic_demo"}

    @app.post("/route")
    @torch.inference_mode()
    def route(request: Request):
        raw=ACTIONS[model(encode([request.conversation],vocab)).argmax(-1).item()]
        action=guard(request.conversation,raw)
        return {"action":action,"response":REPLIES[action],"stage":stage,"guard_applied":action!=raw}

    return app

if __name__=="__main__":
    import uvicorn
    cli=argparse.ArgumentParser()
    cli.add_argument("--run",type=Path,default=Path("runs/demo"))
    cli.add_argument("--port",type=int,default=8000)
    opts=cli.parse_args()
    uvicorn.run(create_app(opts.run),host="127.0.0.1",port=opts.port,access_log=False)
