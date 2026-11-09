# Care Navigation Eval Lab

A local ML work sample for a **fictional care-navigation agent**. It trains a small PyTorch Transformer to pick one of six structured actions. The API turns an action into a fixed, non-clinical template. It uses no patient records and does not provide medical advice or perform actual human handoffs.

## What it tests

The dataset contains 282 fictional conversation snippets across scheduling, practical resources, clarification, clinical review, urgent review, and privacy review. The train, validation, and test sets use different prompt families (216 / 18 / 48 snippets), and the vocabulary is built from training text only. The dataset audit checks duplicate prompts, labels, and split family isolation. All conversations are authored fixtures; none were adapted from patient interactions.

1. Warm-start the approximately 22K-parameter causal Transformer on routine intents.
2. Freeze the backbone and supervised-fine-tune the 184 trainable parameters in a rank-4 LoRA output head.
3. Apply direct preference optimization (DPO) to chosen/rejected safety actions, using the frozen supervised checkpoint as the reference. This is a one-action-token completion, not open-ended text generation.
4. Compare versioned checkpoints with raw and rule-assisted metrics. A release gate checks safety routing and regressions on ordinary navigation. The API serves the selected checkpoint and accepts only explicit synthetic-demo requests.

## Reproduce

Use Python 3.12. No model downloads or GPU are required.

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python train.py --seed 23 --out runs/demo
python -m unittest discover -s tests -v
python serve.py --run runs/demo
```

The run exports base, SFT, and DPO checkpoints, a JSONL loss trace, a fixture fingerprint, and case-level failures in `runs/demo/report.json`. Open the FastAPI docs at http://127.0.0.1:8000/docs or send:

```bash
curl -s http://127.0.0.1:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"mode":"synthetic_demo","conversation":"For this fictional scenario, I need a ride to the clinic."}'
```

## Reference run (CPU, seed 23)

All values are from 48 held-out fixture windows, across 12 scenario families. Safety and routine navigation have 24 snippets each.

| Checkpoint | Raw action accuracy | Raw safety routing | Rule-assisted safety routing | Rule-assisted routine navigation |
| --- | ---: | ---: | ---: | ---: |
| Routine base | 16.7% | 0.0% | 100.0% | 33.3% |
| LoRA SFT | 22.9% | 12.5% | 100.0% | 33.3% |
| LoRA SFT + DPO | 14.6% | 29.2% | 100.0% | 0.0% |

The DPO candidate improved raw safety routing but regressed on ordinary requests. The gate blocked that candidate and selected SFT for the local demo. The rule-assisted score reflects hand-written cue patterns in a small fictional suite; it does not measure clinical safety, adversarial robustness, or helpfulness with actual users. The fixture variants within a scenario family are correlated, and all raw scores reveal the limits of this model and vocabulary. See [reference report](examples/reference_report.json) and [Hugging Face’s DPO documentation](https://huggingface.co/docs/trl/dpo_trainer) for the reference-based objective. No prompts or responses are persisted by the API.
