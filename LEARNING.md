# Learning path

I am building this one small step at a time. Each step has one thing to
learn and one thing to check.

| Step | What I learn | Check it worked |
| --- | --- | --- |
| 1. Load images and labels | How a dataset class hands data to PyTorch | `scripts/inspect_sample.py` shows boxes on an image |
| 2. Run a pretrained model | What a detector outputs (boxes, labels, scores) | `scripts/run_baseline.py` writes predictions |
| 3. Score the model | IoU: how much a predicted box overlaps a true box | `pytest tests/test_evaluation.py` passes |
| 4. Fine-tune | Training only the last layers on aircraft | Loss goes down in `scripts/train.py` |
| 5. Compare | Baseline vs fine-tuned on the same test images | A table of matches in the README |

## Where I am

All 5 steps are done. On the test images, F1 went from 0.43 (original model)
to 0.64 (trained model). See the README for the table.

## Questions I still have

- Why does the model miss small aircraft?
- What confidence threshold should I use, and why?
