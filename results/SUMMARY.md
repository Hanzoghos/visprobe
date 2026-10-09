# Experiment Results Summary

Protocol: macro (equal-weight) average over three datasets; VQAv2 uses a seeded (42) random 1/3 test subset;
fine-tuning = 100k mixed examples (33,333 per dataset), 1 epoch, LoRA r=8.
Metrics: acc = accuracy; ECE = expected calibration error (lower better); Brier = Brier score (lower better).

## Per-model baseline / after

| Model | Stage | SNLI-VE acc/ECE/Brier | VQAv2 acc/ECE/Brier | Visual7W acc/ECE/Brier | Macro acc |
|---|---|---|---|---|---|
| Qwen3-VL-2B | baseline | 0.6918/0.0608/0.5390 | 0.6618/0.0909/0.5506 | 0.8512/0.0492/0.2387 | 0.7350 |
| Qwen3-VL-2B | after | 0.8998/0.0365/0.1676 | 0.8558/0.0524/0.2337 | 0.8875/0.0379/0.1758 | 0.8810 |
| Qwen3-VL-8B | baseline | 0.8474/0.0218/0.2775 | 0.8504/0.0250/0.2665 | 0.8802/0.0236/0.2082 | 0.8593 |
| Qwen3-VL-8B | after | 0.9160/0.0366/0.1383 | 0.8793/0.0350/0.2028 | 0.9171/0.0257/0.1340 | 0.9041 |
| Molmo2-4B | baseline | 0.8720/0.0551/0.2041 | 0.8671/0.1283/0.1890 | 0.8774/0.0793/0.1784 | 0.8722 |
| Molmo2-4B | after | 0.9096/0.0369/0.1486 | 0.8867/0.0343/0.1897 | 0.9159/0.0284/0.1369 | 0.9041 |
| Qwen2.5-VL-7B (baseline only) | baseline | 0.7388/0.1243/0.3770 | 0.8253/0.1906/0.2388 | 0.8823/0.0842/0.1711 | 0.8155 |
| Qwen2.5-VL-7B (baseline only) | after | — | — | — | — |

## Fine-tuning gains (after−baseline, macro average; models with complete pairs only)

| Model | baseline macro acc | after macro acc | Δacc | ΔECE | ΔBrier |
|---|---|---|---|---|---|
| Qwen3-VL-2B | 0.7350 | 0.8810 | +0.146 | -0.025 | -0.250 |
| Qwen3-VL-8B | 0.8593 | 0.9041 | +0.045 | +0.009 | -0.092 |
| Molmo2-4B | 0.8722 | 0.9041 | +0.032 | -0.054 | -0.032 |

Notes: Qwen2.5-VL-3B is retired (training-stability incident; baseline retained). 
Qwen2.5-VL-7B has baseline only at release time; after-results will be appended when fine-tuning completes.
