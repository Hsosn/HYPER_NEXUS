# Model Optimizer — Advanced Level

**Level:** Advanced  
**Category:** AI/ML

## Overview
Real executable PyTorch optimization toolkit with actual pruning, quantization, distillation, and hyperparameter search — not config generators.

## Capabilities
- **HyperparameterOptimizer**: Random/grid/Bayesian search with real training
- **Model Pruning**: Unstructured (global L1), structured (L2), iterative
- **Quantization**: Dynamic int8, Quantization-Aware Training (QAT)
- **Knowledge Distillation**: Teacher-student with soft targets (KL div)
- **LR Finder**: Real LR range test with training steps

## Usage
```bash
python model_optimizer.py tune --method bayesian --trials 50
python model_optimizer.py prune --model model.pt --sparsity 0.5
python model_optimizer.py quantize --model model.pt --type qat
python model_optimizer.py distill
```

## Key Classes
| Class | Description |
|-------|-------------|
| `HyperparameterOptimizer` | Grid/random search |
| `Distiller` | Knowledge distillation |
| `apply_unstructured_pruning()` | Global L1 pruning |
| `apply_structured_pruning()` | L2 row/column pruning |
| `find_optimal_lr()` | LR range test |
| `apply_dynamic_quantization()` | int8 quantization |
