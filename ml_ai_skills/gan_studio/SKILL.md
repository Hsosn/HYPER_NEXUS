# GAN Studio — Advanced Level

**Level:** Advanced  
**Category:** AI/ML

## Overview
Generative Adversarial Network toolkit with real architecture designs, training patterns, and evaluation metrics for image generation.

## Architectures
- **DCGAN**: Deep Convolutional GAN with upsampling generator
- **WGAN-GP**: Wasserstein GAN with gradient penalty
- **CGAN**: Conditional GAN with label conditioning
- **CycleGAN**: Unpaired image-to-image translation
- **StyleGAN**: Style-based generator with AdaIN

## Evaluation Metrics
- **FID**: Fréchet Inception Distance
- **IS**: Inception Score
- **LPIPS**: Learned Perceptual Image Patch Similarity

## Usage
```bash
python gan_studio.py design --type dcgan --latent-dim 128 --image-size 64
python gan_studio.py design --type stylegan
python gan_studio.py train --config gan_config.json
python gan_studio.py generate --model generator.pt --num-samples 10
python gan_studio.py evaluate
```

## Best Practices
- Optimizer: Adam(lr=2e-4, betas=(0.5, 0.999))
- Use spectral normalization for stability
- Label smoothing (real=0.9 instead of 1.0)
- Monitor D accuracy (~0.5 = good balance)
