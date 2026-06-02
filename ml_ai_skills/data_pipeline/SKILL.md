# Data Pipeline — Advanced Level

**Level:** Advanced  
**Category:** AI/ML

## Overview
End-to-end data processing for ML with PyTorch DataLoader creation, preprocessing, feature engineering, and train/val/test splitting.

## Capabilities
- **Preprocessing**: Missing values (median/mode), outlier detection (IQR), categorical encoding
- **Feature Engineering**: Statistical features, polynomial features, ratio features, log transforms
- **Data Splitting**: Stratified train/val/test with configurable ratios
- **DataLoader**: PyTorch-native DataLoader with performance tuning tips

## Usage
```bash
python data_pipeline.py preprocess --input data.csv --clip-outliers --encode-categorical
python data_pipeline.py features --input data.csv --method auto
python data_pipeline.py split --input data.csv --test-size 0.2 --stratify label
python data_pipeline.py dataloader --input train.csv --batch-size 32 --num-workers 4
```

## Requirements
- pandas, numpy, scikit-learn
- PyTorch (torch.utils.data)
