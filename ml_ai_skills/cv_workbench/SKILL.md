# CV Workbench — Advanced Level

**Level:** Advanced  
**Category:** AI/ML

## Overview
Computer vision tools and pipelines using PyTorch and torchvision for image classification, object detection, segmentation, augmentation, and preprocessing.

## Capabilities
- **Image Classification**: Simple CNN, ResNet, EfficientNet, ViT with transfer learning
- **Object Detection**: Faster R-CNN, SSD, RetinaNet, YOLO (via ultralytics)
- **Segmentation**: U-Net, FCN, DeepLabV3, Mask R-CNN
- **Augmentation**: MixUp, CutMix, AutoAugment, RandAugment
- **Preprocessing**: Standard ImageNet normalization, resize/crop pipelines

## Usage
```bash
python cv_workbench.py classify --architecture transfer_resnet --num-classes 10
python cv_workbench.py detect --model detector.pt --image test.jpg
python cv_workbench.py segment --model seg_model.pt --image test.jpg
python cv_workbench.py augment --input images/ --output augmented/ --transforms flip_h,rotation,zoom
python cv_workbench.py preprocess --input images/ --size 224,224
```

## Key Configurations
- **Normalization**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Augmentation**: AutoAugment, RandAugment, MixUp, CutMix
- **Transfer Learning**: torchvision pretrained models + replace classifier head
