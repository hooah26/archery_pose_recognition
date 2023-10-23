import os
import cv2
import json
from PIL import Image
import torch
import numpy as np
from datasets import load_metric
from torchvision.transforms import RandomRotation, Compose, RandomHorizontalFlip, RandomCrop, RandomPerspective, AutoAugment, AutoAugmentPolicy
from transformers import ViTFeatureExtractor

image_dir = "../images"
# Feature Extractor 사전 학습 모델 불러오기
model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
aug_func = Compose(
    [   #AutoAugment(AutoAugmentPolicy.IMAGENET),
         # RandomRotation(degrees=10),
         RandomHorizontalFlip(),
    ]
)
def transform(example_batch):
    inputs = feature_extractor([np.array(aug_func(Image.open(os.path.join(image_dir,x)))) for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)