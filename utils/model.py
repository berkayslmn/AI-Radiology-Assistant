import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from utils.dataset import NUM_CLASSES

DROPOUT_P = 0.5


def build_densenet121(pretrained: bool = False) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(num_ftrs, NUM_CLASSES),
    )
    return model


def load_trained_densenet121(weights_path: str, device: torch.device) -> nn.Module:
    model = build_densenet121(pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def predict_with_tta(model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    flipped_tensor = torch.flip(input_tensor, dims=[3])

    with torch.no_grad():
        probs_original = torch.sigmoid(model(input_tensor)[0])
        probs_flipped = torch.sigmoid(model(flipped_tensor)[0])

    return ((probs_original + probs_flipped) / 2).cpu()


def get_pos_weights(df, class_names) -> torch.Tensor:
    pos_weights = []
    total_samples = len(df)
    for class_name in class_names:
        pos_count = df["Finding Labels"].fillna("").apply(
            lambda labels: class_name in str(labels).split("|")
        ).sum()
        neg_count = total_samples - pos_count
        weight = neg_count / (pos_count + 1e-7)
        pos_weights.append(weight)
    return torch.tensor(pos_weights, dtype=torch.float32)