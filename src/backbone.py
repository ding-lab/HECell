"""Load a timm backbone, optionally with local weights."""
import timm
import torch


def load_backbone(name, weights, device, num_classes=0):
    model = timm.create_model(name, pretrained=(weights is None), num_classes=num_classes)
    if weights:
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=False)
    return model.to(device)
