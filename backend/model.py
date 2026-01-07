from __future__ import annotations

from pathlib import Path
from typing import Tuple
import ssl
from urllib.error import URLError

import torch
from PIL import Image
from torchvision import models, transforms


class PneumoniaClassifier:
    """
    Wrapper around a ResNet-18 model fine-tuned for binary
    classification (Normal vs. Pneumonia).
    """

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.device = self._get_device()
        self.class_names = ["Normal", "Pneumonia"]
        self.model = self._load_model(checkpoint_path)
        self.preprocess = self._build_preprocess()

    @staticmethod
    def _get_device() -> torch.device:
        """
        Prefer Apple M1/M2 Metal (MPS) if available, then CUDA, else CPU.
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self, checkpoint_path: Path | None) -> torch.nn.Module:
        """
        Load a pre-trained ResNet-18 and adapt it for binary classification.
        Optionally load fine-tuned weights from a checkpoint path.
        """
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except (URLError, ssl.SSLCertVerificationError, Exception) as exc:
            # Fallback: load model without pretrained weights so the API can still start.
            model = models.resnet18(weights=None)

        num_features = model.fc.in_features
        # Add dropout for regularization before final layer
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(num_features, 2),
        )

        if checkpoint_path is not None and checkpoint_path.is_file():
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # Allow for checkpoints that may contain additional keys
            missing, unexpected = model.load_state_dict(
                state_dict, strict=False
            )
            if missing or unexpected:
                # In a real system, you might log this instead of printing
                print(
                    f"Loaded checkpoint with missing={missing}, unexpected={unexpected}"
                )

        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _build_preprocess() -> transforms.Compose:
        """
        Build torchvision transforms for chest X-ray preprocessing.
        These settings are reasonable defaults and can be adjusted
        when you fine-tune the model.
        """
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        Run inference on a single PIL image and return:
        - predicted class label ("Normal" or "Pneumonia")
        - confidence score (float in [0, 1])
        """
        with torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, idx = torch.max(probs, dim=0)

        label = self.class_names[int(idx)]
        return label, float(confidence.item())

    def predict_from_path(self, image_path: Path) -> Tuple[str, float]:
        """
        Convenience method to run prediction given an image file path.
        """
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)


def load_default_model() -> PneumoniaClassifier:
    """
    Factory to load the default classifier instance.
    Loads trained pneumonia_model.pth if available, otherwise uses pretrained weights.
    """
    # Try to load the trained model weights
    checkpoint_path = Path(__file__).parent / "pneumonia_model.pth"
    if not checkpoint_path.is_file():
        checkpoint_path = None
        print(
            "Warning: pneumonia_model.pth not found. Using pretrained weights. "
            "Run train_model.py to train a custom model."
        )
    else:
        print(f"Loading trained model from {checkpoint_path}")
    
    return PneumoniaClassifier(checkpoint_path=checkpoint_path)

