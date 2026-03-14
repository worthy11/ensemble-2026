import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

class DotterModel(nn.Module):
    """
    Model Dotter: Architektura U-Net z gotowym koderem ResNet34 z biblioteki segmentation-models-pytorch.
    Korzysta z pre-trenowanych wag na ImageNet co znacznie przyspiesza i ułatwia trening
    oraz poprawia skuteczność segmentacji siatki milimetrowej EKG.
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=1):
        super(DotterModel, self).__init__()
        # Inicjalizacja modelu U-Net z biblioteki SMP
        self.model = smp.Unet(
            encoder_name=encoder_name,        # Wybór kodera (np. resnet34)
            encoder_weights=encoder_weights,  # Pre-trenowane wagi z ImageNet
            in_channels=in_channels,          # Zwykle 3 dla RGB
            classes=out_channels,             # Liczba kanałów wyjściowych maski (1 dla binarnej)
            activation='sigmoid'              # Od razu dodajemy sigmoid na wyjściu by uzyskać wartości [0, 1]
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Testowy kod by skontrolować wymiary
    x = torch.randn((1, 3, 256, 256)).cuda() if torch.cuda.is_available() else torch.randn((1, 3, 256, 256))
    model = DotterModel().cuda() if torch.cuda.is_available() else DotterModel()
    preds = model(x)
    print(preds.shape) # Spodziewane: (1, 1, 256, 256)
