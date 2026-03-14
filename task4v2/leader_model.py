import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

class LeaderModel(nn.Module):
    """
    Model Leader: Architektura U-Net z ResBlocks (resnet34).
    Przyjmuje wyprostowany obraz EKG i generuje maskę śladów sygnału (linii odprowadzeń),
    bez tła krzywych i wydruku papieru.
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=1):
        super(LeaderModel, self).__init__()
        # Inicjalizacja modelu U-Net z SMP z modułami ResBlock w backendzie kodera
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation='sigmoid'
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Szybki test zgodności układu wymiarów
    x = torch.randn((1, 3, 512, 512)).cuda() if torch.cuda.is_available() else torch.randn((1, 3, 512, 512))
    model = LeaderModel().cuda() if torch.cuda.is_available() else LeaderModel()
    preds = model(x)
    print(preds.shape) # Spodziewane: (1, 1, 512, 512)
