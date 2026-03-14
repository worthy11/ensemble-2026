import os
import cv2
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from leader_model import LeaderModel
from dotter_model import DotterModel
from grid_processor import Gridder, Undistortion

# ---------------------------------------------------------------------------
# 1. Dedykowany Dataset dla The Leadera (W Locie)
# ---------------------------------------------------------------------------
class LeaderDataset(Dataset):
    """
    Dataset dedykowany dla uczenia sieci Leader.
    Wymaga surowych obrazow PNG oraz odpowiadajacych etykiet JSON.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Tylko bezpieczne pliki obrazów (omijamy śmieci macOS)
        self.image_files = sorted([f for f in os.listdir(data_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')])
    
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        basename = os.path.splitext(filename)[0]
        
        img_path = os.path.join(self.data_dir, filename)
        json_path = os.path.join(self.data_dir, f"{basename}.json")
        
        # Wczytujemy surowy obraz BGR
        raw_image = cv2.imread(img_path)
        if raw_image is None:
            raise ValueError(f"Nie znaleziono obrazu: {img_path}")
            
        # UWAGA: Specyfikacja twierdzi, że JSON dotyczy WYLACZNIE obrazu po transformacji
        # Dlatego wewnątrz Datasets tylko przygotowujemy dane. Gridder i Undistortion
        # uruchomimy na kartcie graficznej tuż przed pętlą dla ekstremalnej szybkosci.
        
        # Wczytywanie etykiet współrzędnych JSON z podziałem na odprowadzenia
        target_pts_dict = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                target_pts_dict = json.load(f)
                
        return raw_image, target_pts_dict, filename

# ---------------------------------------------------------------------------
# 2. Funkcja mieszanej straty (Loss) do ostrej segmentacji ścieżek
# ---------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        return bce_loss + dice_loss

# ---------------------------------------------------------------------------
# 3. Renderowanie maski Ground Truth z JSONa
# ---------------------------------------------------------------------------
def render_target_mask_from_json(target_pts_dict, w, h):
    """
    Rysuje perfekcyjną, białą maskę EKG 2D na podstawie wektorów z JSONa
    do wykorzystania jako The Ground Truth dla sieci Leader na wymiar (W, H).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Standard format: { "lead_name": [[x1, y1], [x2, y2], ...] }
    for lead_name, points in target_pts_dict.items():
        if not points: continue
        
        # Konwersja na format OpenCV int32
        pts = np.array(points, dtype=np.int32)
        
        # Rysowanie grubych krzywych na czarnym tle by U-Net latwiej to złapał (grubosć = 3 px)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=3)
        
    # Normalizacja 0.0 - 1.0 dla PyTorcha
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
    return mask_tensor

def pad_tensor_for_unet(tensor):
    h, w = tensor.shape[2], tensor.shape[3]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor, h, w, pad_h, pad_w

# ---------------------------------------------------------------------------
# 4. Główna Pętla Treningowa The Leadera
# ---------------------------------------------------------------------------
def train_leader(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TRAIN-LEADER] Rozpoczynam ekspresowy trening na urządzeniu: {device.type.upper()}")
    
    # Zamrożony pre-processing model the Dotter
    dotter = DotterModel(in_channels=3, out_channels=1).to(device)
    if os.path.exists("weights/dotter_weights.pth"):
        dotter.load_state_dict(torch.load("weights/dotter_weights.pth", map_location=device))
    dotter.eval() # Nie trenujemy Dottera w tym skrypcie!
    
    gridder = Gridder()
    undistorter = Undistortion()

    # Model docelowy - The Leader
    model = LeaderModel(in_channels=3, out_channels=1).to(device)
    if args.resume and os.path.exists(args.weights_out):
        model.load_state_dict(torch.load(args.weights_out, map_location=device))
        
    dataset = LeaderDataset(args.input_dir)
    # Wykorzystujemy maly Collate Fn dla list słownikow JSON
    def custom_collate(batch):
        return batch # Zwróc liste tupli, rozwiniemy ją sami dla bezpieczenstwa
        
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=0, collate_fn=custom_collate # Multithreading moze stwarzac problemy z JSON i OpenCV
    )
    
    criterion = DiceBCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoka Leadera {epoch}/{args.epochs}", unit="b") as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                batch_loss = 0.0
                valid_items = 0
                
                # Iteracja sekwencyjna po batchu (ze wzgledu na rozne wymiary z-unwarpowanych zdjec)
                for raw_img_bgr, json_pts, filename in batch:
                    # 1. FAZA ZAMROZONA: Unwarping za pomoca modelu Dotter i prawidel matematycznych
                    img_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)
                    img_t = torch.tensor((img_rgb.astype(np.float32)/255.0).transpose(2,0,1)).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        preds_grid = dotter(img_t).squeeze().cpu().numpy()
                        b_mask = (preds_grid > 0.5).astype(np.uint8) * 255
                    
                    g_mat = gridder.process(b_mask)
                    unwarped_bgr = undistorter.process(raw_img_bgr, g_mat)
                    
                    # 2. FAZA TRENOWALNA: The Leader i JSON maski w z-unwarpowanej przestrzeni
                    u_h, u_w = unwarped_bgr.shape[0], unwarped_bgr.shape[1]
                    
                    # Rysowanie celu 
                    target_mask_tensor = render_target_mask_from_json(json_pts, u_w, u_h).unsqueeze(0).to(device)
                    
                    # Przygotowanie prawidlowego wejscia u-netowego
                    unwarped_rgb = cv2.cvtColor(unwarped_bgr, cv2.COLOR_BGR2RGB)
                    unwarped_t = torch.tensor((unwarped_rgb.astype(np.float32)/255.0).transpose(2,0,1)).unsqueeze(0).to(device)
                    
                    padded_t, h, w, ph, pw = pad_tensor_for_unet(unwarped_t)
                    padded_target, _, _, _, _ = pad_tensor_for_unet(target_mask_tensor)
                    
                    # Predykcja
                    preds = model(padded_t)
                    loss = criterion(preds, padded_target)
                    
                    batch_loss += loss
                    valid_items += 1
                
                if valid_items > 0:
                    batch_loss = batch_loss / valid_items
                    batch_loss.backward()
                    optimizer.step()
                    epoch_loss += batch_loss.item()
                    pbar.set_postfix(**{"Loss": batch_loss.item()})
                    
                pbar.update(1)
                
        avg_loss = epoch_loss / len(train_loader)
        print(f"-> Zakończono {epoch} Epokę The Leadera. Średni Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), args.weights_out)
        
    print("\n[SUKCES] The Leader został nauczony w trybie Fast-Track!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/train")
    parser.add_argument("--batch_size", type=int, default=2) # 2 wystarczy na skrocony trening i oszczedzi VRAM mniejszych kart
    parser.add_argument("--epochs", type=int, default=5) # 5 epok to okolo 5-10 min oczekiwania dla pierwszych wynikow 
    parser.add_argument("--lr", type=float, default=2e-4) # Agresywny Learning rate, the leader uczy sie prosciej niz siatka
    parser.add_argument("--weights_out", type=str, default="weights/leader_weights.pth")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    train_leader(args)
