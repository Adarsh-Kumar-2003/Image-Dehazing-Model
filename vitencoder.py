

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from transformers import ViTModel, ViTMAEForPreTraining
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


transform_rgb = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


def extract_7_channels(img_rgb):
    V = torch.max(img_rgb, dim=0, keepdim=True)[0]

    def dark_channel(img, patch):
        min_rgb, _ = torch.min(img, dim=0, keepdim=True)
        pad = patch // 2
        return -F.max_pool2d(-min_rgb, patch, 1, pad)

    scales = [3, 5, 7, 9, 11, 13]
    dcp_maps = [dark_channel(img_rgb, s) for s in scales]

    return torch.cat([V] + dcp_maps, dim=0)  



def load_models_with_freeze():
    encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    mae = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    decoder = mae.decoder

    for name, param in encoder.named_parameters():
        if "encoder.layer.10" in name or "encoder.layer.11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return encoder, decoder


proj7to3 = nn.Conv2d(7, 3, kernel_size=1)



def patches_to_image(patch_tokens, nH, nW, patch=16):
    B, N, flat = patch_tokens.shape
    C = 3
    P = patch

    patches = patch_tokens.view(B, N, C, P, P)
    patches = patches.view(B, nH, nW, C, P, P)

    img = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, C, nH * P, nW * P)
    return img



def forward_pass(encoder, decoder, haze7):
    rgb = proj7to3(haze7)                      
    enc_out = encoder(pixel_values=rgb).last_hidden_state  
    patch_pixels = decoder(enc_out)            

    B, N, _ = patch_pixels.shape
    side = int(N ** 0.5)

    pred = patches_to_image(patch_pixels, side, side)
    return pred



def loss_fn(pred, gt):
    return F.l1_loss(pred, gt)



def train_step(encoder, decoder, optimizer, haze7, clean):
    optimizer.zero_grad()

    pred = forward_pass(encoder, decoder, haze7)
    loss = loss_fn(pred, clean)

    loss.backward()
    optimizer.step()

    return loss.item()



def train_model(encoder, decoder, train_loader, epochs=5):

    params = list(proj7to3.parameters())

    for name, param in encoder.named_parameters():
        if param.requires_grad:
            params.append(param)

    params += list(decoder.parameters())

    optimizer = optim.Adam(params, lr=2e-4)

    for epoch in range(epochs):
        total_loss = 0

        for haze7, clean in train_loader:
            haze7 = haze7.cuda()
            clean = clean.cuda()

            loss = train_step(encoder, decoder, optimizer, haze7, clean)
            total_loss += loss

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")



def load_reside_train(root):
    haze_dir = os.path.join(root, "hazy")
    gt_dir = os.path.join(root, "GT")

    filenames = sorted(os.listdir(haze_dir))

    haze_list, clean_list = [], []

    for fname in filenames:
        haze_path = os.path.join(haze_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(gt_path):
            continue

        hazy_img = Image.open(haze_path).convert("RGB")
        clean_img = Image.open(gt_path).convert("RGB")

        hazy_rgb = transform_rgb(hazy_img)
        clean_rgb = transform_rgb(clean_img)

        haze7 = extract_7_channels(hazy_rgb)

        haze_list.append(haze7)
        clean_list.append(clean_rgb)

    haze_tensor = torch.stack(haze_list)
    clean_tensor = torch.stack(clean_list)

    return haze_tensor, clean_tensor


def get_train_loader(train_root, batch_size=2):
    haze, clean = load_reside_train(train_root)
    dataset = TensorDataset(haze, clean)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



def test_single_image(encoder, decoder, img_path, save_path="output.png"):
    img = Image.open(img_path).convert("RGB")
    rgb = transform_rgb(img)

    haze7 = extract_7_channels(rgb).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = forward_pass(encoder, decoder, haze7)

    out = pred.squeeze().permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    out = (out * 255).astype("uint8")

    Image.fromarray(out).save(save_path)
    print("Saved:", save_path)



if __name__ == "__main__":
    print("ViT")
    encoder, decoder = load_models_with_freeze()
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    proj7to3.cuda()

    print("pre processing training data")
    train_loader = get_train_loader("reside-6k/train", batch_size=2)

    print("training started")
    train_model(encoder, decoder, train_loader, epochs=15)

    print("training ended")
    torch.save(encoder.state_dict(), "encoder_trained.pth")
    torch.save(decoder.state_dict(), "decoder_trained.pth")
    torch.save(proj7to3.state_dict(), "proj7to3.pth")

    print("testing")
    test_single_image(
        encoder, decoder,
        "reside-6k/test/hazy/1.jpg",
        "clean_1_output.png"
    )
