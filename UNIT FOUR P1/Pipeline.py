import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from realesrgan import RealESRGAN

class ImgSet(Dataset):
    def __init__(self, root, dim=256):
        self.dir = root
        self.files = sorted(os.listdir(root))
        self.fx = transforms.Compose([
            transforms.Resize((dim, dim)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = os.path.join(self.dir, self.files[idx])
        x = Image.open(p).convert("RGB")
        return self.fx(x), self.files[idx]

def upscale_batch(model, imgs):
    out = []
    for i in imgs:
        r = model(i.unsqueeze(0))[0]
        out.append(r)
    return torch.stack(out)

def chunk_map(x, k=3):
    b, c, h, w = x.shape
    s = []
    for i in range(0, h - k, k):
        for j in range(0, w - k, k):
            s.append(x[:, :, i:i+k, j:j+k])
    s = torch.stack(s, dim=1)
    b, n, c, h, w = s.shape
    return s.view(b, n, c*h*w)

def extract_vec(enc, x):
    with torch.no_grad():
        z = enc(x)
    return chunk_map(z)

def train_mem(mem):
    d = mem.reshape(-1, mem.shape[-1]).cpu().numpy()
    est = EmpiricalCovariance().fit(d)
    return est

def eval_dist(est, q):
    q = q.reshape(q.shape[0]*q.shape[1], -1).cpu().numpy()
    return est.mahalanobis(q)

def run_pipeline(folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sr = RealESRGAN(device, scale=4)
    sr.load_weights("RealESRGAN_x4.pth")

    data = ImgSet(folder, dim=128)
    loader = DataLoader(data, batch_size=4, shuffle=False)

    encoder = resnet50(pretrained=True).to(device)
    encoder = torch.nn.Sequential(*list(encoder.children())[:-2])

    bank = []

    for imgs, _ in tqdm(loader):
        imgs = imgs.to(device)
        hi = upscale_batch(sr, imgs)
        vec = extract_vec(encoder, hi)
        bank.append(vec.cpu())

    bank = torch.cat(bank, dim=0)
    model = train_mem(bank)
    return model, encoder, sr

def detect(folder, model, encoder, sr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = ImgSet(folder, dim=128)
    loader = DataLoader(data, batch_size=1, shuffle=False)

    out = {}

    for x, name in loader:
        x = x.to(device)
        hi = sr(x)[0].unsqueeze(0)
        v = extract_vec(encoder, hi)
        d = eval_dist(model, v)
        score = float(np.max(d))
        out[name[0]] = score

    return out

model, encoder, sr = run_pipeline("train_imgs")
scores = detect("test_imgs", model, encoder, sr)
print(scores)
