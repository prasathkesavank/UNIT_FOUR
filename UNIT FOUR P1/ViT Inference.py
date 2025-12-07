import torch, timm, torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import torch.nn as nn, torch.optim as optim
from tqdm import tqdm

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prep = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

data = datasets.CIFAR10("/content/cifar10", download=True, transform=prep)
idx1 = list(range(1000))
idx2 = list(range(1000,1200))

tr = Subset(data, idx1)
vl = Subset(data, idx2)

tr_loader = DataLoader(tr, batch_size=32, shuffle=True)
vl_loader = DataLoader(vl, batch_size=64)

def train(m, ep=5):
    m = m.to(dev)
    opt = optim.Adam(m.parameters(), lr=1e-4)
    lossf = nn.CrossEntropyLoss()
    for i in range(ep):
        m.train()
        s = 0
        for x, y in tqdm(tr_loader):
            x, y = x.to(dev), y.to(dev)
            o = m(x)
            l = lossf(o, y)
            opt.zero_grad()
            l.backward()
            opt.step()
            s += l.item()
        print("epoch", i+1, s/len(tr_loader))
    m.eval()
    a = 0
    t = 0
    with torch.no_grad():
        for x, y in vl_loader:
            x, y = x.to(dev), y.to(dev)
            p = m(x).argmax(1)
            a += (p == y).sum().item()
            t += y.size(0)
    acc = a / t
    print("val:", acc)
    return acc

cnn = torchvision.models.resnet18(pretrained=True)
cnn.fc = nn.Linear(cnn.fc.in_features, 10)

vt = timm.create_model("vit_base_patch16_224", pretrained=True)
vt.head = nn.Linear(vt.head.in_features, 10)

print("resnet")
acc_cnn = train(cnn, ep=3)
print("vit")
acc_vit = train(vt, ep=3)
print("resnet:", acc_cnn, "vit:", acc_vit)
