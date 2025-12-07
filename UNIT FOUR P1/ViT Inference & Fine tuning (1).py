import torch, timm, torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import torch.nn as nn, torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prep = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

data = datasets.CIFAR10("/content/cifar10", download=True, transform=prep)
idx_train = list(range(1000))
idx_val = list(range(1000, 1200))
train_set = Subset(data, idx_train)
val_set = Subset(data, idx_val)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

def run_model(net, epochs=5):
    net = net.to(device)
    optimz = optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        net.train()
        s = 0
        for imgs, lbls in tqdm(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = net(imgs)
            loss = loss_fn(preds, lbls)
            optimz.zero_grad()
            loss.backward()
            optimz.step()
            s += loss.item()
        print(f"Epoch {ep+1}: loss {s/len(train_loader):.4f}")

    net.eval()
    c, t = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = net(imgs).argmax(dim=1)
            c += (out == lbls).sum().item()
            t += lbls.size(0)

    score = c/t
    print("Val accuracy:", score)
    return score

cnn = torchvision.models.resnet18(pretrained=True)
cnn.fc = nn.Linear(cnn.fc.in_features, 10)

vit = timm.create_model("vit_base_patch16_224", pretrained=True)
vit.head = nn.Linear(vit.head.in_features, 10)

print("Running ResNet")
resnet_acc = run_model(cnn, epochs=3)

print("Running ViT")
vit_acc = run_model(vit, epochs=3)

print("Final:", resnet_acc, vit_acc)
