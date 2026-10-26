"""E5: train LV segmentation on EchoNet and test whether Dice predicts EF failure at scale.

Replicates the CAMUS E1 decoupling on a second institution with 20x the patients, and
needs no quality labels (the CAMUS quality grader did not transfer; see e4a).

EF here is monoplane Simpson from the predicted mask, using the same disc construction the
tracings themselves encode. Our reconstruction of EchoNet's own EF from expert tracings has
r = 0.971 / MAE 1.05 EF points, so measured error is dominated by segmentation, not by the
EF computation.
"""
import sys, argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unet import UNet, dice_ce_loss, dice_per_class

R = Path(__file__).resolve().parents[1]; dev="cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="e5_echonet")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
out = R/"results"/a.tag; out.mkdir(parents=True, exist_ok=True)
(out/"config.json").write_text(json.dumps(vars(a), indent=2))

C = R/"data/cache_echonet"
imgs = np.load(C/"images_112.npy", mmap_mode="r")
msks = np.load(C/"masks_112.npy", mmap_mode="r")
idx  = pd.read_csv(C/"index_112.csv")

class DS(Dataset):
    def __init__(s, split, aug=False):
        s.pos = np.asarray(idx.index[idx.split==split]); s.aug=aug
    def __len__(s): return len(s.pos)
    def __getitem__(s, i):
        j = int(s.pos[i])
        x = torch.from_numpy(np.array(imgs[j], dtype=np.float32))
        y = torch.from_numpy(np.array(msks[j], dtype=np.int64))
        if s.aug:
            rng = np.random.default_rng(a.seed*7919+i)
            k1,k2 = int(rng.integers(-6,7)), int(rng.integers(-6,7))
            x = torch.roll(x,(k1,k2),(0,1)); y = torch.roll(y,(k1,k2),(0,1))
        return x[None], y, j

tl = DataLoader(DS("TRAIN",True), batch_size=a.batch_size, shuffle=True,
                num_workers=8, pin_memory=True, persistent_workers=True)
vl = DataLoader(DS("VAL"), batch_size=128, num_workers=4)
sl = DataLoader(DS("TEST"), batch_size=128, num_workers=4)
print(f"[E5] train {len(tl.dataset)} val {len(vl.dataset)} test {len(sl.dataset)} frames", flush=True)

model = UNet(1, 2, width=a.width).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
scaler = torch.cuda.amp.GradScaler()

def evaluate(loader):
    model.eval(); rows=[]
    with torch.no_grad():
        for x,y,j in loader:
            x,y = x.to(dev,non_blocking=True), y.to(dev,non_blocking=True)
            p = model(x).argmax(1)
            d = dice_per_class(p, y, n_classes=2).cpu().numpy()[:,0]
            for i in range(len(d)):
                rows.append({"row": int(j[i]), "dice": float(d[i]),
                             "area_pred": float((p[i]==1).float().mean()),
                             "area_gt": float((y[i]==1).float().mean())})
    return pd.DataFrame(rows)

best=-1
for ep in range(1, a.epochs+1):
    model.train(); tot=0.
    for x,y,_ in tl:
        x,y = x.to(dev,non_blocking=True), y.to(dev,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(): loss = dice_ce_loss(model(x), y, n_classes=2)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot+=loss.item()*len(x)
    sch.step()
    if ep%5==0 or ep==a.epochs:
        v = evaluate(vl)["dice"].mean()
        if v>best: best=v; torch.save(model.state_dict(), out/"best.pt")
        print(f"  ep {ep:3d} loss {tot/len(tl.dataset):.4f}  val Dice {v:.4f}{'  *' if v==best else ''}", flush=True)

model.load_state_dict(torch.load(out/"best.pt"))
res = pd.concat([evaluate(sl).assign(split="TEST"), evaluate(vl).assign(split="VAL")])
res = res.join(idx, on="row", rsuffix="_i")
res.to_csv(out/"per_frame.csv", index=False)
print(f"\ntest Dice {res[res.split=='TEST'].dice.mean():.4f}  (published EchoNet ~0.92)")
print(f"-> {out}")
