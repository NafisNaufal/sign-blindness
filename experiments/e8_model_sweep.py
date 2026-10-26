"""R2 (roadmap item 3): does a HIGHER-DICE model give lower EF error?

The submitted paper measured decoupling within one model across patients, but its
recommendations implied the across-model claim. Training several models spanning a Dice
range settles which claim the data supports.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unet import UNet, dice_ce_loss, dice_per_class
R = Path(__file__).resolve().parents[1]; dev="cuda"
OUT = R/"results/e8_sweep"; OUT.mkdir(parents=True, exist_ok=True)

C = R/"data/cache_echonet"
imgs = np.load(C/"images_112.npy", mmap_mode="r")
msks = np.load(C/"masks_112.npy", mmap_mode="r")
idx  = pd.read_csv(C/"index_112.csv")

class DS(Dataset):
    def __init__(s, split, aug=False, seed=0):
        s.pos=np.asarray(idx.index[idx.split==split]); s.aug=aug; s.seed=seed
    def __len__(s): return len(s.pos)
    def __getitem__(s,i):
        j=int(s.pos[i])
        x=torch.from_numpy(np.array(imgs[j],dtype=np.float32))
        y=torch.from_numpy(np.array(msks[j],dtype=np.int64))
        if s.aug:
            rng=np.random.default_rng(s.seed*7919+i)
            k1,k2=int(rng.integers(-6,7)),int(rng.integers(-6,7))
            x=torch.roll(x,(k1,k2),(0,1)); y=torch.roll(y,(k1,k2),(0,1))
        return x[None],y,j

def disc_volume(mask, n=20):
    pts=np.argwhere(mask>0).astype(np.float64)
    if len(pts)<20: return np.nan
    c=pts.mean(0); u=np.linalg.svd(pts-c,full_matrices=False)[2][0]
    nn=np.array([-u[1],u[0]]); t=(pts-c)@u; s=(pts-c)@nn
    t0,t1=t.min(),t.max(); L=t1-t0
    if L<4: return np.nan
    e=np.linspace(t0,t1,n+1); d=np.zeros(n)
    for i in range(n):
        b=(t>=e[i])&(t<e[i+1])
        if b.any(): d[i]=s[b].max()-s[b].min()
    return float(np.pi/4*(d**2).sum()*(L/n))

# models chosen to span a Dice range: capacity and training length both varied
CONFIGS=[("tiny-6ep",6,8),("small-12ep",12,12),("mid-20ep",20,20),("full-35ep",35,32)]
rows=[]
for name,epochs,width in CONFIGS:
    torch.manual_seed(0); np.random.seed(0)
    tl=DataLoader(DS("TRAIN",True),batch_size=64,shuffle=True,num_workers=8,
                  pin_memory=True,persistent_workers=True)
    model=UNet(1,2,width=width).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    sc=torch.cuda.amp.GradScaler()
    for ep in range(epochs):
        model.train()
        for x,y,_ in tl:
            x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(): loss=dice_ce_loss(model(x),y,n_classes=2)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        sch.step()
    model.eval()
    ev=DataLoader(DS("TEST"),batch_size=128,num_workers=4)
    per={}
    with torch.no_grad():
        for x,y,j in ev:
            p=model(x.to(dev)).argmax(1).cpu().numpy(); g=y.numpy()
            for k in range(len(j)):
                jj=int(j[k]); r_=idx.iloc[jj]
                inter=((p[k]==1)&(g[k]==1)).sum(); den=(p[k]==1).sum()+(g[k]==1).sum()
                per.setdefault(r_.key,[]).append(
                    dict(dice=1.0 if den==0 else 2*inter/den,
                         vp=disc_volume(p[k]==1), vg=disc_volume(g[k]==1)))
    ds,es=[],[]
    for key,v in per.items():
        if len(v)!=2: continue
        vp=[a["vp"] for a in v]; vg=[a["vg"] for a in v]
        if not all(np.isfinite(vp+vg)): continue
        efp=(max(vp)-min(vp))/max(vp)*100; efg=(max(vg)-min(vg))/max(vg)*100
        ds.append(np.mean([a["dice"] for a in v])); es.append(abs(efp-efg))
    ds,es=np.array(ds),np.array(es)
    rows.append({"model":name,"epochs":epochs,"width":width,"n":len(ds),
                 "dice":ds.mean(),"ef_mae":es.mean(),"catastrophic":100*(es>10).mean()})
    print(f"  {name:12s} w={width:2d} ep={epochs:2d}  Dice {ds.mean():.4f}  "
          f"EF MAE {es.mean():5.2f}  catastrophic {100*(es>10).mean():5.1f}%", flush=True)
    del model; torch.cuda.empty_cache()

df=pd.DataFrame(rows); df.to_csv(OUT/"sweep.csv",index=False)
print("\n"+"="*74); print("R2  ACROSS-MODEL: does higher Dice mean lower EF error?"); print("="*74)
print(df.round(4).to_string(index=False))
from scipy.stats import pearsonr, spearmanr
if len(df)>2:
    r,p=pearsonr(df.dice,df.ef_mae); rho,pr=spearmanr(df.dice,df.ef_mae)
    print(f"\n  across models: Pearson r(Dice, EF MAE) = {r:+.3f} (p={p:.3f}); "
          f"Spearman {rho:+.3f}")
    print(f"  Dice range spanned: {df.dice.min():.4f} to {df.dice.max():.4f} "
          f"({df.dice.max()-df.dice.min():.4f})")
    print(f"  EF MAE range      : {df.ef_mae.min():.2f} to {df.ef_mae.max():.2f}")
    print(f"  --> across models, higher Dice {'DOES' if r<-0.5 else 'does NOT reliably'} "
          f"track lower EF error")
