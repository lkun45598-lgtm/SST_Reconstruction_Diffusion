"""Training driver. modes:
  ostia_full    : complete native OSTIA, synthetic cloud holes, loss on hidden pixels (pretraining)
  ostia_masked  : OSTIA with an additional UNKNOWN cloud set A removed from the diffusion state
                  (validation of the masked-data objective: labels on A are never used)
  ostia_naive   : INVALID control: A pixels filled with the visible mean and used as labels
  jaxa          : genuine JAXA frames; true NaN pixels are unknown; hidden = genuine obs under a cloud pattern
"""
import argparse, json, os, time, torch
from ufno import UFNO
from edm import Denoiser, masked_loss, visible_mean
from data import Ostia, Jaxa, CloudLib, roles_complete, roles_masked

p = argparse.ArgumentParser()
p.add_argument('--mode', required=True, choices=['ostia_full', 'ostia_masked', 'ostia_naive', 'jaxa'])
p.add_argument('--out', required=True); p.add_argument('--init', default=None)
p.add_argument('--steps', type=int, default=6000); p.add_argument('--bs', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--seed', type=int, default=0)
p.add_argument('--scale', type=float, default=None); p.add_argument('--val-every', type=int, default=500)
p.add_argument('--ostia-years', nargs=2, type=int, default=None)
p.add_argument('--replay', type=float, default=0.0, help='jaxa mode: fraction of batches drawn from complete OSTIA')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
RANK = int(os.environ.get('RANK', 0)); WORLD = int(os.environ.get('WORLD_SIZE', 1)); LOCAL = int(os.environ.get('LOCAL_RANK', 0))
if WORLD > 1:
    import torch.distributed as dist
    torch.cuda.set_device(LOCAL); dist.init_process_group('nccl')
dev = f'cuda:{LOCAL}'; torch.manual_seed(a.seed + 1000 * RANK)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
log = open(f'{a.out}/train.log', 'a') if RANK == 0 else None
def P(*s):
    if RANK != 0: return
    m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()

lib = CloudLib(dev)
ostia = Ostia(dev, years=a.ostia_years) if (a.mode.startswith('ostia') or a.replay > 0) else None
jaxa = Jaxa(dev, lib) if a.mode == 'jaxa' else None

def get_batch(split, B):
    """-> x (kelvin, NaN allowed), V, H, E, geo"""
    if a.mode == 'jaxa' and not (split == 'train' and a.replay > 0 and torch.rand(1).item() < a.replay):
        x, ocean, g = jaxa.batch(B, split, 160)
        P_ = lib.crops(B, 160, 160)
        K = ~torch.isnan(x)
        V, H_, E = roles_masked(x, ocean, P_, A=~K)
    else:
        x, ocean, g = ostia.batch(B, split, 64)
        P_ = lib.crops(B, 160, 64)
        if a.mode == 'ostia_masked':
            A = lib.crops(B, 160, 64, frac=(0.1, 0.5))
            V, H_, E = roles_masked(x, ocean, P_, A=A)
        elif a.mode == 'ostia_naive':
            A = lib.crops(B, 160, 64, frac=(0.1, 0.5))
            V0, _, _ = roles_complete(x, ocean, P_)
            V = V0 & ~A
            fill = visible_mean(x, V)                 # pretend A is known: fill it and use it as label
            x = torch.where(A & ocean, fill.expand_as(x), x)
            H_, E = ocean & P_, ocean
        else:
            V, H_, E = roles_complete(x, ocean, P_)
    ok = (H_.float().sum((1, 2, 3)) >= 8) & (V.float().sum((1, 2, 3)) >= 8)
    if not ok.all():  # drop degenerate samples
        k = ok.nonzero().flatten()
        if len(k) == 0: return get_batch(split, B)
        x, V, H_, E = x[k], V[k], H_[k], E[k]
        g = dict(coords=g['coords'][k], spacing=g['spacing'][k], extent=g['extent'])
    return x, V, H_, E, g

# normalisation scale: std of (x - visible mean) over training data, /0.5 so that sigma_data = 0.5
if a.scale is None:
    assert WORLD == 1, 'pass --scale for multi-GPU runs'
    vals = []
    for _ in range(30):
        x, V, H_, E, g = get_batch('train', 32)
        mu = visible_mean(x, V); d = torch.nan_to_num(x - mu)[E & ~torch.isnan(x)]
        vals.append(d.float())
    a.scale = float(torch.cat(vals).std() / 0.5)
P('scale (K per normalised unit):', a.scale)
net = UFNO().to(dev); den = Denoiser(net).to(dev)
if a.init:
    ck = torch.load(a.init, map_location=dev); den.load_state_dict(ck['model']); P('init from', a.init)
cfg = dict(vars(a), width=48, mults=[1, 2, 2], modes=12, emb=128, sigma_data=0.5)
json.dump(cfg, open(f'{a.out}/config.json', 'w'), indent=1)
ddp = den
if WORLD > 1:
    from torch.nn.parallel import DistributedDataParallel as DDP
    ddp = DDP(den, device_ids=[LOCAL])
opt = torch.optim.AdamW(den.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (0.5 * (1 + __import__('math').cos(3.14159 * s / a.steps)) * 0.95 + 0.05))

@torch.no_grad()
def validate(n=8):
    den.eval(); tot = 0.0
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(123)
        for _ in range(n):
            x, V, H_, E, g = get_batch('val', a.bs)
            l, _ = masked_loss(den, x, V, H_, E, g, a.scale); tot += l.item()
    den.train(); return tot / n

P('mode', a.mode, 'params(M)', sum(q.numel() for q in net.parameters()) / 1e6)
t0 = time.time(); run = 0.0
for step in range(1, a.steps + 1):
    x, V, H_, E, g = get_batch('train', a.bs)
    loss, _ = masked_loss(ddp, x, V, H_, E, g, a.scale)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0); opt.step(); sched.step()
    run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
    if step % 100 == 0: P(f'step {step} loss {run:.4f} {(time.time()-t0)/step:.3f}s/it')
    if (step % a.val_every == 0 or step == a.steps) and RANK == 0:
        P(f'  val_loss@{step} {validate():.4f}')
        torch.save(dict(model=den.state_dict(), scale=a.scale, step=step, cfg=cfg), f'{a.out}/ckpt.pt')
P('done', time.time() - t0)
if WORLD > 1: dist.destroy_process_group()
