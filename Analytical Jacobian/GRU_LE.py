# ================================================================
#  gru_smnist_lyap_repro.py  –  exact SMNIST-GRU pipeline (Vogt+24)
# ================================================================
"""
Train a 1-layer GRU on *row-wise* Sequential-MNIST (SMNIST) and compute
its full Lyapunov spectrum, faithfully matching the protocol in
Vogt et al., “Lyapunov-Guided Representation …” (arXiv:2204.04876).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse, math, os, random, pathlib, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# Data utilities
def get_loaders(batch=128, root="./torch_datasets"):
    tfm = transforms.ToTensor()
    root = pathlib.Path(root).expanduser()
    ds_tr = MNIST(root, train=True,  download=True, transform=tfm)
    ds_te = MNIST(root, train=False, download=True, transform=tfm)
    n_tr  = int(0.8 * len(ds_tr))               # 80 % train / 20 % val
    ds_tr, ds_va = random_split(ds_tr, [n_tr, len(ds_tr) - n_tr])
    def mk(ds, shuf): return DataLoader(ds, batch, shuffle=shuf, drop_last=True)
    return mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)


# Model
class GRUSMNIST(nn.Module):
    def __init__(self, hidden=64, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.gru    = nn.GRU(28, hidden, batch_first=True)   # 28 × 28 → 28-step row stream
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden, 10)

    def forward(self, x):
        B = x.size(0)
        seq = x.view(B, 28, 28)             # row-wise unfold
        h0  = torch.zeros(1, B, self.hidden, device=x.device, dtype=x.dtype)
        y, _ = self.gru(seq, h0)
        y    = self.drop(y)                 # dropout matches repo
        return self.fc(y[:, -1])


# Initialization
def init_edge_of_chaos(model: nn.Module) -> float:
    p = torch.empty(1).uniform_(0.1, 30).item()
    # 1 ⟶ small orthogonal weights
    nn.init.orthogonal_(model.gru.weight_ih_l0)        # input -> hidden
    nn.init.orthogonal_(model.gru.weight_hh_l0)        # hidden -> hidden
    nn.init.zeros_(model.gru.bias_ih_l0)               # keep IH biases 0

    H = model.hidden
    with torch.no_grad():                              # 2 -> edge-of-chaos bias
        model.gru.bias_hh_l0[0*H:1*H] = -p             # reset-gate r
        model.gru.bias_hh_l0[1*H:2*H] =  p             # update-gate z
        model.gru.bias_hh_l0[2*H:3*H].zero_()          # candidate-n
    return p


# Training loop
def run_epoch(net, loader, crit, opt, train, device, desc):
    net.train(train)
    tot_loss = 0.0; correct = 0; seen = 0
    for X, y in tqdm(loader, desc=desc, leave=False):
        X, y = X.to(device), y.to(device)
        if train: opt.zero_grad()
        logit = net(X)
        loss  = crit(logit, y)
        if train:
            loss.backward(); clip_grad_norm_(net.parameters(), 5.0); opt.step()
        tot_loss += loss.item() * X.size(0)
        correct  += (logit.argmax(1) == y).sum().item()
        seen     += X.size(0)
    return tot_loss / seen, correct / seen


# Analytic Jacobian  (no batch dimension)
def gru_jacobian(cell: nn.GRUCell,
                 x_t: torch.Tensor,          # shape (28,)
                 h_prev: torch.Tensor):      # shape (H,)
    """
    Return J = ∂h_t / ∂h_{t-1} for a single GRUCell step.
    Shapes:  x_t (28,)   h_prev (H,)   ->  J (H,H)
    """
    # weight splits (PyTorch gate order: r, z, n)
    Wi_r, Wi_z, Wi_n = cell.weight_ih.chunk(3, 0)
    Wh_r, Wh_z, Wh_n = cell.weight_hh.chunk(3, 0)
    bi_r, bi_z, bi_n = cell.bias_ih.chunk(3, 0)
    bh_r, bh_z, bh_n = cell.bias_hh.chunk(3, 0)

    # forward pass
    r = torch.sigmoid(x_t @ Wi_r.T + h_prev @ Wh_r.T + bi_r + bh_r)  # (H,)
    z = torch.sigmoid(x_t @ Wi_z.T + h_prev @ Wh_z.T + bi_z + bh_z)  # (H,)
    n_input = x_t @ Wi_n.T + r * (h_prev @ Wh_n.T + bh_n) + bi_n
    n = torch.tanh(n_input)                                          # (H,)

    # useful diagonals
    diag  = lambda v: torch.diag_embed(v)        # (H,) -> (H,H)
    D_r   = diag(r * (1 - r))
    D_z   = diag(z * (1 - z))
    D_n   = diag(1 - n ** 2)

    # partials
    d_r_dh = D_r @ Wh_r            # (H,H)
    d_z_dh = D_z @ Wh_z
    d_n_dh = D_n @ (r.unsqueeze(1) * Wh_n)
    d_n_dr = D_n @ (h_prev @ Wh_n.T + bh_n)      # (H,)

    # total Jacobian
    J  = (1 - z).unsqueeze(1) * d_n_dh           # ∂[(1-z)⊙n] / ∂h
    J -= diag(n) @ d_z_dh                        #  −n ⊙ ∂z/∂h
    J += diag(z)                                 #  + z * I
    J += (1 - z).unsqueeze(1) * (d_n_dr.unsqueeze(1) * d_r_dh)  # r→n path

    return J                                     # (H,H)


# helper – make a GRUCell state-dict from a 1-layer GRU
def gru_to_cell_state(gru: nn.GRU) -> dict[str, torch.Tensor]:
    """Return a state_dict whose keys match nn.GRUCell."""
    sd = gru.state_dict()
    return {
        'weight_ih': sd['weight_ih_l0'],
        'weight_hh': sd['weight_hh_l0'],
        'bias_ih'  : sd['bias_ih_l0'],
        'bias_hh'  : sd['bias_hh_l0'],
    }


# Lyapunov spectrum  (batch-free, uses Jacobian above)
@torch.no_grad()
def lyap_spectrum(model,
                  seq: torch.Tensor,   # (T, 28)
                  *, warm: int = 500) -> torch.Tensor:
    """
    Full Lyapunov spectrum of a trained 1-layer GRU `model`
    on input sequence `seq` (T,28).  Returns tensor (H,).
    """
    H   = model.hidden
    dev = seq.device
    dty = seq.dtype

    # --- matching GRUCell ---------------------------------------------
    cell = nn.GRUCell(28, H, device=dev, dtype=dty)
    cell.load_state_dict({
        'weight_ih': model.gru.weight_ih_l0,
        'weight_hh': model.gru.weight_hh_l0,
        'bias_ih'  : model.gru.bias_ih_l0,
        'bias_hh'  : model.gru.bias_hh_l0,
    })

    # --- containers ----------------------------------------------------
    h   = torch.zeros(H, device=dev, dtype=dty)     # (H,)
    Q   = torch.eye(H, device=dev, dtype=dty)       # (H,H)
    le_sum = torch.zeros(H, device=dev, dtype=dty)
    steps  = 0

    # --- warm-up -------------------------------------------------------
    for t in range(warm):
        h = cell(seq[t], h)

    # --- QR loop with progress bar ------------------------------------
    for t in tqdm(range(warm, seq.size(0)), desc="Lyap-QR", leave=False):
        J = gru_jacobian(cell, seq[t], h)           # (H,H)
        Q, R = torch.linalg.qr(J @ Q)               # reduced QR
        le_sum += torch.log(torch.abs(torch.diagonal(R)))
        h = cell(seq[t], h)
        steps += 1

    return le_sum / steps       # (H,)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--batch',  type=int, default=128)
    ap.add_argument('--lr',     type=float, default=1e-2)
    ap.add_argument('--trials', type=int, default=100,
                    help='number of independent runs to average')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    trL, vaL, teL = get_loaders(args.batch)
    criterion = nn.CrossEntropyLoss()

    all_LE = []        # will collect one (H,) array per trial
    all_p  = []        # keep the p that was drawn each run

    for run in range(1, args.trials + 1):
        # build & init model
        net = GRUSMNIST(args.hidden).to(device)
        p   = init_edge_of_chaos(net)          # bias trick
        all_p.append(p)
        opt = optim.Adam(net.parameters(), lr=args.lr)

        print(f"\n=== Trial {run}/{args.trials}  (p = {p:.3f}) ===")

        # train
        for ep in range(1, args.epochs + 1):
            run_epoch(net, trL, criterion, opt, True,
                      device, f"[T{run}] {ep}/{args.epochs} train")
            _, va = run_epoch(net, vaL, criterion, opt, False,
                              device, f"[T{run}] {ep}/{args.epochs} val  ")
            print(f"  epoch {ep:02d}: val-acc = {va*100:5.2f}%")

        # prepare sequence for LE
        imgs, _ = next(iter(teL))
        seq = imgs[:15].to(device).view(-1, 28)      # (420,28)
        seq = seq.repeat(2, 1)                       # (840,28)

        # Lyapunov spectrum
        torch.set_default_dtype(torch.float64)
        net = net.double()
        LE = lyap_spectrum(net, seq, warm=500).cpu().numpy()
        all_LE.append(LE)

        # save individual run
        np.save(f"lyap_spectrum_T{run}.npy", LE)

        print(f"  λ₁ = {LE[0]:+.6f}   λ_H = {LE[-1]:+.6f}")

    # average over trials
    mean_LE = np.mean(all_LE, axis=0)
    np.save("lyap_spectrum_mean.npy", mean_LE)

    print("\n===  A v e r a g e  over 100 trials  ===")
    print(f"λ₁̄ = {mean_LE[0]:+.6f}   λ̄_H = {mean_LE[-1]:+.6f}")
    print(f"p values drawn: {', '.join(f'{x:.3f}' for x in all_p)}")

    # plot mean spectrum
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4.5,3.2))
    plt.plot(range(1, len(mean_LE)+1), mean_LE,
             marker='o', markersize=3, lw=1.5, label='mean of 100 trials')
    plt.axhline(0, color='black', lw=.8, ls='--')
    plt.xlabel('Exponent index  $i$')
    plt.ylabel(r'$\bar{\lambda}_i$')
    plt.title('Mean Lyapunov spectrum (100 trials)')
    plt.tight_layout()
    plt.savefig('lyap_spectrum_mean.png', dpi=300)
    print("Saved  lyap_spectrum_mean.npy  and  lyap_spectrum_mean.png")

if __name__ == '__main__':
    main()
