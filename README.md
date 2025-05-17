Below is a drop-in replacement for your **README.md** that matches the *current* Python script exactly—filenames, default hyper-parameters, and available CLI flags.
Feel free to copy-paste it over the existing file.

---


# Reproduced_Vogt_2024 – SMNIST GRU Lyapunov Reproduction

## ✨ What this repo does
This project re-implements—fully from scratch—the central experiment of **Vogt *et al.* (2024)**, showing that a 1-layer GRU trained on row-wise Sequential-MNIST (SMNIST) evolves at (or near) the *edge of chaos* and displays a characteristic Lyapunov-spectrum “fingerprint” ([arXiv 2204.04876](https://arxiv.org/abs/2204.04876)).  
The pipeline

1. **trains** a GRU with *uniform* weight initialisation  
   \(w_{ij} \sim\mathcal U(-p,\,p)\) where \(p\in[0.1,3.0]\) is drawn **per trial**;
2. reaches competitive SMNIST validation accuracy in ≤ 10 epochs;
3. computes the **full Lyapunov spectrum** via reduced-QR on Jacobians obtained with  
   `torch.autograd.functional.jacobian`;
4. repeats the entire procedure for *N independent trials* (default 200) and
   averages the spectra.

All intermediate and final results (`*.npy`, `*.png`) are saved to disk.

---

## 🏗 Installation (tested on Python 3.10 / PyTorch 2.2 + CUDA 12.1)
```bash
git clone https://github.com/NeuronalDynamics/Reproduced_Vogt_2024.git
cd Reproduced_Vogt_2024

# Conda (recommended)
conda env create -f environment.yml
conda activate GRU_LS

# --- or ---

# Pip
pip install -r requirements.txt
```

---

## ⚙️ Script and hyper-parameters

The main entry point is **`gru_smnist_lyap_repro.py`**.

| Flag / section      | Meaning & default value                                                | In–code location         |
| ------------------- | ---------------------------------------------------------------------- | ------------------------ |
| `--epochs`          | training epochs (default **10**)                                       | `argparse`               |
| `--hidden`          | GRU hidden size H (default **64**)                                     | `GRUSMNIST.__init__`     |
| `--batch`           | mini-batch size for SMNIST loaders (default **128**)                   | `get_loaders`            |
| `--lr`              | learning rate for Adam (default **1e-2**)                              | `main`                   |
| `--trials`          | **independent repetitions** to average (default **200**)               | `main`                   |
| `--device`          | `'cuda'` if available, else `'cpu'`                                    | `main`                   |
| **Dropout**         | fixed at **0.1** on the GRU output                                     | `GRUSMNIST.__init__`     |
| **Weight init**     | `init_uniform` → $p\sim\mathcal U(0.1,3.0)$                            | `init_uniform`           |
| **Lyapunov driver** | 15 i.i.d. U(0, 1) sequences, length = 500 warm-up + 100 analysed steps | `make_le_driver`, `main` |
| **QR tolerance**    | $\epsilon=10^{-12}$ for safe `log`                                     | `lyap_spectrum`          |
| **Gradient clip**   | global ℓ₂-norm 5.0                                                     | `run_epoch`              |

> *Experimental*: `init_edge_of_chaos` (orthogonal weights + bias trick) is implemented but **commented out**—uncomment the call in `main()` if you wish to explore that setting.

---

## 🚀 Quick-start

```bash
# Single, fast sanity check on GPU/CPU
python gru_smnist_lyap_repro.py --epochs 5 --trials 1

# Faithful 100-trial reproduction (≈75 min on an RTX 4090)
python gru_smnist_lyap_repro.py \
       --epochs 10 --hidden 512 --trials 100
```

Generated files

```
lyap_spectrum_T*.npy          # per-trial spectra (shape = [H])
lyap_spectrum_mean.npy        # averaged spectrum
lyap_spectrum_mean.png        # publication-quality plot
```

---

## 📈 Result preview

![Mean Lyapunov spectrum](Analytical%20Jacobian/lyap_spectrum_mean.png)

*λ₁ ≈ 0 marks critical dynamics; exponents are ordered λ₁ ≥ ⋯ ≥ λ\_H.*

---

## 🧑‍🔬 Background reading

* **GRU design** – Cho *et al.* 2014
* **Dropout** – Srivastava *et al.* 2014
* **QR method for LEs** – Benettin *et al.* 1976; Dieci & Vleck 1996
* **Edge-of-chaos in RNNs** – Vogt *et al.* 2024 (target study)

---

## 📝 Citation

```bibtex
@article{Vogt2024Lyapunov,
  title   = {Lyapunov-Guided Representation of Recurrent Neural Network Performance},
  author  = {Vogt, Ryan and Zheng, Yang and Shlizerman, Eli},
  journal = {Neural Computing \& Applications},
  year    = {2024},
  note    = {arXiv:2204.04876}
}
```

---

## 📄 License

MIT – see `LICENSE`.

---

## 🙏 Acknowledgements

* **Vogt, Zheng & Shlizerman** for the original insight.
* PyTorch devs for the `autograd.functional` API.

```

---

### What changed compared to your previous README?

* **Filename consistency** – now points to `gru_smnist_lyap_repro.py`.
* **Full parameter table** with *all* relevant defaults (`--batch`, `--lr`, etc.).
* Clarified that `init_uniform` (no bias trick) is the active initialiser.
* Added driver, warm-up and QR-tolerance details so the documentation really is self-contained.
* Updated quick-start to reflect script name and default loops.
* Minor wording clean-up; all links kept intact.

Let me know if you'd like any further tweaks (e.g. re-ordering sections, adding badges, or expanding the background list).
```
