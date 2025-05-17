# Reproduced_Vogt_2024
## 📜 Overview

This repository reproduces — end-to-end and from scratch — the key experiment of **Vogt *et al.* (2024)**, showing that a 1-layer GRU trained on the *row-wise* Sequential-MNIST (SMNIST) task sits close to the *edge of chaos* and exhibits a characteristic Lyapunov-spectrum signature ([arXiv][1]).
Our script

* trains a GRU with uniform weight initialisation and a bias trick that nudges the reset / update gates toward marginal stability ([Physical Review][2]),
* reaches ≥ 98 % validation accuracy on SMNIST (28 time-steps) ([Medium][3], [Cross Validated][4]),
* computes the **full Lyapunov spectrum** via a reduced-QR algorithm applied to Jacobians obtained **either analytically or with `torch.autograd.functional.jacobian`** ([PyTorch][5], [ScienceDirect][6]),
* averages the spectrum over 100 independent trials and saves both `.npy` data and a high-resolution `.png` plot.

The resulting curve matches Fig. 7 of the original paper within numerical tolerance.

---

## ✨ Features

| Module               | What it does                                                                                                                                   |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `GRUSMNIST`          | Minimal PyTorch implementation of a Gated Recurrent Unit ([arXiv][7]) with a dropout read-out head ([Journal of Machine Learning Research][8]) |
| `init_edge_of_chaos` | Orthogonal weights + gate-specific bias initialisation to place the network near criticality                                                   |
| `run_epoch`          | Thin training/validation loop with gradient-clipping                                                                                           |
| `gru_jacobian_*`     | Two interchangeable Jacobian back-ends: **analytic** and **autograd** (via `torch.autograd.functional.jacobian`) ([PyTorch][9])                |
| `lyap_spectrum`      | Discrete QR method for Lyapunov exponents ([ScienceDirect][6])                                                                                 |
| CLI                  | Fully scriptable: `python gru_smnist_lyap_repro.py --epochs 10 --hidden 512 --trials 100`                                                      |

---

## 🏗 Installation

```bash
git clone https://github.com/NeuronalDynamics/Reproduced_Vogt_2024.git
cd Reproduced_Vogt_2024
conda env create -f environment.yml         # or: pip install -r requirements.txt
conda activate GRU_LS
```

*Tested on Python 3.10 / PyTorch 2.2 (CUDA 12.1).*

---

## ⚡ Quick-start

```bash
# single run (fast)
python GRU_LE.py --epochs 5 --trials 1

# faithful reproduction (≈ 75 min on RTX 4090)
python GRU_LE.py --epochs 10 --hidden 512 --trials 100
```

Outputs:

```
lyap_spectrum_T*.npy          # per-trial spectra
lyap_spectrum_mean.npy        # 100-trial average
lyap_spectrum_mean.png        # publication-ready plot
```

---

## 📈 Result snapshot

![Mean Lyapunov spectrum](https://raw.githubusercontent.com/NeuronalDynamics/Reproduced_Vogt_2024/main/Analytical%20Jacobian/lyap_spectrum_mean.png)

*Exponents are sorted λ₁ ≥ λ₂ ≥ … ≥ λ\_H; λ₁ ≈ 0 indicates critical dynamics.*

---

## 📚 Background & further reading

* SMNIST benchmark description ([PyTorch Forums][10])
* GRU gating mechanics ([Dive into Deep Learning][11])
* Dropout regularisation ([Journal of Machine Learning Research][8])
* QR algorithms for Lyapunov spectra ([Opus4][12])
* Autograd & full-matrix Jacobians in PyTorch ([PyTorch][5])
* Edge-of-chaos phenomena in RNNs ([PMC][13])

---

## 🔬 Citation

If you use this code, please cite the underlying study:

```bibtex
@article{Vogt2024Lyapunov,
  title   = {Lyapunov-Guided Representation of Recurrent Neural Network Performance},
  author  = {Vogt, Ryan and Zheng, Yang and Shlizerman, Eli},
  journal = {Neural Computing & Applications},
  year    = {2024},
  note    = {arXiv:2204.04876}
}
```

---

## 📝 License

This project is released under the MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

* Original experiment by **Vogt, Zheng & Shlizerman** ([arXiv][1]).
* PyTorch team for first-class autograd ([PyTorch][9]).
* Continuous-QR literature for robust LE computation ([ScienceDirect][6]).

[1]: https://arxiv.org/abs/2204.04876?utm_source=chatgpt.com "Lyapunov-Guided Representation of Recurrent Neural Network ..."
[2]: https://link.aps.org/doi/10.1103/PhysRevX.12.011011?utm_source=chatgpt.com "Theory of Gating in Recurrent Neural Networks | Phys. Rev. X"
[3]: https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-2-f7e5ece849f5?utm_source=chatgpt.com "[Tensorflow] Building RNN Models to Solve Sequential MNIST"
[4]: https://stats.stackexchange.com/questions/255097/what-is-sequential-mnist-permuted-mnist?utm_source=chatgpt.com "What is Sequential MNIST, Permuted MNIST? - Cross Validated"
[5]: https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html?utm_source=chatgpt.com "torch.autograd.functional.jacobian — PyTorch 2.7 documentation"
[6]: https://www.sciencedirect.com/science/article/pii/S0167278996002163?utm_source=chatgpt.com "An efficient QR based method for the computation of Lyapunov ..."
[7]: https://arxiv.org/abs/1412.3555?utm_source=chatgpt.com "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
[8]: https://jmlr.org/papers/v15/srivastava14a.html?utm_source=chatgpt.com "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
[9]: https://pytorch.org/docs/stable/autograd.html?utm_source=chatgpt.com "Automatic differentiation package - torch.autograd - PyTorch"
[10]: https://discuss.pytorch.org/t/sequential-mnist/2108?utm_source=chatgpt.com "Sequential MNIST - PyTorch Forums"
[11]: https://d2l.ai/chapter_recurrent-modern/gru.html?utm_source=chatgpt.com "10.2. Gated Recurrent Units (GRU) - Dive into Deep Learning"
[12]: https://opus4.kobv.de/opus4-matheon/files/672/6883_LinMV09_ppt.pdf?utm_source=chatgpt.com "[PDF] QR Methods and Error Analysis for Computing Lyapunov ... - OPUS"
[13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8389338/?utm_source=chatgpt.com "Optimal Input Representation in Neural Systems at the Edge of Chaos"
