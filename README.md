# Generative Path Candidate Sampler for Faster Point-to-Point Ray Tracing

[![arXiv link][arxiv-badge]][arxiv-url]
[![Colab link][colab-badge]][colab-url]

This repository accompanies the paper "[Transform-Invariant Generative Ray Path Sampling for Efficient Radio Propagation Modeling with Point-to-Point Ray Tracing][arxiv-url]" by Jérome Eertmans, Enrico Maria Vitucci, Vittorio Degli Esposti, Nicola Di Cicco, Claude Oestges and Laurent Jacques.

It provides:
- The source code for the model described in the paper, implemented in JAX, in [`src/sampling_paths`](src/sampling_paths), including a script to train and evaluate the model on synthetic data, in [`__main__.py`](src/sampling_paths/__main__.py).
- Pre-trained model weights, available at [this link](todo).
- Tests files in [`tests/`](tests/) to verify the correctness of the implementation.
- A tutorial notebook, available at [this link][colab-url], demonstrating how to use the model for path sampling.

## Installation

After cloning the repository, run:

```bash
pip install .
```

Alternatively, you can avoid manually cloning the repository by installing directly from GitHub:

```bash
pip install git+https://github.com/jeertmans/sampling-paths.git
```

## Usage

After installation, you can train and evaluate the model using:

```bash
train-path-sampler --help
```

## Getting help

For any question about the method or its implementation, make sure to first read the related [paper](todo).

If you want to report a bug in this library or the underlying algorithm, please open an issue on this [GitHub repository](https://github.com/jeertmans/fpt-jax/issues). If you want to request a new feature, please consider opening an issue on [DiffeRT's GitHub repository](https://github.com/jeertmans/DiffeRT) instead.

## Citing

If you use this library in your research, please cite our paper:

```bibtex
TODO
```

[arxiv-badge]: https://img.shields.io/badge/arXiv-2510.16172-b31b1b.svg
[arxiv-url]: https://arxiv.org/abs/2510.16172
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-url]: https://colab.research.google.com/github/jeertmans/sampling-paths/blob/main/tutorial.ipynb
