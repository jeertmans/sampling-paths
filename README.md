# Generative Path Candidate Sampler for Faster Point-to-Point Ray Tracing

[![arXiv link][arxiv-badge]][arxiv-url]
[![Colab link][colab-badge]][colab-url]

This repository accompanies the paper [*Transform-Invariant Generative Ray Path Sampling for Efficient Radio Propagation Modeling*][arxiv-url] by Jérome Eertmans, Enrico Maria Vitucci, Vittorio Degli Esposti, Nicola Di Cicco, Laurent Jacques and Claude Oestges.

It provides:
- The source code for the model described in the paper, implemented in JAX, in [`src/sampling_paths`](src/sampling_paths), including a script to train and evaluate the model on synthetic data, in [`__main__.py`](src/sampling_paths/__main__.py).
- Pre-trained model weights, available at [this link](https://github.com/jeertmans/sampling-paths/releases/tag/npjwt2026).
- Tests files in [`tests/`](tests/) to verify the correctness of the implementation.
- A tutorial notebook, viewable [here](https://differt.rtfd.io/npjwt2026/notebooks/sampling-paths.html), demonstrating how to use the model for path sampling.

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

## Getting Help

For any question about the method or its implementation, make sure to first read the related [paper][arxiv-url].

If you want to report a bug in this library or the underlying algorithm, please open an issue on this [GitHub repository](https://github.com/jeertmans/sampling-paths/issues). If you want to request a new feature, please consider opening an issue on [DiffeRT's GitHub repository](https://github.com/jeertmans/DiffeRT) instead.

## Citing

If you use this library in your research, please cite our paper:

```bibtex
@misc{eertmans2026,
	title         = {Transform-Invariant Generative Ray Path Sampling for Efficient Radio Propagation Modeling},
	author        = {Jérome Eertmans and Enrico M. Vitucci and Vittorio Degli-Esposti and Nicola Di Cicco and Laurent Jacques and Claude Oestges},
	year          = 2026,
	url           = {https://arxiv.org/abs/2603.01655},
	eprint        = {2603.01655},
	archiveprefix = {arXiv},
	primaryclass  = {cs.LG}
}
```

[arxiv-badge]: https://img.shields.io/badge/arXiv-2603.01655-b31b1b.svg
[arxiv-url]: https://arxiv.org/abs/2603.01655
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-url]: https://colab.research.google.com/github/jeertmans/sampling-paths/blob/main/notebooks/tutorial.ipynb
