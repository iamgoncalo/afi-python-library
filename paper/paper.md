---
title: 'AFI: A Python Library for Computing Freedom as the Ratio of Perception to Distortion'
tags:
  - Python
  - complex systems
  - swarm intelligence
  - freedom
  - navigability
  - exploration-exploitation
  - gradient dynamics
authors:
  - name: Gonçalo Melo
    orcid: 0009-0008-6255-7724
    affiliation: 1
affiliations:
  - name: Independent Researcher, Porto, Portugal
    index: 1
date: 3 March 2026
bibliography: paper.bib
---

# Summary

`afi` is a Python library that computes Freedom — the structural availability of paths — through the ratio law $F = P/D$, where $P$ (Perception) is a system's differentiation capacity and $D$ (Distortion) is its resistance to traversal. The library provides modules for computing $P$ and $D$ from domain-specific inputs (swarm intelligence algorithms, sensor networks, weighted graphs), composing multi-factor Distortion both multiplicatively and additively, detecting exploration–exploitation transitions via the coefficient of variation of the Freedom field, simulating agent movement under the gradient law $\dot{x} = -P(x) \cdot \nabla D(x)$, and evaluating convergence bounds. All computations are built on NumPy and SciPy, with built-in validation routines that compare $F = P/D$ against four alternative models and test noise robustness.

# Statement of Need

Swarm intelligence research has produced over 200 algorithms [@wang2025swarm], each with its own parameters, update rules, and convergence criteria. Researchers studying these algorithms lack a shared, algorithm-agnostic metric for comparing structural dynamics — particularly the exploration–exploitation transition, which every SI algorithm exhibits but each measures differently [@asghari2024resource].

No existing Python package provides this. Libraries like `pymoo` [@blank2020pymoo] and `mealpy` [@van2023mealpy] implement optimization algorithms but do not expose the underlying structural dynamics that govern convergence. `SALib` [@herman2017salib] provides sensitivity analysis but not the domain-specific Freedom computation. `NetworkX` [@hagberg2008networkx] handles graph topology but not navigability as a ratio of perception to resistance.

`afi` fills this gap by implementing the ratio law $F = P/D$ from the Architecture of Freedom Intelligence framework [@melo2026afi] as a computational toolkit. It enables researchers to: (1) compute Freedom from domain-specific Perception and Distortion measurements using factory functions for ACO, PSO, and sensor networks; (2) detect exploration–exploitation transitions with the universal quantifier $E(t) = \sigma[F] / \mu[F]$; (3) compare multiplicative versus additive Distortion composition empirically; (4) validate all results against alternative models ($F = P - D$, $F = P \times D$, $F = P^2/D$, random baseline) in a single function call.

The library is designed for researchers in swarm intelligence, complex systems, building performance simulation, and anyone studying navigability, optimization landscapes, or adaptive systems. It prioritizes correctness (40 tests covering mathematical properties including scale-covariance and monotonicity), interpretability (every result decomposes into $\Delta P$ and $\Delta D$ contributions), and honest comparison (alternative models are always tested alongside).

# Comparison to Related Packages

| Package | What it does | What `afi` adds |
|---------|-------------|-----------------|
| `pymoo` | Multi-objective optimization | Algorithm-agnostic Freedom metric; E(t) transition detection |
| `mealpy` | SI algorithm implementations | Structural dynamics analysis, not just optimization output |
| `SALib` | Sensitivity analysis | Domain-specific P and D computation; gradient law simulation |
| `NetworkX` | Graph algorithms | Navigability as F = P/D, not just shortest path |

# Acknowledgements

This work was supported by the Portuguese Foundation for Science and Technology (FCT) through Project 2025.00020.AIVLAB.DEUCALION, providing access to the Deucalion supercomputer at the National Advanced Computing Centre (MACC), Guimarães, Portugal.

During the preparation of this work, the author used Claude (Anthropic) for literature search assistance, code development, mathematical verification, and manuscript preparation. The author reviewed and edited all content and takes full responsibility for the content of the publication.

# References
