---
title: 'afi: A Python Library for Freedom-Based Structural Analysis of Swarm Intelligence'
tags:
  - Python
  - swarm intelligence
  - exploration-exploitation
  - complex adaptive systems
  - structural analysis
  - navigability
authors:
  - name: Gonçalo Melo
    orcid: 0009-0008-6255-7724
    affiliation: 1
affiliations:
  - name: Independent Researcher, Porto, Portugal
    index: 1
date: 4 March 2026
bibliography: paper.bib
---

# Summary

`afi` is a Python library that computes Freedom $F = P/D$ — the ratio of Perception (differentiation capacity) to Distortion (structural resistance) — across adaptive systems. The library provides seven modules implementing the core equations of the Architecture of Freedom Intelligence (AFI) framework [@melo2026afi; @melo2026swarm]: `perception` computes $P = \log_2(N) \times T$ from distinguishable states and temporal depth; `distortion` implements both multiplicative ($D = R^\alpha \cdot O^\beta \cdot T^\gamma$) and additive ($D = \alpha R + \beta O + \gamma T$) composition forms; `freedom` computes $F = P/D$ and the intelligent regime variant $F = (P_\text{ext} \times P_\text{rec}) / (D_\text{ext} \times D_\text{int})$; `exploration` detects the exploration–exploitation transition via $E(t) = \sigma[F] / \mu[F]$; `gradient` implements the movement law $dx/dt = -P(x) \cdot \nabla D(x)$; `convergence` evaluates the bound $t_\text{conv} \leq \bar{D} \cdot \ln|S| / P_\text{mean}$; and `validation` compares five competing structural models and tests noise robustness. Built on NumPy and SciPy with 1,303 lines of library code and 40 passing tests, `afi` provides the first computational toolkit for analysing *why* swarm algorithms converge, not merely *whether* they do.

# Statement of Need

Swarm intelligence (SI) research produces hundreds of new bio-inspired optimisers annually [@wang2025swarm; @vanthieu2023mealpy; @chen2025iwsn]. Each algorithm — Ant Colony Optimisation, Particle Swarm Optimisation, Bacterial Foraging, and their many variants — uses idiosyncratic metrics to track convergence: pheromone concentration, inertia weight, chemotactic step size. This creates a fragmented landscape where comparing *structural dynamics* across algorithms is effectively impossible [@li2024cooperative; @vanthieu2024mafese]. Every researcher answers the same question differently: is my algorithm still exploring, or has it begun exploiting?

No existing Python library provides an algorithm-agnostic metric for detecting the exploration–exploitation transition. Libraries such as mealpy [@vanthieu2023mealpy] and NiaPy (Vrbančič et al., JOSS 2018) implement algorithms but offer no structural diagnosis of convergence dynamics. PySwarms (Miranda, JOSS 2018) tracks PSO-specific velocity histories. pymoo (Blank & Deb, IEEE Access 2020) addresses multi-objective Pareto fronts. SALib performs sensitivity analysis on parameter spaces. NetworkX computes graph-theoretic centrality. None answers the question: given any swarm algorithm on any landscape, *when* does exploration give way to exploitation, and *why*?

`afi` fills this gap. It defines a universal structural vocabulary — Perception $P$, Distortion $D$, Freedom $F = P/D$ — that maps onto any algorithm's quantities, then computes the coefficient of variation $E(t) = \sigma[F]/\mu[F]$ as a transition detector. Researchers can run ACO with mealpy, PSO with PySwarms, and then use `afi` to compare their structural dynamics on a common scale.

# State of the Field

The Python ecosystem for swarm intelligence is mature at the optimiser layer. mealpy offers 233 metaheuristic implementations with convergence plotting and benchmark support [@vanthieu2023mealpy]. NiaPy provides a minimalist microframework for 70+ algorithms (Vrbančič et al., JOSS 2018). pymoo specialises in multi-objective optimisation with NSGA-II/III (Blank & Deb, IEEE Access 2020). These are excellent *execution* libraries.

At the analysis layer, the landscape is sparse. SALib computes Sobol and Morris sensitivity indices over parameter spaces, but is domain-agnostic and lacks SI-specific constructs. NetworkX provides graph algorithms and centrality measures, but does not connect topology to navigability.

The critical gap is structural analysis of convergence dynamics. When an algorithm slows, is it because the landscape is hard (high $D$) or because information is scarce (low $P$)? When convergence accelerates, is it genuine exploitation or premature collapse? No library currently decomposes these causes. `afi` occupies this niche: it is a *diagnostic* layer that sits atop any optimiser, providing theory-grounded structural metrics.

Table 1 summarises the positioning. Every existing library answers "which algorithm solves this?" or "how sensitive is this parameter?". `afi` answers "what structural conditions govern convergence across all algorithms?".

| Capability | mealpy | NiaPy | pymoo | SALib | afi |
|---|---|---|---|---|---|
| Algorithm execution | 233 | 70+ | Multi-obj | — | — |
| Convergence curves | ✓ | ✓ | ✓ | — | — |
| Sensitivity analysis | — | — | — | ✓ | — |
| $P$/$D$ decomposition | — | — | — | — | ✓ |
| $E(t)$ transition metric | — | — | — | — | ✓ |
| Anomaly attribution | — | — | — | — | ✓ |
| Multi-model comparison | — | — | — | — | ✓ |

# Software Design

`afi` follows a modular architecture with three layers. The **core** layer (`perception`, `distortion`, `freedom`) implements the fundamental ratio law. The **extensions** layer (`exploration`, `gradient`, `convergence`) derives predictions from the core. The **validation** layer provides built-in honesty mechanisms.

Domain-specific factory functions bridge the gap between raw algorithm data and universal $P$/$D$ quantities. For ACO, `perception_from_aco(pheromone, visibility, alpha, beta)` computes per-edge perception as $\tau^\alpha \eta^\beta$. For PSO, `perception_from_pso(velocity, pbest_diff, gbest_diff)` computes information magnitude per particle. For building sensors, `perception_from_sensors(quantization_levels, temporal_depth)` computes joint perception from sensor specifications.

A design principle of the library is mandatory honest comparison. The `compare_models()` function always tests $F = P/D$ against four alternatives — additive ($P - D$), multiplicative ($P \times D$), nonlinear ($P^2/D$), and random — reporting $R^2$ and Pearson $r$ for all five. Similarly, `compare_distortion_composition()` tests multiplicative against additive distortion forms. If an alternative model wins, the function reports this transparently rather than suppressing the result.

Mathematical invariants are enforced at the module level. `Perception` rejects $N < 2$ or $T < 1$. `DistortionFactor` rejects values below 1 (the neutral element for multiplicative composition). `compute_freedom` rejects non-positive inputs. Scale covariance ($F(\lambda P, \lambda D) = F(P,D)$) is verified across 100 random trials in the test suite.

All code uses NumPy for vectorised computation and SciPy for statistical tests, with no heavy or exotic dependencies. The library targets Python $\geq$ 3.9 and is released under the MIT license.

# Research Impact Statement

`afi` enables any researcher to computationally test the AFI framework's predictions. The library implements the six swarm equations (S1–S6) proposed in [@melo2026swarm], each of which makes a falsifiable prediction testable through `afi` functions: $E(t) = \sigma[F]/\mu[F]$ crossing 1.0 near convergence (S3), convergence time bounded by $\bar{D} \cdot \ln|S| / P_\text{mean}$ (S4), and stigmergic dominance conditions (S6).

The library has been used in three submitted research papers: a cross-domain validation of $F = P/D$ across swarm intelligence, building performance, and complex adaptive systems [@melo2026afi]; a derivation of structural laws for swarm convergence [@melo2026swarm]; and an anomaly detection study using Freedom-based causal decomposition for IoT-enabled smart buildings.

By providing the $E(t)$ metric — the first proposed algorithm-agnostic exploration–exploitation quantifier — `afi` contributes a concrete, computable tool to a challenge that the SI community has discussed extensively but measured inconsistently for three decades [@wang2025swarm; @itmconf2025si; @li2024cooperative].

# AI Usage Disclosure

During the development of this software, the author used Claude (Anthropic) for code generation assistance, mathematical verification, test design, documentation drafting, and literature search. The author reviewed, tested, and validated all code, designed the software architecture, and takes full responsibility for the content of this publication and the correctness of the software.

# Acknowledgements

This work was supported by the Portuguese Foundation for Science and Technology (FCT) through Project 2025.00020.AIVLAB.DEUCALION, providing access to the Deucalion supercomputer at the National Advanced Computing Centre (MACC), Guimarães, Portugal.

# References
