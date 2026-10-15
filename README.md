# UTCF reference implementation and experiments

Supplementary material for "UTCF: A Fact-Based, Bitemporal Framework for Governing
Indirect-Tax Classification Across Jurisdictions" (submitted to IEEE Access).

Requirements: Python 3.12+ (standard library only) for the experiments; matplotlib for figures.

    python utcf_sim.py all            # writes results/*.json (about 40 s on one core)
    python make_figures.py figures    # draws Figs. 3-6 from results/

Experiments (Section VII): size (Exp. 1, Fig. 3), change (Exp. 2, Table 6, Fig. 4),
errors (Exp. 3, Table 7, Fig. 5), scale (Exp. 4, Fig. 6), sensitivity (Exp. 5, Table 8).
All experiments use fixed seeds; runtimes in `scale` depend on hardware
(paper: Intel Xeon 2.80 GHz, Python 3.13, single core).
