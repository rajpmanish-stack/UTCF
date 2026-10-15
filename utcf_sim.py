"""Reference implementation and experiments for the Universal Taxability
Classification Framework (UTCF).

Standard library only (Python 3.12+). Every experiment uses fixed seeds, so running
    python utcf_sim.py all
reproduces every number, table and figure input reported in the paper.
Outputs are written as JSON to ./results/.

Model of the world (Section VI-A of the paper)
---------------------------------------------
* A catalog of N items in F product families. Each item has eight binary
  taxability facts (prevalence 4%-35%) and one of 200 external product codes
  aligned to families but not to tax treatment.
* A jurisdiction defines C = 40 categories. A category is a set of one to
  three families plus up to two fact conditions (fact == value). Categories
  may overlap; each jurisdiction applies them in its own precedence order, and
  an item belongs to the first category that matches (or to a residual
  category). Each category carries one of four treatments.
* Harmonized regime: all jurisdictions share one library of category
  definitions and differ only in precedence and treatment (as SSUTA members
  share the Library of Definitions). Independent regime: each jurisdiction
  draws its own definitions.
* The UTC is the coarsest common refinement (meet) of the jurisdictional
  category partitions, computed by Algorithm 1 on distinct fact vectors.
"""
import json
import os
import random
import statistics as st
import sys
import time
from collections import defaultdict

F = 12                      # product families
NF = 8                      # recorded binary facts
PREV = [0.04 + i * (0.31 / (NF - 1)) for i in range(NF)]   # 4% .. 35%
NCODES = 200                # external product codes
C = 40                      # categories per jurisdiction
T = 4                       # treatments
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------- catalog
def make_catalog(n, rng):
    codes = {f: [c for c in range(NCODES) if c % F == f] for f in range(F)}
    fam = [rng.randrange(F) for _ in range(n)]
    facts = [tuple(1 if rng.random() < p else 0 for p in PREV) for _ in range(n)]
    code = [rng.choice(codes[fam[i]]) for i in range(n)]
    return fam, facts, code


def make_category(rng, nfacts=NF):
    fams = frozenset(rng.sample(range(F), rng.randint(1, 3)))
    conds = tuple((q, rng.randint(0, 1)) for q in rng.sample(range(nfacts), rng.randint(0, 2)))
    return fams, conds


def make_library(rng):
    return [make_category(rng) for _ in range(C)]


def make_jurisdiction(rng, library):
    order = list(range(len(library)))
    rng.shuffle(order)
    treat = {k: rng.randrange(T) for k in range(len(library))}
    treat[-1] = 0
    return {"lib": library, "order": order, "treat": treat}


def matches(cat, f, x):
    fams, conds = cat
    return f in fams and all(x[q] == v for q, v in conds)


def classify(j, f, x):
    """c_j(p): first matching category in j's precedence order, else -1."""
    for k in j["order"]:
        if matches(j["lib"][k], f, x):
            return k
    return -1


def make_regime(rng, nj, harmonized):
    shared = make_library(rng)
    return [make_jurisdiction(rng, shared if harmonized else make_library(rng)) for _ in range(nj)]


# ------------------------------------------------------------- Algorithm 1
def build_utc(fam, facts, juris):
    """Algorithm 1: UTC as the meet of jurisdictional partitions.

    Classification depends on an item only through its fact vector
    key = (family, facts), so signatures are computed once per distinct key
    (M keys) instead of once per item: O(N + M*J*C) instead of O(N*J*C).
    Returns item->class index, class signatures, and the number of keys M.
    """
    keys = {}
    item_key = [None] * len(fam)
    for i in range(len(fam)):
        k = (fam[i], facts[i])
        if k not in keys:
            keys[k] = len(keys)
        item_key[i] = keys[k]
    sig_of_key = [tuple(classify(j, k[0], k[1]) for j in juris) for k in keys]
    cls_index = {}
    key_class = []
    for s in sig_of_key:
        if s not in cls_index:
            cls_index[s] = len(cls_index)
        key_class.append(cls_index[s])
    item_class = [key_class[item_key[i]] for i in range(len(fam))]
    return item_class, list(cls_index), len(keys)


def build_utc_naive(fam, facts, juris):
    """Item-by-item baseline used only for the runtime comparison."""
    cls_index = {}
    out = []
    for i in range(len(fam)):
        s = tuple(classify(j, fam[i], facts[i]) for j in juris)
        out.append(cls_index.setdefault(s, len(cls_index)))
    return out


def membership_patterns(fam, facts, library):
    """Distinct patterns of library-category membership (harmonized bound)."""
    return len({(f, tuple(matches(c, f, x) for c in library)) for f, x in set(zip(fam, facts))})


# ------------------------------------------------------- change-effort model
def b2_edits(groups, fam_size, nj):
    """Central family code: one row per wholly affected family; otherwise the
    affected items are recoded and the new code mapped in every jurisdiction."""
    e = 0
    for (f, _), n in groups.items():
        e += 1 if n == fam_size[f] else n + nj
    return e


def event(kind, rng, fam, facts, code, juris, item_class, S=3):
    nj = len(juris)
    n = len(fam)
    fam_size = defaultdict(int)
    for f in fam:
        fam_size[f] += 1
    j_idx = rng.randrange(nj)
    j = juris[j_idx]
    cj = [classify(j, fam[i], facts[i]) for i in range(n)]
    members = defaultdict(list)
    for i, k in enumerate(cj):
        if k >= 0:
            members[k].append(i)
    nonempty = [k for k in members if members[k]]

    if kind in ("E1", "E2"):
        if kind == "E1":
            cats = [rng.choice(nonempty)]
        else:
            cats = rng.sample(nonempty, max(1, round(0.4 * len(nonempty))))
        A = [i for k in cats for i in members[k]]
        groups = defaultdict(int)
        for k in cats:
            for i in members[k]:
                groups[(fam[i], k)] += 1
        utcf = len({item_class[i] for i in A})
        return dict(items=len(A), B1=S * len(A), B2=b2_edits(groups, fam_size, nj), UTCF=utcf)

    if kind in ("E3", "E4"):
        # pick a category and a splitting fact that actually splits it
        for _ in range(200):
            k = rng.choice(nonempty)
            if kind == "E3":
                q = rng.randrange(NF)
                val = [facts[i][q] for i in range(n)]
            else:  # a new, unrecorded fact with prevalence 20%
                val = [1 if rng.random() < 0.20 else 0 for _ in range(n)]
            A = [i for i in members[k] if val[i] == 1]
            if 0 < len(A) < len(members[k]):
                break
        groups = defaultdict(int)
        for i in A:
            groups[(fam[i], k)] += 1
        cls_in_k = defaultdict(set)
        for i in members[k]:
            cls_in_k[item_class[i]].add(val[i])
        splits = sum(1 for v in cls_in_k.values() if len(v) == 2)      # s_e new classes
        new_map = len({item_class[i] for i in A})                       # k_e mappings in j
        utcf = splits + new_map
        if kind == "E4":
            fams_k = j["lib"][k][0]
            nF = sum(fam_size[f] for f in fams_k)                       # fact capture
            utcf += nF + 1                                              # + fact definition
        return dict(items=len(A), B1=S * len(A), B2=b2_edits(groups, fam_size, nj), UTCF=utcf)

    if kind == "E5":  # 15% of external codes revised
        revised = set(rng.sample(range(NCODES), int(0.15 * NCODES)))
        A = [i for i in range(n) if code[i] in revised]
        cross = {(item_class[i], code[i]) for i in A}
        return dict(items=len(A), B1=0, B2=len(A), UTCF=len(cross))
    raise ValueError(kind)


def summarize(xs):
    xs = sorted(xs)
    return dict(median=st.median(xs), min=xs[0], max=xs[-1], mean=st.mean(xs))


# ---------------------------------------------------------------- experiments
def exp_size(reps=10, n=5000, js=(1, 2, 5, 10, 25, 50)):
    """Exp. 1: size of the UTC as jurisdictions are added (Fig. 3).

    For each replication one catalog and one 50-jurisdiction regime are drawn;
    the UTC is computed for the first J jurisdictions, so curves are nested."""
    out = {"harmonized": {}, "independent": {}, "M": [], "patterns": []}
    for regime in ("harmonized", "independent"):
        ks = {nj: [] for nj in js}
        for r in range(reps):
            rng = random.Random(1000 + r)
            fam, facts, code = make_catalog(n, rng)
            juris = make_regime(random.Random(2000 + r), max(js), regime == "harmonized")
            for nj in js:
                _, sigs, M = build_utc(fam, facts, juris[:nj])
                ks[nj].append(len(sigs))
            if regime == "harmonized":
                out["M"].append(M)
                out["patterns"].append(membership_patterns(fam, facts, juris[0]["lib"]))
        out[regime] = {nj: summarize(v) for nj, v in ks.items()}
    out["M"] = summarize(out["M"])
    out["patterns"] = summarize(out["patterns"])
    return out


def exp_change(reps=20, n=5000, nj=20, S=3):
    """Exp. 2: records edited per regulatory event (Table 6)."""
    res = {k: defaultdict(list) for k in ("E1", "E2", "E3", "E4", "E5")}
    ksz = []
    for r in range(reps):
        rng = random.Random(7000 + r)
        fam, facts, code = make_catalog(n, rng)
        juris = make_regime(rng, nj, True)
        item_class, sigs, _ = build_utc(fam, facts, juris)
        ksz.append(len(sigs))
        for kind in res:
            e = event(kind, random.Random(9000 + 31 * r + ord(kind[1])), fam, facts, code, juris, item_class, S)
            for key, v in e.items():
                res[kind][key].append(v)
    out = {k: {m: summarize(v) for m, v in d.items()} for k, d in res.items()}
    out["K"] = summarize(ksz)
    # break-even: how many E1-type events recover the E4 fact-capture premium
    be = []
    for i in range(reps):
        prem = res["E4"]["UTCF"][i] - res["E4"]["B1"][i]
        save = res["E1"]["B1"][i] - res["E1"]["UTCF"][i]
        if save > 0:
            be.append(max(0.0, prem / save))
    out["breakeven_E1_events"] = summarize(be)
    return out


def exp_errors(reps=20, mc=2000, n=5000, nj=20, S=3, eps=0.01):
    """Exp. 3: error structure under an E2-type event (Proposition 3)."""
    agg = {"B1": defaultdict(list), "UTCF": defaultdict(list)}
    theory = defaultdict(list)
    for r in range(reps):
        rng = random.Random(7000 + r)
        fam, facts, code = make_catalog(n, rng)
        juris = make_regime(rng, nj, True)
        item_class, _, _ = build_utc(fam, facts, juris)
        erng = random.Random(500 + r)
        j = juris[erng.randrange(nj)]
        cj = [classify(j, fam[i], facts[i]) for i in range(n)]
        members = defaultdict(list)
        for i, k in enumerate(cj):
            if k >= 0:
                members[k].append(i)
        cats = erng.sample(list(members), max(1, round(0.4 * len(members))))
        A = [i for k in cats for i in members[k]]
        csize = defaultdict(int)
        for i in A:
            csize[item_class[i]] += 1
        sizes = list(csize.values())
        ne, ke = len(A), len(sizes)
        theory["ne"].append(ne)
        theory["ke"].append(ke)
        mrng = random.Random(800 + r)
        for _ in range(mc):
            q1 = mrng.binomialvariate(S * ne, eps)
            agg["B1"]["episodes"].append(q1)
            agg["B1"]["volume"].append(q1 / S)
            hit = [s for s in sizes if mrng.random() < eps]
            agg["UTCF"]["episodes"].append(len(hit))
            agg["UTCF"]["volume"].append(sum(hit))
            if hit:
                agg["UTCF"]["size"].extend(hit)
        agg["B1"]["size"].append(1 / S)
    out = {}
    for a in agg:
        out[a] = {
            "episodes_mean": st.mean(agg[a]["episodes"]),
            "volume_mean": st.mean(agg[a]["volume"]),
            "volume_sd": st.pstdev(agg[a]["volume"]),
            "p_zero_error": sum(1 for x in agg[a]["episodes"] if x == 0) / len(agg[a]["episodes"]),
            "episode_size_mean": st.mean(agg[a]["size"]),
        }
    for a in agg:
        v = sorted(agg[a]["volume"])
        out[a]["volume_q"] = {q: v[min(len(v) - 1, int(q / 100 * len(v)))] for q in (50, 90, 99, 99.9)}
        out[a]["volume_samples"] = v
    out["ne_mean"] = st.mean(theory["ne"])
    out["ke_mean"] = st.mean(theory["ke"])
    out["eps"] = eps
    out["S"] = S
    return out


def exp_scale():
    """Exp. 4: runtime of Algorithm 1 and of determination (Fig. 5)."""
    out = {"by_N": {}, "by_J": {}, "naive_by_N": {}, "determination": {}}
    for nn in (1_000, 10_000, 100_000, 1_000_000):
        rng = random.Random(42)
        fam, facts, code = make_catalog(nn, rng)
        juris = make_regime(rng, 50, False)
        t0 = time.perf_counter()
        _, sigs, M = build_utc(fam, facts, juris)
        out["by_N"][nn] = dict(seconds=time.perf_counter() - t0, K=len(sigs), M=M)
        if nn <= 100_000:
            t0 = time.perf_counter()
            build_utc_naive(fam, facts, juris)
            out["naive_by_N"][nn] = dict(seconds=time.perf_counter() - t0)
    rng = random.Random(43)
    fam, facts, code = make_catalog(100_000, rng)
    for nj in (10, 50, 100, 200):
        juris = make_regime(random.Random(44 + nj), nj, False)
        t0 = time.perf_counter()
        _, sigs, _ = build_utc(fam, facts, juris)
        out["by_J"][nj] = dict(seconds=time.perf_counter() - t0, K=len(sigs))
    out["determination"] = determination_benchmark()
    return out


def determination_benchmark(n_lines=200_000):
    """Resolve lines through a bitemporal mapping store and emit evidence."""
    rng = random.Random(45)
    fam, facts, code = make_catalog(5000, rng)
    juris = make_regime(rng, 20, True)
    item_class, sigs, _ = build_utc(fam, facts, juris)
    # mapping store: (class, j) -> list of (valid_from, valid_to, treatment, version, citation)
    store = defaultdict(list)
    for c, s in enumerate(sigs):
        for jj, k in enumerate(s):
            t = juris[jj]["treat"][k]
            store[(c, jj)].append((0, 182, t, 1, f"cite:{jj}:{k}:v1"))
            store[(c, jj)].append((182, 10**9, (t + rng.randrange(2)) % T, 2, f"cite:{jj}:{k}:v2"))
    lines = [(rng.randrange(5000), rng.randrange(20), rng.randrange(365)) for _ in range(n_lines)]
    t0 = time.perf_counter()
    ev = []
    for item, jj, day in lines:
        c = item_class[item]
        for vf, vt, t, ver, cite in store[(c, jj)]:
            if vf <= day < vt:
                ev.append((item, jj, day, c, ver, t, cite))
                break
    dt = time.perf_counter() - t0
    return dict(lines=n_lines, seconds=dt, lines_per_second=n_lines / dt, evidence_records=len(ev))


def exp_sensitivity(reps=10):
    """Exp. 5: E1 edit ratio B1/UTCF as catalog size and systems vary."""
    out = {"by_N": {}, "by_S": {}}
    for nn in (1_000, 5_000, 20_000, 50_000):
        ratios = []
        for r in range(reps):
            rng = random.Random(3000 + r)
            fam, facts, code = make_catalog(nn, rng)
            juris = make_regime(rng, 20, True)
            ic, _, _ = build_utc(fam, facts, juris)
            e = event("E1", random.Random(4000 + r), fam, facts, code, juris, ic, 3)
            ratios.append(e["B1"] / max(1, e["UTCF"]))
        out["by_N"][nn] = summarize(ratios)
    for S in (1, 2, 3, 5):
        ratios = []
        for r in range(reps):
            rng = random.Random(3000 + r)
            fam, facts, code = make_catalog(5000, rng)
            juris = make_regime(rng, 20, True)
            ic, _, _ = build_utc(fam, facts, juris)
            e = event("E1", random.Random(4000 + r), fam, facts, code, juris, ic, S)
            ratios.append(e["B1"] / max(1, e["UTCF"]))
        out["by_S"][S] = summarize(ratios)
    return out


EXPS = {"size": exp_size, "change": exp_change, "errors": exp_errors,
        "scale": exp_scale, "sensitivity": exp_sensitivity}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    todo = list(EXPS) if len(sys.argv) < 2 or sys.argv[1] == "all" else sys.argv[1:]
    for name in todo:
        t0 = time.perf_counter()
        res = EXPS[name]()
        with open(os.path.join(OUT, f"{name}.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(f"{name}: done in {time.perf_counter() - t0:.1f}s")
