# RVV + Quire Extension Design Brief  
*(Float-Compatible and Posit Modes)*

## 1. Overview
This document summarizes the architectural and implementation trade-offs of
adding a **quire-based vector floating-point unit** to a RISC-V core supporting
the **RVV 1.0** vector extension.  
Three operational tiers are compared:

1. **Baseline RVV (IEEE-754)** – conventional vector FPU.  
2. **RVV + Quire (MPP)** – max-precision float pipeline, IEEE-compatible mode.  
3. **RVV + Quire (Pure Posit)** – native posit arithmetic with quire support.

---

## 2. Motivation
Standard IEEE arithmetic rounds after every operation and
produces multiple special values (±0, ±Inf, NaN).  
This causes:
- non-deterministic summations,
- loss of small terms in large accumulations,
- frequent slow-path handling.

The **quire** is an *exact wide accumulator* that eliminates intermediate
rounding and allows one final normalization step.  
It yields deterministic, order-independent results and drastically reduces NaN/Inf generation.

---

## 3. Functional Comparison

| Category | **Baseline RVV (IEEE)** | **RVV + Quire (MPP)** | **RVV + Quire (Posit)** |
|-----------|--------------------------|-----------------------|--------------------------|
| Supported ops | FP16/32/64 add, mul, FMA, div, sqrt | same + exact dot/reductions | Posit 8/16/32/64 equivalents |
| Rounding | after every op | once per reduction | once per reduction |
| Associativity | non-deterministic | deterministic | deterministic |
| NaN / Inf | common (~0.5–1 %) | 10× fewer | none (only NaR) |
| Accuracy gain (dot products) | baseline | ×10³–×10⁶ lower RMS error | comparable to FP64 |
| Determinism | none | yes | yes |
| Special encodings | ±0, ±Inf, NaN | same (compat mode) | single 0 + NaR |
| Hardware addition | – | + Quire banks (64–128 b), SnapF/IRU | + PRU (posit rounder) |
| Software impact | none | new CSR for MPP mode | posit libs + new types |
| Ideal use cases | generic FP workloads | HPC, ML, simulations | physics, AI, embedded robust math |

---

## 4. Area and Power Estimates (7 nm, P670-class core)

| Component | **Baseline** | **+ MPP** | **+ Posit** |
|------------|--------------|-----------|--------------|
| Vector unit | 1.2 mm² (24 % of core) | 1.5 mm² (+0.3) | 1.7 mm² (+0.5) |
| Core total  | 5.0 mm² | 5.3 mm² (+6 %) | 5.5 mm² (+10 %) |
| Vector FP power @ FP-heavy load | 200 mW | 230 mW (+15 %) | 250 mW (+25 %) |

> The “+50 %” datapath overhead translates to **≈ +5–10 %** total core area/power.

---

## 5. Numerical Properties

| Property | IEEE RVV | Quire (MPP) | Quire (Posit) |
|-----------|-----------|--------------|----------------|
| Error growth in N-term dot product | O(N · ε) | O(ε) | O(ε) |
| Reduction reproducibility | order-dependent | order-independent | order-independent |
| Rounding points | each op | once per reduction | once per reduction |
| Exception types | NaN, Inf, overflow, underflow | fewer; same flags | single NaR |
| Determinism across vector width | no | yes | yes |

---

## 6. Implementation Notes

### Quire Architecture
- Width ≈ (2^(es+2)) × n bits (e.g., 512 b for P32E2).  
- Implemented as **banked adders** (64–128 b per bank).  
- Only active banks participate per operation; no long zero shifts.  
- Optional per-lane or shared (banked) instantiation.

### Datapath Modes
1. **IEEE-identical** – SnapF → Quire → IRU (precise IEEE output).  
2. **Max-Precision Float (MPP)** – Decode → Quire → IRU (1 round).  
3. **Pure Posit** – Decode → Quire → PRU (posit rounder).

## 6 a. Block Diagram — RVV FP Lane with Quire
             ┌───────────────────────────────┐
             │        Vector Register File   │
             │   (VLEN bits per lane)        │
             └──────────────┬────────────────┘
                            │
             ┌──────────────┴──────────────┐
             │  Operand Decode & Align     │
             │  • sign / exponent / frac   │
             │  • barrel-shift aligner     │
             └──────────────┬──────────────┘
                            │
      ┌─────────────────────┴────────────────────┐
      │                 Multiplier               │
      │     (produces full-precision product)    │
      └─────────────────────┬────────────────────┘
                            │
             ┌──────────────┴──────────────┐
             │        Quire Fabric         │
             │  • banked wide accumulators │
             │  • exponent→bank decoder    │
             │  • sparse local adders      │
             └──────────────┬──────────────┘
                            │
    ┌───────────────────────┼────────────────────────┐
    │                       │                        │
    │      ┌────▼────┐ ┌────▼────┐ ┌────▼────┐       │
    │      │     PRU │ │ IRU     │ │ SnapF   │       │
    │      │  (posit │ │   (IEEE │ │ (input  │       │
    │      │ rounder)│ │ rounder)│ │ snap)   │       │
    │      └────┬────┘ └────┬────┘ └────┬────┘       │
    │           │           │           │            │
    │     ┌─────────────────┴─────────────────┐      │
    └────▶│ Mode Select & Control (CSR bits)  │◀─────┘
          │ 00=IEEE-compat, 01=MPP, 10=Posit  │
          └───────────────────────────────────┘


### Signal Flow
1. **SnapF** (only in IEEE-compat mode) maps floats to the IEEE lattice.  
2. **Decode → Align → Multiply** generate exact fixed-point products.  
3. **Quire Fabric** accumulates products at full width with no rounding.  
4. **IRU / PRU** perform one final normalization and rounding to the target type.  
5. **CSR-controlled mode** selects which rounding path is active.

---

## 7. Benefits vs Cost Summary

| Metric | **Gain / Effect** |
|---------|-------------------|
| Numerical accuracy | 1–6 orders of magnitude better on reductions |
| NaN/Inf incidence | 10×–100× lower |
| Determinism | full reproducibility across threads / vector widths |
| Core area impact | ≈ +5–10 % |
| Core power impact | ≈ +5–10 % under FP load |
| Design complexity | manageable (shared datapath, extra rounding stages) |

---

## 8. Recommended Deployment Strategy
1. **Expose a CSR bit** to select mode: `00 = IEEE-compat`, `01 = MPP`, `10 = Posit`.  
2. **Maintain full IEEE compliance** in compat mode for ecosystem safety.  
3. **Enable MPP** as default for HPC / ML workloads to reduce NaN/Inf traps.  
4. **Use Pure Posit** in specialized scientific and embedded builds.  
5. Provide compiler flags:  
   - `-mrvv-quire` (enable quire accumulation)  
   - `-mrvv-posit` (enable posit arithmetic)

---

## 9. Takeaway
Adding a quire to an RVV unit provides:
- **Higher precision** and **deterministic results**,
- **Fewer slow exception paths**, and
- **Negligible total core overhead (≈ 5–10 %)**.

For scientific and ML hardware, this represents an exceptional accuracy-per-mm² trade-off.

---

*Prepared for internal architectural evaluation of RVV + Quire design options.*







