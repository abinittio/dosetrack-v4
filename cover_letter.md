# Cover Letter

**Journal of Pharmacokinetics and Pharmacodynamics**

---

Dear Editors,

We submit for your consideration the manuscript entitled *"A Mechanistic Prodrug Pharmacokinetic Model for Lisdexamfetamine with First-Principles Nutrition-Aware Absorption Estimation: Development, Validation, and Open Clinical Simulation Tool"* for publication as an original research article.

Lisdexamfetamine dimesylate (LDX; Vyvanse/Elvanse) is among the most widely prescribed ADHD medications globally, yet no fully open, parameter-fixed mechanistic PK model has been validated against multiple independent datasets and made freely available for clinical and research use. Existing pharmacokinetic analyses have relied on proprietary clinical trial data, precluding independent replication and prospective simulation by the broader research community.

This manuscript makes three contributions of direct relevance to the readership of JPKPD:

1. **A validated, parameter-fixed mechanistic model.** A two-compartment prodrug ODE system with Michaelis-Menten kinetics for both enzymatic conversion and elimination achieves a Cmax MAPE of 9.1% across six independent dose-dataset combinations (30–100 mg; 34–70 kg; four published studies). No parameters were optimised against the validation data. In direct comparison, a one-compartment linear model achieves 74.4% Cmax MAPE on the same datasets (ΔAIC = 19.1), demonstrating that the mechanistic prodrug conversion structure is empirically necessary and not a modelling preference.

2. **A novel first-principles food effect framework.** By inverting the analytical Tmax equation, we back-calculate an effective absorption rate constant for each published cohort. The implied prandial states rank-order the four independent study cohorts in a direction precisely consistent with their published protocols — from fasting paediatric subjects through standardised-meal adults — using only two reference parameter values from the literature.

3. **An open simulation tool.** The model is implemented in an open-source Python package and deployed as a freely accessible web application (dosetrack.dis-solved.com), with a principled validated-horizon design that explicitly communicates model uncertainty to non-specialist users.

The manuscript reports limitations honestly, including AUC overestimation attributable to the two-compartment redistribution dynamics and the small number of external validation points; both are discussed in the context of the model's intended clinical use case.

This work has not been submitted elsewhere and all authors have approved the submission. The source code and data used to generate all results are publicly available at github.com/abinittio/dosetrack-v4 (MIT licence).

Yours sincerely,

**Nabil Yasini**
Dis-Solved, Independent Research
contact@dis-solved.com
ORCID: 0009-0009-4642-7417
