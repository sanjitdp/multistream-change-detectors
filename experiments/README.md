# Experiments

The main text compares external identification methods, studies the time to
identify 80% of changes, examines non-Gaussian observations, and discusses
external global detectors. The Gaussian identification-time table,
simultaneous-change and Poisson-count figures, and non-Gaussian null-error
table appear in the main text. The staggered-change figure appears with its
model and interpretation in the supplement, together with additional external
Gaussian comparisons and the full internal Gaussian curves. Both e-d-BH and
e-d-Bonferroni are our procedures; parallel SR and CUSUM at the stated
thresholds are the corresponding Bonferroni choices.

Run commands from the repository root. The paper's runs used Python 3.11.11,
NumPy 1.26.4, SciPy 1.15.2 and Matplotlib 3.9.2. Create an environment with:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/requirements.txt
```

The global scan also requires a C++17 compiler (`clang++` or `g++`). Its helper
is compiled automatically into a temporary directory. On macOS it links
Apple's Accelerate framework for vectorized math; elsewhere it uses standard
C++ scalar math. No network or private data are needed for the simulations.

Reproduce all simulation studies and their publication tables:

```sh
python3.11 experiments/validate_bayesian_methods.py
python3.11 experiments/external_identification.py
python3.11 experiments/external_tables.py
python3.11 experiments/tradeoff_benchmark.py
python3.11 experiments/diverse_benchmark.py
python3.11 experiments/gaussian_benchmark.py
python3.11 experiments/summarize.py
python3.11 experiments/paper_tables.py
python3.11 experiments/global_baselines.py
```

Run the original Gaussian generator before `global_baselines.py`: the latter
reuses its fitted thresholds and saved paired comparator outcomes. Existing
results can instead be used to regenerate presentation without simulation:

```sh
python3.11 experiments/external_identification.py --summarize-only
python3.11 experiments/external_tables.py --tables-only
python3.11 experiments/tradeoff_benchmark.py --figures-only
python3.11 experiments/diverse_benchmark.py --summarize-only
python3.11 experiments/global_baselines.py --summarize-only
```

All included table and figure captions state the thresholds used by our
procedures. In the global-ARL comparison, the table reports each fitted
statistic threshold rather than the EOP-calibrated thresholds used in the
identification experiments. The null-error table uses LaTeX placement `[H]`
and requires the `float` package. Generated narrative fragments are working
notes; the edited manuscript supplies the final prose and captions.

The scripts create outputs in `results/`. This directory starts empty; run
the simulation sequence above before using the table and figure scripts.
The file descriptions below document those outputs and the study protocol.
The paper's original outcomes and source snapshots remain in the manuscript
workspace.

## External identification methods: main Gaussian comparison

`external_identification_design.txt` records the design. There are 100 streams,
a known Gaussian shift .5, horizon 400 and q=.05, with 3000 paired runs in
each of six settings: no changes; 10 or 50 changes at 101 or 251; and independent
changepoints drawn from the Bayesian prior. The prior has probability 1/2 of
no change; conditional on a finite change, its time is geometric with hazard
.01, starting at observation 1. This same prior is used by the Bayesian
procedures in every setting. Seeds are 62106000 through 62106005 in that order.

The six rules are e-d-BH and e-d-Bonferroni with SR inputs at alpha=q/400;
Gandy--Lau's bounded-CUSUM p-values with ordinary BH at q; a clearly labeled
time-adjusted version using q/400; Chen--Zhang--Poor's MD-FDR; and the posterior
LFDR selector of Lu--Chen--Li. The two Bayesian rules use q=.05 and the stated
prior. All likelihood parameters are known. Their independent implementation
and checks are in `bayesian_methods.py` and `validate_bayesian_methods.py`.

Gandy--Lau uses nearest-grid rounding on {0,.025,...,20}. Gaussian probabilities
over the exact rounding bins give an 801-state transition matrix. Its null
distribution is propagated from zero, yielding inclusive tail p-values for
the rounded chart. Before performance simulation, the cap was raised from
10 to 20 to make the smallest time-adjusted BH cutoff attainable; this recorded
amendment applies to both variants. Validation checks the transition matrix,
direct-CDF recursion, p-value superuniformity, literal selectors and 100000
independent null chart paths. `external_identification.py --validate-only`
runs those checks without the six performance settings.

The primary decision is each method's first nonempty report by 400, or an
empty report if none occurs. We measure FDR and the number of true discoveries
in that same reported set for every method. We also save TDR, report size,
pre-change reports and no-report probabilities. The Bayesian methods have
removed no streams before this first report; no artificial union of their
later decisions is used. Report delay conditions on no earlier report and
counts the first post-change observation as one. A nondetection receives
the remaining duration through 400, so the cap is 400-nu+1. The raw no-report
time is 401, but the delay calculation uses the stated cap. A post-change
report can still be false; its true count and FDR are needed to interpret
its delay. In the prior-generated case the delay starts at the earliest
changepoint in that run and is not pooled with fixed-onset delays.

Our BH rule and the time-adjusted Gandy--Lau rule bound FDR for reporting
times bounded by 400; Bonferroni bounds PFER and hence FDR. Native Gandy--Lau
controls fixed-time FDR. The Bayesian guarantees average over their prior
and are not fixed-configuration guarantees. The results retain both the
fixed-configuration and correct-prior comparisons: e-d-BH gives more true
discoveries and shorter delays than time-adjusted Gandy--Lau in these fixed
settings; posterior LFDR reports sooner under the correct prior. Native
Gandy--Lau's fixed-time null FDR agrees with q within Monte Carlo uncertainty,
although its first-report FDR is much larger.

Files beginning `external_` contain the raw outcomes, exact null chain,
validation results and metadata. `external_readout.txt`
reports every setting; `external_main_table.tex`, `external_sparse_table.tex`,
`external_delay_table.tex` and `external_prior_table.tex` supply the manuscript
tables. `external_fixedtime.pdf` shows secondary current-set FDR/TDR for the
four frequentist rules. Conditional sample sizes and uncertainty are retained.

Primary sources:

- [Gandy--Lau (2013), Algorithm 2 and Sections 2--4](https://arxiv.org/pdf/1204.4333).
- [Chen--Zhang--Poor (2020), MD-FDR Algorithm 1](https://par.nsf.gov/servlets/purl/10194694).
- [Lu--Chen--Li (2022), Algorithms 3--4](https://arxiv.org/pdf/2206.08435).

## Bayesian selector conventions

In `bayesian_methods.py`, the last array axis indexes the original K streams;
leading axes index independent runs. The caller removes each returned batch
from `active` and updates only the remaining streams on the next observation.
Apply each selector once per observation.

The changepoint xi is the first post-change observation, and
`survival(t) = P(xi > t)`. Lu, Chen and Li use tau = xi - 1. The posterior
recursion assumes known likelihoods and priors, independent streams, and
conditional independence over time given each changepoint. Initialize
posterior odds at zero, or log odds at negative infinity. Prior probabilities
may be scalars or arrays broadcastable to the odds; inactive values are
returned unchanged.

For MD-FDR, let d be the number of previously removed streams. Sort active
posterior odds in descending order and select the largest prefix ending at a
rank j with `G[j] >= K / (q * (d + j)) - 1`. K is the original family size.
For the LFDR selector, sort posterior null probabilities `1 / (1 + G)` and
select the largest prefix with mean at most q. Its conditional expected FDP
guarantee applies to that batch under the assumed model.

## Internal Gaussian comparison: identification times and further curves

`tradeoff_design.txt` records the design before outcomes were generated.
We use 100 independent unit-variance Gaussian streams, horizon 400, and
4000 runs per case. Either no streams change, or 10 or 50 streams change
from mean zero to .5 at observation 101. Each method sees the same data.
With q=.05 and alpha=q/400, e-d-BH uses SR thresholds K/(r alpha), parallel
SR uses K/alpha, and separate per-stream SR uses 1/alpha. SHEREF applies
e-BH to SR/t at fixed-time FDR level q. The first two provide across-stream
FDR/PFER guarantees at every reporting time bounded by 400; the separate
charts instead bound each stream's probability of a false alarm by q.

The main identification-time table includes both changed-set sizes and all
four methods. The supplementary figure pairs the probability of any declaration
by the horizon under the global null with the expected proportion of changed
streams currently declared in the 50-change case; another figure includes
SHEREF and the 10-change case. Saved summaries also include FDR, FWER, PFER and TDR at
time 400 and at the first one or ten declarations capped at 400. Identification
times count post-change observations until the current set contains 80% or
all changed streams; nondetections receive time 401 and are retained.
There were no nondetections for the 80% target in these cases.

`tradeoff_summary.csv`, `tradeoff_delay.csv`, `tradeoff_curves.csv` and
`tradeoff_replicates_m*.npz` retain all results. Metadata, seeds and versions
are in `tradeoff_metadata.json`. Figures and the numerical table have names
beginning `tradeoff_`. Monte Carlo standard errors use independent runs as
the units and retain within-run dependence across streams.

## Earlier Gaussian study and the original ARL comparators

Run `python3.11 experiments/gaussian_benchmark.py`, then
`python3.11 experiments/summarize.py` from the repository root. Python 3.11,
NumPy 1.26.4 and Matplotlib 3.9.2 were used. No network or private data are
required. `results/metadata.json` records the simulation configuration and
software versions; CSV files and compressed replicate-level arrays accompany
the figures and LaTeX tables. The generator's deterministic checks compare SR
and CUSUM recursions with their definitions and test the BH step-up rule.

## Data and methods

There are 50 streams. Independent observation vectors have marginal variance
one, equicorrelation rho = 0 or .5, and means zero before each stream's change
and mu = .4 or .75 afterwards. The likelihood ratio is exp(mu X - mu²/2).
The null configuration has no changes; the sparse and dense configurations
have respectively 5 and 25 changes at observation 101. The staggered
configuration has 10 changes at 101 and another 15 at 151. Other streams never
change. The simulation ends at 400 and uses 2000 independent replications per
configuration. Contemporaneous correlation leaves the marginal likelihood
ratios conditionally mean one under the global past filtration in unchanged
streams, since observation vectors are independent over time.

This earlier study uses alpha = .001 for e-d-BH and e-d-Bonferroni, with
SR statistics R[k,t] = (1+R[k,t-1]) exp(mu X[k,t] - mu²/2), R[k,0]=0.
e-d-BH uses the step-up thresholds 50/(r alpha); Bonferroni uses 50/alpha.
The latter is precisely a collection of parallel SR charts. This identity is
explicit, rather than counting the two names as different methods.
SR/t BH uses level .05 with e-values R[k,t]/t and fixed sensors. This is the
unboosted SHEREF rule (equations 7--8 in Lee, Chen and Li), with the declaration
indexed by the observation just used. It targets fixed-time FDR, which differs
from the EOP target of the constant-threshold rules.

A separate comparison calibrates e-d-BH, SR-max, CUSUM-max, and Sum-CUSUM to
global-null ARL 1000 under independent streams. The CUSUM statistics are
W[k,t]=max(0,W[k,t-1]+mu X[k,t]-mu²/2). CUSUM-max raises an alarm if any W[k,t]
crosses its threshold, declaring those streams. Sum-CUSUM raises a global
alarm if the sum of W[k,t] crosses its threshold; it makes no stream-level
declarations. The respective global statistics are max_r(log R^[r]+log r),
max_k log R[k], max_k W[k], and sum_k W[k]. Thresholds are tuned separately
for each mu using 800 independent null paths, by bisection on the empirical
restricted ARL capped at 10001 observations. `calibration.csv` reports both
the restricted calibration mean and the number censored. A fresh set of
1500 null paths is then simulated until *every* first alarm occurs, including
the two theoretical-threshold rules. Reported validation ARLs are uncensored
sample means, with Monte Carlo standard errors. Calibration error is visible
in this validation, and no exact matched ARL or theoretical ARL guarantee is
claimed for the fitted thresholds. For BH and SR-max the level actually
implied by calibration is 50 exp(-threshold), included in the calibration CSV.
In the current manuscript, SR-max and CUSUM-max are labeled e-d-Bonferroni/SR
and e-d-Bonferroni/CUSUM (ours). Their legacy identifiers remain in the original
raw files. Sum-CUSUM is the external global detector in this initial study.
The matched thresholds are retained in the correlated runs; those runs are
sensitivity checks, not comparisons at a matched ARL under correlation.

Primary sources for the baselines:

- Mei (2010), equations 7--9: [author-hosted published paper](https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/6/1466/files/2020/09/mei_biometrika2010.pdf).
- Lee, Chen and Li, equations 7--8 and fixed-sensor identity immediately following equation 9: [arXiv version 1](https://arxiv.org/html/2504.06526v1).

## External global detectors

`global_external_design.txt` fixes the extension of the ARL comparison.
`global_baselines.py` and `global_baselines.cpp` add Xie--Siegmund's T2 and
Chan's transformed CUSUM. Both use 400 independent null training paths
(seed 73100) and 750 fresh validation paths (seed 74100), with no retuning
on validation data. Calibration retains record maxima until an upper bracket
is crossed; lower thresholds are fitted on those same uncensored paths.
An analytic ARL approximation initializes the Xie--Siegmund search only.
Validation continues until every alarm occurs. The original 800/1500 fitting
and validation runs for our procedures and Sum-CUSUM are unchanged.

Xie--Siegmund's T2 scans lengths 1 through min(t,199), sums the transformed
positive Gaussian window statistics across streams, and uses mixture weight
p0=.1. It estimates positive shifts rather than receiving their true size.
Chan's rule sums log[1+p0(lambda_M exp(R/2)-1)] over local CUSUM scores, with
published design choices mu0=1, lambda_M=.64 and p0=.1. These choices are fixed
across alternatives. Our methods and Sum-CUSUM use the true shift.

All six methods use exactly the same 2000 observation paths in each of the
four alternatives: shifts .4 or .75 in 5 or 25 of 50 streams at time 101.
The generator verifies exact reproduction of the old first-alarm arrays.
The new methods validated at ARLs 955.0 (SE 35.0) and 1009.3 (36.3), respectively.
These actual values, not an assumption of exact ARL matching, accompany the
delay comparison. Global aggregation is especially effective for dense
changes, but its pre-change alarm probabilities differ from those of our
procedures. No stream-identification sets are assigned to any global-only
method. The study reports conditional global delay, pre-change alarm
probability, conditioning sample size and nondetection probability. Nondetections
by 400 receive time 401, or delay 301; none occurred for the two new methods.

Files beginning `global_external_` contain the frozen protocol, record maxima,
raw training/validation/alternative arrays, summary CSVs, readout and table.
`global_external_table.tex` is included in the manuscript; the accompanying
delay plot remains available. `--checks-only` compares the compiled statistics
with independent direct window calculations and records computation time.

Primary sources: [Xie--Siegmund (2013), equations 3.5--3.7](https://arxiv.org/pdf/1207.2386)
and [Chan (2017), equations 5.1--5.3](https://arxiv.org/pdf/1506.08504).

## Estimands in the earlier Gaussian study

All stream-level errors and powers use the *current* declaration set; a
declaration is not carried forward after its statistic falls below threshold.
At time t a false discovery is a declared stream whose change has not yet
occurred. FDR is mean FDP, PFER is mean false count, and FWER is probability
of at least one false discovery. TDR is the mean proportion of already
changed streams currently declared. Its complement is the mean fraction of
missed changed streams. `prob80_t` and `proball_t` concern simultaneous
declarations at t, not declarations accumulated across time.

We also evaluate two prespecified bounded stopping rules: the first time
there is at least one declaration, capped at 400, and the first time there
are at least five declarations, capped at 400. The ratio estimates divide
mean FDP or mean false count by mean bounded stopping time. These are
estimates for these particular rules, **not estimates of the supremum in
EOP and not empirical verification of an all-stopping-times theorem**.
Ratio standard errors use the paired influence values (error - ratio*time)
divided by the mean time. Ordinary means use replicate-level sample standard
deviation divided by sqrt(n), including clustering across streams within
each replication.

Delay counts the first post-change observation as one. Per-stream delays
start at that stream's change time, and a mean is taken within each
replication before computing the Monte Carlo standard error. Times to 80%
and all identification are the first times the instantaneous declaration
set contains that fraction of all eventually changed streams, and delays
are measured from observation 101. For staggered changes this means waiting
until the second group has actually changed before 80% or all can be reached.
No detection by time 400 is assigned time 401; restricted delays and the
nondetection proportion are both saved. These restricted means include all
replications and must not be described as uncensored expected delays.
Global detection delays instead use the first global alarm conditional on
no alarm before 101, including post-change nondetections at time 401. The
conditioning sample size and nondetection proportions are also saved.

All methods see the same data in a replication; independent random-number
seeds are used for calibration, validation and each data configuration. The
seeds and sample sizes appear in CSV rows. Marginal error bars are 1.96
Monte Carlo standard errors. They quantify simulation uncertainty, not
uncertainty from fitting the threshold on a separate calibration sample.

## Files

- `calibration.csv`: threshold, training restricted ARL and censoring count.
- `arl_validation.csv`: fresh uncensored ARL means and standard errors.
- `summary.csv`: errors, stopping-rule ratios, powers and restricted delays.
- `curves.csv`: instantaneous metrics every five observations.
- `replicates_*.npz`, `arl_mu*.npz`: replicate-level outcomes for reanalysis.
- `type1.pdf`, `power.pdf`: publication figures with corresponding PNGs.
- `results.tex`: concise model description, generated results and limitations.
- `error_table.tex`, `delay_table.tex`: generated tables.

The earlier experiments study known positive Gaussian shifts, two effect sizes,
one stream count and two dependence structures. The added identification and
global studies above provide the scan, mixture and Bayesian comparisons used
in the revised manuscript. Their different statistical tasks and guarantees
are retained rather than treating every method as an identification procedure.

## Prespecified non-Gaussian experiments

Run `python3.11 experiments/diverse_benchmark.py` to reproduce the separate
study in `diverse-design.txt`; use `--summarize-only` to regenerate its plots
from saved CSV files. The design was written before the outcomes were
generated, and its SHA256 hash is recorded in `results/diverse_metadata.json`.
All 15 configurations use 1500 independent replications and a monitoring
horizon of 400. All configurations are retained.

The heavy-tail experiment uses 100 streams, with independent-in-time vectors
formed from a shared Student t3 variable and stream-specific Student t3
variables. Both components are divided by sqrt(3); their weights produce
unit marginal variance and cross-stream correlation 0 or .5. Twenty streams
have alternating positive shifts .45 and .9, either all at time 101 or with
10 changes at 101 and another 10 at 201. The other 80 streams remain null.
Separate global-null configurations use the same two correlations. Evidence
factors are `1 + .5 sign(X)`. Their conditional expectation is one for every
still-null stream because its conditional distribution is symmetric. This
argument does not require a known t likelihood or a moment-generating
function; symmetry with respect to the global past is the needed assumption.

The count experiment uses K=20,100,500, with known intercepts alternating .1
and 1. Given the observed past, `X[t,k]` is Poisson with mean
`q[t,k] * lambda[t,k]`, where `lambda[t,k] = a[k] + .3 X[t-1,k]`. The multiplier
q changes from 1 to 1.75 at time 101 in either 10% or 50% of streams; each
affected subset contains equal numbers from both intercept groups. A null
configuration is also run at each K. Both regimes are stable because the
largest autoregressive coefficient is 1.75*.3=.525. A 200-observation null
warmup supplies the observed initial history; only detector statistics are
reset. The exact conditional likelihood factor is
`exp(X[t,k]*log(1.75) - .75*lambda[t,k])`. It has conditional mean one for every
still-null stream. Streams are independent across k but serially dependent.
The intensity model and its parameters are assumed known, not estimated.

All four choices use these same factors: e-d-BH (ours), e-d-Bonferroni/SR
(ours), e-d-Bonferroni/CUSUM (ours), and SHEREF (fixed-time FDR). SR uses `(1+R_previous)*L`; the restart
CUSUM uses `max(1,C_previous)*L`, which can be below one and is exactly the
maximum likelihood product over nonempty restart windows. BH uses step-up
thresholds K/(r*alpha), while both parallel charts use K/alpha, with
alpha=q/H=.05/400. Thus BH controls FDR at level q for stopping rules bounded
by H=400, while the parallel charts instead control PFER at level q. SHEREF
applies e-BH to SR/t at fixed-time FDR level q=.05. These different guarantees
are stated explicitly; achieved ordinary FDR and ARL were not matched. The
figures report the **same measured ordinary FDR and TDR for every method**,
so an identification gain must be read alongside its false-discovery cost.
SHEREF has weaker thresholds than BH before H and hence at least as many
true declarations, with equality at H; its guarantee concerns deterministic
reporting times rather than all stopping rules bounded by H.

`diverse_heavy.pdf` and `diverse_staggered.pdf` pair FDR and TDR curves under
both dependence structures. TDR counts only streams already changed, so it
can drop when the second group changes. `diverse_counts.pdf` compares FDR
and TDR at the prespecified time 200 as K grows, with separate sparse and
dense rows. This is a study of statistical performance as the number of
streams grows, not a computational-runtime benchmark. Shading/bars represent
1.96 estimated Monte Carlo standard errors, with rates clipped to [0,1].
These are simulation-error summaries, not exact coverage guarantees. An
estimated rate and standard error of zero mean that no relevant errors
occurred in this finite sample; they do not establish a true error rate of
zero.

`diverse_summary.csv` contains all reporting times, including null errors
and probabilities of identifying 80% or all currently changed streams.
`diverse_curves.csv` contains each fifth monitoring time. `diverse_report.txt`
lists the prespecified reporting-time results with standard errors; the
`diverse_*.npz` files contain replicate-level outcomes. Delays retain
nondetections by assigning detection time 401, as in the original Gaussian
study. Selected stopping-rule ratios are saved for auditing, but are not
plotted or interpreted as estimates of the EOP supremum. The seeds,
software versions, design hash and execution time are recorded separately.

The first diverse-study run used alpha=.001 and three methods (without
SHEREF). To align with the main paper protocol, a recorded amendment aligned the
thresholds with the main paper's finite-horizon q=.05 budget and added
SHEREF. Every scenario, seed, sample size, effect size and data-generation
choice was retained. This was a protocol alignment, not optimization based
on results. All initial-level CSV, raw outcomes, plots, metadata, design and
source are archived in the manuscript workspace. The amended design
was saved before the new run and its hash appears in the aligned metadata.

The manuscript uses `diverse_heavy.pdf` and `diverse_counts.pdf` in the main
text and `diverse_staggered.pdf` in the supplement. These retain ordinary FDR,
SHEREF, independent and dependent observations, and both affected fractions.
`diverse_main.pdf` is an earlier compact overview and is not included in the
manuscript. The generated `diverse_appendix.tex`, `diverse_main_caption.tex`
and `diverse_results.tex` are working reference fragments; the edited prose
and captions in the manuscript determine the compiled paper.

## Relation to the Gaussian optimality theorem

The manuscript proves a matching minimax lower bound:
the worst conditional expected delay to r simultaneous true declarations
is asymptotic to 2 log(1/alpha)/mu^2 as alpha decreases to zero. K, m, r and
mu are fixed; all Gaussian observations are independent; the shift is known.
The admissible procedure must control FDR EOP over every configuration,
including those with fewer than m changed streams. Monitoring continues after
earlier declarations. Both e-d-BH and e-d-Bonferroni, with SR or CUSUM inputs,
attain the bound. This is not a first-alarm conditional-delay result or a claim
of optimality with growing K, and it does not distinguish smaller-order BH
and Bonferroni differences. The simulations examine finite-sample behavior
and distinct global/identification decisions; they do not estimate the EOP
supremum or serve as a proof of the asymptotic theorem.

## Exploratory public sensor analysis

`home_sensor_study.py` implements the fixed protocol in `home_sensor_design.txt`
for UCI's [Gas sensors for home activity monitoring](https://doi.org/10.24432/C5BS4F).
Create `experiments/data/home-sensors/` and save the
[UCI archive](https://archive.ics.uci.edu/static/public/362/gas%2Bsensors%2Bfor%2Bhome%2Bactivity%2Bmonitoring.zip)
as `experiments/data/home-sensors/source.zip`, then run
`python3.11 experiments/home_sensor_study.py`. The source SHA256 is checked.
The script writes processed bins, provenance, inclusions, outcomes and model
diagnostics. Downloaded archives are ignored by git.

All three methods alerted before the reference time in every retained record.
Held-out residual diagnostics did not support the conditional-symmetry model
used to construct evidence. These results are retained as exploratory work
and are not used as real-data validation in the manuscript. The dataset lacks
stream-specific ground truth, so its flagged-channel proportions are not TDR.
See `results/home_sensor_readout.txt`.
