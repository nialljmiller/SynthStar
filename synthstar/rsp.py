"""MESA RSP (radial stellar pulsation) models of the followed pulsators of a field (night_movie.py).

  python -m synthstar.rsp                       M3: tune, run to the limit cycle, write the high-resolution cycles
  python -m synthstar.rsp V1 V3                 only these
(make_stars.py runs it for Gaia variables and hand-made pulsators, in process)

Written for M3 and described for it below; make_stars.py's pulsators use the same steps with their own composition, mass,
distance and reddening (data/<field>/field.json). For each followed star (data/<field>/followed.json):

1. Physical parameters. Composition and mass are M3's: X = 0.75, Z = 0.001 ([Fe/H] ~ -1.5,
   alpha-enhanced; a09 opacities), M = 0.68 Msun for RR Lyrae (M3 horizontal branch). L starts
   from Gaia: the median mean G of M3's RR Lyrae at M3's distance and reddening (bolometric
   correction ~0 for F stars), ~50 Lsun. Convection: RSP set A (MESA's default).
2. Teff is tuned with RSP's linear nonadiabatic analysis (LINA) until the linear period of the
   star's mode -- fundamental for RRab, first overtone for RRc -- equals the Gaia period to 0.1%.
   That mode must be the only unstable one of the two lowest, else the nonlinear model drifts to
   the other mode or double-mode pulsation; if not, L is lowered in 5% steps, which moves the
   period-matched model to a cooler Teff. Result: the four RRab needed L = 0.80 x cluster (39.7
   Lsun, Teff 6406-6503 K), the RRc 0.90 and 0.95 x (Teff ~7240-7270 K). So the RRab models are
   ~0.25 mag fainter than the cluster's RR Lyrae level: a tension between these set-A models and
   M3's short-period RRab, left visible rather than rescaled.
3. Nonlinear RSP from a kick near the expected full amplitude (40 km/s RRab, 20 km/s RRc; from a
   small kick these weakly unstable models grow only ~0.3% per period), at 300 steps/cycle, until
   the kinetic-energy amplitude changes < 0.05% per period (RSP_GREKM_avg_abs_limit), or 800
   periods. RRc V7 (Gaia 1454784...; overtone growth only +0.002/period) converges too slowly for
   that: its limit cycle was bracketed by runs from four kicks and the run whose amplitude hardly
   drifts (dR 0.096 vs the bracketed ~0.100 Rsun) is used (params.json records this).
4. Reloaded, the limit cycle is run for 3 more periods at <= 25 s per step (half the Argus 60 s
   cadence or better) with the colors module writing the photospheric SED (f_lambda at 10 pc)
   at every step, packed into runs/rsp_<label>/pulsation.h5.

Amplitudes and light-curve shapes are predictions of the RSP models, not fitted to Gaia: the
RRab models' rho amplitudes (~0.53 mag) are smaller than the real stars' (Gaia G ~1.1 mag),
a known tendency of set-A models. The block's 15.3 d Type II Cepheid could not be modelled:
with default and set C/D convection, gentle and strong kicks, and higher mass/viscosity, its
envelope either cools below the opacity tables (logT < 2.7) or diverges within ~10 cycles.
"""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

from astropy.table import Table
import h5py
import numpy as np

from . import config as C
from .seds import read_sed_csv

# Cluster parameters (literature; assumptions of the models). use() selects one.
CLUSTERS = {
    # M3: 10.18 kpc (Baumgardt & Vasiliev 2021); E(B-V) Harris (1996, 2010 ed.); [Fe/H] -1.5 alpha-enhanced
    "m3": dict(dm=15.04, ebv=0.01, z=0.001, mass_rr=0.68, prefix=""),
    # M15: 10.71 kpc (Baumgardt & Vasiliev 2021); E(B-V) 0.10 and [Fe/H] -2.37 (Harris); alpha-enhanced
    # [M/H] ~ -2.08 (Salaris et al. 1993) -> Z ~ 0.00015; very metal-poor HB RR Lyrae ~0.75 Msun
    "m15": dict(dm=15.15, ebv=0.10, z=0.00015, mass_rr=0.75, prefix="m15_"),
    # field RRab Gaia DR3 308108245007332224: distance from its parallax (0.269 +/- 0.021 mas, +0.017 zero point
    # -> 3.50 kpc), E(B-V) Schlafly & Finkbeiner (2011), Gaia's photometric [M/H] = -0.06 (solar-scaled, not
    # alpha-enhanced: a disk star), Z = 0.0169 x 10^-0.06; metal-rich RR Lyrae ~0.55 Msun
    "rr308108": dict(dm=12.72, ebv=0.044, z=0.0147, mass_rr=0.55, prefix="rr308108_", kap="gs98"),
    # field RRab Gaia DR3 416445908838640384: parallax 0.253 +/- 0.025 mas (+0.017 zero point -> 3.70 kpc), E(B-V)
    # Schlafly & Finkbeiner (2011), Gaia photometric [M/H] = -0.33 +/- 0.22 -> Z = 0.0169 x 10^-0.33 (solar-scaled);
    # mass 0.58 Msun assumed (between the metal-rich 0.55 and M3's 0.68)
    "rr416445": dict(dm=12.84, ebv=0.203, z=0.0079, mass_rr=0.58, prefix="rr416445_", kap="gs98"),
}
KAP = {"a09": ["kap_file_prefix = 'a09'", "kap_lowT_prefix = 'lowT_fa05_a09p'", "kap_CO_prefix = 'a09_co'"],
       "gs98": ["kap_file_prefix = 'gs98'", "kap_lowT_prefix = 'lowT_fa05_gs98'", "kap_CO_prefix = 'gs98_co'"]}
KAP_MIX = "a09"
CLUSTER = "m3"
TARGETS = C.DATA / "m3" / "followed.json"
DM = CLUSTERS["m3"]["dm"]
EBV = CLUSTERS["m3"]["ebv"]
PREFIX = ""
A_G_PER_EBV = 2.74
X, Z = 0.75, 0.001
# classical Cepheids (DCEP) carry their own mass (synthstar/gaia.py: Bono et al. 2000b mass-luminosity relation);
# BL Her and W Vir default to the 0.55 Msun of MESA's rsp_Type_II_Cepheid test model
MASS = {"RRab": 0.68, "RRc": 0.68, "RRd": 0.68, "Type II Cepheid": 0.55, "BL Her": 0.55, "W Vir": 0.55}

MODE = {"RRab": 0, "RRc": 1, "RRd": 0, "Type II Cepheid": 0, "DCEP F": 0, "DCEP 1O": 1, "BL Her": 0, "W Vir": 0,
        "HADS": 0}
STAR = C.STAR_BIN
RUN_LIMIT_S = 4 * 3600  # wall-clock budget of one MESA run; a star whose run exceeds it gets no model (held constant)
DM_RUN_LIMIT_S = 8 * 3600  # the long RRd double-mode runs
SED_INSTRUMENT = C.REPO / "filters" / "SEDonly" / "SEDonly"
KICK = {"RRab": 40.0, "RRc": 20.0, "RRd": 30.0, "Type II Cepheid": 5.0,  # km/s, near full amplitude
        "DCEP F": 15.0, "DCEP 1O": 10.0, "BL Her": 4.5, "W Vir": 5.0, "HADS": 10.0}
LIMIT_PERIODS = {"RRab": 800, "RRc": 800, "RRd": 800, "Type II Cepheid": 300,  # backstops
                 "DCEP F": 400, "DCEP 1O": 400, "BL Her": 2000, "W Vir": 300, "HADS": 3000}
# steps per cycle in the limit-cycle run (MESA's test suite: 600 default; rsp_Type_II_Cepheid 1200)
LIMIT_STEPS = {"DCEP F": 600, "DCEP 1O": 600, "BL Her": 1200, "W Vir": 1200, "HADS": 600}
# extra controls from MESA's test-suite models, in every stage (rsp_Type_II_Cepheid: zoning of the extended envelope)
# HADS: purely radiative (no convection). With RSP's turbulent convection (default or rsp_Delta_Scuti's mixing
# length 1.2) the modes of a HADS sit at neutral stability (growth ~ -1e-5 per period over L x 0.7-1.5), so no
# limit cycle forms; radiative models pulsate, but nothing convective limits their amplitude (no red edge)
EXTRA_CONTROLS = {"HADS": ["RSP_alfa = 0d0"],
                  "BL Her": ["RSP_nz = 200", "RSP_nz_outer = 60", "RSP_T_anchor = 11d3"],
                  # W Vir: the static envelope build needs a looser anchor tolerance (MESA's own suggestion)
                  "W Vir": ["RSP_nz = 200", "RSP_nz_outer = 60", "RSP_T_anchor = 11d3", "RSP_T_anchor_tolerance = 1d-5"]}
# non-linear acceptance: the limit cycle's period within this fraction of Gaia's (the F/1O period ratio is
# ~0.70-0.75, so this also confirms the mode)
MODE_TOL = 0.02
MODE_WINDOW = 0.10  # within this, a limit cycle is in the star's mode and its Teff is corrected
MAX_CORRECTIONS = 2
MAX_NONLINEAR_ATTEMPTS = 4  # per star, across luminosities, Teff corrections and kick trials


class NonlinearAttemptLimit(RuntimeError):
    """The star has used its full nonlinear-search budget."""


def _attempt_budget(base, stage=None):
    """Read or claim a persistent attempt, locked across threads/processes.

    Older runs have no ledger: seed their count from recorded rejected attempts.
    A claimed attempt remains counted after interruption/restart. Linear tuning
    and final SED extraction do not consume nonlinear-search attempts.
    """
    base.mkdir(parents=True, exist_ok=True)
    path = base / "nonlinear_attempts.json"
    with (base / ".nonlinear_attempts.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            budget = json.loads(path.read_text())
        else:
            rejected = base / "rejected.json"
            legacy = len(json.loads(rejected.read_text())) if rejected.exists() else 0
            budget = dict(legacy_rejected=legacy, attempts=[])
        used = budget["legacy_rejected"] + len(budget["attempts"])
        if stage is not None:
            if used >= MAX_NONLINEAR_ATTEMPTS:
                raise NonlinearAttemptLimit(f"{used} nonlinear attempts used; limit {MAX_NONLINEAR_ATTEMPTS}")
            budget["attempts"].append(dict(stage=stage, started=time.strftime("%Y-%m-%d %H:%M:%S")))
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(budget, indent=1) + "\n")
            os.replace(temp, path)
            used += 1
        return max(0, MAX_NONLINEAR_ATTEMPTS - used)
# weakly unstable models (growth ~1e-5 per period: HADS, some RRc) meet the GREKM stop at once and would need
# ~1e5 periods to reach their limit cycle: bracket instead, from these multiples of the kind's kick, over
# BRACKET_PERIODS periods each (in parallel); the limit cycle lies where the amplitude drift changes sign
BRACKET_FACTORS = (0.25, 0.5, 1.0, 2.0, 3.0)
BRACKET_PERIODS = 150
# RSP convective parameter sets (Paxton et al. 2019, ApJS 243, 10, Table 4); set A is RSP's default
CONVECTION_SETS = {
    "A": ["RSP_alfa = 1.5d0", "RSP_alfam = 0.25d0", "RSP_alfap = 0.0d0", "RSP_alfat = 0.0d0", "RSP_gammar = 0.0d0"],
    "B": ["RSP_alfa = 1.5d0", "RSP_alfam = 0.50d0", "RSP_alfap = 0.0d0", "RSP_alfat = 0.0d0", "RSP_gammar = 1.0d0"],
    "C": ["RSP_alfa = 1.5d0", "RSP_alfam = 0.40d0", "RSP_alfap = 1.0d0", "RSP_alfat = 0.01d0", "RSP_gammar = 0.0d0"],
    "D": ["RSP_alfa = 1.5d0", "RSP_alfam = 0.70d0", "RSP_alfap = 1.0d0", "RSP_alfat = 0.01d0", "RSP_gammar = 1.0d0"],
}
# double-mode RR Lyrae: "an unsolved problem" (Paxton et al. 2019); attempted in every set from a mixed F + 1O kick
DM_PERIODS = 1500       # periods of the non-linear run
DM_MIN_RATIO = 0.10     # each mode's luminosity amplitude >= this fraction of the stronger one
DM_RATIO_STABILITY = 0.20  # the amplitude ratio may change by at most this between the two quarters analysed
DM_FINE_DAYS = 1.0      # SEDs over this much model time (a double-mode light curve does not repeat each cycle)
MAX_FINE_STEPS = 4000  # SEDs per cycle in the fine run (long-period stars: >25 s apart, still < the 60 s cadence)
FINE_STEP_S = 25.0


def use(cluster):
    """Point every module-level parameter at one cluster (default m3), or at an old argus_field.py field
    (data/<field>/field.json; its targets carry their own distance, reddening, Z, mass and opacities)."""
    global CLUSTER, TARGETS, DM, EBV, Z, PREFIX, KAP_MIX
    if cluster not in CLUSTERS:
        CLUSTERS[cluster] = json.loads((C.DATA / cluster / "field.json").read_text())["rsp"]
    p = CLUSTERS[cluster]
    KAP_MIX = p.get("kap", "a09")
    CLUSTER, TARGETS, DM, EBV, Z, PREFIX = cluster, C.DATA / cluster / "followed.json", p["dm"], p["ebv"], p["z"], p["prefix"]
    for kind in ("RRab", "RRc", "RRd"):
        MASS[kind] = p["mass_rr"]


def m_bol_to_lsun(m_bol):
    return 10 ** ((4.74 - m_bol) / 2.5)


def luminosity(target):
    """L (Lsun) from Gaia mean G at M3's distance; RR Lyrae share the cluster's horizontal-branch level.
    A field star with its own distance (target["dm"]) uses its own Gaia intensity-mean G; a star made from parameters
    (make_stars.py) may give L itself."""
    if "L" in target:
        return float(target["L"])
    if "dm" in target:  # A_G: the star's own (synthstar/gaia.py physical()); older targets 2.74 E(B-V)
        return m_bol_to_lsun(target["G_mean"] - target["dm"] - target.get("A_G", A_G_PER_EBV * target["ebv"]))
    if target["kind"].startswith("RR"):
        rr = Table.read(C.DATA / CLUSTER / "gaia_rrlyrae.ecsv")
        col = rr["Gmagavg"]
        g = np.nanmedian(np.asarray(col.filled(np.nan) if hasattr(col, "filled") else col, float))
    else:
        g = target["G"]
    return m_bol_to_lsun(g - DM - A_G_PER_EBV * EBV)


RSP_INSTRUMENT = C.REPO / "filters" / "RSP_ArgusGaia" / "RSP_ArgusGaia"


def build_sed_instrument():
    """The RSP runs' instruments, from `python -m synthstar setup`'s ArgusGaiaSDSS: Gaia + Argus (no SDSS), and a
    one-filter instrument with a short table for the SED cycles (MESA pads each SED CSV to the filter's length)."""
    full = C.REPO / "filters" / "ArgusGaiaSDSS" / "ArgusGaiaSDSS"
    if not RSP_INSTRUMENT.exists():
        RSP_INSTRUMENT.mkdir(parents=True)
        names = ["G.dat", "Gbp.dat", "Grp.dat", "beta.dat", "rho.dat"]
        for f in names:
            shutil.copy(full / f, RSP_INSTRUMENT / f)
        (RSP_INSTRUMENT / RSP_INSTRUMENT.name).write_text("\n".join(names) + "\n")
    if SED_INSTRUMENT.exists():
        return
    SED_INSTRUMENT.mkdir(parents=True)
    src = np.loadtxt(C.REPO / "filters" / "ArgusGaiaSDSS" / "ArgusGaiaSDSS" / "rho.dat", delimiter=",", skiprows=1)
    w = np.arange(5500.0, 8300.0, 20.0)
    t = np.interp(w, src[:, 0], src[:, 1], left=0.0, right=0.0)
    lines = ["Wavelength,Transmission"] + [f"{a:.6f},{b:.12e}" for a, b in zip(w, t)]
    (SED_INSTRUMENT / "rho_sed.dat").write_text("\n".join(lines) + "\n")
    (SED_INSTRUMENT / "SEDonly").write_text("rho_sed.dat\n")


def inlist(params, stage):
    cd = C.colors_data()
    star_job = ["show_log_description_at_start = .false.", "save_model_when_terminate = .true.",
                f"save_model_filename = '{stage}.mod'",
                f"history_columns_file = '{C.mesa_dir() / 'star/test_suite/rsp_RR_Lyrae/history_columns.list'}'"]
    if stage == "fine":
        star_job += ["load_saved_model = .true.", "load_model_filename = 'limit.mod'"]
    else:
        star_job += ["create_RSP_model = .true.", "initial_zfracs = 6", "set_initial_age = .true.", "initial_age = 0",
                     "set_initial_model_number = .true.", "initial_model_number = 0"]
    controls = [f"RSP_mass = {params['mass']}d0", f"RSP_Teff = {params['teff']:.4f}d0", f"RSP_L = {params['L']:.5f}d0",
                f"RSP_X = {params.get('x', X)}d0", f"RSP_Z = {params.get('z', Z)}d0", "photo_interval = 100000000", "write_profiles_flag = .false.",
                "terminal_interval = 2000"] + list(params.get("extra_controls", []))
    if stage == "lina":
        controls += ["max_model_number = 1", "history_interval = 1000"]
    elif stage == "validate":  # the test suite's own settings: MESA defaults for kick, steps per cycle and stopping
        controls += [f"RSP_max_num_periods = {params['limit_periods']}", "history_interval = 200"]
    elif stage == "limit":
        # stop at the limit cycle: kinetic-energy amplitude changing < 0.05% per period (checked from period 10)
        # start near the limit-cycle amplitude (weakly unstable models grow ~0.3%/period from a small
        # kick: thousands of periods); converge at 300 steps/cycle, the SED cycles are run finer
        controls += [f"RSP_max_num_periods = {params['limit_periods']}", f"history_interval = {params.get('history_interval', 200)}",
                     f"RSP_GREKM_avg_abs_limit = {params.get('grekm', '5d-4')}",  # 0d0 (brackets): run all periods
                     f"RSP_target_steps_per_cycle = {params.get('limit_steps', 300)}",
                     f"RSP_kick_vsurf_km_per_sec = {params.get('kick', 5.0)}d0",
                     f"RSP_fraction_1st_overtone = {params.get('fraction_1o', 1.0 if params['mode'] == 1 else 0.0)}d0"]
    else:
        controls += [f"RSP_target_steps_per_cycle = {params['steps']}", f"RSP_max_num_periods = {params.get('fine_periods', 3)}",
                     "history_interval = 1"]
    make_csv = ".true." if stage == "fine" else ".false."
    instrument = SED_INSTRUMENT if stage == "fine" else RSP_INSTRUMENT

    def block(name, lines):
        return f"&{name}\n" + "".join(f"      {ln}\n" for ln in lines) + "/\n"

    return (block("star_job", star_job) + "&eos\n/\n"
            + block("kap", [f"Zbase = {params.get('z', Z)}d0"] + KAP[params.get("kap", KAP_MIX)])
            + block("colors", ["use_colors = .true.", f"instrument = '{instrument}'",
                               f"vega_sed = '{cd / 'stellar_models' / 'vega_flam.csv'}'",
                               f"stellar_atm = '{cd / 'stellar_models' / 'Kurucz2003all'}/'",
                               "distance = 3.0857d19", f"make_csv = {make_csv}", f"sed_per_model = {make_csv}",
                               "colors_results_directory = 'SED'"])
            + block("controls", controls) + "&pgstar\n/\n")


# OpenMP threads per RSP run. Measured on a 32-core node (RRab and Cepheid, 20 periods; results identical at every
# thread count): speed-up 2.65x at 4 threads, 3.8x at 8, 4.3x at 12, i.e. per-core efficiency 66%, 47%, 36%.
# Auto mode (config.toml [run] rsp_threads = 0) gives each run, as it starts, the free cores / the runs waiting,
# between 1 and AUTO_MAX_THREADS: threads only when cores would otherwise sit idle, never past 4, so a run holding
# many cores for an hour does not hold up later runs.
AUTO_MAX_THREADS = 4
SETTLE_SECONDS = 1.0  # a star submits its runs together (convection sets, kicks): let them all register first
_pool = threading.Condition()
_free = None
_waiting = 0


def cores():
    """Cores this process's RSP runs may use: $ARGUS_CORES (set by tools/stage_batch.py to share the machine between
    fields run at once), else config.toml [run] max_star_procs, else all."""
    if os.environ.get("ARGUS_CORES"):
        return int(os.environ["ARGUS_CORES"])
    return int(C.config()["run"]["max_star_procs"]) or os.cpu_count()


def take_cores():
    """Block until cores are free; return the OpenMP thread count for this run (also the cores it holds)."""
    global _free, _waiting
    fixed = int(C.config()["run"]["rsp_threads"])
    with _pool:
        if _free is None:
            _free = cores()
        _waiting += 1
    time.sleep(SETTLE_SECONDS)
    with _pool:
        need = min(fixed, cores()) if fixed else 1
        _pool.wait_for(lambda: _free >= need)
        n = need if fixed else max(1, min(AUTO_MAX_THREADS, _free // _waiting))
        _free -= n
        _waiting -= 1
    return n


def give_cores(n):
    global _free
    with _pool:
        _free += n
        _pool.notify_all()


def run(directory: Path, text: str, log="out.txt", limit_s=RUN_LIMIT_S):
    """Run MESA in directory; every RSP run goes through here, so the core pool covers all stars, convection sets
    and kicks of this process. The thread count used is written to directory/omp_threads."""
    if directory.name == "limit" or directory.name.startswith(("bracket_", "dm_")):
        _attempt_budget(directory.parent, directory.name)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inlist").write_text(text)
    n = take_cores()
    (directory / "omp_threads").write_text(f"{n}\n")
    try:
        with open(directory / log, "w") as out:
            subprocess.run([str(STAR)], cwd=directory, stdout=out, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS=str(n)), timeout=limit_s)
    finally:
        give_cores(n)  # also after a timeout, or the other stars wait for these cores forever
    return (directory / log).read_text(errors="replace")


def linear(params, directory):
    """Periods (d) and growth rates per period of the lowest modes (RSP's LINA), or None if the static model fails."""
    shutil.rmtree(directory / "LOGS", ignore_errors=True)  # never read a previous run's results
    run(directory, inlist(params, "lina"), "lina.txt")
    out = directory / "LOGS" / "LINA_period_growth.data"
    if not out.exists():
        return None
    table = np.loadtxt(out)
    return table[:, 1], table[:, 2]


def tune_teff(label, target, base, L):
    """The linear model at L with Teff tuned until the period of the star's mode equals Gaia's to 0.1%, or None
    if the linear analysis fails (e.g. the static envelope cannot be built)."""
    mode, p_target = MODE[target["kind"]], target["period"]
    params = dict(mass=target["mass"] if "mass" in target else MASS[target["kind"]], L=L, mode=mode, z=target.get("z", Z),
                  kap=target.get("kap", KAP_MIX), x=target.get("x", X), extra_controls=EXTRA_CONTROLS.get(target["kind"], []))
    # first guess from the period-Teff scaling (P ~ Teff^-3.5 at fixed L, M), anchored on a 6400 K RRab
    p0 = p_target / (0.745 if mode == 1 else 1.0)
    teffs, periods = [6400.0 * (0.594 * (params["L"] / 47.0) ** 0.84 * (0.68 / params["mass"]) ** 0.68 / p0) ** (1 / 3.48)], []
    for _ in range(8):
        params["teff"] = teffs[-1]
        lina = linear(params, base / "lina")
        if lina is None:
            print(f"  {label}: linear analysis failed at Teff {params['teff']:.0f} K, L {L:.2f}; see {base / 'lina' / 'lina.txt'}",
                  flush=True)
            return None
        p, growth = lina
        periods.append(p[mode])
        if abs(periods[-1] / p_target - 1) < 1e-3:
            break
        if len(periods) == 1:
            teffs.append(teffs[-1] * (periods[-1] / p_target) ** (1 / 3.48))
        else:  # secant in log Teff - log P
            slope = (np.log(periods[-1]) - np.log(periods[-2])) / (np.log(teffs[-1]) - np.log(teffs[-2]))
            teffs.append(float(np.exp(np.log(teffs[-1]) + (np.log(p_target) - np.log(periods[-1])) / slope)))
    params.update(teff=teffs[len(periods) - 1], linear_periods=p.tolist(), growth=growth.tolist(), linear_period=periods[-1])
    return params


def candidates(label, target, base):
    """Period-matched linear models with the star's own mode unstable, in the order to try: the star's L first,
    then L stepped down 5% at a time to 0.70 x (which moves the period-matched model to a cooler Teff), or up to
    1.50 x for an RRc whose fundamental is unstable (too cool); the non-linear run decides the rest."""
    L0 = luminosity(target)
    mode = MODE[target["kind"]]
    up, down = (1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50), (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
    p0 = tune_teff(label, target, base, L0)
    growth = p0["growth"] if p0 else [0.0, 0.0, 0.0]
    order = up if mode == 1 and growth[0] > 0 else down
    if order is down and growth[mode] <= 0 and any(growth):
        # the star's mode is stable at its own L: step L the way that raises its growth rate
        p_up, p_down = tune_teff(label, target, base, L0 * 1.05), tune_teff(label, target, base, L0 * 0.95)
        if p_up is None or p_down is None:
            return
        g_up, g_down = p_up["growth"][mode], p_down["growth"][mode]
        order = up if g_up > g_down else down
        print(f"  {label}: mode {mode} stable at L; growth {g_down:.3g} at L x 0.95, {g_up:.3g} at L x 1.05: stepping "
              f"L {'up' if order is up else 'down'}", flush=True)
    for factor in (1.0,) + order:
        params = tune_teff(label, target, base, L0 * factor)
        if params is None:
            continue
        if params["growth"][mode] <= 0:
            print(f"  {label}: L x {factor}: mode {mode} is linearly stable at Teff {params['teff']:.0f} K "
                  f"(growth {params['growth']})", flush=True)
            continue
        params.update(cluster_L=L0, L_factor=factor)
        yield params


def mode_amplitudes(history, periods, frac=(0.5, 1.0)):
    """Luminosity amplitude of the F and 1O modes over a fraction of the run's time span. Each mode's frequency is
    located (single-sinusoid amplitude maximised within +/-5% of the given period's frequency, resolution a tenth of
    1/span), then both are fitted together with their harmonics and combinations (least squares)."""
    h = np.genfromtxt(history, skip_header=5, names=True)
    t = h["star_age"] * 365.25
    y = 10 ** h["log_L"]
    sel = (t >= t[0] + frac[0] * (t[-1] - t[0])) & (t <= t[0] + frac[1] * (t[-1] - t[0]))
    t, y = t[sel], y[sel] / np.mean(y[sel]) - 1
    span = t[-1] - t[0]

    def amp_at(f):
        a = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)])
        c = np.linalg.lstsq(a, y, rcond=None)[0]
        return np.hypot(c[0], c[1])

    found = []
    for p_ in periods:
        f0 = 1 / p_
        grid = np.arange(0.95 * f0, 1.05 * f0, 0.1 / span)
        found.append(float(grid[np.argmax([amp_at(f) for f in grid])]))
    f1, f2 = found
    freqs = [f1, f2, 2 * f1, 2 * f2, f1 + f2, f2 - f1]
    cols = [np.ones_like(t)] + [fn(2 * np.pi * f * t) for f in freqs for fn in (np.sin, np.cos)]
    coef = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]
    return float(np.hypot(coef[1], coef[2])), float(np.hypot(coef[3], coef[4]))


def accept_double_mode(label, target, base):
    """A double-mode (F + 1O) limit cycle, or None.

    Linear: Teff tuned so F matches Gaia's fundamental period, at the first L (factors 1, 0.95, 1.05, ...) where F and
    1O are both unstable; the model's 1O period is compared with Gaia's. Non-linear: in each convection set (A-D, in
    parallel) a mixed F + 1O kick, DM_PERIODS periods; accepted if both modes keep >= DM_MIN_RATIO of the stronger's
    luminosity amplitude with a ratio stable to DM_RATIO_STABILITY between the third and fourth quarters of the run."""
    L0 = luminosity(target)
    p_1o = target.get("period2")
    found = None
    for factor in (1.0, 0.95, 1.05, 0.90, 1.10, 0.85, 1.15, 0.80, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50, 1.60):
        params = tune_teff(label, target, base, L0 * factor)
        if params is None:
            continue
        g = params["growth"]
        if g[0] > 0 and g[1] > 0:
            found = dict(params, cluster_L=L0, L_factor=factor)
            break
        print(f"  {label}: L x {factor}, Teff {params['teff']:.0f} K: F and 1O not both unstable (growth {g[:2]})", flush=True)
    if found is None:
        print(f"  {label}: no L where F and 1O are both linearly unstable at Gaia's fundamental period", flush=True)
        return None
    lin_1o = found["linear_periods"][1]
    print(f"  {label}: L x {found['L_factor']}, Teff {found['teff']:.0f} K: linear P1O {lin_1o:.5f} d"
          + (f" vs Gaia {p_1o:.5f} d (P1O/PF {lin_1o / target['period']:.4f} vs {p_1o / target['period']:.4f})" if p_1o else ""),
          flush=True)

    def one(name):
        p = dict(found, extra_controls=list(found.get("extra_controls", [])) + CONVECTION_SETS[name], fraction_1o=0.5,
                 kick=KICK["RRd"], limit_periods=DM_PERIODS, grekm="0d0", history_interval=20, limit_steps=300)
        d = base / f"dm_{name}"
        shutil.rmtree(d, ignore_errors=True)
        try:
            text = run(d, inlist(p, "limit"), limit_s=DM_RUN_LIMIT_S)
        except NonlinearAttemptLimit:
            return dict(set=name, ok=False, reason="nonlinear attempt budget exhausted")
        if "NO CONVERGENCE IN RELAX_ENV" in text:  # MESA's own suggestion: a looser envelope-relaxation tolerance
            p["extra_controls"] = p["extra_controls"] + ["RSP_relax_dm_tolerance = 1d-5"]
            shutil.rmtree(d, ignore_errors=True)
            try:
                text = run(d, inlist(p, "limit"), limit_s=DM_RUN_LIMIT_S)
            except NonlinearAttemptLimit:
                return dict(set=name, ok=False, reason="nonlinear attempt budget exhausted")
        lines = period_lines(text)
        if len(lines) < 200 or not (d / "limit.mod").exists():
            return dict(set=name, ok=False, reason="run failed")
        per = (found["linear_periods"][0], found["linear_periods"][1])
        a3 = mode_amplitudes(d / "LOGS" / "history.data", per, (0.5, 0.75))
        a4 = mode_amplitudes(d / "LOGS" / "history.data", per, (0.75, 1.0))
        r3, r4 = a3[1] / max(a3[0], 1e-12), a4[1] / max(a4[0], 1e-12)
        weak = min(a4) / max(a4)
        ok = weak >= DM_MIN_RATIO and abs(r4 / r3 - 1) <= DM_RATIO_STABILITY
        why = ("double-mode" if ok else
               "F only" if a4[1] < DM_MIN_RATIO * a4[0] else "1O only" if a4[0] < DM_MIN_RATIO * a4[1] else
               f"mode amplitude ratio not stable ({r3:.2f} -> {r4:.2f})")
        return dict(set=name, ok=ok, reason=why, amp_F=a4[0], amp_1O=a4[1], ratio_q3=r3, ratio_q4=r4, params=p)

    with ThreadPoolExecutor(len(CONVECTION_SETS)) as ex:
        res = list(ex.map(one, CONVECTION_SETS))
    for r in res:
        print(f"  {label} convection set {r['set']}: {r['reason']}"
              + (f" (L amplitudes F {r['amp_F']:.3f}, 1O {r['amp_1O']:.3f}; 1O/F {r['ratio_q3']:.2f} -> {r['ratio_q4']:.2f})"
                 if "amp_F" in r else ""), flush=True)
    (base / "double_mode.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "params"} for r in res], indent=1))
    good = [r for r in res if r["ok"]]
    if not good:
        print(f"  {label}: no convection set gives stable double-mode pulsation (see {base / 'double_mode.json'}); "
              "double-mode pulsation is an unsolved problem in RSP (Paxton et al. 2019)", flush=True)
        return None
    pick = max(good, key=lambda r: min(r["amp_F"], r["amp_1O"]) / max(r["amp_F"], r["amp_1O"]))
    (base / "limit").mkdir(exist_ok=True)
    shutil.copy(base / f"dm_{pick['set']}" / "limit.mod", base / "limit" / "limit.mod")
    return dict(pick["params"], convection_set=pick["set"], nonlinear_period=target["period"], double_mode=True,
                limit_cycle_note=f"double-mode in convection set {pick['set']}: 1O/F luminosity amplitude "
                                 f"{pick['ratio_q3']:.2f} -> {pick['ratio_q4']:.2f}", mode_check="both modes present")


def bracket_limit(label, target, base, params):
    """Limit cycle of a weakly unstable model by bracketing (see BRACKET_FACTORS). Returns the accepted params, with
    base/limit/limit.mod taken from the run nearest the drift's sign change, or None if the period is not the star's."""
    kicks = [round(KICK[target["kind"]] * f, 3) for f in BRACKET_FACTORS]
    kicks = kicks[:_attempt_budget(base)]
    if not kicks:
        raise NonlinearAttemptLimit(f"nonlinear attempt budget exhausted (limit {MAX_NONLINEAR_ATTEMPTS})")
    print(f"  {label}: weakly unstable (convergence stop met at once): bracketing kicks {kicks} km/s, "
          f"{BRACKET_PERIODS} periods each", flush=True)

    def one(kick):
        p = dict(params, kick=kick, limit_periods=BRACKET_PERIODS, grekm="0d0")
        d = base / f"bracket_{kick:g}"
        shutil.rmtree(d, ignore_errors=True)
        lines = period_lines(run(d, inlist(p, "limit")))
        if len(lines) < 30 or not (d / "limit.mod").exists():
            return dict(kick=kick, ok=False)
        n, per, dr = (np.array(x) for x in zip(*lines))
        return dict(kick=kick, ok=True, drift=float(np.polyfit(n[10:], dr[10:], 1)[0] / dr[10:].mean()),
                    period=float(per[-20:].mean()), period_scatter=float(per[-20:].std()), delta_R=float(dr[-20:].mean()))

    with ThreadPoolExecutor(len(kicks)) as ex:
        res = [r for r in ex.map(one, kicks) if r["ok"]]
    for r in res:
        print(f"  {label} kick {r['kick']:g}: dR {r['delta_R']:.4f} Rsun, drift {r['drift']:+.2e} per period, "
              f"P {r['period']:.5f} +/- {r['period_scatter']:.5f} d", flush=True)
    (base / "bracket.json").write_text(json.dumps(res, indent=1))
    res = [r for r in res if abs(r["period"] / target["period"] - 1) <= MODE_TOL]
    if not res:
        return None
    signs = [i for i in range(len(res) - 1) if res[i]["drift"] > 0 >= res[i + 1]["drift"]]
    if signs:
        i = signs[0]
        pick = min(res[i:i + 2], key=lambda r: abs(r["drift"]))
        note = (f"bracketed: amplitude drift changes sign between kicks {res[i]['kick']:g} and {res[i + 1]['kick']:g} "
                f"km/s ({BRACKET_PERIODS} periods each); adopted kick {pick['kick']:g} (drift {pick['drift']:+.1e}/period)")
    else:
        pick = min(res, key=lambda r: abs(r["drift"]))
        note = (f"NOT bracketed: no sign change of the amplitude drift over kicks {[r['kick'] for r in res]} km/s; "
                f"adopted the smallest drift (kick {pick['kick']:g}, {pick['drift']:+.1e}/period): amplitude weakly constrained")
    (base / "limit").mkdir(exist_ok=True)
    shutil.copy(base / f"bracket_{pick['kick']:g}" / "limit.mod", base / "limit" / "limit.mod")
    return dict(params, kick=pick["kick"], nonlinear_period=pick["period"], delta_R=pick["delta_R"],
                periods_run=BRACKET_PERIODS, limit_cycle_note=note, mode_check=f"non-linear period within {MODE_TOL:.0%}")


def accept_nonlinear(label, target, base):
    """The first candidate whose limit cycle pulsates at Gaia's period (within MODE_TOL), i.e. in the star's mode, or None.

    A limit cycle clearly in the star's mode (period within MODE_WINDOW; F and 1O differ by ~30%) but off by more
    than MODE_TOL (the non-linear period at full amplitude differs from the linear one tuned to) gets its Teff
    corrected by that offset (P ~ Teff^-3.48 at fixed L, M) and the limit cycle rerun, up to MAX_CORRECTIONS times.
    Rejected attempts are recorded in base/rejected.json."""
    rejected = []
    for params in candidates(label, target, base):
        params.update(limit_periods=LIMIT_PERIODS[target["kind"]], kick=KICK[target["kind"]],
                      limit_steps=LIMIT_STEPS.get(target["kind"], 300))
        for attempt in range(MAX_CORRECTIONS + 1):
            limit_dir = base / "limit"
            shutil.rmtree(limit_dir, ignore_errors=True)
            text = run(limit_dir, inlist(params, "limit"))
            lines = period_lines(text)
            if (limit_dir / "limit.mod").exists() and len(lines) < 20 and "stop because GREKM" in text:
                bracketed = bracket_limit(label, target, base, params)  # weakly unstable: not a failure
                if bracketed is not None:
                    (base / "rejected.json").write_text(json.dumps(rejected, indent=1))
                    return bracketed
                rejected.append(dict(L_factor=params["L_factor"], teff=params["teff"], reason="bracketing: off the star's period"))
                break
            p_nl = lines[-1][1] if lines else np.nan
            ran = (limit_dir / "limit.mod").exists() and len(lines) >= 20
            if ran and abs(p_nl / target["period"] - 1) <= MODE_TOL:
                amp = np.array([d for _, _, d in lines])
                params.update(nonlinear_period=p_nl, amplitude_change_last10=float(abs(amp[-1] / amp[-10] - 1)),
                              periods_run=len(lines), delta_R=float(amp[-1]), teff_corrections=attempt,
                              mode_check=f"non-linear period within {MODE_TOL:.0%}")
                (base / "rejected.json").write_text(json.dumps(rejected, indent=1))
                return params
            why = "run failed" if not ran else f"non-linear period {p_nl:.5f} d vs Gaia {target['period']:.5f} d"
            print(f"  {label}: L x {params['L_factor']}, Teff {params['teff']:.0f} K: {why}", flush=True)
            rejected.append(dict(L_factor=params["L_factor"], teff=params["teff"], reason=why))
            (base / "rejected.json").write_text(json.dumps(rejected, indent=1))
            if _attempt_budget(base) == 0:
                raise NonlinearAttemptLimit(f"nonlinear attempt budget exhausted (limit {MAX_NONLINEAR_ATTEMPTS})")
            if not ran or abs(p_nl / target["period"] - 1) > MODE_WINDOW or attempt == MAX_CORRECTIONS:
                break
            params["teff"] *= (p_nl / target["period"]) ** (1 / 3.48)
            print(f"  {label}: correcting Teff to {params['teff']:.0f} K for the non-linear period", flush=True)
    print(f"  {label}: no candidate pulsates in the star's mode at its period; see {base / 'rejected.json'}", flush=True)
    return None


def period_lines(text):
    """RSP's per-period terminal summary: (period number, period d, delta R)."""
    return [(int(m[1]), float(m[2]), float(m[3])) for m in
            re.finditer(r"period\s+(\d+)\s+([0-9.Ee+-]+)\s+delta R\s+([0-9.Ee+-]+)", text)]


def model_star(label, target):
    """Model one star to runs/rsp_<label>/pulsation.h5. A star with no model gets a FAILED file instead, which is
    final until the directory is removed (make_stars.py then reports it not built)."""
    base = C.RUNS / f"rsp_{PREFIX}{label}"
    done = base / "pulsation.h5"
    if done.exists() or (base / "FAILED").exists():
        return
    try:
        build(label, target, base, done)
    except NonlinearAttemptLimit as err:
        (base / "FAILED").write_text(f"{label}: {err}\n")
        print(f"  {label}: FAILED, {err}", flush=True)
    except subprocess.TimeoutExpired:
        (base / "FAILED").write_text(f"{label}: a MESA run exceeded its time limit\n")
        print(f"  {label}: FAILED, a MESA run exceeded its time limit", flush=True)


def build(label, target, base, done):
    params_file = base / "params.json"
    if params_file.exists():  # resume (models made before the non-linear mode check keep their own route)
        params = json.loads(params_file.read_text())
    else:
        params = (accept_double_mode if target["kind"] == "RRd" else accept_nonlinear)(label, target, base)
        if params is None:
            (base / "FAILED").write_text(f"{label}: no model (rejected.json / double_mode.json say why)\n")
            return
        params_file.write_text(json.dumps(params, indent=1))
    limit_dir = base / "limit"
    if not (limit_dir / "limit.mod").exists():
        lines = period_lines(run(limit_dir, inlist(params, "limit")))
        amp = np.array([d for _, _, d in lines])
        params.update(nonlinear_period=lines[-1][1], amplitude_change_last10=float(abs(amp[-1] / amp[-10] - 1)),
                      periods_run=len(lines), delta_R=float(amp[-1]))
        params_file.write_text(json.dumps(params, indent=1))
    fine_dir = base / "fine"
    period_s = params.get("nonlinear_period", target["period"]) * 86400
    params["steps"] = (int(np.clip(np.ceil(period_s / FINE_STEP_S), 1000, MAX_FINE_STEPS))
                       if target["kind"] != "Type II Cepheid" else 3000)
    if target["kind"] == "RRd":  # a stretch of model time, not one cycle
        params["fine_periods"] = int(np.ceil(DM_FINE_DAYS / params.get("nonlinear_period", target["period"]))) + 1
    if not (fine_dir / "fine.mod").exists():
        fine_dir.mkdir(parents=True, exist_ok=True)
        (fine_dir / "limit.mod").write_bytes((limit_dir / "limit.mod").read_bytes())
        run(fine_dir, inlist(params, "fine"))
    pack(fine_dir, done, params, target)
    params_file.write_text(json.dumps(params, indent=1))


def pack(fine_dir, out, params, target):
    """SEDs of every high-resolution step, with time and stellar parameters, in one file."""
    h = np.genfromtxt(fine_dir / "LOGS" / "history.data", skip_header=5, names=True)
    row = {int(m): i for i, m in enumerate(h["model_number"])}
    files = {int(re.search(r"_(\d+)\.csv$", p.name)[1]): p for p in (fine_dir / "SED").glob("*_SED_*.csv")}
    models = sorted(m for m in files if m in row)
    seds = [read_sed_csv(files[m]) for m in models]
    wave = seds[0][:, 0]
    idx = [row[m] for m in models]
    with h5py.File(out, "w") as f:
        f.attrs.update(label=out.parent.name, kind=target["kind"], gaia_source=target["source"],
                       gaia_period=target["period"], gaia_epoch_bjd=target["epoch_bjd"], mass=params["mass"],
                       L=params["L"], teff_static=params["teff"], X=X, Z=params.get("z", Z), linear_period=params["linear_period"],
                       nonlinear_period=params.get("nonlinear_period", np.nan), distance_modulus=target.get("dm", DM),
                       ebv=target.get("ebv", EBV),
                       flux_units="erg s^-1 cm^-2 A^-1 (f_lambda at 10 pc)")
        f.create_dataset("wavelength", data=wave)
        f.create_dataset("flux", data=np.array([s[:, 1] for s in seds]), compression="gzip", compression_opts=4)
        f.create_dataset("time_days", data=h["star_age"][idx] * 365.25)
        for name in ("log_Teff", "log_L", "log_R", "log_g"):
            f.create_dataset(name, data=h[name][idx])
    for p in (fine_dir / "SED").glob("*.csv"):
        p.unlink()


# MESA test-suite models (star/test_suite/rsp_*), run through this module's inlists: the expected period is the
# one the test suite checks (its run_star_extras), at the period number it checks it
TEST_CASES = {
    "cepheid": dict(params=dict(mass=4.165, L=1438.8, teff=6050.0, x=0.73, z=0.007, kap="a09", mode=0, kick=5.0),
                    expected_period=3.92305, at_period=10, source="rsp_Cepheid (CEP-227, Pilecki et al. 2013)"),
    # the test's own settings, including its non-standard convection (chaotic by design): a check of the machinery
    "type2": dict(params=dict(mass=0.55, L=136.0, teff=6410.0, x=0.76, z=0.0001, kap="a09", mode=0,
                              extra_controls=["RSP_alfap = 0.0d0", "RSP_alfat = 0.0d0", "RSP_alfam = 0.05d0",
                                              "RSP_gammar = 0d0", "RSP_target_steps_per_cycle = 1200",
                                              "RSP_kick_vsurf_km_per_sec = 4.5d0", "RSP_nz = 200", "RSP_nz_outer = 60",
                                              "RSP_T_anchor = 11d3", "RSP_do_check_omega = .false.",
                                              "RSP_min_deltaR_for_periods = 0.2"]),
                  expected_period=1.73724, at_period=10, source="rsp_Type_II_Cepheid"),
    "delta_scuti": dict(params=dict(mass=2.0, L=30.0, teff=6900.0, x=0.70, z=0.02, kap="a09", mode=0,
                                    extra_controls=["RSP_alfam = 0.25d0", "RSP_alfa = 1.2d0",
                                                    "RSP_kick_vsurf_km_per_sec = 3.0d0", "RSP_fraction_1st_overtone = 0.5d0",
                                                    "RSP_fraction_2nd_overtone = 0d0"]),
                        expected_period=0.14715, at_period=10, source="rsp_Delta_Scuti"),
}


def validate(name):
    """Run a test-suite model with this module's inlists; compare RSP's period with the test suite's value."""
    build_sed_instrument()
    case = TEST_CASES[name]
    params = dict(case["params"], limit_periods=case["at_period"])
    base = C.RUNS / f"rsp_validate_{name}"
    periods, growth = linear(params, base / "lina")
    text = run(base / "limit", inlist(params, "validate"))
    lines = period_lines(text)
    got = next((p for n, p, _ in lines if n == case["at_period"]), lines[-1][1] if lines else np.nan)
    ok = abs(got - case["expected_period"]) <= 0.01 * case["expected_period"]
    print(f"{name} ({case['source']}): linear periods {np.round(periods[:3], 5).tolist()} d, growth "
          f"{np.round(growth[:3], 4).tolist()}; nonlinear period at period {case['at_period']}: {got:.5f} d vs test suite "
          f"{case['expected_period']:.5f} d -> {'PASS' if ok else 'FAIL'} (1%)")
    return ok


def main(labels=None):
    build_sed_instrument()
    targets = json.loads(TARGETS.read_text())
    todo = {k: v for k, v in targets.items() if not labels or k in labels}
    with ThreadPoolExecutor(int(C.config()["run"]["workers"])) as ex:  # stars modelled at once
        list(ex.map(lambda kv: model_star(*kv), todo.items()))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--validate"]:
        sys.exit(0 if all(validate(n) for n in args[1:] or TEST_CASES) else 1)
    if args[:1] == ["--field"]:
        use(args[1])
        args = args[2:]
    main(args)
