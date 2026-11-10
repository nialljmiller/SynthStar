"""Evolve -> save -> branch an AGB structure, so the expensive pre-AGB evolution is computed once.

The user's workflow: evolve an intermediate-mass star once to the early AGB and `save_model` there (the costly part);
then branch several AGB continuations from that one saved model, each run cheaply to a different AGB-tip luminosity.
The generator `AGB variable` (synthstar/sources/agb.py) turns each branch's seds.h5 into a 200+ d variable star.

  python -m synthstar run agb_branch        # run the trunk, then its branches (runner skips finished ones)

The trunk (`_agb_trunk`) evolves 2 Msun, [M/H] 0 past core-He burning onto the AGB and stops at log L/Lsun 3.4,
saving early_agb.mod. Each branch loads that model and continues to a higher log L limit (3.6, 3.7, 3.8), giving AGB
tips of growing radius -> growing fundamental period. Run the trunk first (its save_model must exist before a branch
loads it); the runner's resume handles the rest.

Design notes (MESA load_model traps):
 * every branch sets continue_past_he_burning, or the he4 central-depletion stop would end it at step 0;
 * load_model_filename is given as an absolute path (MESA resolves it relative to the branch run directory);
 * changing initial_mass after a load does nothing -- the branch inherits the saved model's mass. The only
   meaningful branch levers with this physics template are the stop luminosity (below) and anything set through
   extra_controls that the template does not also hard-code (the Blocker wind is hard-coded, so vary it in
   physics.inlist, not here);
 * NOTE the 2-3 Msun AGB tip is ~3200-3450 K, below the 3500 K Kurucz atmosphere edge: these branches' luminous
   rows fall outside the atmosphere grid, so their SEDs are grey-clamped (see synthstar/sources/agb.py). The
   branches are structurally correct; optical SED fidelity awaits a cool-star atmosphere grid.
"""
from synthstar import config as C

MASS, FEH = 2.0, 0.0
SAVE = "early_agb.mod"
TRUNK_STOP = 3.4           # log L/Lsun at the early AGB, where the trunk saves its model
BRANCH_STOPS = (3.6, 3.7, 3.8)


def _stop(logl):
    return dict(continue_past_he_burning=True, accepted_terminations=["log_L_upper_limit"],
                star_job_extra=["required_termination_code_string(1) = 'log_L_upper_limit'"],
                extra_controls=[f"log_L_upper_limit = {logl:g}d0"],
                history_interval=10, photo_interval=500, resume_from_photo=True, resume_attempts=4)


def tracks(make):
    trunk = make(MASS, FEH, make.alpha_sun, tag="agb_trunk", kind="AGB trunk (saved early-AGB model)",
                 population="AGB evolve->save->branch", why="evolve once to the early AGB and save the model",
                 save_model=SAVE, save_seds=True, **_stop(TRUNK_STOP))
    saved = str(C.RUNS / trunk["name"] / SAVE)  # absolute: MESA resolves load_model relative to the run dir
    out = [trunk]
    for logl in BRANCH_STOPS:
        out.append(make(MASS, FEH, make.alpha_sun, tag=f"agb_branch_L{logl:g}", kind="AGB branch",
                        population="AGB evolve->save->branch",
                        why=f"branch from {trunk['name']}: continue to log L {logl:g}",
                        load_model=saved, save_seds=True, **_stop(logl)))
    return out
