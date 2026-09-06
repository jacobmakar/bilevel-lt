"""Cell tags for the ladder: one string per cell, used for the JSON file name, for
resumption (a cell whose JSON exists is skipped) and by scripts/sweep_eta.py to match
PROGRESS lines to job-list commands. Torch-free so the sweep tooling can import it with a
bare system python. Only arguments that change a cell's result belong here; a non-default
value appends a suffix, so tags of older cells stay valid when a new dial is added.
"""
DEFAULTS = dict(head='linear', seed=1, mode='cert', point='zero', estimator='identity', outer_opt='adam',
                ridge=1e-4, k_list='200,600', polish_iters=0, damp_rel=1e-3, damp_track=0.5,
                outer_steps=200, val_per_class=100, imbalance=100, ref_steps=None)
TYPES = {k: (type(v) if v is not None else int) for k, v in DEFAULTS.items()}


def cell_tag(a: dict) -> str:
    """`a` maps argument names to values (strings are fine: they are cast by TYPES)."""
    def g(k):
        v = a.get(k)
        return DEFAULTS[k] if v is None else TYPES[k](v)
    tag = f"{g('head')}_s{g('seed')}_{g('mode')}_{g('point')}"
    if g('mode') == 'loop':
        tag += f"_{g('estimator')}_{g('outer_opt')}"
    if g('ridge') != DEFAULTS['ridge']:
        tag += f"_lam{g('ridge'):g}"
    if g('mode') == 'cert' and g('k_list') != DEFAULTS['k_list']:
        tag += f"_k{g('k_list').replace(',', '-')}"
    if g('polish_iters') > 0:
        tag += f"_polish{g('polish_iters')}"
    if g('damp_rel') != DEFAULTS['damp_rel']:
        tag += f"_damp{g('damp_rel'):g}"
    if g('mode') == 'loop' and g('estimator') == 'damped_track' and g('damp_track') != DEFAULTS['damp_track']:
        tag += f"_c{g('damp_track'):g}"
    if g('mode') == 'loop' and g('outer_steps') != DEFAULTS['outer_steps']:
        tag += f"_T{g('outer_steps')}"
    if g('val_per_class') != DEFAULTS['val_per_class']:
        tag += f"_val{g('val_per_class')}"
    if g('imbalance') != DEFAULTS['imbalance']:
        tag += f"_imb{g('imbalance')}"
    if g('mode') == 'ref' and a.get('ref_steps') is not None:
        tag += f"_ref{g('ref_steps')}"
    return tag
