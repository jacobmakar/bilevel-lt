import os
import sys

from bilevel_lt.ladder import parse_args
from bilevel_lt.tags import cell_tag

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from sweep_eta import parse_cmd, tag_of  # noqa: E402

CMDS = [
    "python -m bilevel_lt.ladder --head dlin16 --mode cert --point la1 --k_list 200,600 --seed 2 --out_root x",
    "python -m bilevel_lt.ladder --head dlin16 --mode loop --point zero --estimator damped_track --damp_track 2 --ridge 0.1 --seed 7 --threads 2",
    "python -m bilevel_lt.ladder --head linear --mode ref --val_per_class 25 --seed 4",
    "python -m bilevel_lt.ladder --head relu16 --mode cert --point zero --k_list 600,2000 --polish_iters 500 --damp_rel 0.01 --imbalance 50 --seed 1",
]


def test_driver_and_sweep_tool_agree():
    for cmd in CMDS:
        assert cell_tag(vars(parse_args(cmd.split()[3:]))) == tag_of(parse_cmd(cmd))


def test_tag_shapes():
    assert tag_of(parse_cmd(CMDS[0])) == 'dlin16_s2_cert_la1'
    assert tag_of(parse_cmd(CMDS[1])) == 'dlin16_s7_loop_zero_damped_track_adam_lam0.1_c2'
    assert tag_of(parse_cmd(CMDS[2])) == 'linear_s4_ref_zero_val25'
    assert tag_of(parse_cmd(CMDS[3])) == 'relu16_s1_cert_zero_k600-2000_polish500_damp0.01_imb50'
