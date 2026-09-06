import torch
from torch import nn
from torch.func import functional_call, grad

from bilevel_lt.autobalance import Flattener, sgd_step, step_lr


def test_flattener_roundtrip():
    m = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    params = {k: v.detach() for k, v in m.named_parameters()}
    fl = Flattener(params)
    v = fl.flat(params)
    assert v.shape == (fl.p,) and fl.p == 4 * 3 + 3 + 3 * 2 + 2
    back = fl.unflat(v)
    assert all(torch.equal(back[k], params[k]) for k in params)


def test_sgd_step_matches_torch_sgd():
    torch.manual_seed(0)
    m = nn.Linear(5, 3)
    x, y = torch.randn(8, 5), torch.randint(0, 3, (8,))
    ref = nn.Linear(5, 3)
    ref.load_state_dict(m.state_dict())
    opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-2)
    params = {k: v.detach().clone() for k, v in m.named_parameters()}
    buf = None
    loss = lambda p: nn.functional.cross_entropy(functional_call(m, p, (x,)), y)
    for _ in range(3):
        opt.zero_grad()
        nn.functional.cross_entropy(ref(x), y).backward()
        opt.step()
        params, buf = sgd_step(params, grad(loss)(params), buf, lr=0.1, wd=1e-2)
    for k, v in ref.named_parameters():
        assert torch.allclose(params[k], v.detach(), atol=1e-6)


def test_step_lr_schedule():
    assert step_lr(0.1, 0, 100) == 0.1 and step_lr(0.1, 79, 100) == 0.1
    assert abs(step_lr(0.1, 80, 100) - 0.01) < 1e-12 and abs(step_lr(0.1, 90, 100) - 0.001) < 1e-12
