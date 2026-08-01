"""CPU-only checks for the optimizers, the federated driver, and the plumbing.

    python3 -m tests.test_code     # run everything, print a report
    pytest tests/test_code.py      # or under pytest

Both from the ``code/`` directory.

No GPU, no dataset download, about a minute. What it pins down:

* the Newton-Schulz helper does not scribble on its input (it used to, whenever
  the input was already bfloat16, because ``Tensor.to`` returns ``self``);
* it really does approximate the polar factor;
* every sign/LMO method is invariant to a positive rescaling of the gradient --
  the property that makes the paper's heavy-ball main text and its EMA algorithm
  boxes describe the same trajectories;
* the paper's Theorem 1-3 descent inner products, computed with torch;
* **the federated driver with one client reproduces the centralized optimizer
  exactly**, for all eight matrix-parameter rules. This is the load-bearing test:
  it ties ``federated_algorithms.run_federated`` to ``optimizers.py`` so the two
  cannot silently diverge;
* several clients holding identical data reproduce the single-client run;
* EF21-MuonSign keeps its exact model ``X`` distinct from the broadcast ``W``;
* the metrics schema and the multi-seed aggregation.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from federated.algorithms import METHODS, run_federated
from common.optimizers import (
    EF21MuonSign,
    EF21MuonUSign,
    EF21SignMuon,
    Muon,
    MuonSign,
    MuonUSign,
    SignMuon,
    SignSGD,
    muon_lmo,
    zeropower_via_newtonschulz5,
)
from common.utils import History, split_param_names

CENTRAL_CLASSES = {
    "signmuon": SignMuon,
    "ef21signmuon": EF21SignMuon,
    "muonusign": MuonUSign,
    "muonsign": MuonSign,
    "ef21muonusign": EF21MuonUSign,
    "ef21muonsign": EF21MuonSign,
    "muon": Muon,
    "signsgd": SignSGD,
}

TOL = 1e-5


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class TinyNet(nn.Module):
    """Exactly one matrix parameter plus a 2-tensor 'head', so the routing is
    unambiguous: ``fc1.weight`` is the matrix, everything else is auxiliary."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TallNet(nn.Module):
    """Same shape of test as ``TinyNet``, but with a TALL matrix parameter.

    ``fc1.weight`` is ``(12, 4)``, so ``sqrt(max(1, m/n)) = sqrt(3) != 1``. This
    matters: on ``TinyNet`` the only matrix is ``(4, 6)``, where the aspect factor
    is exactly 1 and the whole "the factor lives in ``lambda`` federated and in
    ``scale_aspect`` centrally, and the two coincide bit for bit" claim is
    vacuously true. Here it has to actually hold.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 12)
        self.fc2 = nn.Linear(12, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def tall_data(n: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 4, generator=g),
            torch.randint(0, 3, (n,), generator=g))


def tall_loader(n: int = 8, seed: int = 0):
    return [tall_data(n, seed)]


def tiny_data(n: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 6, generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return x, y


def tiny_loader(n: int = 8, seed: int = 0):
    """One deterministic full batch per epoch (``shuffle=False``)."""
    x, y = tiny_data(n, seed)
    return DataLoader(TensorDataset(x, y), batch_size=n, shuffle=False)


def exact_polar(Y: torch.Tensor) -> torch.Tensor:
    """Rank-truncated ``U V^T`` via SVD, in float64."""
    U, S, Vh = torch.linalg.svd(Y.double(), full_matrices=False)
    r = int((S > 1e-9 * S[0]).sum())
    return U[:, :r] @ Vh[:r, :]


# --------------------------------------------------------------------------
# Newton-Schulz helper
# --------------------------------------------------------------------------


def test_newton_schulz_does_not_mutate_input():
    for dtype in (torch.float32, torch.bfloat16):
        G = torch.randn(5, 7, dtype=dtype)
        before = G.clone()
        zeropower_via_newtonschulz5(G, steps=5, dtype=dtype)
        assert torch.equal(G, before), f"input mutated for dtype={dtype}"


def _controlled_spectrum(shape, kappa: float, seed: int) -> torch.Tensor:
    """``U diag(s) V^T`` with ``s`` log-spaced over ``[1/kappa, 1]``.

    Testing on Gaussian matrices conflates the implementation with the *input's*
    conditioning -- a Gaussian's condition number is heavy-tailed, and 5 Newton-Schulz
    steps genuinely cannot lift a singular value that starts too small. Controlling
    the spectrum separates the two.
    """
    g = torch.Generator().manual_seed(seed)
    m, n = shape
    r = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(m, r, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, r, generator=g))
    s = torch.logspace(math.log10(1.0 / kappa), 0.0, r)
    return U @ torch.diag(s) @ V.T


def test_newton_schulz_gets_the_direction_but_not_the_scale():
    """What 5 steps of the Muon quintic actually deliver.

    The quintic is *not* a convergent iteration: it drives the singular values into
    a band around 1 and oscillates inside it. Two consequences, both measured over
    80 (shape, seed) cases:

    * On an **already orthogonal** input the direction is exact -- ``NS(Q) = c*Q`` for
      a positive scalar ``c``, since ``A = QQ^T`` is a multiple of the identity and
      each step just rescales. So cosine is 1.0000 and every sign agrees. But the
      *relative Frobenius error reaches 0.31*, because ``c`` oscillates in
      ``[0.69, 1.07]``. Asserting a small Frobenius error would therefore be wrong
      even for a perfect implementation -- the quintic does not fix the scale, which
      is exactly why magnitude is handled separately by ``lambda`` / ``scale_aspect``.
    * On a conditioned input the sign pattern agrees only 84-91% of the time. So
      ~1 entry in 8 of ``sign(polar(G))`` flips under the practical oracle -- the
      reason the sign-family methods are sensitive to the LMO approximation
      (cf. ``--lmo-dtype`` and ``counterexamples/verify_ns_oracle.py``).

    The assertions below therefore test *direction* and *approximate orthogonality*,
    with margins taken from the 80-case sweep.
    """
    shapes = [(8, 8), (16, 16), (64, 64), (32, 96)]

    # (a) orthogonal input: direction must be essentially exact.
    for shape in shapes:
        for seed in (0, 1):
            G = _controlled_spectrum(shape, kappa=1.0, seed=seed)
            D = muon_lmo(G, ns_steps=5, dtype=torch.float32, scale_aspect=False)
            P = exact_polar(G).float()
            cos = float((D * P).sum() / (D.norm() * P.norm()))
            agree = float((torch.sign(D) == torch.sign(P)).float().mean())
            assert cos > 0.999, f"{shape}: cosine {cos:.5f} on an orthogonal input"
            assert agree > 0.98, f"{shape}: {agree:.1%} sign agreement on an " \
                                 f"orthogonal input (should be exact)"

    # (b) moderately conditioned input: direction close, scale within the band.
    for shape in shapes:
        for seed in (0, 1, 2):
            G = _controlled_spectrum(shape, kappa=4.0, seed=seed)
            D = muon_lmo(G, ns_steps=5, dtype=torch.float32, scale_aspect=False)
            P = exact_polar(G).float()
            cos = float((D * P).sum() / (D.norm() * P.norm()))
            agree = float((torch.sign(D) == torch.sign(P)).float().mean())
            sv = torch.linalg.svdvals(D.double())
            assert cos > 0.95, f"{shape}: cosine {cos:.4f} -- wrong direction"
            assert agree > 0.70, f"{shape}: only {agree:.1%} of signs agree"
            assert 0.4 < float(sv.min()) and float(sv.max()) < 1.5, \
                f"{shape}: singular values [{sv.min():.3f}, {sv.max():.3f}] -- not " \
                f"approximately orthogonal"


def test_muon_lmo_shapes():
    for shape in [(4, 6), (6, 4), (5, 5), (3, 2, 2, 2)]:
        G = torch.randn(*shape)
        assert muon_lmo(G, dtype=torch.float32).shape == G.shape
    v = torch.randn(7)                       # ndim < 2 passes straight through
    assert torch.equal(muon_lmo(v), v)


def test_aspect_ratio_scaling_is_invisible_to_output_signing():
    """SignMuon/MuonSign sign the LMO output, so Muon's sqrt(max(1,m/n)) factor
    cannot change their step; Muon/MuonUSign it does scale."""
    torch.manual_seed(1)
    G = torch.randn(10, 4)
    a = muon_lmo(G, dtype=torch.float32, scale_aspect=True)
    b = muon_lmo(G, dtype=torch.float32, scale_aspect=False)
    assert torch.equal(torch.sign(a), torch.sign(b))
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------
# Gradient-scale invariance (the momentum-convention equivalence)
# --------------------------------------------------------------------------


def _drive(cls, grads, lr=0.1, momentum=0.7, nesterov=False):
    p = nn.Parameter(torch.zeros(4, 5))
    opt = cls([p], lr=lr, momentum=momentum, nesterov=nesterov,
              lmo_dtype=torch.float32)
    traj = []
    for g in grads:
        p.grad = g.clone()
        opt.step()
        traj.append(p.detach().clone())
    return traj


def test_gradient_scale_invariance():
    """Every method's iterates are unchanged when G -> c*G for c > 0.

    This is exactly why the paper's main-text heavy-ball momentum
    (``mu*M + G``) and its algorithm boxes' EMA (``mu*M + (1-mu)*G``) give
    identical trajectories: the two buffers differ by the constant factor
    ``1 - mu``, and sign, polar and the EF21 recursion are all positively
    homogeneous.
    """
    torch.manual_seed(2)
    grads = [torch.randn(4, 5) for _ in range(6)]
    for name, cls in CENTRAL_CLASSES.items():
        for nesterov in (False, True):
            base = _drive(cls, grads, nesterov=nesterov)
            scaled = _drive(cls, [3.0 * g for g in grads], nesterov=nesterov)
            for t, (u, v) in enumerate(zip(base, scaled)):
                assert torch.allclose(u, v, atol=1e-4), \
                    f"{name} (nesterov={nesterov}) not scale invariant at step {t}"


# --------------------------------------------------------------------------
# The paper's counterexamples, in torch
# --------------------------------------------------------------------------


def _signmuon_instance(sigma1: float):
    O = torch.tensor([[101., 20., 2., -2.],
                      [-20., 97., 20., -20.],
                      [-2., 20., 2., 101.],
                      [-2., 20., -101., -2.]], dtype=torch.float64) / 103.0
    u1 = torch.tensor([10., -3., 10., 10.], dtype=torch.float64).reshape(-1, 1) / math.sqrt(309.)
    v1 = torch.tensor([10., 3., -10., 10.], dtype=torch.float64).reshape(-1, 1) / math.sqrt(309.)
    return sigma1 * (u1 @ v1.T) + O, O


def _muonsign_instance(M: float, eps: float = 1.0):
    S = torch.tensor([[-1., -1., 1., 1., 1.],
                      [-1., -1., 1., -1., -1.],
                      [1., -1., 1., 1., -1.],
                      [1., 1., -1., -1., 1.],
                      [1., 1., 1., -1., 1.]], dtype=torch.float64)
    G = eps * S.clone()
    G[3, 1] = eps * S[3, 1] + (M - eps)
    return G, S


def test_theorem1_exact_oracle():
    """<G, sign(polar(G))> = (-43*sigma1 + 532)/103."""
    for sigma1 in (100.0, 1000.0):
        G, O = _signmuon_instance(sigma1)
        P = exact_polar(G)
        assert (P - O).norm() < 1e-10, "polar(G) should equal O"
        got = float((G * torch.sign(P)).sum())
        want = (-43.0 * sigma1 + 532.0) / 103.0
        assert abs(got - want) < 1e-6, f"sigma1={sigma1}: {got} != {want}"
        assert got < 0, "SignMuon must ascend under the exact oracle"


def test_theorems23_exact_oracle():
    G, S = _muonsign_instance(100.0)
    assert torch.equal(torch.sign(G), S)
    D = exact_polar(S)
    assert abs(float(D[3, 1]) - (-0.2425)) < 1e-3, float(D[3, 1])
    assert abs(float((G * D).sum()) - (-13.888)) < 1e-2          # MuonUSign, Thm 2
    assert abs(float((G * torch.sign(D)).sum()) - (-76.0)) < 1e-9  # MuonSign, Thm 3


def test_exact_and_newton_schulz_oracles_differ_on_the_instances():
    """Regression test for a measured fact, not a defect.

    The theorems are stated for the exact LMO and the counterexample code runs the
    exact LMO; networks are trained with Newton-Schulz, as practitioners do. The
    two oracles are different maps, and on these instances they disagree: at the
    published constants (sigma1=1000, M=100) the 5-step oracle does not ascend for
    Theorems 1-2, while sigma1=100 / M=500 ascend under both. Theorem 3 is
    oracle-robust. Pinned here so the numbers cannot drift; see
    ``counterexamples/verify_ns_oracle.py``.
    """
    def after(G):                       # SignMuon
        return float((G * torch.sign(muon_lmo(G.float(), 5, torch.float32, False))).sum())

    def before(G, S):                   # MuonUSign
        return float((G * muon_lmo(S.float(), 5, torch.float32, False).double()).sum())

    def both(G, S):                     # MuonSign
        return float((G * torch.sign(muon_lmo(S.float(), 5, torch.float32, False)).double()).sum())

    assert after(_signmuon_instance(1000.0)[0]) > 0, "sigma1=1000 does not ascend at 5 NS steps"
    assert after(_signmuon_instance(100.0)[0]) < 0, "sigma1=100 ascends at 5 NS steps"

    G100, S = _muonsign_instance(100.0)
    G500, _ = _muonsign_instance(500.0)
    assert before(G100, S) > 0, "M=100 does not ascend at 5 NS steps"
    assert before(G500, S) < 0, "M=500 ascends at 5 NS steps"
    assert both(G100, S) < 0 and both(G500, S) < 0, "Theorem 3 is oracle-robust"


# --------------------------------------------------------------------------
# Federated <-> centralized equivalence
# --------------------------------------------------------------------------


def _centralized_reference(method, rounds, lr, lr_aux, momentum, seed=0,
                           weight_decay=0.0, net=TinyNet, loader_fn=None):
    """Drive the centralized optimizer by hand, one full-batch step per round."""
    torch.manual_seed(seed)
    model = net()
    matrix_names, aux_names = split_param_names(model, 2)
    named = dict(model.named_parameters())

    opt = CENTRAL_CLASSES[method]([named[n] for n in matrix_names],
                                  lr=lr, momentum=momentum,
                                  weight_decay=weight_decay,
                                  decoupled_weight_decay=True,
                                  lmo_dtype=torch.float32)
    # weight_decay=0.0, not `weight_decay`: `centralized.train.build_optimizers`
    # never decays the auxiliary group -- decaying normalization scales is not
    # standard practice, and keeping it at zero for every method is what makes
    # `--weight-decay` describe the matrix parameters alone. This reference used
    # to decay it, which pinned the federated driver to a convention the real
    # centralized path does not use.
    aux = torch.optim.AdamW([named[n] for n in aux_names], lr=lr_aux,
                            weight_decay=0.0)

    loader = (loader_fn or tiny_loader)()
    criterion = nn.CrossEntropyLoss()
    for _ in range(rounds):
        for x, y in loader:
            model.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            opt.step()
            aux.step()
    if hasattr(opt, "restore_exact"):
        opt.restore_exact()
    return model


def _federated_run(method, rounds, lr, lr_aux, momentum, n_clients=1, seed=0,
                   weight_decay=0.0, net=TinyNet, loader_fn=None):
    torch.manual_seed(seed)
    model = net()
    make = loader_fn or tiny_loader
    loaders = [make() for _ in range(n_clients)]
    run_federated(
        method, model, loaders, [make()],
        rounds=rounds, n_steps=1, lr=lr, lr_aux=lr_aux, momentum=momentum,
        weight_decay=weight_decay, eval_freq=10 ** 9, device="cpu",
        cosine_schedule=False, lmo_dtype=torch.float32, verbose=False,
        # EXPLICIT, not inherited: the centralized reference above is built with
        # ``scale_aspect=True`` and ``lambda_mult=1``, which IS the legacy
        # convention. Relying on the driver's default would make this test's
        # meaning depend on a default that has already changed once.
        lr_scaling="legacy",
    )
    return model


def test_federated_one_client_equals_centralized():
    """The load-bearing test: N=1 federated == the centralized optimizer.

    ``sgd`` and ``adam`` are excluded: they are torch optimizers on the
    centralized side and server-side aggregation rules on the federated side, and
    ``sgd`` deliberately keeps the heavy-ball momentum convention.
    """
    failures = []
    for method in CENTRAL_CLASSES:
        ref = _centralized_reference(method, rounds=6, lr=0.05, lr_aux=0.01, momentum=0.8)
        fed = _federated_run(method, rounds=6, lr=0.05, lr_aux=0.01, momentum=0.8)
        for (n, a), (m, b) in zip(ref.named_parameters(), fed.named_parameters()):
            assert n == m
            if not torch.allclose(a, b, atol=TOL):
                failures.append(f"{method}/{n}: max|diff| = {(a - b).abs().max():.3e}")
    assert not failures, "federated != centralized:\n  " + "\n  ".join(failures)


def test_federated_one_client_equals_centralized_on_a_tall_matrix():
    """The same equivalence where the aspect factor is not 1.

    ``TinyNet``'s only matrix is ``(4, 6)``, so ``sqrt(max(1, m/n)) == 1`` and the
    test above cannot distinguish "the factor is applied once, outside the oracle"
    from "the factor is not applied at all". ``TallNet``'s ``(12, 4)`` gives
    ``sqrt(3)``, so the centralized ``scale_aspect=True`` convention and the
    federated ``scale_aspect=False`` + ``lambda`` convention have to genuinely
    agree -- which is the bit-for-bit claim the READMEs make about ``legacy``.
    """
    m, n = 12, 4
    assert math.sqrt(max(1.0, m / n)) > 1.0, "fixture no longer exercises the factor"
    failures = []
    for method in CENTRAL_CLASSES:
        ref = _centralized_reference(method, rounds=6, lr=0.05, lr_aux=0.01,
                                     momentum=0.8, net=TallNet, loader_fn=tall_loader)
        fed = _federated_run(method, rounds=6, lr=0.05, lr_aux=0.01, momentum=0.8,
                             net=TallNet, loader_fn=tall_loader)
        for (a_name, a), (b_name, b) in zip(ref.named_parameters(),
                                            fed.named_parameters()):
            assert a_name == b_name
            if not torch.allclose(a, b, atol=TOL):
                failures.append(
                    f"{method}/{a_name}: max|diff| = {(a - b).abs().max():.3e}")
    assert not failures, "federated != centralized on a tall matrix:\n  " + \
        "\n  ".join(failures)


def test_federated_muonserver_equals_centralized_muon_at_one_client():
    """``muonserver`` is excluded from ``CENTRAL_CLASSES``; pin it separately.

    It has no centralized class of its own because it differs from ``muon`` only
    in *where* the LMO runs relative to the average -- and with a single client
    there is nothing to average, so the two must coincide exactly. That makes
    centralized ``Muon`` the right reference, and leaves no federated matrix
    method unpinned.
    """
    for net, loader_fn in ((TinyNet, tiny_loader), (TallNet, tall_loader)):
        ref = _centralized_reference("muon", rounds=6, lr=0.05, lr_aux=0.01,
                                     momentum=0.8, net=net, loader_fn=loader_fn)
        fed = _federated_run("muonserver", rounds=6, lr=0.05, lr_aux=0.01,
                             momentum=0.8, net=net, loader_fn=loader_fn)
        for (a_name, a), (b_name, b) in zip(ref.named_parameters(),
                                            fed.named_parameters()):
            assert a_name == b_name
            assert torch.allclose(a, b, atol=TOL), (
                f"muonserver != Muon at N=1 on {net.__name__}/{a_name}: "
                f"max|diff| = {(a - b).abs().max():.3e}")


def test_the_two_drivers_agree_on_the_weight_decay_convention():
    """Same equivalence, now with weight decay switched on.

    The zero-decay test above passes under *either* convention, so it cannot see a
    coupled/decoupled mismatch between the two drivers -- which is exactly the
    discrepancy that existed. Decoupled is the only well-posed choice here: every
    step direction is positively homogeneous of degree zero, so folding ``wd * X``
    into the gradient leaves the step length untouched and merely rotates it.
    """
    failures = []
    for method in CENTRAL_CLASSES:
        kw = dict(rounds=6, lr=0.05, lr_aux=0.01, momentum=0.8, weight_decay=0.02)
        ref = _centralized_reference(method, **kw)
        fed = _federated_run(method, **kw)
        moved = max((p - q).abs().max().item()
                    for p, q in zip(_centralized_reference(method, rounds=6, lr=0.05,
                                                           lr_aux=0.01, momentum=0.8,
                                                           weight_decay=0.0).parameters(),
                                    ref.parameters()))
        # A decay too small to move the parameters would make this test vacuous.
        assert moved > 10 * TOL, f"{method}: wd=0.02 changed nothing ({moved:.2e})"
        for (n, a), (m, b) in zip(ref.named_parameters(), fed.named_parameters()):
            assert n == m
            if not torch.allclose(a, b, atol=TOL):
                failures.append(f"{method}/{n}: max|diff| = {(a - b).abs().max():.3e}")
    assert not failures, ("the drivers disagree once weight decay is nonzero:\n  "
                          + "\n  ".join(failures))


def test_decoupled_decay_is_not_scaled_by_the_per_layer_multiplier():
    """``X *= 1 - lr*wd``, with NO ``lambda`` in it. Pinned at lambda != 1.

    Both drivers apply decay unscaled by the per-layer factor, on purpose: decay
    is a property of the parameter, not of the step geometry. The equality tests
    above cannot see this -- ``TinyNet``'s only matrix is ``(4, 6)`` at the
    default ``legacy`` rule, so every multiplier there is exactly 1 and scaling
    by it is a no-op. Scaling decoupled decay by ``lambda_mult`` passes every
    other test in this file.
    """
    lr, wd, lam = 0.1, 0.5, 0.25
    torch.manual_seed(0)
    for cls in CENTRAL_CLASSES.values():
        p = nn.Parameter(torch.ones(4, 4))
        # A zero gradient no longer implies a zero step: under the randomized-zero
        # convention the sign family transmits +-1 everywhere, so every coordinate
        # moves by lr*lambda. Capture the direction and subtract it, which pins the
        # decay factor itself -- the thing this test is about -- for every method.
        p.grad = torch.zeros(4, 4)
        opt = cls([{"params": [p], "lambda_mult": lam}], lr=lr, momentum=0.0,
                  weight_decay=wd, decoupled_weight_decay=True,
                  lmo_dtype=torch.float32, scale_aspect=False)
        opt.capture_direction = True
        opt.step()
        if hasattr(opt, "restore_exact"):
            opt.restore_exact()
        d = opt.state[p]["last_direction"]
        expected = (1.0 - lr * wd) - lr * lam * d
        got = p.detach()
        assert torch.allclose(got, expected, atol=1e-6), (
            f"{cls.__name__}: decoupled decay gave {float(got[0, 0]):.6f}, expected "
            f"{float(expected[0, 0]):.6f}. Scaled by lambda the decay factor would "
            f"be {1 - lr * wd * lam:.6f} rather than {1 - lr * wd:.6f}")


def test_the_two_drivers_agree_on_weight_decay_at_a_nontrivial_multiplier():
    """The driver-level decay equivalence, where ``lambda`` is not 1.

    ``TallNet`` under ``unit-gain`` gives the LMO family ``sqrt(3)`` and the sign
    family ``1/2`` at once, so a mismatch in *where* the multiplier sits relative
    to the decay cannot cancel. The ``legacy``/``TinyNet`` version of this test
    runs entirely at ``lambda == 1``.
    """
    failures = _compare_drivers_under_rule("unit-gain", TallNet, tall_loader,
                                           weight_decay=0.02)
    assert not failures, ("the drivers disagree at lambda != 1 with decay on:\n  "
                          + "\n  ".join(failures))


def test_the_exact_step_maps_pin_the_step_length():
    """The claim the appendix's weight-decay argument rests on, on the exact maps.

    ``sign`` and the polar factor are positively homogeneous of degree **zero**, and
    their Frobenius norms are constants of the shape alone -- ``sqrt(m n)`` and
    ``sqrt(min(m, n))``. Hence a step along either has a length that no additive
    perturbation of the gradient can change: coupled weight decay can only rotate it.

    Stated on ``torch.linalg.svd``, not on the shipped Newton-Schulz iteration --
    NS-5 approximates the polar factor's *direction* well but its norm only to a few
    percent, which is what the next test measures.
    """
    def exact_polar(A):
        U, _, Vh = torch.linalg.svd(A, full_matrices=False)
        return U @ Vh

    torch.manual_seed(0)
    for (m, n) in [(16, 12), (12, 16), (64, 27), (33, 33)]:
        A = torch.randn(m, n, dtype=torch.float64)
        r = min(m, n)
        assert abs(exact_polar(A).norm().item() - math.sqrt(r)) < 1e-9
        assert abs(torch.sign(A).norm().item() - math.sqrt(m * n)) < 1e-9
        for c in (1e-6, 1e-3, 1.0, 1e3, 1e6):
            assert torch.equal(torch.sign(c * A), torch.sign(A)), c
            assert (exact_polar(c * A) - exact_polar(A)).abs().max() < 1e-9, c
        # and the norm is untouched by an ADDITIVE perturbation of any size
        B = torch.randn(m, n, dtype=torch.float64) * 1e4
        assert abs(exact_polar(A + B).norm().item() - math.sqrt(r)) < 1e-9
        assert abs(torch.sign(A + B).norm().item() - math.sqrt(m * n)) < 1e-9


def test_coupled_decay_does_not_shrink_the_implemented_step():
    """The same claim on the real optimizers, with the honest caveats.

    * **Sign-terminated** steps (SignMuon, MuonSign, SignSGD) land on
      ``{-1,+1}^{m x n}`` whatever the input, so the invariance is *exact* here.
    * **LMO-terminated** steps are pinned to ``sqrt(min(m,n))`` by the exact polar
      factor, but the shipped Newton-Schulz iteration only approximates it, and
      coupled decay changes the input's *spectrum* -- so the step norm moves a little.
      That residual is Newton-Schulz's approximation error, not decay.
    * **EF21-SignMuon** steps along the EF21 estimator of the Muon direction, whose
      norm reaches ``sqrt(r)`` only asymptotically, so it is excluded from the norm
      claim by construction (the appendix says so too).

    In every case the direction *is* rotated, which is the actual harm.
    """
    from common.lr_scaling import FAMILY_SIGN

    torch.manual_seed(0)
    m, n = 16, 12

    # How far the shipped NS-5 iteration's output norm may sit from the exact
    # sqrt(min(m,n)). MEASURED here rather than hard-coded, over the three input
    # distributions the methods actually feed it (raw momentum, a sign matrix, a
    # scaled-sign EF21 payload), so this test cannot go stale if the iteration or
    # its coefficients change.
    def ns_norm(A):
        return muon_lmo(A, ns_steps=5, dtype=torch.float32,
                        scale_aspect=False).norm().item() / math.sqrt(min(m, n))

    probe = []
    for _ in range(16):
        A = torch.randn(m, n)
        probe += [ns_norm(A), ns_norm(torch.sign(A)), ns_norm(A.abs().mean() * torch.sign(A))]
    NS_BAND = max(0.05, 2.0 * max(abs(x - 1.0) for x in probe))
    for name, cls in CENTRAL_CLASSES.items():
        base = torch.randn(m, n)
        g = torch.randn(m, n) * 1e-3        # small gradient => large rho
        steps = {}
        for tag, wd in (("coupled", 0.5), ("plain", 0.0)):
            param = nn.Parameter(base.clone())
            opt = cls([param], lr=0.1, momentum=0.0, lmo_dtype=torch.float32,
                      weight_decay=wd, decoupled_weight_decay=False,
                      scale_aspect=False)
            before = param.detach().clone()
            param.grad = g.clone()
            opt.step()
            if hasattr(opt, "restore_exact"):
                opt.restore_exact()
            steps[tag] = param.detach() - before
        n_c, n_p = steps["coupled"].norm().item(), steps["plain"].norm().item()

        if cls.family == FAMILY_SIGN:
            # 1e-4 relative is float32 differencing noise; a real shrinkage would be
            # O(lr*wd) = 5e-2 relative, ~500x larger, so the test still has teeth.
            assert abs(n_c - n_p) <= 1e-4 * n_p, (
                f"{name}: coupled decay changed a sign step's norm "
                f"{n_p:.8g} -> {n_c:.8g}; that is exactly impossible")
            assert abs(n_p / (0.1 * math.sqrt(m * n)) - 1.0) <= 1e-4, n_p
        elif name != "ef21signmuon":
            exact = 0.1 * math.sqrt(min(m, n))
            for tag, got in (("coupled", n_c), ("plain", n_p)):
                assert abs(got / exact - 1.0) <= NS_BAND, (
                    f"{name}/{tag}: step norm {got:.6g} is {got / exact:.3f}x the "
                    f"exact polar value {exact:.6g} -- outside the Newton-Schulz band")
            # No systematic shrinkage: the gap is bounded by the NS error, not by wd.
            assert abs(n_c - n_p) <= 2 * NS_BAND * exact, (
                f"{name}: {n_p:.6g} -> {n_c:.6g} exceeds twice the NS band")

        # ... but the direction really is rotated, which is the harm.
        cos = (steps["coupled"] * steps["plain"]).sum().item() / (n_c * n_p + 1e-30)
        assert cos < 0.999, f"{name}: coupled decay had no effect at all (cos={cos:.4f})"


def test_identical_clients_reduce_to_one():
    """N clients with identical data give the same model as N=1.

    Holds for every uplink: the majority vote over identical signs is that sign,
    and the average of identical EF21 payloads is that payload.
    """
    failures = []
    for method in METHODS:
        one = _federated_run(method, rounds=4, lr=0.05, lr_aux=0.01, momentum=0.8, n_clients=1)
        many = _federated_run(method, rounds=4, lr=0.05, lr_aux=0.01, momentum=0.8, n_clients=3)
        for (n, a), (_, b) in zip(one.named_parameters(), many.named_parameters()):
            if not torch.allclose(a, b, atol=TOL):
                failures.append(f"{method}/{n}: max|diff| = {(a - b).abs().max():.3e}")
    assert not failures, "N=3 identical clients != N=1:\n  " + "\n  ".join(failures)


def test_all_federated_methods_run():
    for method in METHODS:
        model = TinyNet()
        history = run_federated(
            method, model, [tiny_loader()], [tiny_loader()],
            rounds=3, n_steps=2, lr=0.01, lr_aux=0.01, momentum=0.9,
            weight_decay=1e-4, eval_freq=1, device="cpu",
            lmo_dtype=torch.float32, verbose=False,
        )
        assert history.steps == [0, 1, 2, 3], f"{method}: {history.steps}"
        for value in history.series["test_acc"]:
            assert value is not None and 0.0 <= value <= 100.0
        for p in model.parameters():
            assert torch.isfinite(p).all(), f"{method} produced non-finite parameters"


def test_eval_freq_records_only_evaluated_rounds():
    history = run_federated(
        "signmuon", TinyNet(), [tiny_loader()], [tiny_loader()],
        rounds=5, n_steps=1, lr=0.01, eval_freq=2, device="cpu",
        lmo_dtype=torch.float32, verbose=False,
    )
    assert history.steps == [0, 2, 4, 5], history.steps


# --------------------------------------------------------------------------
# EF21-MuonSign bookkeeping
# --------------------------------------------------------------------------


def test_ef21muonsign_separates_exact_and_broadcast_models():
    torch.manual_seed(3)
    p = nn.Parameter(torch.zeros(4, 4))
    opt = EF21MuonSign([p], lr=0.1, momentum=0.0, lmo_dtype=torch.float32)
    for _ in range(4):
        p.grad = torch.randn(4, 4)
        opt.step()

    W = p.detach().clone()
    X = opt.state[p]["exact_model"]
    assert not torch.allclose(W, X), "W and X should differ once the downlink compresses"

    with opt.using_exact():
        assert torch.allclose(p.data, X), "using_exact must expose X"
    assert torch.allclose(p.data, W), "using_exact must put W back"

    opt.restore_exact()
    assert torch.allclose(p.data, X), "restore_exact must install X"


def test_ef21_estimator_tracks_a_constant_target():
    """With a constant gradient the EF21 estimator converges to it, which is why
    compressing before the oracle restores the descent property."""
    p = nn.Parameter(torch.zeros(6, 6))
    opt = EF21MuonUSign([p], lr=0.0, momentum=0.0, lmo_dtype=torch.float32)
    target = torch.randn(6, 6)
    for _ in range(1000):
        p.grad = target.clone()
        opt.step()
    est = opt.state[p]["grad_estimator"]
    assert (est - target).norm() / target.norm() < 1e-3, (est - target).norm()


# --------------------------------------------------------------------------
# Per-layer learning-rate scaling
# --------------------------------------------------------------------------


def test_unit_gain_reproduces_the_shipped_muon_factor():
    """The criterion's strongest validation.

    Applying the unit-gain criterion to an LMO step must reproduce
    ``sqrt(max(1, fan_out/fan_in))``, the aspect factor in the reference Muon
    implementation. If this failed, the criterion would be ad hoc.
    """
    from common.lr_scaling import FAMILY_LMO, layer_multiplier, resolve_rule

    ug, legacy = resolve_rule("unit-gain"), resolve_rule("legacy")
    for shape in [(64, 3, 3, 3), (512, 512, 3, 3), (512, 256, 1, 1), (10, 512)]:
        a = layer_multiplier(ug, FAMILY_LMO, shape)
        b = layer_multiplier(legacy, FAMILY_LMO, shape)
        m, n = shape[0], math.prod(shape[1:]) if len(shape) > 1 else 1
        assert abs(a - b) < 1e-12, (shape, a, b)
        assert abs(a - math.sqrt(max(1.0, m / n))) < 1e-12


def test_unit_gain_equalizes_the_per_step_gain_exactly():
    """The invariant the whole rule exists to enforce.

    ``unit-gain`` is exactly ``lambda = sqrt(fan_out)/||s||_F``, so the per-step RMS
    gain ``gamma(eta_0 * lambda * s) = eta_0 * lambda * ||s||_F / sqrt(fan_out)``
    equals ``eta_0`` on *every* layer and in *both* families. If this ever fails the
    rule has been mis-implemented.
    """
    from common.lr_scaling import (FAMILY_LMO, FAMILY_SIGN, RESNET18_SHAPES,
                                   resolve_rule)

    rule = resolve_rule("unit-gain")
    for _, m, n in RESNET18_SHAPES:
        for family, s_fro in ((FAMILY_LMO, math.sqrt(min(m, n))),
                              (FAMILY_SIGN, math.sqrt(m * n))):
            lam = rule.multiplier(family, m, n)
            assert abs(lam - math.sqrt(m) / s_fro) < 1e-12, (m, n, family)
            gain = lam * s_fro / math.sqrt(m)
            assert abs(gain - 1.0) < 1e-12, (m, n, family, gain)


def test_initialization_gain_is_shape_independent():
    """The rule's premise: a fan-in-scaled init has the same gain at every shape, so
    the constant of proportionality is absorbed into eta_0 rather than varying."""
    from common.lr_scaling import RESNET18_SHAPES

    for _, m, n in RESNET18_SHAPES:
        he = math.sqrt(2 / n) * math.sqrt(m * n) / math.sqrt(m)          # He normal
        torch_default = (1 / math.sqrt(3 * n)) * math.sqrt(m * n) / math.sqrt(m)
        assert abs(he - math.sqrt(2)) < 1e-12
        assert abs(torch_default - 1 / math.sqrt(3)) < 1e-12


def test_unit_gain_sign_family_is_inverse_sqrt_fan_in():
    """And it must be independent of fan-out, which is what the algebra says."""
    from common.lr_scaling import FAMILY_SIGN, layer_multiplier, resolve_rule

    rule = resolve_rule("unit-gain")
    for shape in [(64, 3, 3, 3), (512, 512, 3, 3), (128, 64, 1, 1)]:
        n = math.prod(shape[1:])
        assert abs(layer_multiplier(rule, FAMILY_SIGN, shape) - 1 / math.sqrt(n)) < 1e-12
    # same fan-in, different fan-out => same multiplier
    a = layer_multiplier(rule, FAMILY_SIGN, (64, 64, 3, 3))
    b = layer_multiplier(rule, FAMILY_SIGN, (512, 64, 3, 3))
    assert abs(a - b) < 1e-12


def test_power_presets_match_named_rules():
    from common.lr_scaling import FAMILY_SIGN, layer_multiplier, resolve_rule

    shape = (256, 128, 3, 3)
    for spec, named in [("power:0.5", "unit-gain"), ("power:1", "mup"),
                        ("power:0.5,0.5", "mishra-analysis"), ("power:0", "none")]:
        a = layer_multiplier(resolve_rule(spec), FAMILY_SIGN, shape)
        b = layer_multiplier(resolve_rule(named), FAMILY_SIGN, shape)
        assert abs(a - b) < 1e-12, (spec, named, a, b)


def test_rms_gain_identity():
    """``gamma(A) = ||A||_F / sqrt(m)`` really is the RMS gain, and the two families
    have the exact Frobenius norms the derivation assumes."""
    torch.manual_seed(7)
    m, n = 40, 90
    M = torch.randn(m, n)
    P = exact_polar(M).float()
    assert abs(float(P.norm()) - math.sqrt(min(m, n))) < 1e-4       # ||UV^T||_F = sqrt(r)
    S = torch.sign(P)
    assert abs(float(S.norm()) - math.sqrt(m * n)) < 1e-4           # +-1 entries

    # rms(A u) / rms(u) -> ||A||_F / sqrt(m) in expectation
    u = torch.randn(n, 2048)
    for A in (P, S):
        emp = float((A @ u).norm() / math.sqrt(m)) / float(u.norm() / math.sqrt(n))
        pred = float(A.norm()) / math.sqrt(m)
        assert abs(emp - pred) / pred < 0.05, (emp, pred)


def test_lambda_mult_equals_folding_the_factor_into_the_lmo():
    """Putting the aspect factor in ``lambda_mult`` (with ``scale_aspect=False``)
    is exactly equivalent to leaving it inside the LMO -- so the refactor that
    moved it out cannot have changed any Muon-family result."""
    torch.manual_seed(11)
    # Both regimes: fan_out > fan_in (factor sqrt(2) != 1, so the test bites) and
    # fan_out < fan_in (factor clipped to 1).
    for shape in [(16, 2, 2, 2), (20, 5), (64, 64, 3, 3)]:
        n = math.prod(shape[1:])
        lam = math.sqrt(max(1.0, shape[0] / n))
        grads = [torch.randn(*shape) for _ in range(4)]

        p_in = nn.Parameter(torch.zeros(*shape))
        opt_in = Muon([p_in], lr=0.1, momentum=0.5, lmo_dtype=torch.float32,
                      scale_aspect=True)
        p_out = nn.Parameter(torch.zeros(*shape))
        opt_out = Muon([{"params": [p_out], "lambda_mult": lam}], lr=0.1, momentum=0.5,
                       lmo_dtype=torch.float32, scale_aspect=False)

        for g in grads:
            p_in.grad, p_out.grad = g.clone(), g.clone()
            opt_in.step()
            opt_out.step()
            assert torch.allclose(p_in, p_out, atol=1e-6), \
                f"{shape} (lambda={lam:.4f}): {(p_in - p_out).abs().max()}"
        assert p_in.abs().sum() > 0, "the test must actually take steps"


def test_optimizer_families_are_tagged():
    from common.lr_scaling import FAMILY_LMO, FAMILY_SIGN

    expected = {
        SignMuon: FAMILY_SIGN, MuonSign: FAMILY_SIGN, SignSGD: FAMILY_SIGN,
        Muon: FAMILY_LMO, MuonUSign: FAMILY_LMO, EF21SignMuon: FAMILY_LMO,
        EF21MuonUSign: FAMILY_LMO, EF21MuonSign: FAMILY_LMO,
    }
    for cls, fam in expected.items():
        assert cls.family == fam, (cls.__name__, cls.family, fam)


def test_build_optimizers_assigns_one_group_per_matrix_param():
    import argparse

    from centralized.train import build_optimizers

    args = argparse.Namespace(
        optimizer="signmuon", lr=0.1, lr_aux=0.01, momentum=0.9, nesterov=False,
        weight_decay=0.0, ns_steps=5, lmo_dtype="float32", lr_scaling="unit-gain",
        head_adamw="always", n_head_tensors=2)
    model = TinyNet()
    opt_main, opt_aux, info = build_optimizers(model, args)

    assert len(opt_main.param_groups) == 1                  # TinyNet has one matrix
    group = opt_main.param_groups[0]
    assert group["name"] == "fc1.weight"
    assert abs(group["lambda_mult"] - 1 / math.sqrt(6)) < 1e-12    # fan_in = 6
    assert group["scale_aspect"] is False, "the factor must not be applied twice"
    assert opt_aux is not None and len(opt_aux.param_groups[0]["params"]) == 3
    assert info["rule"] == "unit-gain" and info["family"] == "sign"


# --------------------------------------------------------------------------
# Protocol: validation split and metric helpers
# --------------------------------------------------------------------------


def test_val_split_is_disjoint_and_seed_stable():
    from centralized.data import _split_indices

    tr1, va1 = _split_indices(50_000, 5_000, val_seed=12345)
    tr2, va2 = _split_indices(50_000, 5_000, val_seed=12345)
    assert (tr1 == tr2).all() and (va1 == va2).all(), "split must be deterministic"
    assert len(va1) == 5_000 and len(tr1) == 45_000
    assert not (set(tr1.tolist()) & set(va1.tolist())), "train and val must be disjoint"

    _, va3 = _split_indices(50_000, 5_000, val_seed=999)
    assert set(va1.tolist()) != set(va3.tolist()), "a different val_seed must differ"


def test_history_selection_helpers():
    h = History()
    for step, (va, te) in enumerate([(80.0, 79.0), (85.0, 84.0), (83.0, 90.0),
                                     (84.0, 86.0), (84.5, 87.0)]):
        h.record(step, val_acc=va, test_acc=te)

    assert h.argbest("val_acc", "max") == 1          # best val, not best test
    assert h.at("test_acc", 1) == 84.0               # the number you'd report
    assert abs(h.last_k_mean("test_acc", 3) - (90.0 + 86.0 + 87.0) / 3) < 1e-12
    assert h.steps_to_target("test_acc", 86.0) == 2
    assert h.steps_to_target("test_acc", 99.0) is None
    assert h.values("val_acc") == [80.0, 85.0, 83.0, 84.0, 84.5]


def test_split_param_names():
    model = TinyNet()
    matrix, aux = split_param_names(model, n_head_tensors=2)
    assert matrix == ["fc1.weight"]
    assert aux == ["fc1.bias", "fc2.weight", "fc2.bias"]
    assert set(matrix) | set(aux) == {n for n, _ in model.named_parameters()}


def test_history_pads_missing_series():
    h = History()
    h.record(0, test_acc=1.0)
    h.record(5, test_acc=2.0, test_loss=0.5)
    assert h.to_dict() == {"steps": [0, 5], "test_acc": [1.0, 2.0],
                           "test_loss": [None, 0.5]}
    assert h.last("test_acc") == 2.0


def test_aggregate_groups_by_seed(tmp_path=None):
    import json
    import tempfile
    from pathlib import Path

    import aggregate

    root = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    for seed, acc in ((0, 80.0), (1, 82.0), (2, 84.0)):
        d = root / "run" / f"seed{seed}"
        d.mkdir(parents=True)
        with open(d / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"config": {"algorithm": "signmuon", "lr": 0.01, "seed": seed,
                                  "device": f"cuda:{seed}"},
                       "history": {"steps": [0, 1], "test_acc": [10.0, acc]}}, f)

    runs = aggregate.load_runs([root])
    assert len(runs) == 3
    keys = {aggregate.group_key(r["config"]) for r in runs}
    assert len(keys) == 1, "seed and device must not split the group"

    agg = aggregate.aggregate_group(runs, "test_acc")
    assert agg["steps"] == [0, 1] and agg["n_runs"] == 3
    assert abs(agg["mean"][-1] - 82.0) < 1e-12
    assert abs(agg["std"][-1] - 2.0) < 1e-12          # sample std of 80, 82, 84


def test_seeds_from_two_machines_stay_one_group():
    """``hardware`` must not split a seed set.

    ``save_run`` stamps the GPU, driver and commit into every config, so before
    ``hardware`` was ignored a five-seed sweep spread over two boxes grouped as two
    groups of one -- and every tool downstream then reported "1 seed" and printed a
    blank std, which reads as agreement rather than as a missing measurement.
    """
    import aggregate

    base = {"algorithm": "signmuon", "lr": 0.05, "rounds": 2000}
    a = dict(base, seed=0, hardware={"gpu_name": "A100", "torch_version": "2.8.0"})
    b = dict(base, seed=1, hardware={"gpu_name": "RTX A4000", "torch_version": "2.9.0"})
    assert aggregate.group_key(a) == aggregate.group_key(b)

    # ...but a real difference still splits, or the grouping would be useless.
    c = dict(base, seed=2, lr=0.02, hardware=a["hardware"])
    assert aggregate.group_key(a) != aggregate.group_key(c)


def test_budget_rebalancing_keeps_every_requested_phase_or_says_it_did_not():
    """A phase the user asked for is either run or reported as dropped.

    `autobalance` re-sorts `args.phases` into a canonical order at the end, and that
    list was spelled out by hand. When `rules` was added it went into the argparse
    choices, the schedule and the phase sequence but not into that one list, so
    **any** --budget-hours deleted it -- at 24 h against a 14 h plan, with ten hours
    of slack and no note. Silent is the part that matters: a dropped phase the
    report names is a decision, one it does not is a lie about what ran.
    """
    import sys

    from federated import overnight as ov

    for budget in (0, 24, 12, 8, 4):
        argv = sys.argv
        try:
            sys.argv = ["x", "--phases", *ov.PHASE_ORDER, "--budget-hours", str(budget)]
            args = ov.get_args()
        finally:
            sys.argv = argv
        asked = set(args.phases)
        notes = ov.autobalance(args, 0.127, ov.Budget(budget))
        blob = " ".join(notes)
        for phase in asked - set(args.phases):
            said = ov.PHASE_LABELS.get(phase, phase)
            assert said in blob, (
                f"at --budget-hours {budget} the '{phase}' phase was dropped without "
                f"a note; notes were {notes}")
        # Every sheddable phase must have a label, or the check above passes
        # vacuously on the next phase somebody adds.
        assert set(ov.PHASE_LABELS) <= set(ov.PHASE_ORDER)
        assert {"rules", "wd", "verify"} <= set(ov.PHASE_LABELS)
        assert list(args.phases) == [p for p in ov.PHASE_ORDER if p in args.phases], \
            "phases must come back in canonical order"

    # ...and a shortened tuning horizon is a warning, not a neutral adjustment: it
    # puts back the proxy that picked the wrong rate for both methods checked.
    argv = sys.argv
    try:
        sys.argv = ["x", "--phases", "lr", "final", "--budget-hours", "6"]
        args = ov.get_args()
    finally:
        sys.argv = argv
    notes = ov.autobalance(args, 0.127, ov.Budget(6))
    if args.tune_rounds < args.final_rounds:
        assert any("PROXY" in n for n in notes), notes
        assert "verify" in args.phases, \
            "a proxy horizon must re-enable the check that catches a bad proxy"


def test_the_export_keeps_a_baseline_and_its_scaled_ablation_apart():
    """`adam` and `adam --scale-baselines` are two rows, not one.

    They share an ``algorithm`` and, in the reported protocol, a seed count, so
    keying a table row on the algorithm made the winner a dict-ordering accident.
    The 2026-07-30 export did exactly that and printed the ablation, at eta_0=0.05,
    as the Adam baseline -- 1.4 points above the plain method the paper reports.
    """
    from pathlib import Path

    from federated.export_article import Run, group_reported

    def mk(scaled, lr, seed, acc):
        cfg = {"algorithm": "adam", "lr": lr, "seed": seed, "split": "full",
               "weight_decay": 0.0, "last_k": 2, "scale_baselines": scaled,
               "lr_scaling": "unit-gain", "dataset": "cifar10", "model": "cnn2"}
        hist = {"steps": [0, 100], "test_acc": [10.0, acc]}
        return Run(Path(f"/tmp/{'s' if scaled else 'p'}{seed}/metrics.json"),
                   {"config": cfg, "history": hist})

    runs = ([mk(False, 0.001, s, 77.0 + s * 0.1) for s in range(5)]
            + [mk(True, 0.05, s, 78.4 + s * 0.1) for s in range(5)])
    groups = group_reported(runs)
    assert set(groups) == {"adam", "adam+scaled"}, groups
    assert len(groups["adam"]) == 5 and len(groups["adam+scaled"]) == 5
    assert all(not r.config["scale_baselines"] for r in groups["adam"])
    # ...and the ablation sorts after its baseline rather than above it.
    assert list(groups) == ["adam", "adam+scaled"]


def test_the_rule_ablation_stays_out_of_the_reported_table():
    """`rules`-phase finals are full-50k, zero-decay runs like any other.

    Only ``lr_scaling`` tells them apart, so without the inference in ``scan`` they
    land in `tab:exp_3` as extra rows. And the ordering they are compared on has to
    span the same methods under every rule: the ablation re-tunes three, while the
    reported rule has eleven, and comparing those two lists finds a difference every
    time. Both of these were live bugs.
    """
    from pathlib import Path

    from federated.export_article import (Run, group_reported, phase_of, rule_rows,
                                          scan)

    def mk(alg, rule, seed, acc, tmp):
        cfg = {"algorithm": alg, "lr": 0.05, "seed": seed, "split": "full",
               "weight_decay": 0.0, "last_k": 2, "lr_scaling": rule,
               "scale_baselines": False, "dataset": "cifar10", "model": "cnn2"}
        hist = {"steps": [0, 100], "test_acc": [10.0, acc]}
        return Run(Path(tmp) / f"{alg}_{rule}_{seed}" / "metrics.json",
                   {"config": cfg, "history": hist})

    # Eight methods at the reported rule, three of them also re-tuned under `none`.
    reported = [mk(a, "unit-gain", s, acc, "/t")
                for a, acc in (("muon", 86.0), ("signmuon", 85.9), ("muonusign", 84.5),
                               ("signsgd", 81.7), ("muonsign", 81.4))
                for s in range(3)]
    ablation = [mk(a, "none", s, acc, "/t")
                for a, acc in (("signmuon", 85.2), ("signsgd", 81.5), ("muonsign", 81.0))
                for s in range(3)]
    for r in ablation:                       # what scan's second pass does
        r.phase = phase_of(r.config, "unit-gain")

    assert {r.phase for r in ablation} == {"rules"}
    assert set(group_reported(reported + ablation)) == {
        "muon", "signmuon", "muonusign", "signsgd", "muonsign"}, \
        "the ablation must not add rows to the reported table"

    rows = rule_rows(reported + ablation)
    assert {r["algorithm"] for r in rows} == {"signmuon", "signsgd", "muonsign"}, \
        "only the methods the ablation actually re-tuned belong in it"
    per_rule = {}
    for r in rows:
        per_rule.setdefault(r["rule"], []).append((r["acc_mean"], r["algorithm"]))
    assert len(per_rule) == 2 and all(len(v) == 3 for v in per_rule.values())


def test_the_export_drops_runs_made_under_a_superseded_sign_convention():
    """A night's leftovers under the old zero convention are not comparable.

    The tree on the GPU box accumulates. When the 2026-07-31 federated study
    re-ran under `--uplink-zeros random --mv-ties random`, the previous night's
    tuning jobs, made under `keep`/`zero`, were still sitting beside it at the
    same nominal `(method, eta_0, rounds)`. They landed in the bundle's tuning
    table as duplicate rows at a different accuracy, one of them above the rate
    the finals actually used, which reads as a selection failure rather than as
    the two experiments it is.
    """
    import json
    import tempfile
    from pathlib import Path

    from federated.export_article import scan

    def write(root, name, uplink, ties, split, lr):
        d = root / name / "seed0"
        d.mkdir(parents=True)
        cfg = {"algorithm": "signmuon", "lr": lr, "seed": 0, "rounds": 200,
               "split": split, "weight_decay": 0.0, "last_k": 2, "run_name": name,
               "target_acc": 80.0, "dataset": "cifar10", "model": "cnn2",
               "lr_scaling": "unit-gain", "scale_baselines": False,
               "uplink_zeros": uplink, "mv_ties": ties}
        hist = {"steps": [0, 100, 200], "test_acc": [10.0, 80.0, 85.0]}
        (d / "metrics.json").write_text(json.dumps({"config": cfg, "history": hist}))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "federated"
        write(root, "final_signmuon_e200_f", "random", "random", "full", 0.1)
        write(root, "tune_lr_signmuon_0.1_e200_t", "random", "random", "tune", 0.1)
        write(root, "tune_verify_signmuon_0.05_e200_t", "keep", "zero", "tune", 0.05)

        runs, bad = scan(root)
        assert [r.config["run_name"] for r in runs] == [
            "final_signmuon_e200_f", "tune_lr_signmuon_0.1_e200_t"], \
            "the leftover run must not reach the tuning table"
        assert len(bad) == 1 and "sign convention" in bad[0][1]

    # A tree that predates the convention fields entirely is one experiment, not a
    # tree of leftovers: nothing is dropped when every run agrees by default.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "federated"
        for name, split in (("final_signmuon_e200_f", "full"),
                            ("tune_lr_signmuon_0.1_e200_t", "tune")):
            d = root / name / "seed0"
            d.mkdir(parents=True)
            cfg = {"algorithm": "signmuon", "lr": 0.1, "seed": 0, "rounds": 200,
                   "split": split, "weight_decay": 0.0, "last_k": 2, "run_name": name,
                   "target_acc": 80.0, "dataset": "cifar10", "model": "cnn2",
                   "lr_scaling": "unit-gain", "scale_baselines": False}
            hist = {"steps": [0, 100, 200], "test_acc": [10.0, 80.0, 85.0]}
            (d / "metrics.json").write_text(json.dumps({"config": cfg, "history": hist}))
        runs, bad = scan(root)
        assert len(runs) == 2 and not bad


def test_the_export_bundle_round_trips_to_a_plottable_tree():
    """export_article -> .zip -> open_bundle -> a tree the plotters can read.

    This is the whole federated workflow off a remote box: compute, bundle, download
    one file, unpack, plot. The load-bearing part is that the bundle keeps the
    ``metrics.json`` tree shape, so ``--bundle`` is just ``--root`` on the unpacked
    copy and no figure code has to know that bundles exist.
    """
    import json
    import tempfile
    from pathlib import Path

    import aggregate
    from federated import export_article

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "federated"
        for alg, lr, accs in (("signmuon", 0.05, (84.0, 84.4)),
                              ("muon", 0.1, (85.8, 86.2))):
            for seed, acc in enumerate(accs):
                d = root / f"fed_{alg}" / f"seed{seed}"
                d.mkdir(parents=True)
                cfg = {"algorithm": alg, "lr": lr, "seed": seed, "rounds": 200,
                       "split": "full", "weight_decay": 0.0, "last_k": 2,
                       "target_acc": 80.0, "dataset": "cifar10", "model": "cnn2",
                       "n_parties": 11, "uplink_zeros": "random",
                       "n_matrix_params": 762_560, "n_aux_params": 2_146,
                       "n_matrix_layers": 3, "device": "cuda:0"}
                hist = {"steps": [0, 100, 200],
                        "test_acc": [10.0, acc - 1.0, acc],
                        "test_loss": [2.3, 0.6, 0.5],
                        "gain_spread": [1.0, 1.0, 1.0]}
                (d / "metrics.json").write_text(
                    json.dumps({"config": cfg, "history": hist}), encoding="utf-8")
                # A model.pt is what makes the tree too big to move; it must be
                # left behind rather than bundled.
                (d / "model.pt").write_bytes(b"\x00" * 4096)

        out = Path(td) / "federated_export"
        rc = export_article.main(["--root", str(root), "--out", str(out),
                                  "--overnight", str(Path(td) / "nope")])
        assert rc == 0, "export failed"

        for name in ("SUMMARY.md", "MANIFEST.json", "runs.csv", "curves.csv",
                     "table_federated.csv", "communication.csv", "configs.json"):
            assert (out / name).is_file(), f"{name} missing from the bundle"

        table = (out / "table_federated.csv").read_text(encoding="utf-8")
        assert "signmuon" in table and "muon" in table
        # The accounting comes from `communication_bits`, not a second copy of it.
        comm = (out / "communication.csv").read_text(encoding="utf-8")
        assert "29.4" in comm, comm

        archive = out.parent / "federated_export_results.zip"
        assert archive.is_file(), "the .zip is the file to download"
        assert not any(p.name == "model.pt" for p in out.rglob("*")), \
            "model weights must not travel in the bundle"

        # Unpack somewhere else entirely, as a download would.
        dest = Path(td) / "downloaded"
        dest.mkdir()
        bundle = export_article.open_bundle(archive, unpack_to=dest)
        runs_dir = export_article.runs_root(bundle)
        runs = aggregate.load_runs([runs_dir])
        assert len(runs) == 4, f"expected 4 runs in the unpacked bundle, got {len(runs)}"
        keys = {aggregate.group_key(r["config"]) for r in runs}
        assert len(keys) == 2, "two methods, two seeds each"


def test_aggregate_labels_are_unique():
    """Two groups differing only in an unprinted field must not share a label.

    ``describe`` prints a fixed shortlist of keys, so groups that differ only in,
    say, ``lr_scaling`` used to render identically -- and the curves dict, keyed
    on that label, silently kept whichever was written last. On the CIFAR sweep
    that discarded seven groups, including three-seed runs overwritten by
    single-seed ones.
    """
    import aggregate

    base = {"dataset": "cifar10", "optimizer": "muon", "epochs": 75, "lr": 0.05}
    groups = {
        ("a",): [{"config": dict(base, lr_scaling="unit", seed=0)}],
        ("b",): [{"config": dict(base, lr_scaling="none", seed=0)}],
        ("c",): [{"config": {"dataset": "cifar10", "optimizer": "sgd", "lr": 0.02}}],
    }
    labels = aggregate.unique_labels(groups)

    assert len(set(labels.values())) == 3, "labels still collide"
    assert "lr_scaling=unit" in labels[("a",)]
    assert "lr_scaling=none" in labels[("b",)]
    # An uncontested label is left exactly as ``describe`` renders it, and only
    # the fields that actually differ are appended.
    assert labels[("c",)] == aggregate.describe(groups[("c",)][0]["config"])
    assert "seed" not in labels[("a",)], "ignored fields must not leak into the suffix"
    assert "epochs" not in labels[("a",)].split("lr_scaling")[1]


# --------------------------------------------------------------------------
# Federated protocol: per-layer scaling, data split, GPU augmentation
# --------------------------------------------------------------------------


def test_federated_legacy_rule_is_the_old_convention():
    """``lr_scaling='legacy'`` must reproduce ``scale_aspect=True``, bit for bit.

    The driver now applies the shape factor *outside* the oracle
    (``scale_aspect=False`` plus a per-layer multiplier), as the centralized path
    does. Under ``legacy`` the LMO family's multiplier IS the aspect factor, so
    the two conventions have to coincide -- otherwise switching the plumbing would
    silently have changed every published federated number.
    """
    from common.optimizers import muon_lmo

    torch.manual_seed(0)
    for shape in [(10, 4), (4, 10), (7, 7), (6, 3, 2, 2)]:
        G = torch.randn(*shape)
        inside = muon_lmo(G, ns_steps=5, dtype=torch.float32, scale_aspect=True)
        m = shape[0]
        n = 1
        for s in shape[1:]:
            n *= s
        outside = muon_lmo(G, ns_steps=5, dtype=torch.float32, scale_aspect=False) \
            * math.sqrt(max(1.0, m / n))
        assert torch.allclose(inside, outside, atol=1e-6), shape


def _compare_drivers_under_rule(rule_name, net, loader_fn, weight_decay=0.0,
                                rounds=5, lr=0.05, lr_aux=0.01, momentum=0.8):
    """N=1 federated vs centralized under ``rule_name``; returns the mismatches.

    The centralized side gets one param group per matrix tensor carrying that
    tensor's ``lambda_mult``, with ``scale_aspect=False`` -- i.e. the convention
    ``centralized/train.py`` actually uses. That is what makes this a test of the
    two drivers agreeing rather than of either one alone.
    """
    from common.lr_scaling import layer_multiplier, resolve_rule
    from federated.algorithms import method_family

    rule = resolve_rule(rule_name)
    failures = []
    for method in CENTRAL_CLASSES:
        torch.manual_seed(0)
        ref_model = net()
        matrix_names, aux_names = split_param_names(ref_model, 2)
        named = dict(ref_model.named_parameters())
        family = method_family(method)
        groups = [{"params": [named[n]],
                   "lambda_mult": layer_multiplier(rule, family, tuple(named[n].shape))}
                  for n in matrix_names]
        opt = CENTRAL_CLASSES[method](groups, lr=lr, momentum=momentum,
                                      weight_decay=weight_decay,
                                      decoupled_weight_decay=True,
                                      lmo_dtype=torch.float32, scale_aspect=False)
        aux = torch.optim.AdamW([named[n] for n in aux_names], lr=lr_aux,
                                weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        for _ in range(rounds):
            for x, y in loader_fn():
                ref_model.zero_grad(set_to_none=True)
                criterion(ref_model(x), y).backward()
                opt.step()
                aux.step()
        if hasattr(opt, "restore_exact"):
            opt.restore_exact()

        torch.manual_seed(0)
        fed_model = net()
        run_federated(method, fed_model, [loader_fn()], [loader_fn()],
                      rounds=rounds, n_steps=1, lr=lr, lr_aux=lr_aux,
                      momentum=momentum, weight_decay=weight_decay,
                      eval_freq=10 ** 9, device="cpu", cosine_schedule=False,
                      lmo_dtype=torch.float32, lr_scaling=rule_name, verbose=False)
        for (n, a), (_, b) in zip(ref_model.named_parameters(),
                                  fed_model.named_parameters()):
            if not torch.allclose(a, b, atol=TOL):
                failures.append(f"{method}/{n}: max|diff| = {(a - b).abs().max():.3e}")
    return failures


def test_federated_per_layer_scaling_matches_centralized():
    """N=1 federated under a rule == the centralized optimizer under the same rule.

    The load-bearing equivalence, extended past ``legacy``: with the shape factor
    switched on, the two drivers must still agree, or the federated and
    centralized tables would be reporting different algorithms under one name.
    Run on BOTH fixtures -- ``TinyNet``'s only matrix is ``(4, 6)``, where the LMO
    family's multiplier is exactly 1 and only the sign family is exercised.
    """
    for net, loader_fn in ((TinyNet, tiny_loader), (TallNet, tall_loader)):
        failures = _compare_drivers_under_rule("unit-gain", net, loader_fn)
        assert not failures, (f"unit-gain federated != centralized on "
                              f"{net.__name__}:\n  " + "\n  ".join(failures))


def test_method_families_are_tagged_consistently():
    """The family decides the multiplier, so a wrong tag mis-scales a whole method."""
    from common.lr_scaling import FAMILY_LMO, FAMILY_SIGN
    from federated.algorithms import METHODS, method_family

    expected = {"signmuon": FAMILY_SIGN, "muonsign": FAMILY_SIGN, "signsgd": FAMILY_SIGN,
                "ef21signmuon": FAMILY_LMO, "muonusign": FAMILY_LMO,
                "ef21muonusign": FAMILY_LMO, "ef21muonsign": FAMILY_LMO,
                "muon": FAMILY_LMO, "muonserver": FAMILY_LMO,
                "sgd": None, "adam": None}
    assert set(expected) == set(METHODS)
    for name, want in expected.items():
        assert method_family(name) == want, name
    # --scale-baselines is the only thing that gives SGD/Adam a rule.
    assert method_family("adam", scale_baselines=True) == FAMILY_SIGN
    assert method_family("muon", scale_baselines=True) == FAMILY_LMO


def test_federated_step_norms_match_the_family():
    """A sign-family step has ``||s||_F = sqrt(mn)``; an LMO one ``~sqrt(min(m,n))``.

    This is the premise the whole per-layer rule rests on, checked on the real
    driver rather than on the step maps in isolation.
    """
    from common.lr_scaling import FAMILY_SIGN
    from federated.algorithms import method_family

    for method in ("signmuon", "muonsign", "signsgd", "muonusign", "ef21muonusign"):
        torch.manual_seed(0)
        model = TinyNet()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        run_federated(method, model, [tiny_loader()], [tiny_loader()],
                      rounds=1, n_steps=1, lr=1.0, lr_aux=0.0, momentum=0.0,
                      eval_freq=10 ** 9, device="cpu", cosine_schedule=False,
                      lmo_dtype=torch.float32, lr_scaling="none", verbose=False)
        p = dict(model.named_parameters())["fc1.weight"]
        norm = (p.detach() - before["fc1.weight"]).norm().item()
        m, n = p.shape
        if method_family(method) == FAMILY_SIGN:
            assert abs(norm / math.sqrt(m * n) - 1.0) < 1e-5, f"{method}: {norm}"
        else:
            # Newton-Schulz only approximates the polar factor's norm.
            assert 0.7 < norm / math.sqrt(min(m, n)) < 1.3, f"{method}: {norm}"


def test_majority_vote_ties_are_recorded():
    """An even client split abstains, and the driver has to say how often.

    ``sign(sum_j s^(j))`` is zero wherever the clients split evenly, so before a
    tie-break the aggregate is not in ``{-1,+1}`` at even ``N``. Under the default
    ``mv_ties="random"`` it is, and the recorded fraction is therefore a raw
    pre-break diagnostic rather than a property of the transmitted vote. It is
    recorded rather than silently absorbed into the step size. The paper runs odd
    ``N``, where the vote cannot tie at all.
    """
    torch.manual_seed(0)
    loaders = [tiny_loader(seed=s) for s in range(4)]        # even N, different data
    h = run_federated("signmuon", TinyNet(), loaders, [tiny_loader()],
                      rounds=3, n_steps=1, lr=0.01, eval_freq=1, device="cpu",
                      lmo_dtype=torch.float32, verbose=False)
    ties = h.values("mv_tie_frac")
    assert len(ties) == 3, "one tie fraction per evaluated training round"
    assert all(0.0 <= t <= 1.0 for t in ties)

    # ... and the EF21 uplink has no vote at all, so nothing is recorded.
    h2 = run_federated("ef21muonusign", TinyNet(), loaders, [tiny_loader()],
                       rounds=2, n_steps=1, lr=0.01, eval_freq=1, device="cpu",
                       lmo_dtype=torch.float32, verbose=False)
    assert not h2.values("mv_tie_frac")


def test_the_uplink_rule_turns_the_compressor_zeros_into_one_bit():
    """The compressor emits `{-1, 0, +1}`; the uplink rule makes the channel `+-1`.

    Not a corner case: `polar(M)` has an exactly-zero column wherever `M` does, and
    `M` does wherever a feature was zero across the whole local batch -- which after
    ReLU and MaxPool is common (measured at 0.1-3.0% of entries on CNN2). Left alone,
    that would mean the uplink is not literally one bit per parameter, and an ODD
    client count would not by itself prevent ties, a zero vote not being `+-1`.

    So it is not left alone: `uplink_zeros` defaults to `random`, which maps each
    zero to a fair `+-1`, and the paper's accounting assumes exactly that. The
    ternary alphabet is now the opt-in (`--uplink-zeros keep`) and is retained only
    as a diagnostic; the raw zero rate is still recorded either way, which is what
    the last assertion here pins. `communication_bits` is what covers `keep`.
    """
    M = torch.randn(6, 8)
    M[2] = 0.0                                    # a dead unit's gradient row
    s = torch.sign(muon_lmo(M, ns_steps=5, dtype=torch.float32, scale_aspect=False))
    assert int((s == 0).sum()) >= 8, "a zero row of M must give zeros in sign(polar(M))"

    # ... and the driver measures it rather than assuming it away.
    loaders = [tiny_loader(seed=s) for s in range(5)]        # ODD client count
    h = run_federated("signmuon", TinyNet(), loaders, [tiny_loader()],
                      rounds=2, n_steps=1, lr=0.01, eval_freq=1, device="cpu",
                      lmo_dtype=torch.float32, verbose=False)
    assert len(h.values("uplink_zero_frac")) == 2
    assert all(0.0 <= z <= 1.0 for z in h.values("uplink_zero_frac"))

    # Forcing the zeros to +-1 makes the channel a genuine one bit, and then an
    # odd count really cannot tie.
    for rule in ("random", "positive"):
        h2 = run_federated("signmuon", TinyNet(), loaders, [tiny_loader()],
                           rounds=2, n_steps=1, lr=0.01, eval_freq=1, device="cpu",
                           lmo_dtype=torch.float32, uplink_zeros=rule, verbose=False)
        assert all(t == 0.0 for t in h2.values("mv_tie_frac")), \
            f"uplink_zeros={rule} at odd N must never tie, got {h2.values('mv_tie_frac')}"
        # The zero fraction still reports the RAW compressor output, so the
        # diagnostic does not vanish when the rule hides the zeros.
        assert h2.values("uplink_zero_frac") == h.values("uplink_zero_frac")


def test_random_tie_break_yields_a_true_sign_matrix():
    """`--mv-ties random` restores ||s||_F = sqrt(mn), which unit-gain assumes.

    Also pins the two properties that make it safe to offer: the recorded tie
    fraction still measures the *vote* (it is counted before the break), and the
    run stays reproducible because the coin comes from the seeded global RNG.
    """
    def go(rule, seed):
        torch.manual_seed(seed)
        model = TinyNet()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        h = run_federated("signmuon", model, [tiny_loader(seed=s) for s in range(4)],
                          [tiny_loader()], rounds=1, n_steps=1, lr=1.0, lr_aux=0.0,
                          momentum=0.0, eval_freq=1, device="cpu",
                          cosine_schedule=False, lr_scaling="none",
                          lmo_dtype=torch.float32, mv_ties=rule, verbose=False)
        p = dict(model.named_parameters())["fc1.weight"]
        return (p.detach() - before["fc1.weight"]), h.values("mv_tie_frac")[-1]

    step_zero, frac_zero = go("zero", 0)
    step_rand, frac_rand = go("random", 0)
    m, n = step_zero.shape

    assert frac_zero > 0, "the fixture must actually produce ties at N=4"
    assert abs(frac_rand - frac_zero) < 1e-12, \
        "mv_tie_frac must count the vote, not the post-break result"
    # Abstaining shortens the step; breaking the tie restores the full sign norm.
    assert step_zero.norm().item() < 0.999 * math.sqrt(m * n)
    assert abs(step_rand.norm().item() / math.sqrt(m * n) - 1.0) < 1e-5
    # Every entry is a unit sign step. Compared with a tolerance, not for exact
    # equality: the step is recovered as `p - p_before`, and that subtraction
    # carries float32 rounding of order 1e-7.
    assert (step_rand.abs() - 1.0).abs().max().item() < 1e-6

    # Reproducible: same seed, same coins.
    again, _ = go("random", 0)
    assert torch.equal(step_rand, again)


def test_averaging_polar_factors_shrinks_the_step_but_polar_of_the_average_does_not():
    """Why there are two full-precision Muon controls, not one.

    `muon` orthogonalizes on the worker and the server averages the polar factors;
    `muonserver` averages first and orthogonalizes once. The second is the
    uncompressed control for MuonUSign / EF21-MuonUSign / EF21-MuonSign, because it
    is what those methods become when the compressor is the identity.

    They are not interchangeable: averaging near-orthogonal matrices *shortens* the
    step, by an amount that grows with the client count and with heterogeneity,
    while `polar(mean)` keeps its norm at every `N`. Comparing the server-LMO
    family against `muon` would therefore confound "what does the 1-bit uplink
    cost?" with "what does averaging orthogonal matrices cost?".
    """
    m, n = 40, 96
    r = math.sqrt(min(m, n))
    torch.manual_seed(0)
    shared = torch.randn(m, n)

    def norms(N, q):
        Ms = [shared + q * torch.randn(m, n) for _ in range(N)]
        avg = sum(muon_lmo(M, 5, torch.float32, False) for M in Ms) / N
        srv = muon_lmo(sum(Ms) / N, 5, torch.float32, False)
        return avg.norm().item() / r, srv.norm().item() / r

    one_avg, one_srv = norms(1, 1.0)
    many_avg, many_srv = norms(11, 1.0)

    assert many_avg < 0.85 * one_avg, (
        f"averaging polar factors must shorten the step: {one_avg:.3f} -> {many_avg:.3f}")
    assert abs(many_srv / one_srv - 1.0) < 0.05, (
        f"polar(mean) must be N-independent: {one_srv:.3f} -> {many_srv:.3f}")

    # ... and the shrinkage grows with heterogeneity, so a tuned eta_0 -- which is
    # one constant -- cannot absorb it.
    mild, _ = norms(11, 0.1)
    wild, _ = norms(11, 3.0)
    assert wild < mild, f"shrinkage must grow with heterogeneity: {mild:.3f} vs {wild:.3f}"


def test_communication_accounting_is_honest_about_the_round_trip():
    """The "32x reduction" holds per channel, not for every method's round trip.

    A downlink is one bit exactly when the object the server distributes is already
    +-1-valued: the majority vote (SignMuon, SignSGD), a signed LMO output
    (MuonSign), or a primal EF21 residual (EF21-MuonSign). The three methods whose
    server-side quantity is dense -- polar(.) or a scaled average of signs -- must
    broadcast a full-precision model, so their round trip stays under 2x however
    good the uplink is. Reading this off ``spec.downlink`` gets SignMuon wrong,
    which is why ``compresses_downlink`` exists.

    The alphabet is the run's, not an assumption: a raw zero rate inflates the cost
    only under ``--uplink-zeros keep``, since the default randomizes the zeros to
    +-1 and the channel is then a genuine bit whatever that rate is. Charging the
    ternary entropy to a run that transmits +-1 is what this used to do, and it put
    the run logs 0.37 bits above the paper's own table.
    """
    from federated.algorithms import communication_bits

    n_mat, n_aux, n_lay = 762_560, 2_146, 3            # CNN2

    # Even a strictly +-1 uplink costs MORE than 1 bit per parameter model-wide,
    # because the auxiliary group rides along uncompressed. On CNN2 that group is
    # 0.28% of the parameters and adds 0.09 bits; on a model with a larger head it
    # would dominate. This is the "+ epsilon" in "1 bit per parameter".
    clean = communication_bits("signmuon", n_mat, n_aux, uplink_zero_frac=0.0)
    assert 1.0 < clean["uplink_bits_per_param"] < 1.12, clean["uplink_bits_per_param"]
    assert 29.0 < clean["uplink_reduction"] < 29.9, clean["uplink_reduction"]

    # A raw zero rate does NOT move the default accounting: those zeros are
    # transmitted as +-1. This is the paper's Table "Communication per client per
    # round", and a run log has to agree with it.
    measured = communication_bits("signmuon", n_mat, n_aux, uplink_zero_frac=0.15)
    assert measured == clean, "randomized zeros must cost exactly one bit"

    # Only the legacy ternary channel pays the entropy, and only on the
    # majority-vote uplink -- the EF21 residual goes through sign_pm1 either way.
    legacy = communication_bits("signmuon", n_mat, n_aux, uplink_zero_frac=0.10,
                                uplink_zeros="keep")
    assert 1.40 < legacy["uplink_bits_per_param"] < 1.50, legacy["uplink_bits_per_param"]
    assert legacy["uplink_reduction"] < clean["uplink_reduction"]
    assert 20 < legacy["uplink_reduction"] < 25, legacy["uplink_reduction"]
    for name in ("ef21signmuon", "ef21muonusign", "ef21muonsign"):
        assert (communication_bits(name, n_mat, n_aux, 0.10, uplink_zeros="keep")
                == communication_bits(name, n_mat, n_aux, 0.10)), name

    # Dense server-side quantity: the round trip is dominated by the 32-bit
    # downlink. polar(.) for the two server-LMO methods, a scaled average of
    # signs for EF21-SignMuon.
    for name in ("muonusign", "ef21muonusign", "ef21signmuon"):
        c = communication_bits(name, n_mat, n_aux, 0.10)
        assert c["downlink_reduction"] < 1.01, name
        assert c["round_trip_reduction"] < 2.0, (name, c["round_trip_reduction"])

    # +-1-valued server-side quantity: an order of magnitude in BOTH directions.
    # signmuon and signsgd are here because the majority vote is itself the
    # broadcast -- the server never sends the model.
    for name in ("signmuon", "signsgd", "muonsign", "ef21muonsign"):
        c = communication_bits(name, n_mat, n_aux, 0.0)
        assert c["downlink_reduction"] > 20, (name, c["downlink_reduction"])
        assert c["round_trip_reduction"] > 20, (name, c["round_trip_reduction"])

    # Error feedback pays one full-precision scale per matrix layer, on each EF21
    # channel: 32*3 bits on the uplink for every EF21 method, and again on the
    # downlink for the bidirectional one. Small on CNN2, but it must be counted,
    # and it must NOT be charged to a method that has no EF21 channel.
    for name in ("ef21signmuon", "ef21muonusign", "ef21muonsign"):
        bare = communication_bits(name, n_mat, n_aux, 0.0, n_layers=0)
        with_s = communication_bits(name, n_mat, n_aux, 0.0, n_layers=n_lay)
        assert with_s["uplink_bits_per_param"] > bare["uplink_bits_per_param"], name
    down_scaled = communication_bits("ef21muonsign", n_mat, n_aux, 0.0, n_layers=n_lay)
    down_bare = communication_bits("ef21muonsign", n_mat, n_aux, 0.0, n_layers=0)
    assert down_scaled["downlink_bits_per_param"] > down_bare["downlink_bits_per_param"]
    for name in ("signmuon", "muonsign", "signsgd", "muon"):
        a = communication_bits(name, n_mat, n_aux, 0.0, n_layers=0)
        b = communication_bits(name, n_mat, n_aux, 0.0, n_layers=n_lay)
        assert a == b, f"{name} has no error-feedback channel to carry a scale"

    # The uncompressed references save nothing, in either direction.
    for name in ("muon", "muonserver", "sgd", "adam"):
        c = communication_bits(name, n_mat, n_aux, 0.0)
        assert abs(c["round_trip_reduction"] - 1.0) < 1e-9, name

    # The auxiliary group is never compressed, so it caps the achievable ratio.
    huge_head = communication_bits("muonsign", n_mat, n_mat, 0.0)
    assert huge_head["round_trip_reduction"] < 2.0, \
        "a model that is half uncompressed head cannot reach 32x"


def test_federated_val_split_is_held_out_before_partitioning():
    """No client may hold a validation image, and the split must match centrally."""
    import numpy as np

    from centralized.data import _split_indices
    from federated.data import partition_indices

    y_train = np.arange(1000) % 10
    y_test = np.arange(200) % 10
    train_pool, val_idx = _split_indices(1000, 100, val_seed=12345)

    tr, te = partition_indices(y_train, y_test, "homo", 5,
                               rng=np.random.default_rng(0), train_pool=train_pool)
    held = set(val_idx.tolist())
    for j, idx in tr.items():
        assert not (set(idx.tolist()) & held), f"client {j} holds validation images"
    assert sum(len(v) for v in tr.values()) == 900
    assert sum(len(v) for v in te.values()) == 200

    # Dirichlet honours the pool too, and every client keeps some data.
    tr2, _ = partition_indices(y_train, y_test, "noniid-labeldir", 5, beta=0.5,
                               rng=np.random.default_rng(0), train_pool=train_pool)
    assert not (set(np.concatenate(list(tr2.values())).tolist()) & held)
    assert min(len(v) for v in tr2.values()) >= 10


def test_gpu_shard_augmentation_matches_torchvision():
    """The device-resident augmentation must be the torchvision one.

    ``RandomCrop(32, padding=4)`` + ``RandomHorizontalFlip()`` is reimplemented as
    tensor ops so the dataset can live on the GPU. The randomness differs (a
    different generator), so the check is distributional plus an exact check of
    the two deterministic corners: zero offset with no flip is the identity, and
    the normalization matches.
    """
    from federated.data import GpuEvalSet, GpuShard

    torch.manual_seed(0)
    x = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.uint8)
    y = torch.randint(0, 10, (64,))

    ev = GpuEvalSet(x, y, batch_size=32, dataset="cifar10")
    batches = list(ev)
    assert len(batches) == 2 and batches[0][0].shape == (32, 3, 32, 32)
    # The eval path is exactly ToTensor + Normalize.
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
    std = torch.tensor((0.2023, 0.1994, 0.2010)).view(1, 3, 1, 1)
    want = (x[:32].float() / 255.0 - mean) / std
    assert torch.allclose(batches[0][0], want, atol=1e-6)

    sh = GpuShard(x, y, batch_size=16, dataset="cifar10", seed=0)
    xb, yb = sh.next_batch()
    assert xb.shape == (16, 3, 32, 32) and yb.shape == (16,)
    # Cropping from a zero-padded image can only introduce zero-valued *pixels*,
    # which normalize to -mean/std; nothing else may appear.
    floor = float((-mean / std).min())
    assert xb.min().item() >= floor - 1e-4
    # A shard sweeps its whole content before repeating.
    seen = {int(i) for _ in range(4) for i in sh.next_batch()[1]}
    assert len(seen) <= 10, "labels are only 0-9, so this just checks it runs"

    # Augmentation off (MNIST has padding 0) is the identity plus normalization.
    xm = torch.randint(0, 256, (8, 1, 28, 28), dtype=torch.uint8)
    shm = GpuShard(xm, y[:8], batch_size=8, dataset="mnist", seed=0)
    xbm, _ = shm.next_batch()
    assert xbm.shape == (8, 1, 28, 28)


def test_gpu_crop_and_flip_match_torchvision_distributionally():
    """The device-side crop must have torchvision's offset distribution.

    An off-by-one in the padding would still produce plausible-looking images, so
    shape checks cannot catch it. This compares the per-row probability that an
    output row came from the zero padding -- which is exactly what the offset
    distribution determines -- against ``RandomCrop(32, padding=4)`` itself, over
    2000 draws each.
    """
    from torchvision import transforms

    from federated.data import GpuShard

    n, size, pad = 2000, 32, 4
    ones = torch.full((n, 3, size, size), 255, dtype=torch.uint8)

    # torchvision: how often is output row i entirely padding?
    crop = transforms.RandomCrop(size, padding=pad)
    torch.manual_seed(0)
    ref = torch.zeros(size)
    for i in range(n):
        out = crop(ones[i].float())
        ref += (out[0].sum(dim=1) == 0).float()
    ref /= n

    # ours: same statistic, straight off the shard
    sh = GpuShard(ones, torch.zeros(n, dtype=torch.long), batch_size=n,
                  dataset="cifar10", seed=0)
    sh.mean = torch.zeros(1, 3, 1, 1)      # strip normalization for the comparison
    sh.std = torch.ones(1, 3, 1, 1)
    xb, _ = sh.next_batch()
    got = (xb[:, 0].sum(dim=2) == 0).float().mean(dim=0)

    assert torch.allclose(ref, got, atol=0.04), \
        f"padding-row profile differs:\n  torchvision {ref[:6].tolist()}\n  ours {got[:6].tolist()}"
    # Both must actually pad sometimes, or the test would pass on two no-ops.
    assert ref[0] > 0.3 and ref[size // 2] == 0.0

    # And the flip is a fair coin applied per sample. Probed with the crop
    # neutralized (pad = 0 makes the offsets degenerate), so a shifted window
    # cannot be mistaken for a flip.
    asym = torch.zeros(64, 3, 8, 8, dtype=torch.uint8)
    asym[:, :, :, 0] = 255
    sf = GpuShard(asym, torch.zeros(64, dtype=torch.long), batch_size=64,
                  dataset="cifar10", seed=3)
    sf.pad = 0
    assert sf.augment, "cifar10 shards must augment"
    flipped = 0
    for _ in range(20):
        xb, _ = sf.next_batch()
        flipped += int((xb[:, 0, 0, -1] > xb[:, 0, 0, 0]).sum())
    assert 0.4 < flipped / (20 * 64) < 0.6, flipped / (20 * 64)

    # MNIST is deliberately NOT augmented -- `get_mnist_transform` never was.
    plain = GpuShard(torch.zeros(8, 1, 28, 28, dtype=torch.uint8),
                     torch.zeros(8, dtype=torch.long), batch_size=8,
                     dataset="mnist", seed=0)
    assert not plain.augment


def test_gpu_shard_sweeps_its_whole_shard():
    """Every sample must be visited once per epoch, in a seed-stable order."""
    from federated.data import GpuShard

    x = torch.arange(20 * 1 * 4 * 4, dtype=torch.uint8).reshape(20, 1, 4, 4)
    y = torch.arange(20)
    a = GpuShard(x, y, batch_size=5, dataset="mnist", seed=7, augment=False)
    b = GpuShard(x, y, batch_size=5, dataset="mnist", seed=7, augment=False)
    seen = []
    for _ in range(4):
        ya = a.next_batch()[1]
        assert torch.equal(ya, b.next_batch()[1]), "same seed must give same order"
        seen += ya.tolist()
    assert sorted(seen) == list(range(20)), "one epoch must cover the shard exactly"


def test_federated_anchors_transport_between_rules():
    """A rule changes what eta_0 means; the anchor has to move with it.

    Under ``legacy`` the transported anchor is the published value itself, and
    under ``unit-gain`` the LMO family is untouched (its multiplier is the same
    aspect factor) while the sign family moves by the geometric-mean fan-in.
    """
    from federated.tune import LEGACY_ANCHORS, anchor_for

    shapes = [("conv1", (64, 3, 5, 5)), ("conv2", (128, 64, 5, 5)),
              ("fc1", (120, 4608))]
    for m, want in LEGACY_ANCHORS.items():
        assert abs(anchor_for(m, "legacy", shapes) - want) < 1e-12, m

    # LMO family: unchanged, because lambda is the aspect factor under both rules.
    for m in ("muon", "muonusign", "ef21muonusign", "ef21muonsign", "ef21signmuon"):
        assert abs(anchor_for(m, "unit-gain", shapes) - LEGACY_ANCHORS[m]) < 1e-12, m

    # Sign family: multiplied by the geometric mean of sqrt(fan_in).
    boost = math.exp(sum(math.log(math.sqrt(n)) for n in (75, 1600, 4608)) / 3)
    for m in ("signmuon", "muonsign", "signsgd"):
        assert abs(anchor_for(m, "unit-gain", shapes) / LEGACY_ANCHORS[m] - boost) < 1e-9, m
    assert 25.0 < boost < 30.0, boost

    # The baselines have no family, so no rule moves them.
    for m in ("sgd", "adam"):
        for rule in ("legacy", "unit-gain", "mup"):
            assert anchor_for(m, rule, shapes) == LEGACY_ANCHORS[m]
    assert anchor_for("adam", "unit-gain", shapes, scale_baselines=True) > \
        LEGACY_ANCHORS["adam"]


# --------------------------------------------------------------------------
# Synthetic benchmark
# --------------------------------------------------------------------------


def test_quadratic_constants_are_exact():
    """``L``, ``sigma`` and the gradient are closed forms, not estimates.

    The Hessian of ``F(X) = 1/2 <X-C, A(X-C)B>`` is ``B (x) A``, so its extreme
    eigenvalues are the extreme products of the two spectra. The benchmark reads
    ``L`` off that identity and never estimates it; if the identity broke, every
    ``L``-relative number the sweeps print would be silently wrong.
    """
    from synthetic.benchmark import Quadratic

    prob = Quadratic(9, 7, torch.device("cpu"), seed=3, spectrum="uniform")

    la = torch.linalg.eigvalsh(prob.A)
    lb = torch.linalg.eigvalsh(prob.B)
    prod = torch.outer(la, lb)
    # float32 throughout, so compare relatively.
    assert abs(prob.L / float(prod.max()) - 1) < 1e-5, prob.L
    assert abs(prob.sigma / float(prod.min()) - 1) < 1e-4, prob.sigma

    # The closed-form gradient must agree with autograd, since the benchmark
    # replaced the autograd path with it.
    X = torch.randn(9, 7, dtype=torch.double, requires_grad=True)
    A, B = prob.A.double(), prob.B.double()
    f = 0.5 * torch.sum(X * (A @ X @ B))
    f.backward()
    assert torch.allclose(X.grad, A @ X.detach() @ B, atol=1e-10)

    f_val, g = prob.value_and_grad(X.detach().float())
    assert abs(f_val - f.item()) < 1e-4
    assert torch.allclose(g.double(), X.grad, atol=1e-4)


def test_logspace_spectrum_hits_the_requested_condition_number():
    from synthetic.benchmark import Quadratic

    for kappa in (1e2, 1e4, 1e6):
        prob = Quadratic(16, 16, torch.device("cpu"), seed=0,
                         spectrum="logspace", kappa=kappa)
        assert abs(prob.L - 1.0) < 1e-5, prob.L
        assert abs(prob.kappa / kappa - 1.0) < 1e-3, (prob.kappa, kappa)


def test_lr_grid_parsing_linear_and_log():
    from synthetic.benchmark import parse_lr_grid

    linear = parse_lr_grid("1e-4:1e-3:1e-4")     # Table 3's form
    assert len(linear) == 10
    assert abs(linear[0] - 1e-4) < 1e-15 and abs(linear[-1] - 1e-3) < 1e-15
    assert 2e-4 in linear                        # the paper's SignMuon optimum

    log = parse_lr_grid("1e-4:1e-1:x6")          # 6 points per decade, 3 decades
    assert len(log) == 19
    assert abs(log[0] - 1e-4) < 1e-15 and abs(log[-1] - 1e-1) < 1e-12
    ratios = [log[i + 1] / log[i] for i in range(len(log) - 1)]
    assert max(ratios) - min(ratios) < 1e-9      # geometric


def test_step_norms_match_the_lr_scaling_module():
    """``||s||_F`` is what turns a floor slope into a claim about ``rho``."""
    from synthetic.benchmark import step_norm

    m, n = 12, 20
    for method in ("signmuon", "muonsign", "signsgd"):
        assert abs(step_norm(method, m, n) - math.sqrt(m * n)) < 1e-12
    for method in ("muon", "muonusign", "ef21muonusign", "ef21muonsign"):
        assert abs(step_norm(method, m, n) - math.sqrt(min(m, n))) < 1e-12

    # The sign step realizes its norm exactly -- a +-1 matrix has no slack.
    G = torch.randn(m, n)
    assert abs(float(torch.sign(G).norm()) - math.sqrt(m * n)) < 1e-4

    # The exact LMO realizes sqrt(r) exactly ...
    U, _, Vt = torch.linalg.svd(G, full_matrices=False)
    assert abs(float((U @ Vt).norm()) - math.sqrt(min(m, n))) < 1e-4


def test_every_lr_grid_follows_its_method_step_norm():
    """The search window has to be keyed off ``||s||_F``, not written by hand.

    A sign step has length ``eta*sqrt(mn)`` and an LMO step ``eta*sqrt(r)``, so
    at ``100x100`` their optimal ``eta`` differ by a factor of ten and they need
    different windows. Enumerated per method, the two drifted: ``muonusign`` and
    ``ef21signmuon`` take an LMO-length step but had been given the sign
    window, which censored ``muonusign``'s optimum at the upper edge and left
    ``ef21signmuon``'s smallest floor point unsettled after 60k iterations.
    """
    from synthetic.benchmark import (DEFAULT_LR_GRIDS, DEFAULT_METHODS,
                                     LR_GRID_FAMILIES, parse_lr_grid, step_norm)

    m = n = 100
    assert set(DEFAULT_LR_GRIDS) == set(DEFAULT_METHODS)
    for family, members in LR_GRID_FAMILIES.items():
        specs = {DEFAULT_LR_GRIDS[method] for method in members}
        assert len(specs) == 1, (family, specs)
        norms = {round(step_norm(method, m, n), 9) for method in members}
        assert len(norms) == 1, (family, norms)

    # ... and the two windows are a decade apart at both ends, as their step
    # norms are.
    sign = parse_lr_grid(DEFAULT_LR_GRIDS["signmuon"])
    lmo = parse_lr_grid(DEFAULT_LR_GRIDS["muon"])
    ratio = step_norm("signmuon", m, n) / step_norm("muon", m, n)
    assert abs(lmo[0] / sign[0] - ratio) < 1e-9
    assert abs(lmo[-1] / sign[-1] - ratio) < 1e-9

    # Neither window stops below the stability edge that ``--mode stability``
    # measures for its family at this size. A stable eta the grid cannot reach
    # is a censored optimum waiting to happen, which is what a window stopping
    # at 0.01 did to MuonUSign. Points above the edge are nearly free: they
    # diverge in a few steps and the runner retires them.
    SIGN_EDGE, LMO_EDGE = 0.1343, 1.894        # largest eta_max in each family
    assert sign[-1] >= SIGN_EDGE, (sign[-1], SIGN_EDGE)
    assert lmo[-1] >= LMO_EDGE, (lmo[-1], LMO_EDGE)

    # Both still clear the values that were censored in the pre-fix run.
    assert sign[-1] > 0.01 and lmo[-1] > 0.1


def test_newton_schulz_step_is_measurably_shorter_than_the_exact_lmo():
    """Five NS steps leave the singular values in a band around 1, not at 1.

    So every LMO-family run takes a step 6-22% shorter than the ``sqrt(r)`` its
    theory assumes, which rescales the effective learning rate by the same
    factor. ``step_norm`` documents the gap; this pins it down so it cannot
    drift unnoticed if the coefficients are ever retuned.
    """
    torch.manual_seed(0)
    ratios = {}
    for m, n in ((64, 64), (100, 100), (64, 576), (500, 500)):
        r = math.sqrt(min(m, n))
        got = sum(float(muon_lmo(torch.randn(m, n), dtype=None,
                                 scale_aspect=False).norm())
                  for _ in range(3)) / 3
        ratios[(m, n)] = got / r

    assert 0.75 < min(ratios.values()) < 0.95, ratios
    assert max(ratios.values()) < 0.99, ratios          # never reaches sqrt(r)
    assert ratios[(64, 576)] < ratios[(500, 500)], ratios   # worse when oblong


def _reference_trajectory(method: str, grads, lr, mu, m, n):
    """The paper's algorithm boxes, transcribed literally and independently.

    Deliberately NOT written in terms of ``common.optimizers``: this is the
    reference the implementation is checked against, so sharing code with it
    would defeat the point. ``grads`` is a fixed sequence, independent of the
    iterate, so the recursion is a pure function of the box.

    ``L`` is the LMO with ``scale_aspect=False`` -- the aspect factor is a step
    *rescaling* that the drivers apply outside the oracle, and it is not part of
    any box (see Algorithm ``alg:muon_lmo``, which has no such factor).
    """
    def L(Y):
        return muon_lmo(Y, ns_steps=5, dtype=torch.float32, scale_aspect=False)

    X = torch.zeros(m, n)
    M = torch.zeros(m, n)
    d_est = torch.zeros(m, n)      # EF21 estimator of the LMO output
    g_est = torch.zeros(m, n)      # EF21 estimator of the momentum
    W = torch.zeros(m, n)          # broadcast model, EF21-MuonSign only
    for G in grads:
        M = mu * M + (1.0 - mu) * G                       # EMA momentum
        if method == "muon":
            d = L(M)
        elif method == "signsgd":
            d = torch.sign(M)
        elif method == "signmuon":                        # sign AFTER the LMO
            d = torch.sign(L(M))
        elif method == "muonusign":                       # sign BEFORE the LMO
            d = L(torch.sign(M))
        elif method == "muonsign":                        # sign on BOTH sides
            d = torch.sign(L(torch.sign(M)))
        elif method == "ef21signmuon":                    # EF21 on the LMO OUTPUT
            delta = L(M) - d_est
            d_est = d_est + delta.abs().mean() * torch.sign(delta)
            d = d_est
        elif method in ("ef21muonusign", "ef21muonsign"):  # EF21 BEFORE the LMO
            delta = M - g_est
            g_est = g_est + delta.abs().mean() * torch.sign(delta)
            d = L(g_est)
        else:
            raise AssertionError(method)
        X = X - lr * d
        if method == "ef21muonsign":                      # downlink EF21-P
            shift = X - W
            W = W + shift.abs().mean() * torch.sign(shift)
    return X


def test_each_direction_is_the_documented_formula():
    """Pin every step rule to the paper's box, not just to the other driver.

    The federated<->centralized equality tests pin the two *implementations* to
    each other, so a bug applied consistently to both is invisible to them:
    swapping SignMuon with MuonSign, or swapping EF21-before-the-oracle with
    EF21-after-the-oracle -- which is exactly the distinction Theorem 4 turns on
    -- leaves all of them passing. This test is the one that would fail.
    """
    m, n = 6, 5
    lr, mu, steps = 0.05, 0.8, 4
    torch.manual_seed(0)
    # A FIXED gradient sequence: iterate-independent, so the reference above is a
    # closed-form recursion rather than a re-implementation of the training loop.
    grads = [torch.randn(m, n) for _ in range(steps)]

    trajectories = {}
    for method, cls in CENTRAL_CLASSES.items():
        p = nn.Parameter(torch.zeros(m, n))
        opt = cls([p], lr=lr, momentum=mu, weight_decay=0.0,
                  lmo_dtype=torch.float32, scale_aspect=False)
        for G in grads:
            p.grad = G.clone()
            opt.step()
        if hasattr(opt, "restore_exact"):
            opt.restore_exact()            # compare on X, the iterate the theory bounds
        want = _reference_trajectory(method, grads, lr, mu, m, n)
        assert torch.allclose(p.detach(), want, atol=1e-5), (
            f"{method}: implementation departs from its algorithm box, "
            f"max|diff| = {(p.detach() - want).abs().max():.3e}")
        trajectories[method] = p.detach().clone()

    # Non-vacuity: the rules must be mutually distinguishable, or a consistent
    # swap would satisfy the assertions above by accident. The one legitimate
    # coincidence is EF21-MuonUSign vs EF21-MuonSign, whose exact models agree
    # when the gradient does not depend on the iterate -- they differ only in the
    # downlink, which is what `test_ef21muonsign_separates_exact_and_broadcast_models`
    # covers.
    allowed = {frozenset({"ef21muonusign", "ef21muonsign"})}
    names = list(trajectories)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if frozenset({a, b}) in allowed:
                continue
            assert not torch.allclose(trajectories[a], trajectories[b], atol=1e-5), (
                f"{a} and {b} produce the same trajectory, so this test cannot "
                f"tell them apart -- pick a harder gradient sequence")


def test_the_exact_svd_lmo_truncates_to_the_rank():
    """The counterexample oracle must rank-truncate; ``sign(G)`` is often low-rank.

    ``U @ Vt`` from a full SVD appends an arbitrary orthonormal completion of the
    null space, which is non-unique on rank-deficient input -- and would silently
    change the answer on exactly the instances the theorems are stated about.
    """
    import numpy as np
    from counterexamples.optimizers import muon_lmo as exact_lmo

    R = np.outer(np.ones(4), np.array([1.0, -1.0, 1.0, -1.0]))   # rank 1
    D = exact_lmo(R)
    assert np.linalg.matrix_rank(D) == 1, D
    # sqrt(r) = 1, not sqrt(min(m, n)) = 2, which is what no truncation would give
    assert abs(np.linalg.norm(D) - 1.0) < 1e-12, np.linalg.norm(D)
    # and <Y, D> is the nuclear norm, the defining property of the LMO
    assert abs(float(np.sum(R * D)) - np.linalg.norm(R, "nuc")) < 1e-10
    # scale invariance survives the relative rank threshold
    assert np.allclose(exact_lmo(1e-8 * R), D, atol=1e-12)
    assert np.allclose(exact_lmo(3.0 * R), D, atol=1e-12)


def test_the_counterexample_package_reproduces_the_theorem_constants():
    """The published Theorem 1-3 numbers come from ``counterexamples/``, not here.

    The torch tests above re-derive the same inner products with their own local
    oracle. Nothing tied that to the module that actually prints the numbers in
    the paper, so the two could drift apart silently.
    """
    import numpy as np
    from counterexamples.optimizers import muon_lmo as exact_lmo
    from counterexamples.problems import (muonsign_counterexample,
                                          signmuon_counterexample)

    G1, info1 = signmuon_counterexample(sigma1=1000.0)
    assert np.allclose(exact_lmo(G1), info1["O"], atol=1e-12)     # polar(G) == O
    assert abs(float(np.sum(G1 * np.sign(exact_lmo(G1)))) + 42468 / 103) < 1e-9

    G2, info2 = muonsign_counterexample(eps=1.0, M=100.0)
    assert np.array_equal(np.sign(G2), info2["S"])
    D2 = exact_lmo(info2["S"])
    assert abs(float(np.sum(G2 * D2)) - (-13.8879)) < 1e-3        # Theorem 2
    assert abs(float(np.sum(G2 * np.sign(D2))) + 76.0) < 1e-9     # Theorem 3, exact
    assert float(np.sum(G2 * exact_lmo(G2))) > 0                  # Muon descends


def test_the_counterexample_constants_do_not_depend_on_the_sign_convention():
    """No matrix the theorems sign has a zero entry, so ``sign(0)`` never arises.

    ``counterexamples/`` follows the paper and maps an exact zero to a random
    ``+-1``, which makes every sign channel a strict bit but would make the
    printed constants seed-dependent if any of them were reached through a tie.
    None is: the instances and their oracle outputs are entrywise nonzero, so the
    randomized and the ternary convention agree on all of Theorems 1-3. Pinned
    because the alternative -- discovering it from a constant that moves between
    runs -- is expensive.
    """
    import numpy as np
    from counterexamples.optimizers import muon_lmo as exact_lmo, sign_pm1
    from counterexamples.problems import (muonsign_counterexample,
                                          signmuon_counterexample)

    G1, info1 = signmuon_counterexample(sigma1=1000.0)
    G2, info2 = muonsign_counterexample(eps=1.0, M=100.0)
    for name, M in (("O", info1["O"]), ("polar(G)", exact_lmo(G1)),
                    ("S", info2["S"]), ("polar(S)", exact_lmo(info2["S"]))):
        assert np.all(M != 0.0), f"{name} has a zero entry"

    # ... and the tie-break itself never emits a zero, whatever it is handed.
    rng = np.random.default_rng(0)
    s = sign_pm1(np.zeros((8, 8)), rng)
    assert set(np.unique(s)) == {-1.0, 1.0}, np.unique(s)
    assert np.array_equal(sign_pm1(G2, rng), np.sign(G2))         # no ties -> identical


def test_the_ef21_counterexample_rate_and_nonconvexity_witness():
    """Pins the two constants the Theorem 4 appendix quotes against the code.

    (i) On the assembled ``f`` the loss gains exactly ``49/240`` per period
    from ``t = 3`` on (the published per-step rate ``49/480``), for every
    momentum and both variants.  (ii) The nonconvexity remark claims the run
    itself is incompatible with convexity, through the exact monotonicity
    violation ``<grad f(X6) - grad f(X3), X6 - X3> = -9A/50``.  Pinned so the
    remark's constant cannot drift from the construction that prints it.
    """
    import numpy as np
    from counterexamples.optimizers import EF21SignMuon
    from counterexamples.problems import ef21_signmuon_counterexample

    for mu, nesterov in ((0.0, False), (0.5, False), (0.5, True), (0.9, True)):
        grad_fn, loss_fn, shape, info = ef21_signmuon_counterexample(
            mu=mu, nesterov=nesterov)
        opt = EF21SignMuon(shape, eta=1.0, mu=mu, nesterov=nesterov)
        Xs = [opt.X.copy()]
        for _ in range(9):
            opt.step(grad_fn(opt.grad_point()))
            Xs.append(opt.X.copy())
        for t in (3, 4, 5, 6, 7):                       # exact tail ascent
            assert abs((loss_fn(Xs[t + 2]) - loss_fn(Xs[t])) - 49 / 240) < 1e-9
        ip = float(np.vdot(grad_fn(Xs[6]) - grad_fn(Xs[3]), Xs[6] - Xs[3]))
        assert abs(ip + 9 * info["A"] / 50) < 1e-9      # = -9A/50 exactly
        assert ip < 0                                   # no convex f does this


def test_capture_direction_records_the_step_actually_taken():
    """``alignment`` mode is only meaningful if ``d_t`` is the realized step."""
    torch.manual_seed(0)
    for name, cls in CENTRAL_CLASSES.items():
        p = nn.Parameter(torch.randn(6, 5))
        opt = cls([p], lr=0.1, momentum=0.0, lmo_dtype=None)
        opt.capture_direction = True
        p.grad = torch.randn(6, 5)

        # EF21MuonSign advances the exact model X rather than p.data, and
        # creates it lazily inside the first step -- but it is initialized from
        # p.data, so before the first step the two agree.
        before = p.data.clone()
        opt.step()
        after = (opt.state[p]["exact_model"] if name == "ef21muonsign" else p.data)

        d = opt.state[p]["last_direction"]
        assert torch.allclose(before - after, 0.1 * d, atol=1e-5), name

    # And the hook stays off unless asked for.
    p = nn.Parameter(torch.randn(6, 5))
    opt = SignMuon([p], lr=0.1)
    p.grad = torch.randn(6, 5)
    opt.step()
    assert "last_direction" not in opt.state[p]


def test_batched_runner_reproduces_the_sequential_one():
    """The load-bearing test for ``synthetic.batched``.

    The batched runner advances a whole ``(lr, momentum, schedule)`` grid as one
    ``[B, m, n]`` trajectory, which is the only reason the sweeps finish in
    minutes rather than hours -- but it is a second implementation of all ten
    update rules, so nothing stops it drifting from ``run_one`` except this.
    ``run_one`` is the reference: it drives the real ``common.optimizers``
    classes and ``torch.optim``, so agreement here ties the fast path back to
    the same code the deep-learning experiments use.

    The grid deliberately mixes all three schedules, a shorter per-config budget
    and a wildly oversized step, and runs with ``stop_at_target`` both off and
    on, because those are the paths where the two drivers' *control flow*
    differs rather than their arithmetic: ``run_one`` breaks out of its loop,
    the batched runner masks the slice and drops it at the next compaction.

    **Why the horizon is short.** A batched matmul reduces in a different order
    from a single one, so the two runners see gradients differing at the last
    float32 bit. Measured on this problem that stays at ``1e-7`` for ~20 steps
    and then amplifies, because ``sign`` is discontinuous: an entry of ``M`` or
    of ``polar(M)`` within rounding of zero flips and the step changes by
    ``O(1)``. That is a property of the methods -- the one this paper is about --
    not of batching, and any change of GPU or BLAS does the same. So this pins
    the update rules, where the two must agree exactly, and leaves the
    long-horizon behaviour to the aggregates, which are stable: measured at
    ``100x100`` over 800 steps the two agree to ``3e-3`` on ``best_f`` and
    ``1e-5`` on the alignment statistics.

    Nothing here is asserted to be bit-identical across machines, because it is
    not: a different BLAS reduces in a different order. What is asserted is that
    the two *trajectories* agree step by step to a tolerance scaled by the
    trajectory, which is what a wrong update rule would violate immediately.
    """
    from synthetic.benchmark import Quadratic, run_grid, DEFAULT_METHODS

    prob = Quadratic(9, 7, torch.device("cpu"), seed=1337, spectrum="uniform")
    configs = [
        {"lr": 3e-3, "momentum": 0.0, "schedule": "const"},
        {"lr": 1e-2, "momentum": 0.9, "schedule": "sqrt"},
        {"lr": 5e-2, "momentum": 0.5, "schedule": "linear"},
        # Short per-config budget. Its step is deliberately modest: at lr = 2e-1
        # Muon sits near its stability edge on this instance and amplifies a
        # last-bit difference by two orders of magnitude in 9 steps, which would
        # be measuring the edge rather than the runners.
        {"lr": 2e-2, "momentum": 0.0, "schedule": "const", "max_iters": 9},
        # Wild step and heavy momentum: the only config that trips the
        # divergence exit (SGD does; a normalized step oscillates at a large
        # radius rather than blowing up). Its *values* are not compared -- see
        # UNSTABLE below -- only its control flow, which is the point of it.
        {"lr": 5e+1, "momentum": 0.95, "schedule": "const", "max_iters": 8},
    ]
    # Configs whose trajectory is compared only through its control flow. On an
    # oversized step F swings over three orders of magnitude between
    # consecutive iterates, so a float32 last-bit difference between the two
    # runners becomes O(1) within a few steps -- for EF21-SignMuon by step 2.
    # Comparing values there would be measuring that amplification, which is a
    # property of the method (and Theorem 4's subject), not of the runner.
    UNSTABLE = {4}
    # Tolerances are scaled by the largest value on the trajectory, not by the
    # value being compared. Rounding enters at ``ulp * max|F|`` and stays there;
    # a step that drops F from 1225 to 0.031 -- which the oversized config does
    # -- turns three ulp at the top into 1e-4 *relative* at the bottom, and a
    # pointwise relative tolerance would be measuring that cancellation rather
    # than any disagreement between the runners.
    floats = {"best_f": "loss_history", "final_loss": "loss_history",
              "best_gnorm": "grad_norm_history"}

    def close(x, y, series):
        # 1e-3 of the trajectory's scale. The measured gap between two float32
        # reduction orders is ~1e-7 here, so this leaves four orders of
        # headroom for a different BLAS -- and a genuinely wrong update rule
        # differs by O(1), three orders the other way.
        scale = max([abs(v) for v in series if v == v] + [abs(x), 1e-12])
        return abs(x - y) <= 1e-3 * scale

    for method in DEFAULT_METHODS:
        for stop_at_target in (False, True):
            kw = dict(target_loss=1e-2, max_iters=20, init_seed=42,
                      lmo_dtype="float32", keep_history=True,
                      capture_alignment=True, stop_at_target=stop_at_target)
            torch.manual_seed(0)
            seq = run_grid(method, [prob], configs, "sequential", **kw)[0]
            torch.manual_seed(0)
            # A compaction period that divides none of the budgets, so slices
            # are carried dead for a while and then dropped mid-run.
            bat = run_grid(method, [prob], configs, "batched",
                           compact_every=7, **kw)[0]

            for i, (a, b) in enumerate(zip(seq, bat)):
                where = f"{method}, config {i}, stop={stop_at_target}"
                assert a["diverged"] == b["diverged"], (
                    f"{where}: diverged {a['diverged']} != {b['diverged']}")
                # One step of slack on the threshold crossing, and none on the
                # trajectory itself. `iters_to_converge` is the first t with
                # F <= 1e-2, and F crosses that threshold *gradually* here: of
                # the ten crossings in this grid, four land within 5% of the
                # target and one within 0.5%. A different BLAS moving the
                # trajectory by a fraction of a percent therefore moves the
                # crossing a whole step -- which says nothing about whether the
                # two runners agree, and the elementwise comparison below is
                # what actually answers that.
                assert abs(a["iters_to_converge"] - b["iters_to_converge"]) <= 1, (
                    f"{where}: iters_to_converge {a['iters_to_converge']} vs "
                    f"{b['iters_to_converge']}")
                assert abs(len(a["loss_history"])
                           - len(b["loss_history"])) <= 1, (
                    f"{where}: recorded {len(a['loss_history'])} steps vs "
                    f"{len(b['loss_history'])} -- the two drivers disagree "
                    f"about when a run stops")
                if i in UNSTABLE:
                    continue
                # The real check: the two trajectories, step by step. A wrong
                # momentum form or a swapped rule shows up here at O(1) on the
                # first step, long before any aggregate could hide it.
                for series in ("loss_history", "grad_norm_history"):
                    for t, (x, y) in enumerate(zip(a[series], b[series])):
                        assert close(x, y, a[series]), (
                            f"{where}: {series}[{t}] {x!r} != {y!r}")
                for key, series in floats.items():
                    if a[key] != a[key]:                      # NaN
                        assert b[key] != b[key], f"{where}: {key}"
                        continue
                    assert close(a[key], b[key], a[series]), (
                        f"{where}: {key} {a[key]!r} != {b[key]!r}")
                assert ("rho" in a) == ("rho" in b), where
                if "rho" in a:
                    # rho lives in [-1, 1], so an absolute 1e-3 is strict; the
                    # oversized config reaches 7e-5 on its own.
                    for key, va in a["rho"].items():
                        assert abs(va - b["rho"][key]) < 1e-3, (
                            f"{where}: rho[{key}] {va!r} != {b['rho'][key]!r}")


def test_batched_ef21_reduces_per_slice_not_over_the_batch():
    """``alpha_t = mean|Delta_t|`` is per configuration, not per batch.

    The failure this guards against is silent: a global ``.mean()`` over the
    stacked residual broadcasts fine and returns plausible numbers, but it hands
    every run in the batch the same compressor magnitude, so a grid point's
    trajectory would depend on which other learning rates happened to be swept
    alongside it. Two batches containing the same configuration must agree.
    """
    from synthetic.batched import ef21_step
    from common.optimizers import _EF21Mixin

    torch.manual_seed(0)
    targets = torch.randn(3, 5, 4)
    targets[1] *= 1000.0                     # a slice that would dominate a global mean

    est = torch.zeros_like(targets)
    ef21_step(targets, est)

    for i in range(3):
        state = {}
        want = _EF21Mixin._ef21_update(targets[i], state, "e")
        assert torch.allclose(est[i], want, atol=1e-6), (
            f"slice {i} differs from the shared single-matrix implementation")


def test_plateau_detector_separates_settled_from_still_descending():
    from synthetic.benchmark import _plateau

    settled = [1.0] * 50 + [0.5] * 50
    level, ok = _plateau(settled)
    assert ok and abs(level - 0.5) < 1e-12

    descending = [2.0 ** -(i / 10.0) for i in range(200)]
    _, ok = _plateau(descending)
    assert not ok


def test_loglog_fit_recovers_a_known_exponent():
    from synthetic.benchmark import loglog_fit

    xs = [1.0, 2.0, 4.0, 8.0, 16.0]
    slope, r2 = loglog_fit(xs, [3.0 * x ** -0.5 for x in xs])
    assert abs(slope + 0.5) < 1e-9 and abs(r2 - 1.0) < 1e-9
    assert math.isnan(loglog_fit([1.0], [1.0])[0])


def test_sign_step_floor_scales_linearly_in_eta():
    """The measurement the appendix's floor table rests on, in miniature.

    A ``+-1`` step has fixed length ``eta*sqrt(mn)``, so a constant step size
    cannot converge: the gradient norm settles at a level proportional to
    ``eta``. Halving ``eta`` must halve that level.
    """
    from synthetic.benchmark import Quadratic, run_one, _plateau

    prob = Quadratic(20, 20, torch.device("cpu"), seed=11,
                     spectrum="logspace", kappa=1e3)
    levels = []
    for lr in (4e-3, 2e-3, 1e-3):
        r = run_one("signsgd", {"lr": lr, "momentum": 0.0}, prob,
                    target_loss=0.0, max_iters=int(6000 * 4e-3 / lr),
                    lmo_dtype="float32", keep_history=True)
        level, settled = _plateau(r["grad_norm_history"])
        assert settled, (lr, level)
        levels.append(level)

    for a, b in zip(levels, levels[1:]):
        assert abs(a / b - 2.0) < 0.15, levels


# --------------------------------------------------------------------------
# Double-blind supplement
# --------------------------------------------------------------------------


def test_export_bundles_carry_no_absolute_path():
    """`repo_relative` must strip the writing machine's prefix, and nothing else.

    An export bundle travels with an anonymous submission, and a run tree records
    where it ran: `state.json` stores a log and a metrics path per job, the
    exporters stamp their source tree into a manifest. On our boxes those begin
    with a home directory, so they carry a name. `anonymize.py` excludes
    `results/` from the anonymous bundle, but a bundle copied by hand bypasses
    that, so the paths are rewritten in the files themselves as well.

    The second half of this test is the scar from getting it wrong: normalizing
    separators over the whole string instead of over the matched path turns every
    `\\begin` in a generated `hardware.tex` into `/begin`, and every JSON escape
    with it. Both regressions are silent -- the file still parses.
    """
    import re
    from pathlib import Path

    from common.paths import repo_relative, scrub

    for src, want in (
        ("/home/someone/SignMuon/code/results/centralized", "results/centralized"),
        ("/home/u/S/code/results/tuning_logs/g.log", "results/tuning_logs/g.log"),
        ("D:\\w\\SignMuon\\code\\results\\run\\m.json", "results/run/m.json"),
        ("/x/y/code/results_old/synthetic_20x20/S.md",
         "results_old/synthetic_20x20/S.md"),
        ("* source: `/home/u/S/code/results/centralized`",
         "* source: `results/centralized`"),
        # already relative, and absolute-but-unrelated: both untouched
        ("results/already/relative", "results/already/relative"),
        ("/opt/data/cifar10", "/opt/data/cifar10"),
        # the regression
        ("\\begin{tabular}{@{}llr@{}}", "\\begin{tabular}{@{}llr@{}}"),
        ("GPU & PyTorch / CUDA & Runs \\\\", "GPU & PyTorch / CUDA & Runs \\\\"),
        ('{"a": "l1\\nl2", "b": "say \\"hi\\""}', '{"a": "l1\\nl2", "b": "say \\"hi\\""}'),
        ("\\path{/home/u/code/results/x} \\\\", "\\path{results/x} \\\\"),
    ):
        assert repo_relative(src) == want, (src, repo_relative(src), want)

    assert scrub({"a": ["/home/u/S/code/results/x"], "b": 3}) == {"a": ["results/x"],
                                                                 "b": 3}

    # And the shipped bundles are clean, which is the property that actually
    # matters: the regex is only a means to it.
    leak = re.compile(r"/home/\w+|[A-Za-z]:\\{1,2}Users\\{1,2}\w+")
    binary = {".png", ".pdf", ".pt", ".pth", ".npz", ".gz", ".zip", ".tgz"}
    results = Path(__file__).resolve().parent.parent / "results"
    dirty = []
    for path in sorted(results.rglob("*")) if results.is_dir() else []:
        if not path.is_file() or path.suffix in binary:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if leak.search(text):
            dirty.append(path.relative_to(results).as_posix())
    assert not dirty, ("these would ship a username to a blind reviewer: "
                       + ", ".join(dirty[:10]))


def test_the_shipped_federated_archive_holds_one_sign_convention():
    """The archive that ships must be the run set the appendix describes.

    `app:repro` states that the released tree carries 66 runs from an earlier
    session, that they predate the sign convention of the theory section, and that
    the exporter excludes them from every table and figure. An archive built before
    `_drop_foreign_conventions` existed satisfies none of that: it holds 241 runs
    under two conventions, and its tuning table lists each learning rate twice, at
    two accuracies, which reads as a selection failure.

    Nothing regenerates the archive automatically, so this is the thing that
    notices. Skipped where the archive is absent -- a source checkout has no
    `results/`.
    """
    import json
    import zipfile
    from pathlib import Path

    archive = Path(__file__).resolve().parents[1] / "results/federated_export_results.zip"
    if not archive.exists():
        return

    with zipfile.ZipFile(archive) as z:
        conventions = set()
        for name in z.namelist():
            if name.endswith("metrics.json"):
                config = json.loads(z.read(name))["config"]
                conventions.add((config.get("uplink_zeros") or "keep",
                                 config.get("mv_ties") or "zero"))

    # Not `"excluded" in MANIFEST.json`: that key predates the filter -- it also
    # lists runs dropped for being unreadable -- so it is present in an archive
    # built before the filter existed and proves nothing. The conventions actually
    # inside the archive are the thing to look at.
    assert len(conventions) == 1, (
        f"{archive.name} mixes sign conventions {sorted(conventions)}, so its tuning "
        f"table compares two experiments. Re-export it with "
        f"`python3 -m federated.export_article` on a machine with torch (torch is "
        f"needed only for the communication columns), then rebuild the supplement")


def test_every_code_section_has_a_readme():
    """Each section of the tree documents itself, so the supplement is navigable.

    A reviewer opening `code/` should be able to find the thing they want without
    reading source. Adding a package without a README is the easy way to break that,
    so it fails here.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sections = {"common", "centralized", "federated", "synthetic",
                "counterexamples", "nanogpt", "tests"}
    missing = sorted(name for name in sections
                     if (root / name).is_dir() and not (root / name / "README.md").exists())
    assert not missing, f"no README.md in: {missing}"
    # Not ANONYMIZATION.md: it is withheld from the bundle, and this test runs
    # there. `test_anonymize.py` asserts it exists in a working tree.
    for doc in ("README.md", "REPRODUCE.md"):
        assert (root / doc).exists(), f"code/{doc} is missing"


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


#: Modules that only some tests need and no experiment does. A GPU box
#: provisioned for sweeps has torch but often not these, and counting their
#: absence as a failure takes the whole suite down with it -- which matters
#: because ``synthetic.run_gpu`` and both ``overnight.py`` drivers run this
#: suite as a preflight and refuse to start when it exits non-zero. Skipping
#: them keeps the preflight meaningful instead of forcing --no-selftest, which
#: would skip the optimizer and synthetic checks too. Anything not listed here
#: -- torch above all -- still fails, because its absence means the run itself
#: cannot be trusted.
OPTIONAL_MODULES = frozenset({"matplotlib", "pandas", "scipy", "seaborn"})


# The anonymity checks live in `test_anonymize.py`, which is withheld from the
# anonymous bundle along with the `anonymize.py` it imports. Adopting them here
# keeps one command -- `python3 -m tests.test_code`, what both overnight drivers
# run as a preflight -- and keeps the scan running on every night rather than only
# when someone remembers it. A tree without the module is the bundle, where there
# is nothing for them to check and no import that fails to resolve.
try:
    from tests import test_anonymize as _anonymity
except ModuleNotFoundError:
    pass
else:
    globals().update({name: value for name, value in vars(_anonymity).items()
                      if name.startswith("test_")})


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed, skipped = [], []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed.append((name, str(exc)))
            print(f"FAIL  {name}\n      {exc}")
        except ModuleNotFoundError as exc:
            root = (exc.name or "").split(".")[0]
            if root not in OPTIONAL_MODULES:
                failed.append((name, f"ModuleNotFoundError: {exc}"))
                print(f"ERROR {name}\n      ModuleNotFoundError: {exc}")
                continue
            skipped.append((name, root))
            print(f"skip  {name}\n      needs {root}, which this tree does not have")
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    tail = f", {len(skipped)} skipped" if skipped else ""
    print(f"\n{len(tests) - len(failed) - len(skipped)}/{len(tests)} passed{tail}")
    if skipped:
        print("skipped: " + ", ".join(f"{n} ({m})" for n, m in skipped))
    return 1 if failed else 0




def test_transmitted_signs_are_strictly_one_bit():
    """The paper's randomized-zero convention: every transmitted sign message is
    +-1 valued, even when the momentum -- and hence the LMO output -- has exact
    zero rows (a dead channel). ``sign(0) = 0`` would make the alphabet ternary;
    ``sign_pm1`` maps zeros to random +-1, reproducibly under the seed."""
    from common.optimizers import sign_pm1

    x = torch.zeros(4, 6)
    x[0, 0], x[1, 2] = 1.5, -0.25
    torch.manual_seed(7)
    s1 = sign_pm1(x)
    torch.manual_seed(7)
    s2 = sign_pm1(x)
    assert torch.equal(s1, s2), "randomized zeros must be seed-reproducible"
    assert torch.equal(s1.abs(), torch.ones_like(s1)), "no zeros may survive"
    assert s1[0, 0] == 1 and s1[1, 2] == -1, "nonzero entries keep their sign"
    # Dense inputs must not consume the RNG stream, so a run with no zeros is
    # bit-identical to one under the old convention. Build the input *before*
    # snapshotting the state -- torch.randn advances the stream itself.
    dense = torch.randn(8, 8) + 10.0
    before = torch.get_rng_state()
    sign_pm1(dense)
    assert torch.equal(before, torch.get_rng_state()), (
        "sign_pm1 consumed randomness on a zero-free input")

    # End-to-end: a dead output channel (zero gradient row) still yields a
    # strictly +-1-valued step for every sign-terminated method.
    torch.manual_seed(0)
    for cls in (SignSGD, SignMuon, MuonSign):
        p = nn.Parameter(torch.randn(5, 7))
        opt = cls([p], lr=0.1, momentum=0.0)
        opt.capture_direction = True
        g = torch.randn(5, 7)
        g[2] = 0.0  # dead channel: zero momentum row, zero LMO-output row
        p.grad = g
        opt.step()
        d = opt.state[p]["last_direction"]
        assert torch.equal(d.abs(), torch.ones_like(d)), cls.__name__


def test_hardware_scan_groups_by_experiment_and_machine():
    """The paper needs one row per (experiment, machine), not one per run.

    Different experiments here run on different GPUs, so the table is built from
    what each run recorded; this pins the grouping, including that two runs of
    the same experiment on the same box collapse to a single row with a count.
    """
    import json
    import tempfile
    from pathlib import Path

    from common.hardware import scan_results

    a = {"gpu": "NVIDIA RTX A4000", "gpu_memory_gb": 16.0, "cpu": "AMD EPYC",
         "ram_gb": 128.0, "os": "Linux 6.8", "python": "3.12.3", "torch": "2.7.0",
         "cuda": "12.8"}
    b = dict(a, gpu="NVIDIA A100", gpu_memory_gb=80.0)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel, hw in (("synthetic/run1/metrics.json", a),
                        ("synthetic/run2/metrics.json", a),     # same box -> one row
                        ("federated/run1/metrics.json", b)):    # different box
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"config": {"hardware": hw}}), encoding="utf-8")

        grouped = scan_results(root)
        fams = {f: [(i.get("gpu"), n) for i, n in e] for f, e in grouped.items()}
        assert fams["Synthetic quadratic"] == [("NVIDIA RTX A4000", 2)], fams
        assert fams["Federated CIFAR-10 (CNN2)"] == [("NVIDIA A100", 1)], fams


def test_list_arguments_accept_commas_and_spaces():
    """`--stages grid final`, `grid,final` and `grid, final` must agree.

    A shell splits on spaces, so a comma the user typed for readability arrives
    glued to a token and argparse rejects a name nobody wrote. Validation has to
    happen after the split, which is why these options do not use argparse
    `choices`; this pins both the splitting and the error message.
    """
    from common.utils import split_list_arg

    valid = ["stability", "grid", "final"]
    for spelling in (["grid", "final"], ["grid,final"], ["grid,", "final"],
                     ["grid , final"]):
        assert split_list_arg(spelling, valid, "stage") == ["grid", "final"], spelling

    assert split_list_arg(["grid,grid", "final"], valid, "stage") == ["grid", "final"]
    assert split_list_arg(None, valid, "stage") is None

    try:
        split_list_arg(["grid,flooor"], valid, "stage")
    except ValueError as exc:
        assert "flooor" in str(exc) and "stability" in str(exc), exc
    else:
        raise AssertionError("an unknown stage name must raise, naming the valid set")


def test_momentum_zero_is_not_a_censored_grid_edge():
    """An optimum at momentum 0 is a result, not a grid too narrow.

    The tuner flags an optimum sitting on a grid endpoint, because such a value
    is an upper bound rather than a measurement. Momentum is bounded below by 0
    -- the optimizers reject anything else -- so mom = 0 is a natural bound and
    flagging it sends the reader to widen a grid that cannot be widened. Under
    the paper's own protocol, where a boundary hit disqualifies the row, that
    would have discarded five of ten rows in the first real sweep.
    """
    def flagged(momenta, best_mom):
        boundary = tuple(m for m in (momenta[0], momenta[-1]) if m > 0.0)
        return bool(boundary and best_mom in boundary and len(momenta) > 1)

    assert not flagged([0.0, 0.5, 0.9], 0.0), "mom=0 is a natural bound"
    assert flagged([0.0, 0.5, 0.9], 0.9), "the upper end is genuinely censored"
    assert flagged([0.5, 0.9], 0.5), "a nonzero lower end can still be widened"

    # and the shipped implementation must agree with that rule
    import inspect

    from synthetic import benchmark
    src = inspect.getsource(benchmark)
    assert "if m > 0.0" in src, (
        "the momentum-boundary rule no longer excludes the natural bound at 0")


# --------------------------------------------------------------------------
# The centralized run -> archive -> figures pipeline
# --------------------------------------------------------------------------


def _fake_metrics(tmp, run_name, seed, *, optimizer, lr, epochs, split,
                  top=94.0, weight_decay=0.0, extra_series=None):
    """One synthetic ``metrics.json``, shaped exactly like a real run's."""
    import json

    steps = list(range(epochs + 1))
    hist = {
        "steps": steps,
        "test_acc": [10.0 + (top - 10.0) * (s / epochs) for s in steps],
        "train_acc": [min(100.0, 10.0 + 90.0 * s / epochs) for s in steps],
        "train_loss": [2.3 * (0.9 ** s) for s in steps],
        "test_loss": [2.3 * (0.95 ** s) for s in steps],
        # epoch 0 is an evaluation, not a training epoch, so it has no duration.
        # A reader that forward-fills or zero-fills this column gets the median
        # wrong; the schema stores null and the exporter must drop it.
        "epoch_seconds": [None] + [14.0 for _ in steps[1:]],
    }
    if split == "tune":
        hist["val_acc"] = [a - 0.5 for a in hist["test_acc"]]
    hist.update(extra_series or {})
    cfg = {"dataset": "cifar10", "model": "resnet18", "optimizer": optimizer,
           "epochs": epochs, "lr": lr, "lr_aux": 1e-3, "lr_scaling": "unit-gain",
           "split": split, "weight_decay": weight_decay, "last_k": 5, "seed": seed,
           "run_name": run_name, "batch_size": 128, "momentum": 0.9,
           "hardware": {"gpu": "Test GPU", "cpu": "Test CPU", "os": "Linux 5.15",
                        "python": "3.11.9", "torch": "2.5.1", "cuda": "12.4",
                        "gpu_memory_gb": 16.0, "driver": "550.90", "ram_gb": 64.0}}
    out = tmp / run_name / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"config": cfg, "history": hist}),
                                      encoding="utf-8")


def test_export_attributes_every_phase_the_driver_emits():
    """Every tag ``overnight`` builds must land in the right phase.

    ``export_article.phase_of`` reads the phase off the directory name, and the
    names are built by ``tune.run_one`` from the driver's tag. The two are only
    coupled by convention, so a phase renamed on one side and not the other is
    silent: the runs simply become ``other`` and drop out of the table. The
    ``wd`` case is the sharp one -- ``final_wd_*`` also starts with ``final_``,
    and testing the fallback first would report the decay ablation as a primary
    result.
    """
    from centralized.export_article import phase_of
    from centralized.tune import canonical_tag

    cases = [
        (canonical_tag("gain_signmuon", epochs=20), "tune_", "gain"),
        (canonical_tag("aux_signmuon_lr0.05_aux0.001", epochs=15), "tune_", "aux"),
        (canonical_tag("lr_signmuon_unit-gain_0.05", epochs=75), "tune_", "lr"),
        (canonical_tag("signmuon_unit-gain", epochs=75, split="full", seed=1),
         "final_", "final"),
        (canonical_tag("wd_signmuon_unit-gain", epochs=75, split="full", seed=0),
         "final_", "wd"),
        (canonical_tag("preflight_timing", epochs=2), "tune_", "preflight"),
        # A retired phase: the 2026-07-27 tree still holds these, and an export of
        # it should label them rather than drop them into `other`.
        (canonical_tag("verify_signmuon_unit-gain_0.2", epochs=75), "tune_",
         "verify"),
    ]
    for tag, prefix, want in cases:
        got = phase_of(prefix + tag)
        assert got == want, f"{prefix + tag!r} -> {got!r}, expected {want!r}"


def test_export_table_aggregates_seeds_the_way_the_paper_defines_it(tmp_path=None):
    """Per-seed tail mean first, mean +/- sample std over seeds second.

    Not the same as pooling every seed's epochs and taking one tail: with three
    seeds and ``last_k = 5`` the pooled tail is the last five values of *one*
    seed, so the other two would not enter the number at all. The std must be the
    sample (n-1) one, and must be absent -- not 0.0 -- at a single seed, since one
    seed measured no dispersion and a printed zero reads as perfect agreement.
    """
    import statistics
    import tempfile
    from pathlib import Path

    from centralized.export_article import scan, table_rows

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tops = [94.0, 94.6, 94.2]
        for seed, top in enumerate(tops):
            _fake_metrics(tmp, "final_signmuon_unit-gain_e75_fs%d" % seed, seed,
                          optimizer="signmuon", lr=0.05, epochs=75, split="full",
                          top=top)
        _fake_metrics(tmp, "final_wd_signmuon_unit-gain_e75_fs0", 0,
                      optimizer="signmuon", lr=0.05, epochs=75, split="full",
                      top=93.5, weight_decay=5e-4)
        runs, bad = scan(tmp)
        assert not bad, bad
        rows = {r["phase"]: r for r in table_rows(runs, [90.0])}

        # The expected value, recomputed the long way from the same generator.
        per_seed = []
        for top in tops:
            series = [10.0 + (top - 10.0) * (s / 75) for s in range(76)]
            per_seed.append(sum(series[-5:]) / 5)
        want = sum(per_seed) / 3
        assert abs(rows["final"]["test_acc_mean"] - round(want, 2)) < 5e-3, rows
        assert abs(rows["final"]["test_acc_std"]
                   - round(statistics.stdev(per_seed), 2)) < 5e-3, rows
        assert rows["final"]["n_seeds"] == 3

        # The decay ablation runs at the SAME eta_0 by design, so a key that
        # omitted weight_decay would merge it into the row it is an ablation of.
        assert rows["wd"]["test_acc_std"] is None, "one seed measured no spread"
        assert rows["wd"]["weight_decay"] == 5e-4

        # epoch 0 carries no duration; a run that read the null as 0.0 would
        # report a median of 14.0 -> 7.0 here.
        assert rows["final"]["epoch_seconds_median"] == 14.0


def test_figures_are_a_function_of_the_export_bundle_alone(tmp_path=None):
    """``plot_analysis`` must read the bundle, and only the bundle.

    The whole point of the archive is that the run tree -- gigabytes of
    ``model.pt`` -- never leaves the GPU box. If a figure needed anything outside
    ``runs.csv``/``curves.csv`` it could not be redrawn from what was brought
    home, which is exactly how the pipeline broke before: the figures went
    through an intermediate CSV whose columns the aggregator had since renamed.
    """
    import tempfile
    from pathlib import Path

    from centralized.export_article import main as export_main
    from centralized.plot_analysis import Bundle

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tree, bundle = tmp / "centralized", tmp / "bundle"
        for seed in (0, 1):
            _fake_metrics(tree, f"final_signmuon_unit-gain_e75_fs{seed}", seed,
                          optimizer="signmuon", lr=0.05, epochs=75, split="full",
                          top=94.0 + 0.4 * seed)
        for lr in (0.02, 0.05, 0.1):
            _fake_metrics(tree, f"tune_lr_signmuon_unit-gain_{lr}_e75_t", 0,
                          optimizer="signmuon", lr=lr, epochs=75, split="tune",
                          top=94.0 - 10 * abs(lr - 0.05))
        assert export_main(["--root", str(tree), "--out", str(bundle),
                            "--no-archive", "--quiet"]) == 0

        b = Bundle(bundle)
        rep = b.reported()["signmuon"]
        assert rep["lr"] == 0.05 and rep["n_seeds"] == 2, rep

        # The sweep is drawn from val_acc -- the metric selection actually ranked.
        # Plotting the tune runs' test accuracy would draw a number no decision
        # used, on models trained on 45k rather than 50k.
        sweep = b.sweep()["signmuon"]
        assert set(sweep) == {0.02, 0.05, 0.1}, sweep
        assert max(sweep, key=lambda k: sweep[k]) == 0.05

        curve = b.curve("signmuon", 0.05, "test_acc")
        assert curve["n_seeds"] == 2 and len(curve["steps"]) == 76
        assert all(sd >= 0 for sd in curve["std"])
        assert curve["std"][-1] > 0, "two seeds differ; the band must be nonzero"


def test_every_cross_module_import_resolves():
    """Every ``from <our package> import name`` must name something that exists.

    Pure AST, no imports executed, so it covers modules this suite never loads --
    including the ones behind a CUDA-only path. That is the gap it exists for:
    deleting a helper leaves the importing module broken until something happens
    to import it, and if the only importer is a driver, "something" is the
    overnight run's preflight, twelve hours after you wanted to know.

    That is not hypothetical. ``refine_grid`` was retired from
    ``centralized.tune``; ``federated.tune`` still imported it, unused, and the
    first thing to notice was a preflight on the GPU box.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    packages = {p.name for p in root.iterdir()
                if p.is_dir() and (p / "__init__.py").exists()}

    def bound(target) -> set:
        """Names an assignment target binds, including tuple unpacking.

        ``GRID, AXIS, SURFACE = ...`` in `common/plotting.py` is the case that
        matters: reading only ``ast.Name`` targets would report three real
        constants as missing, and a check that cries wolf gets switched off.
        """
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return {n for t in target.elts for n in bound(t)}
        return set()

    def resolve(module: str):
        """``a.b`` -> ``a/b.py`` or ``a/b/__init__.py``; ``None`` if neither."""
        base = root.joinpath(*module.split("."))
        for cand in (base.with_suffix(".py"), base / "__init__.py"):
            if cand.exists():
                return cand
        return None

    def exported(path: Path):
        """Top-level names a module binds, or ``None`` if it cannot be known.

        A module with a star-import re-exports names this cannot see, so it is
        excused rather than guessed at -- a false alarm here costs more than the
        miss, since the whole value of the check is that it is trusted.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(n for t in node.targets for n in bound(t))
            elif isinstance(node, ast.AnnAssign):
                names |= bound(node.target)
            elif isinstance(node, ast.Import):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if any(a.name == "*" for a in node.names):
                    return None
                names.update(a.asname or a.name for a in node.names)
        return names

    cache, broken = {}, []
    for path in sorted(root.rglob("*.py")):
        if any(part in {"__pycache__", "results", "data"} for part in path.parts):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if node.module.split(".")[0] not in packages:
                continue
            target = resolve(node.module)
            if target is None:
                broken.append(f"{path.relative_to(root)}:{node.lineno} -> no module "
                              f"{node.module}")
                continue
            if target not in cache:
                cache[target] = exported(target)
            if cache[target] is None:               # star-import; cannot be known
                continue
            for alias in node.names:
                if alias.name == "*" or alias.name in cache[target]:
                    continue
                # `from synthetic import benchmark` names a submodule, not
                # something `synthetic/__init__.py` binds. Legal, and common here.
                if resolve(f"{node.module}.{alias.name}") is not None:
                    continue
                broken.append(f"{path.relative_to(root)}:{node.lineno} imports "
                              f"{alias.name!r} from {node.module}, which does "
                              f"not define it")
    assert not broken, "dangling imports:\n  " + "\n  ".join(broken)



def test_grid_anchors_boost_only_the_sign_family():
    """``anchor_for`` is where every grid is centred, for every phase.

    The sign family's eta_0 is a *base* rate that the per-layer rule then divides
    by sqrt(fan_in), so under a scaling rule its anchor moves by the reciprocal of
    the typical multiplier; the LMO family carries the aspect factor under every
    rule and does not move; SGD and Adam are run unscaled. The same three lines
    used to be copy-pasted into each phase and had already drifted once.
    """
    from centralized.tune import (LEGACY_ANCHORS, SCALED_ANCHOR_BOOST, anchor_for,
                                  extend_grid, lattice_value, round_grid)

    for rule in ("legacy", "unit-gain", "mup"):
        boost = SCALED_ANCHOR_BOOST[rule]
        assert anchor_for("signmuon", rule) == LEGACY_ANCHORS["signmuon"] * boost
        assert anchor_for("muon", rule) == LEGACY_ANCHORS["muon"]
        for baseline in ("sgd", "adam"):
            assert anchor_for(baseline, rule) == LEGACY_ANCHORS[baseline], (
                f"{baseline} has no norm-fixed step for the rule to act on")
    try:
        anchor_for("nosuchmethod", "unit-gain")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown method must raise, not silently anchor at 1")

    # Every rate the driver can try is a quotable 1-2-5 lattice point, extensions
    # included -- otherwise a widened grid would have a different resolution from
    # the grid it widens, and the equal-budget claim would be approximate.
    lattice = {f"{lattice_value(k):.6g}" for k in range(-20, 12)}
    grid = round_grid(0.034, points=5)
    for _ in range(4):
        grid = extend_grid(grid, low=False, points=2)
    assert grid == sorted(grid) and len(set(grid)) == len(grid)
    assert all(f"{lr:.6g}" in lattice for lr in grid), grid


if __name__ == "__main__":
    sys.exit(main())
