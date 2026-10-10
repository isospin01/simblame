import simblame as sb


def test_toy_end_to_end(toy_spec):
    h = sb.load_harness(toy_spec)
    r = sb.Runner(h, verbose=False)
    seeds = list(range(40))

    det = sb.check_determinism(r, h.a, seeds[:8])
    assert det.ok

    comp = sb.compare(r, seeds)
    assert comp.significant and comp.n01 > 10 and comp.n10 == 0
    assert comp.verdict == sb.Verdict.REGRESSION

    d = sb.diff_setups(r, seeds)
    cats = {c.category for c in d.physics_changes()}
    assert cats == {"contact", "inertia", "joints"}

    att = sb.attribute(r, d, comp, max_focus=20, controls=5)
    main = att.main_cause()
    assert main is not None and main.name == "contact"
    assert main.role in sb.Role.CAUSES
    assert main.rescue == (main.rescue[1], main.rescue[1])  # reverting friction rescues every seed
    inertia = next(g for g in att.levels["category"] if g.name == "inertia")
    # sliding deceleration is mu*g: the heavier box does nothing on its own
    assert inertia.alone == 0.0 and inertia.role not in sb.Role.STRONG

    div, ta, tb = sb.diverge(r, comp.regressed_seeds[0])
    assert div.first_step is not None and div.first_step >= 1
    assert div.a_success and not div.b_success


def test_identical_setups_say_so(toy_spec):
    h = sb.load_harness(toy_spec)
    h.variants = ("v1.0", "v1.0")
    r = sb.Runner(h, verbose=False)
    comp = sb.compare(r, list(range(10)))
    assert comp.identical == 10
    assert comp.verdict == sb.Verdict.IDENTICAL


def test_open_loop_divergence_and_save(tmp_path, toy_spec):
    h = sb.load_harness(toy_spec)
    r = sb.Runner(h, verbose=False)
    div, ta, tb = sb.diverge(r, 1, open_loop=True)
    assert div.mode == "open-loop"
    ta.save(tmp_path / "a.npz")
    t2 = sb.Trajectory.load(tmp_path / "a.npz")
    assert t2.xpos.shape == ta.xpos.shape and t2.contacts == ta.contacts
    again = sb.compare_trajectories(ta, t2)
    assert again.first_step is None


def test_nondeterminism_is_detected(toy_spec):
    import numpy as np

    h = sb.load_harness(toy_spec)
    rng = np.random.default_rng(0)
    orig = h.reset

    def leaky_reset(env, seed):  # ignores the seed for part of the state
        obs = orig(env, seed)
        env.data.qpos[1] += rng.uniform(-0.01, 0.01)
        return obs

    h.reset = leaky_reset
    r = sb.Runner(h, verbose=False)
    rep = sb.check_determinism(r, h.a, list(range(6)))
    assert not rep.ok and "NOT deterministic" in str(rep)


def test_improvement_direction_is_worded_as_such(toy_spec):
    h = sb.load_harness(toy_spec)
    h.variants = ("v1.1", "v1.0")  # swapped: B is the better setup
    r = sb.Runner(h, verbose=False)
    comp = sb.compare(r, list(range(30)))
    assert comp.verdict == sb.Verdict.IMPROVEMENT and comp.flipped_seeds == comp.improved_seeds
    d = sb.diff_setups(r, list(range(30)))
    att = sb.attribute(r, d, comp, max_focus=10, controls=4)
    assert att.direction == "improvement" and att.main_cause().name == "contact"
    text = str(att)
    assert "improved seeds" in text and "regressed" not in text


def test_few_seeds_never_claim_a_cause(toy_spec):
    h = sb.load_harness(toy_spec)
    r = sb.Runner(h, verbose=False)
    comp = sb.compare(r, [0, 1, 2, 3])
    assert comp.verdict == sb.Verdict.NOT_SIGNIFICANT
    assert comp.delta_ci[0] < 0 < comp.delta_ci[1]  # CI agrees with the test
    assert comp.seeds_needed < 20  # a big one-way change needs few seeds, not thousands
    d = sb.diff_setups(r, [0, 1, 2, 3])
    att = sb.attribute(r, d, comp, controls=1)
    assert att.main_cause() is None
    assert all(g.role not in sb.Role.STRONG for g in att.levels["category"])


def test_depth_three_reaches_fields_of_a_single_entity(toy_spec):
    """The floor is the only contact entity, so field level drills from the category."""
    toy = sb.load_harness(toy_spec)
    module = __import__(type(toy).__module__)

    class FloorEdits(type(toy)):
        def make_env(self, variant):
            xml = module.XML.format(**module.VARIANTS["v1.0"])
            if variant == "v1.1":
                xml = xml.replace('friction="0.5 0.005', 'margin="0.002" friction="0.35 0.005', 1)
            return module.ToyPushEnv(xml)

    h = FloorEdits()
    r = sb.Runner(h, verbose=False)
    seeds = list(range(30))
    comp = sb.compare(r, seeds)
    d = sb.diff_setups(r, seeds)
    att = sb.attribute(r, d, comp, depth=3, max_focus=8, controls=2)
    assert "entity" not in att.levels  # one entity: nothing to split
    fields = {g.name for g in att.levels["field"]}
    assert any(n.endswith("friction") for n in fields) and any(n.endswith("margin") for n in fields)
