"""Which change did it? Counterfactual attribution over the compiled-model diff.

For each group of changes ``g`` (first per physics category, then per entity)
simblame re-runs the seeds that flipped between A and B, twice:

* **reproduces** -- A + g: on how many flipped seeds does ``g`` alone cause the flip?
* **rescues**    -- B - g: on how many does reverting only ``g`` undo the flip?

The group's ``share`` of the change is the average of the two fractions (the
exact Shapley value for two groups, a first-order estimate beyond). A few
*control* seeds, on which A and B agree, are re-run as well: a group that
flips them on its own is being offset by other changes.

==================================  ===========  ===========  =========================
role                                reproduces   rescues      meaning
==================================  ===========  ===========  =========================
primary cause                       >= 70%       >= 70%       explains the change alone
main cause, amplified by others     40-70%       >= 70%       drives it; others add
only harmful together with others   < 40%        >= 70%       needed, but harmless alone
sufficient, but redundant           >= 70%       < 70%        another change does it too
contributing                        >= 20% in either
minor                               >= 5% in either
opposes the change                  pushes success the other way on the evaluated seeds
not involved                        < 5%         < 5%
not attributable (structural)       only topology changes, which can't be patched in place
==================================  ===========  ===========  =========================

Every role above "minor" also needs its counterfactual to differ significantly
(exact McNemar, p < 0.05) from the setup it was applied to; otherwise the group
is reported as ``weak (n.s.)``: more seeds are needed.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .. import stats
from ..harness.results import Outcomes
from ..harness.runner import Runner
from ..model.diff import ModelDiff
from ..model.fields import Kind
from ..model.patch import BACKWARD, FORWARD, Patch
from .compare import Comparison

HIGH, MID, LOW, MINOR = 0.7, 0.4, 0.2, 0.05


class Role:
    PRIMARY = "primary cause"
    MAIN = "main cause, amplified by others"
    TOGETHER = "only harmful together with others"
    REDUNDANT = "sufficient, but redundant"
    CONTRIBUTING = "contributing"
    WEAK = "weak (n.s.)"
    MINOR = "minor"
    OPPOSES = "opposes the change"
    NONE = "not involved"
    STRUCTURAL = "not attributable (structural)"
    NA = "n/a"

    CAUSES = (PRIMARY, MAIN, REDUNDANT)
    STRONG = (PRIMARY, MAIN, TOGETHER, REDUNDANT, CONTRIBUTING)


@dataclasses.dataclass
class GroupResult:
    """Counterfactual results for one group of changes."""

    name: str  # e.g. "contact" or "contact/geom:floor"
    level: str  # all | category | entity | field | joint
    changes: list
    seeds: list
    base_a: float  # success rate of A on the evaluated seeds
    base_b: float
    alone_rate: float  # success rate of A + g
    undo_rate: float  # success rate of B - g
    reproduce: tuple  # (k, n) flipped seeds that flip the same way under A + g
    rescue: tuple  # (k, n) flipped seeds restored under B - g
    p_alone: float = 1.0  # McNemar A vs A + g
    p_undo: float = 1.0  # McNemar B vs B - g
    side_alone: tuple = (0, 0)  # (k, n) control seeds whose outcome A + g changes
    side_undo: tuple = (0, 0)  # (k, n) control seeds whose outcome B - g changes
    skipped: int = 0  # changes that could not be patched in place
    patched: int = 0
    alone_outcomes: Outcomes | None = None
    undo_outcomes: Outcomes | None = None

    @property
    def total(self) -> float:
        return self.base_b - self.base_a

    @property
    def alone(self) -> float:
        """Change in success when only this group is applied to A."""
        return self.alone_rate - self.base_a

    @property
    def undo(self) -> float:
        """Change in success when only this group is reverted in B."""
        return self.undo_rate - self.base_b

    @property
    def alone_share(self) -> float:
        """Fraction of the flipped seeds that A + g reproduces."""
        k, n = self.reproduce
        return k / n if n else float("nan")

    @property
    def undo_share(self) -> float:
        """Fraction of the flipped seeds that B - g rescues."""
        k, n = self.rescue
        return k / n if n else float("nan")

    @property
    def share(self) -> float:
        """Share of the A->B change this group explains (mean of the two fractions)."""
        values = [v for v in (self.alone_share, self.undo_share) if not np.isnan(v)]
        return float(np.mean(values)) if values else float("nan")

    @property
    def opposes(self) -> bool:
        """Moves success *against* the A->B change on the evaluated seeds (significantly)."""
        if not self.total:
            return False
        net = min(self.alone / self.total, -self.undo / self.total)
        return net <= -LOW and min(self.p_alone, self.p_undo) < 0.05

    @property
    def role(self) -> str:
        if self.patched == 0 and self.skipped:
            return Role.STRUCTURAL
        f, b = self.alone_share, self.undo_share
        if np.isnan(f) and np.isnan(b):
            return Role.NA
        f, b = np.nan_to_num(f), np.nan_to_num(b)
        sig_alone, sig_undo = self.p_alone < 0.05, self.p_undo < 0.05
        if f >= HIGH and b >= HIGH:
            return Role.PRIMARY if sig_alone or sig_undo else Role.WEAK
        if b >= HIGH:
            return (Role.MAIN if f >= MID else Role.TOGETHER) if sig_undo else Role.WEAK
        if f >= HIGH:
            return Role.REDUNDANT if sig_alone else Role.WEAK
        if max(f, b) >= LOW:
            return Role.CONTRIBUTING if sig_alone or sig_undo else Role.WEAK
        if self.opposes:
            return Role.OPPOSES
        if max(f, b) >= MINOR:
            return Role.MINOR
        return Role.NONE

    @property
    def involved(self) -> bool:
        return self.role not in (Role.NONE, Role.NA, Role.STRUCTURAL)

    def params(self) -> list:
        """The user-visible changes of this group (no derived companions, no float noise)."""
        return [c for c in self.changes if c.kind != Kind.DERIVED and not c.is_tiny]

    def to_dict(self) -> dict:
        return {
            "group": self.name, "level": self.level, "role": self.role, "share": self.share,
            "reproduce_share": self.alone_share, "rescue_share": self.undo_share,
            "alone_pts": self.alone * 100, "undo_pts": self.undo * 100,
            "control_flips_alone": list(self.side_alone),
            "control_flips_undo": list(self.side_undo),
            "rate_a": self.base_a, "rate_b": self.base_b,
            "rate_a_plus_group": self.alone_rate, "rate_b_minus_group": self.undo_rate,
            "p_alone": self.p_alone, "p_undo": self.p_undo,
            "reproduce": list(self.reproduce), "rescue": list(self.rescue),
            "unpatchable_changes": self.skipped, "seeds": self.seeds,
            "changes": [c.to_dict() for c in self.params()],
        }


@dataclasses.dataclass
class Attribution:
    """Result of :func:`attribute`."""

    status: str  # ok | no_flips | no_model_changes
    direction: str  # regression | improvement
    focus: list  # flipped seeds evaluated
    controls: list  # agreeing seeds evaluated
    all_changes: GroupResult | None = None
    levels: dict = dataclasses.field(default_factory=dict)  # level -> [GroupResult]
    joint: GroupResult | None = None  # greedy combination when no group suffices alone
    joint_names: list = dataclasses.field(default_factory=list)
    flipped_total: int = 0  # all flipped seeds; ``focus`` is a sample when there are many
    a_label: str = "A"
    b_label: str = "B"

    def groups(self, level: str) -> list:
        return sorted(self.levels.get(level, []), key=lambda g: -np.nan_to_num(g.share))

    @property
    def model_share(self) -> float:
        """Fraction of the A->B change reproduced by the compiled-model differences."""
        return self.all_changes.share if self.all_changes else 0.0

    def main_cause(self) -> GroupResult | None:
        """The most specific group that drives the change, or ``None``.

        A child (entity/field) replaces its parent only if it carries nearly all
        of the parent's effect; otherwise the parent is the clearer answer.
        """
        best = None
        for level in ("category", "entity", "field"):
            if best is None:
                candidates = [g for g in self.levels.get(level, []) if g.role in Role.CAUSES]
            else:
                children = [g for g in self.levels.get(level, [])
                            if g.name.startswith(best.name + "/")]
                if not children:  # this level was not drilled for ``best``: look deeper
                    continue
                candidates = [g for g in children if g.share >= 0.9 * best.share]
                if not candidates:
                    break
            if candidates:
                best = max(candidates, key=lambda g: g.share)
        return best

    def children(self, group: GroupResult) -> list:
        return [g for level in ("entity", "field") for g in self.groups(level)
                if g.name.startswith(group.name + "/") and g.involved]

    def others(self, group: GroupResult | None) -> list:
        """Other categories with a significant role (contributing or stronger)."""
        return [g for g in self.groups("category") if g.role in Role.STRONG and
                (group is None or not (g.name == group.name or group.name.startswith(g.name + "/")))]

    def to_dict(self) -> dict:
        main = self.main_cause()
        return {
            "status": self.status, "direction": self.direction, "focus_seeds": self.focus,
            "flipped_total": self.flipped_total, "control_seeds": self.controls, "model_share": self.model_share,
            "main_cause": main.name if main else None,
            "all_changes": self.all_changes.to_dict() if self.all_changes else None,
            "levels": {k: [g.to_dict() for g in v] for k, v in self.levels.items()},
            "joint": self.joint.to_dict() if self.joint else None,
            "joint_names": self.joint_names,
        }

    def __str__(self) -> str:
        from ..report.text import render_attribution

        return render_attribution(self)


def attribute(runner: Runner, diff: ModelDiff, comparison: Comparison, *, depth: int = 2,
              controls: int = 10, max_focus: int = 40, include_rendering: bool = False,
              seed: int = 0) -> Attribution:
    """Hierarchical counterfactual attribution: category -> entity (-> field if depth=3)."""
    return _Attributor(runner, diff, comparison, include_rendering).run(
        depth=depth, controls=controls, max_focus=max_focus, seed=seed)


class _Attributor:
    def __init__(self, runner: Runner, diff: ModelDiff, comparison: Comparison,
                 include_rendering: bool):
        self.runner, self.diff, self.c = runner, diff, comparison
        self.include_rendering = include_rendering
        self.regression = comparison.n01 >= comparison.n10
        self.seeds: list = []  # focus + controls, set by run()
        self.focus: list = []
        self.focus_set: set = set()

    def run(self, depth: int, controls: int, max_focus: int, seed: int) -> Attribution:
        focus, control = self._pick_seeds(controls, max_focus, seed)
        direction = "regression" if self.regression else "improvement"
        flipped = self.c.regressed_seeds if self.regression else self.c.improved_seeds
        result = Attribution("ok", direction, focus, control, flipped_total=len(flipped),
                             a_label=self.c.a.label, b_label=self.c.b.label)
        if not focus:
            result.status = "no_flips"
            return result
        categories = self.diff.groups("category", self.include_rendering)
        if not categories:
            result.status = "no_model_changes"
            return result

        self.seeds, self.focus, self.focus_set = sorted(focus + control), focus, set(focus)
        result.all_changes = self._evaluate("ALL", "all", self.diff.all_changes(self.include_rendering))
        if len(categories) > 1:
            result.levels["category"] = [self._evaluate(n, "category", ch)
                                         for n, ch in categories.items()]
        else:  # one category: it *is* the whole diff
            result.levels["category"] = [dataclasses.replace(
                result.all_changes, name=next(iter(categories)), level="category")]
        if depth >= 2:
            entities = self._drill(result.levels["category"], "entity")
            if entities:
                result.levels["entity"] = entities
            if depth >= 3:
                undrilled = [c for c in result.levels["category"]
                             if not any(e.name.startswith(c.name + "/") for e in entities)]
                fields = self._drill(entities + undrilled, "field")
                if fields:
                    result.levels["field"] = fields
        self._combine(result)
        return result

    def _pick_seeds(self, controls: int, max_focus: int, seed: int) -> tuple:
        flipped = self.c.regressed_seeds if self.regression else self.c.improved_seeds
        if len(flipped) > max_focus:
            rng = np.random.default_rng(seed)
            flipped = sorted(rng.choice(flipped, size=max_focus, replace=False).tolist())
        sa, sb = self.c.a.by_seed(), self.c.b.by_seed()
        agree = [s for s in self.c.a.seeds if sa[s].success == sb[s].success]
        agree.sort(key=lambda s: (not sa[s].success, s))  # successes first: they can break
        return flipped, agree[:controls]

    def _evaluate(self, name: str, level: str, changes: list) -> GroupResult:
        a, b = self.c.a.label, self.c.b.label
        forward = Patch(changes, FORWARD, name)
        alone = self.runner.evaluate(a, self.seeds, policy_variant=a, patch=forward,
                                     label=f"{a} + [{name}]")
        undo = self.runner.evaluate(b, self.seeds, policy_variant=b,
                                    patch=Patch(changes, BACKWARD, name), label=f"{b} - [{name}]")
        oa, ob = self.c.a.by_seed(), self.c.b.by_seed()
        fa, fb = alone.by_seed(), undo.by_seed()
        sa = np.array([oa[s].success for s in self.seeds])
        sb = np.array([ob[s].success for s in self.seeds])
        s_alone = np.array([fa[s].success for s in self.seeds])
        s_undo = np.array([fb[s].success for s in self.seeds])
        controls = [s for s in self.seeds if s not in self.focus_set]

        def flipped(results, reference, seeds):
            return (sum(results[s].success != reference[s].success for s in seeds), len(seeds))

        return GroupResult(
            name, level, changes, list(self.seeds), float(sa.mean()), float(sb.mean()),
            float(s_alone.mean()), float(s_undo.mean()),
            flipped(fa, oa, self.focus), flipped(fb, ob, self.focus),
            stats.mcnemar_exact(*stats.paired_counts(sa, s_alone)),
            stats.mcnemar_exact(*stats.paired_counts(sb, s_undo)),
            flipped(fa, oa, controls), flipped(fb, ob, controls),
            len(forward.skipped), sum(c.kind != Kind.DERIVED for c in forward.changes),
            alone, undo)

    def _drill(self, parents: list, level: str) -> list:
        """Split the strongest parents (max 3) into entities or fields."""
        strong = [p for p in parents if p.role in Role.STRONG]
        candidates = strong or sorted(parents, key=lambda g: -np.nan_to_num(g.share))[:1]
        out = []
        for parent in sorted(candidates, key=lambda g: -np.nan_to_num(g.share))[:3]:
            if level == "entity":
                children = self.diff.groups("entity", self.include_rendering, only=[parent.name])
            else:
                children = {k: v for k, v in self.diff.groups("field", self.include_rendering).items()
                            if k.startswith(parent.name + "/")}
            children = {k: v for k, v in children.items() if any(c.kind != Kind.DERIVED for c in v)}
            if len(children) > 1:
                out.extend(self._evaluate(name, level, ch) for name, ch in children.items())
        return out

    def _combine(self, result: Attribution) -> None:
        """No category reproduces what all changes do -> add categories greedily until one does."""
        total = result.all_changes
        categories = result.groups("category")
        target = 0.9 * np.nan_to_num(total.alone_share)
        if len(categories) < 2 or np.nan_to_num(total.alone_share) < HIGH or \
                np.nan_to_num(categories[0].alone_share) >= target:
            return
        chosen = [categories[0]]
        for group in categories[1:]:
            if max(np.nan_to_num(group.alone_share), np.nan_to_num(group.undo_share)) < MINOR:
                continue
            chosen.append(group)
            if len(chosen) == len(categories):  # same as "ALL model changes": nothing new
                break
            params = [c for g in chosen for c in g.changes if c.kind != Kind.DERIVED]
            names = [g.name for g in chosen]
            result.joint = self._evaluate(" + ".join(names), "joint", self.diff.bundle(params))
            result.joint_names = names
            if np.nan_to_num(result.joint.alone_share) >= target:
                break
