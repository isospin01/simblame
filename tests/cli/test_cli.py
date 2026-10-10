import json

from simblame.cli import main


def test_cli_diff_two_files(tmp_path, xml, capsys):
    a = tmp_path / "a.xml"
    b = tmp_path / "b.xml"
    a.write_text(xml())
    b.write_text(xml(mu=0.25))
    rc = main(["diff", str(a), str(b), "--exit-code", "--json", str(tmp_path / "d.json")])
    out = capsys.readouterr().out
    assert rc == 1 and "friction[sliding]: 1 -> 0.25" in out
    data = json.loads((tmp_path / "d.json").read_text())
    assert data["changes"][0]["category"] == "contact"
    assert main(["diff", str(a), str(a), "--exit-code"]) == 0


def test_cli_run_writes_report(tmp_path, toy_spec, capsys):
    out = tmp_path / "rep"
    rc = main(["run", toy_spec, "--seeds", "24", "--out", str(out), "-q", "--max-focus", "10",
               "--controls", "4"])
    assert rc == 0
    md = (out / "report.md").read_text()
    data = json.loads((out / "report.json").read_text())
    assert "CAUSE" in md and data["attribution"]["main_cause"] == "contact"
    assert data["comparison"]["regressed"] > 0
    assert data["attribution"]["levels"]["category"]


def test_cli_check_and_fingerprint(toy_spec, capsys):
    assert main(["check", toy_spec, "--seeds", "4", "-q"]) == 0
    assert main(["fingerprint", toy_spec]) == 0
    out = capsys.readouterr().out
    assert "physics_hash" in out


def test_cli_init_writes_runnable_template(tmp_path, capsys):
    target = tmp_path / "h.py"
    assert main(["init", str(target)]) == 0
    assert "class MyComparison(sb.Harness)" in target.read_text()
    assert main(["init", str(target)]) == 2  # refuses to overwrite
    assert "already exists" in capsys.readouterr().err


def test_cli_explains_missing_target(capsys):
    assert main(["compare"]) == 2
    assert "say what to compare" in capsys.readouterr().err


def test_cli_explains_common_mistakes(tmp_path, capsys):
    target = tmp_path / "h.py"
    main(["init", str(target)])
    capsys.readouterr()
    assert main(["run", str(target), "--seeds", "2", "-q"]) == 2  # template not filled in
    assert "your harness is not finished" in capsys.readouterr().err
    assert main(["run", "--gym", "Hopper-v5"]) == 2  # nothing to compare
    assert "would be identical" in capsys.readouterr().err
    assert main(["compare", "--libero", "libero_spatial/5"]) == 2
    assert "simblame versions --libero" in capsys.readouterr().err


def test_cli_gym_shortcut(tmp_path, capsys):
    import pytest

    pytest.importorskip("gymnasium.envs.mujoco")
    rc = main(["compare", "--gym", "Hopper-v5", "Hopper-v4", "--policy", "random",
               "--success", "survived", "--max-steps", "100", "--seeds", "3", "-q"])
    assert rc == 0
    assert "IDENTICAL" in capsys.readouterr().out


def test_cli_gym_without_success_signal_is_explained(capsys):
    import pytest

    pytest.importorskip("gymnasium.envs.mujoco")
    rc = main(["compare", "--gym", "Hopper-v5", "Hopper-v4", "--max-steps", "5", "--seeds", "2",
               "-q"])
    assert rc == 2
    assert "--success survived" in capsys.readouterr().err


def test_cli_rejects_bad_input_without_traceback(tmp_path, toy_spec, capsys):
    import pytest

    for bad, message in (("abc", "not a seed"), ("5:3", "empty range"), ("0", "at least one")):
        with pytest.raises(SystemExit) as exit_info:  # argparse: usage + message, exit 2
            main(["compare", toy_spec, "--seeds", bad])
        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err
    assert main(["diff", str(tmp_path / "nope.xml"), str(tmp_path / "nope2.xml")]) == 2
    assert "model file not found" in capsys.readouterr().err


def test_suggested_commands_keep_every_target_option():
    from simblame.cli.main import build_parser
    from simblame.cli.targets import target_text

    args = build_parser().parse_args(
        ["compare", "--gym", "Hopper-v5", "--a-kwargs", "xml_file=a.xml", "--b-kwargs",
         '{"xml_file": "b.xml"}', "--policy", "random", "--success", "survived",
         "--max-steps", "100"])
    again = build_parser().parse_args(["compare", *__import__("shlex").split(target_text(args))])
    for key in ("gym", "a_kwargs", "b_kwargs", "policy", "success", "max_steps"):
        assert getattr(again, key) == getattr(args, key), key
