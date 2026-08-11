"""Static smoke checks for the HRIS_new reproduction scripts.

Run from this directory:
    python test_static_smoke.py
"""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def check_python_files_parse() -> None:
    errors = []
    for path in sorted(ROOT.glob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path.name}: {exc.msg} at line {exc.lineno}")

    if errors:
        raise AssertionError("Python parse failures:\n" + "\n".join(errors))


def check_output_path_is_project_local() -> None:
    offenders = []
    for path in sorted(ROOT.glob("*.py")):
        text = path.read_text(encoding="utf-8-sig")
        if "C:\\Users\\DELL" in text or "Desktop\\HRIS" in text:
            offenders.append(path.name)

    if offenders:
        raise AssertionError(
            "Python files contain hardcoded previous-user paths: "
            + ", ".join(offenders)
        )


def check_authoritative_code_has_no_curve_fabrication_hooks() -> None:
    forbidden = {
        "digitized_curve", "hardcoded_curve", "synthetic_reward",
        "curve_offset", "curve_scale", "paper_y", "target_values",
    }
    offenders = []
    for name in [
        "simulation.py", "experiment_contract.py", "main.py",
        "parameter_evidence.py", "parameter_profiles.py",
        "parameter_identification.py", "cross_figure_validation.py",
        "scientific_preflight.py",
        "prepare_reproduction.py",
    ]:
        path = ROOT / name
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8-sig").lower()
        found = sorted(token for token in forbidden if token in source)
        if found:
            offenders.append(f"{name}: {', '.join(found)}")
    if offenders:
        raise AssertionError(
            "Authoritative code contains forbidden curve-fabrication hooks:\n"
            + "\n".join(offenders)
        )


if __name__ == "__main__":
    check_python_files_parse()
    check_output_path_is_project_local()
    check_authoritative_code_has_no_curve_fabrication_hooks()
    print("Static smoke checks passed.")
