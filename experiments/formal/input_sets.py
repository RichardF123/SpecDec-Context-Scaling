from pathlib import Path

from specdec.engine.family_input import load_manifest

ROOT = Path(__file__).resolve().parents[2]

PILOT_FAMILIES = ("F09", "F03", "F11", "F04")
FORMAL_FAMILIES = tuple(f"FF{i:02d}" for i in range(1,17))

def select_input_set(name):
    if name == "pilot":
        path = ROOT / "docs" / "prompt_families_manifest.json"
        families = PILOT_FAMILIES

    elif name == "formal":
        path = ROOT / "docs" / "prompt_families_formal_manifest.json"
        families = FORMAL_FAMILIES

    else:
        raise ValueError(f"未知输入集合： {name}")

    manifest = load_manifest(path)
    registered = [
        item["family_id"] for item in manifest["families"]
    ]

    if len(registered) != len(set(registered)):
        raise ValueError("Manifest 中存在重复 family")

    missing = set(families) - set(registered)
    if missing:
        raise ValueError(f"Manifest 缺少 family: {sorted(missing)}")

    if name == "formal" and set(registered) != set(families):
        raise ValueError("正式清单必须恰好包含 FF01-FF16")

    return path, families