from experiments.acceptance_diagnostics.analyze import summarize


def row(ctx, position, accepted, mismatch, margin=0.2, rank=2, relevant=True):
    return {
        "ctx": str(ctx), "position": str(position),
        "accepted": str(accepted), "first_mismatch": str(mismatch),
        "decision_relevant": str(relevant), "target_margin": str(margin),
        "draft_token_target_rank": str(rank),
    }


def main() -> int:
    positions = []
    cases = []
    for ctx in (2048, 4096, 8192):
        positions += [row(ctx, 1, True, False), row(ctx, 2, False, True),
                      row(ctx, 3, False, False, relevant=False)]
        cases.append({"ctx": str(ctx), "acceptance": "0.5", "rounds": "2",
                      "mismatch_rounds": "1"})
    result = summarize(positions, cases)
    assert all(item["reached"] == 1 for item in result["position"] if item["position"] <= 2)
    assert all(item["conditional_acceptance"] == 1.0 for item in result["position"] if item["position"] == 1)
    assert all(item["conditional_acceptance"] == 0.0 for item in result["position"] if item["position"] == 2)
    assert all(item["mismatches"] == 1 for item in result["margins"])
    assert len(result["contexts"]) == 3
    print("acceptance diagnostics CPU tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
