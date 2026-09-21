from __future__ import annotations

from .types import FeaturePairing, FeatureSpec


def pair_state_action(
    state_spec: FeatureSpec | None,
    action_spec: FeatureSpec | None,
) -> FeaturePairing:
    """Map each state dim to an action dim for cross-modal metrics.

    Strategy:
    1. Strip common prefixes (`observation.state.`, `action.`) and match remaining
       dot-path substrings exactly. Any joint-named pair like `observation.state.joint_0`
       ↔ `action.joint_0` lines up.
    2. If names are opaque (bare `observation.state` + `action` with positional names
       `0`, `1`, …), fall back to positional pairing when the dims match, and warn.
    3. Otherwise return `method="none"` and leave cross-modal metrics to skip.
    """
    if state_spec is None or action_spec is None:
        return FeaturePairing(
            method="none",
            warnings=["state or action spec missing; cross-modal metrics skipped"],
        )

    state_suffixes = {_strip_prefix(n, "observation.state."): n for n in state_spec.names}
    action_suffixes = {_strip_prefix(n, "action."): n for n in action_spec.names}

    matched: dict[str, str] = {}
    for suffix, s_name in state_suffixes.items():
        if suffix in action_suffixes:
            matched[s_name] = action_suffixes[suffix]

    if matched:
        return FeaturePairing(method="dot_path_match", state_to_action=matched)

    # Loose name-suffix match: pair names where one ends with the other, e.g.
    # `measured_joint_pos_3` ↔ `joint_pos_3`. Common in LeRobot datasets where
    # state sensors carry a `measured_` / `target_` prefix that actions lack.
    loose = _suffix_loose_match(state_spec.names, action_spec.names)
    if loose:
        return FeaturePairing(
            method="dot_path_match",
            state_to_action=loose,
            warnings=[
                "Paired state/action via loose name-suffix match "
                f"({len(loose)} pairs)"
            ],
        )

    common = min(state_spec.dim, action_spec.dim)
    if common > 0:
        positional = {
            state_spec.names[i]: action_spec.names[i] for i in range(common)
        }
        if state_spec.dim == action_spec.dim:
            warnings = [
                "Feature names could not be matched on dot-path; fell back to "
                f"positional pairing (state dim {state_spec.dim} == action dim)",
            ]
        else:
            warnings = [
                "Feature names could not be matched on dot-path; fell back to "
                f"truncated positional pairing "
                f"(state dim {state_spec.dim}, action dim {action_spec.dim}, "
                f"paired first {common})",
            ]
        return FeaturePairing(
            method="positional_fallback",
            state_to_action=positional,
            warnings=warnings,
        )

    return FeaturePairing(
        method="none",
        warnings=[
            f"No dot-path matches and either spec has zero dims "
            f"(state={state_spec.dim}, action={action_spec.dim}); "
            f"cross-modal metrics skipped"
        ],
    )


def _suffix_loose_match(
    state_names: list[str], action_names: list[str]
) -> dict[str, str]:
    """Pair names where one ends with the other (after a non-alphanumeric
    separator). Returns a dict `state_name -> action_name`, each action used
    at most once, preferring longer suffix overlaps."""
    used_actions: set[str] = set()
    pairs: dict[str, str] = {}
    for s in state_names:
        best_a: str | None = None
        best_len = 0
        for a in action_names:
            if a in used_actions:
                continue
            if _ends_with_token(s, a) or _ends_with_token(a, s):
                overlap = min(len(s), len(a))
                if overlap > best_len:
                    best_len = overlap
                    best_a = a
        if best_a is not None:
            pairs[s] = best_a
            used_actions.add(best_a)
    return pairs


def _ends_with_token(haystack: str, needle: str) -> bool:
    """True iff `haystack` ends with `needle`, either exactly or after a
    non-alphanumeric separator (so `joint_1` does not match `joint_11`)."""
    if haystack == needle:
        return True
    if not haystack.endswith(needle):
        return False
    sep_pos = len(haystack) - len(needle) - 1
    if sep_pos < 0:
        return True
    return not haystack[sep_pos].isalnum()


def _strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix):] if name.startswith(prefix) else name
