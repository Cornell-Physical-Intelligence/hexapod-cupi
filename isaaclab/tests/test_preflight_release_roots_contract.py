"""Contract binding the release manifest to the preflight's approved roots.

``isaaclab/deploy/preflight-stage2-spark`` refuses to verify any manifest entry
that falls outside a fixed set of release roots. That allowlist and the manifest
at ``isaaclab/deploy/stage2_pipeline.sha256`` have diverged once already: a path
was pinned in the manifest that the preflight then rejected on the Spark, which
turns a five-second local edit into a failed remote release. Two properties are
pinned here so the divergence fails in the test suite instead:

* the preflight's release-root case line admits exactly ``README.md``,
  ``robot/*``, ``isaaclab/*``, ``packages/*``, and ``ops/*`` — nothing added,
  nothing dropped — and still falls through to a catch-all branch that marks the
  manifest invalid;
* every path in the manifest matches one of those roots, so a future entry
  outside the allowlist fails here rather than during a release.

Standard library only: the preflight is read as text and never executed, and the
manifest hashes are not recomputed — that is ``sha256sum -c``'s job.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ISAACLAB_DIR = Path(__file__).resolve().parents[1]
PREFLIGHT = ISAACLAB_DIR / "deploy" / "preflight-stage2-spark"
MANIFEST = ISAACLAB_DIR / "deploy" / "stage2_pipeline.sha256"

# The approved release roots, in the shell's own glob syntax.
EXPECTED_ROOTS = frozenset(
    {
        "README.md",
        "robot/*",
        "isaaclab/*",
        "packages/*",
        "ops/*",
    }
)

CASE_HEAD_RE = re.compile(r'^\s*case\s+"\$\{manifest_path\}"\s+in\s*$')
CASE_BRANCH_RE = re.compile(r"^\s*(?P<patterns>[^)]+)\)\s*(?P<body>.*)$")
CASE_END_RE = re.compile(r"^\s*esac\s*$")

# Mirrors the preflight's own manifest-line regex.
MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9._/-]+)$")


def _release_root_case_branches() -> list[tuple[list[str], str]]:
    """Return ``(patterns, body)`` for each branch of the release-root case.

    ``body`` is whatever followed the ``)`` on the branch line, which is enough
    to tell the accepting branch (``;;`` alone) from the rejecting one.
    """

    lines = PREFLIGHT.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if CASE_HEAD_RE.match(line)]
    if len(starts) != 1:
        raise AssertionError(
            f"expected exactly one 'case \"${{manifest_path}}\" in', found {len(starts)}"
        )
    branches: list[tuple[list[str], str]] = []
    for line in lines[starts[0] + 1 :]:
        if CASE_END_RE.match(line):
            return branches
        match = CASE_BRANCH_RE.match(line)
        if match is None:
            continue
        patterns = [part.strip() for part in match.group("patterns").split("|")]
        branches.append((patterns, match.group("body").strip()))
    raise AssertionError("release-root case statement is never closed by 'esac'")


def _manifest_paths() -> list[str]:
    paths: list[str] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = MANIFEST_LINE_RE.match(line)
        if match is None:
            raise AssertionError(f"malformed manifest line: {line!r}")
        paths.append(match.group(2))
    return paths


def _matches_root(path: str, root: str) -> bool:
    """Apply one shell ``case`` glob the way bash would to a manifest path."""

    return re.fullmatch(re.escape(root).replace(r"\*", ".*"), path) is not None


class ReleaseRootAllowlistTests(unittest.TestCase):
    def test_case_admits_exactly_the_five_approved_roots(self) -> None:
        branches = _release_root_case_branches()
        accepting = [patterns for patterns, body in branches if patterns != ["*"]]
        self.assertEqual(
            len(accepting), 1, "expected a single accepting release-root branch"
        )
        roots = accepting[0]
        self.assertEqual(len(roots), len(set(roots)), f"duplicate roots: {roots}")
        self.assertEqual(set(roots), set(EXPECTED_ROOTS))

    def test_unlisted_roots_fall_through_to_a_rejecting_branch(self) -> None:
        branches = _release_root_case_branches()
        catch_all = [body for patterns, body in branches if patterns == ["*"]]
        self.assertEqual(len(catch_all), 1, "expected exactly one '*)' branch")
        source = PREFLIGHT.read_text(encoding="utf-8")
        self.assertIn("is outside approved release roots", source)
        self.assertIn("manifest_invalid=1", source)


class ManifestStaysInsideTheAllowlistTests(unittest.TestCase):
    def test_manifest_is_not_empty(self) -> None:
        self.assertGreater(len(_manifest_paths()), 0)

    def test_every_manifest_path_matches_an_approved_root(self) -> None:
        roots = sorted(EXPECTED_ROOTS)
        for path in _manifest_paths():
            self.assertTrue(
                any(_matches_root(path, root) for root in roots),
                f"{path} is outside the approved release roots {roots}",
            )

    def test_manifest_paths_are_relative_and_not_self_referential(self) -> None:
        manifest_rel = MANIFEST.relative_to(ISAACLAB_DIR.parent).as_posix()
        for path in _manifest_paths():
            self.assertFalse(path.startswith("/"), path)
            self.assertFalse(path.startswith("./"), path)
            self.assertNotIn("//", path)
            self.assertNotIn("/../", f"/{path}/")
            self.assertNotEqual(path, manifest_rel)

    def test_every_manifest_target_is_a_regular_non_symlink_file(self) -> None:
        repo_root = ISAACLAB_DIR.parent
        for path in _manifest_paths():
            target = repo_root / path
            self.assertFalse(target.is_symlink(), path)
            self.assertTrue(target.is_file(), path)


if __name__ == "__main__":
    unittest.main()
