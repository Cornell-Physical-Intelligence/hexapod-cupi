"""Command producers: where path planning plugs into the locomotion stack.

The whole package exists to keep one boundary narrow. Planning produces
velocity commands; locomotion consumes them; neither knows anything else about
the other. :mod:`hexapod_nav.producer` defines that contract and
:mod:`hexapod_nav.waypoint` is a small worked example of satisfying it.
"""

from __future__ import annotations

from . import producer, waypoint


__all__ = ["producer", "waypoint"]
