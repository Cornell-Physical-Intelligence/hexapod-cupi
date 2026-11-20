"""Waypoint command producers that consume shared contracts.

Import the module you need, such as ``navigation.waypoint``. The package does not import its
modules on its own: ``grid``, ``region``, ``graph`` and ``planner`` need shapely, and the native
evaluation container, which runs the waypoint follower, has no shapely.
"""
