"""Expose the frozen scorer through the existing evaluation import."""
from .scoring import *
from . import scoring


def __getattr__(name):
    return getattr(scoring, name)
