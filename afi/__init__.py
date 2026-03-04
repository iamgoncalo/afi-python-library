"""
AFI — Architecture of Freedom Intelligence

A Python library for computing Freedom (F = P/D) across domains:
swarm intelligence, building performance, network analysis, and
complex adaptive systems.

F = P / D

where:
    F = Freedom (structural availability of paths)
    P = Perception (differentiation capacity)
    D = Distortion (structural resistance to traversal)
"""

__version__ = "0.1.0"
__author__ = "Gonçalo Melo"

from afi.core import freedom, perception, distortion
from afi.core.freedom import FreedomField
from afi.core.perception import Perception
from afi.core.distortion import Distortion, MultiplicativeDistortion
from afi.exploration import ExplorationExploitation
from afi.gradient import GradientLaw
from afi.convergence import ConvergenceBound

__all__ = [
    "FreedomField",
    "Perception",
    "Distortion",
    "MultiplicativeDistortion",
    "ExplorationExploitation",
    "GradientLaw",
    "ConvergenceBound",
]
