"""
StatForge: A/B Testing & Experimentation Platform.
"""
from .cuped import CUPED
from .statistical_tests import StatisticalTestEngine
from .sequential_testing import SequentialTester
from .srm_detection import SRMDetector
from .novelty_detection import NoveltyDetector
from .power_analysis import PowerAnalysis
from .multiple_testing import MultipleTestingCorrection

__all__ = [
    "CUPED",
    "StatisticalTestEngine",
    "SequentialTester",
    "SRMDetector",
    "NoveltyDetector",
    "PowerAnalysis",
    "MultipleTestingCorrection",
]
