"""How a score becomes a decision. Pure logic, no HTTP or model code, so it's easy to test.

A client can choose:
  - a threshold directly, or a named operating mode (one or the other, not both);
  - an uncertainty margin: scores within +/- margin of the threshold are labelled
    "uncertain" instead of being forced into real/fake;
  - test-time augmentation (average the image with its mirror image).
"""

from dataclasses import dataclass
from enum import Enum


class Label(str, Enum):
    REAL = "real"
    FAKE = "fake"
    UNCERTAIN = "uncertain"


class Mode(str, Enum):
    BALANCED = "balanced"
    HIGH_RECALL = "high_recall"  # catch more fakes, accept more false alarms
    HIGH_PRECISION = "high_precision"  # flag only when very sure


@dataclass(frozen=True)
class DecisionPolicy:
    threshold: float
    mode: Mode | None = None  # set when the threshold came from a preset
    uncertain_margin: float = 0.0
    tta: bool = False

    def classify(self, p_fake: float) -> Label:
        if abs(p_fake - self.threshold) < self.uncertain_margin:
            return Label.UNCERTAIN
        return Label.FAKE if p_fake >= self.threshold else Label.REAL
