import pytest

from app.core.policy import DecisionPolicy, Label


@pytest.mark.parametrize(
    ("p_fake", "expected"),
    [(0.0, Label.REAL), (0.49, Label.REAL), (0.5, Label.FAKE), (1.0, Label.FAKE)],
)
def test_threshold_is_inclusive_for_fake(p_fake: float, expected: Label) -> None:
    assert DecisionPolicy(threshold=0.5).classify(p_fake) is expected


@pytest.mark.parametrize(
    ("p_fake", "expected"),
    [
        (0.39, Label.REAL),
        (0.41, Label.UNCERTAIN),
        (0.5, Label.UNCERTAIN),
        (0.59, Label.UNCERTAIN),
        (0.61, Label.FAKE),
    ],
)
def test_uncertain_band_around_threshold(p_fake: float, expected: Label) -> None:
    assert DecisionPolicy(threshold=0.5, uncertain_margin=0.1).classify(p_fake) is expected


def test_zero_margin_never_returns_uncertain() -> None:
    policy = DecisionPolicy(threshold=0.5)
    assert all(policy.classify(p / 100) is not Label.UNCERTAIN for p in range(101))
