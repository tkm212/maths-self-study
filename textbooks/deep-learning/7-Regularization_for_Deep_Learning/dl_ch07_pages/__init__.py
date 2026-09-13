"""Deep Learning Ch. 7 dashboard pages."""

from __future__ import annotations

from dl_ch07_pages.adversarial import AdversarialPage
from dl_ch07_pages.bagging import BaggingPage
from dl_ch07_pages.dropout import DropoutPage
from dl_ch07_pages.early_stopping import EarlyStoppingPage
from dl_ch07_pages.input_noise import InputNoisePage
from dl_ch07_pages.parameter_sharing import ParameterSharingPage
from dl_ch07_pages.semi_supervised_multitask import SemiSupervisedMultitaskPage
from dl_ch07_pages.tangent_distance import TangentDistancePage
from dl_ch07_pages.weight_decay import WeightDecayPage

__all__ = [
    "AdversarialPage",
    "BaggingPage",
    "DropoutPage",
    "EarlyStoppingPage",
    "InputNoisePage",
    "ParameterSharingPage",
    "SemiSupervisedMultitaskPage",
    "TangentDistancePage",
    "WeightDecayPage",
]
