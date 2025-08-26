import numpy as np

from anfis_toolbox import ANFISClassifier
from anfis_toolbox.membership import GaussianMF


def _make_simple_clf(n_inputs: int = 1, n_mfs: int = 2, n_classes: int = 2) -> ANFISClassifier:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)][:n_mfs]
    return ANFISClassifier(input_mfs, n_classes=n_classes, random_state=0)


def test_classifier_apply_membership_gradients_private_helper():
    clf = _make_simple_clf(n_inputs=1, n_mfs=2, n_classes=2)
    X = np.array([[-0.5], [0.7]])
    # Create gradients by simulating a backward step with dummy dL/dlogits
    logits = clf.forward(X)
    dL_dlogits = np.ones_like(logits) / logits.shape[0]
    clf.backward(dL_dlogits)
    params_before = clf.get_parameters()
    clf._apply_membership_gradients(learning_rate=0.01)
    params_after = clf.get_parameters()
    # Expect some membership parameter to change
    changed = False
    for name in params_before["membership"]:
        for i, mf_before in enumerate(params_before["membership"][name]):
            mf_after = params_after["membership"][name][i]
            if not (
                np.isclose(mf_before["mean"], mf_after["mean"]) and np.isclose(mf_before["sigma"], mf_after["sigma"])
            ):
                changed = True
                break
        if changed:
            break
    assert changed
