import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # use non-interactive backend for tests

from anfis_toolbox import QuickANFIS
from anfis_toolbox.visualization import (
    plot_correlation_heatmap,
    plot_feature_histograms,
    plot_feature_vs_target,
    plot_membership_functions,
    plot_predictions_for_model,
    plot_predictions_vs_target,
    plot_residuals,
    plot_residuals_for_model,
    plot_rule_activations,
    plot_target_distribution,
    plot_training_curve,
    quick_plot_results,
    quick_plot_training,
)


def test_eda_plots_smoke():
    X = np.random.RandomState(0).randn(50, 3)
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * np.random.RandomState(1).randn(50)

    fig1 = plot_feature_histograms(X, bins=10, figsize=(6, 4))
    fig2 = plot_feature_vs_target(X, y, figsize=(8, 3))
    fig3 = plot_correlation_heatmap(X, y, figsize=(5, 5))
    # also exercise save branch for target distribution
    from pathlib import Path

    fig4 = plot_target_distribution(y, figsize=(5, 3), save_path=Path("/tmp") / "ydist.png")

    for fig in (fig1, fig2, fig3, fig4):
        assert hasattr(fig, "savefig")
    plt.close(fig)


def test_model_plots_with_arrays():
    rng = np.random.RandomState(42)
    y_true = rng.randn(100)
    y_pred = y_true + 0.1 * rng.randn(100)

    f1 = plot_training_curve([1.0, 0.7, 0.5, 0.3], figsize=(6, 3))
    f2 = plot_predictions_vs_target(y_true, y_pred, figsize=(5, 4))
    f3 = plot_residuals(y_true, y_pred, figsize=(8, 3))

    for fig in (f1, f2, f3):
        assert hasattr(fig, "savefig")
        plt.close(fig)


def test_input_validation():
    X = np.random.randn(10, 2)
    y = np.random.randn(9)
    with pytest.raises(ValueError):
        plot_feature_vs_target(X, y)


def test_feature_plots_variants(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randn(40, 3)
    y = rng.randn(40)
    # feature_histograms: extra axes, custom names, save
    fh = plot_feature_histograms(X, feature_names=["a", "b", "c"], max_cols=2, save_path=tmp_path / "fh.png")
    assert (tmp_path / "fh.png").exists() and hasattr(fh, "savefig")
    # invalid X ndim
    with pytest.raises(ValueError):
        plot_feature_histograms(X[:, 0])

    # feature_vs_target: extra axes, names, save
    X5 = rng.randn(40, 5)
    fvt = plot_feature_vs_target(
        X5,
        y,
        feature_names=[f"f{i}" for i in range(5)],
        max_cols=3,
        save_path=tmp_path / "fvt.png",
    )
    assert (tmp_path / "fvt.png").exists() and hasattr(fvt, "savefig")

    # correlation heatmap: without target and custom labels
    ch1 = plot_correlation_heatmap(
        X, y, include_target=False, feature_names=["p", "q", "r"], save_path=tmp_path / "ch1.png"
    )
    assert (tmp_path / "ch1.png").exists() and hasattr(ch1, "savefig")
    # correlation heatmap: include target but y=None (no append)
    ch2 = plot_correlation_heatmap(X, None, include_target=True, save_path=tmp_path / "ch2.png")
    assert (tmp_path / "ch2.png").exists() and hasattr(ch2, "savefig")


def test_prediction_plots_variants(tmp_path):
    rng = np.random.RandomState(1)
    y_true = rng.randn(50)
    y_pred = y_true + 0.2 * rng.randn(50)

    # save branches
    pvt = plot_predictions_vs_target(y_true, y_pred, save_path=tmp_path / "pvt.png")
    res = plot_residuals(y_true, y_pred, save_path=tmp_path / "res.png")
    assert (tmp_path / "pvt.png").exists() and (tmp_path / "res.png").exists()
    assert hasattr(pvt, "savefig") and hasattr(res, "savefig")

    # error branches: mismatched lengths
    with pytest.raises(ValueError):
        plot_predictions_vs_target(y_true[:-1], y_pred)
    with pytest.raises(ValueError):
        plot_residuals(y_true[:-1], y_pred)


def test_membership_infer_range_branches(tmp_path):
    rng = np.random.RandomState(3)
    X = rng.randn(20, 1)
    # triangular
    m_t = QuickANFIS.for_regression(X, n_mfs=2, mf_type="triangular")
    plot_membership_functions(m_t, save_path=tmp_path / "m_tri.png")
    # trapezoidal
    m_tr = QuickANFIS.for_regression(X, n_mfs=2, mf_type="trapezoidal")
    plot_membership_functions(m_tr, save_path=tmp_path / "m_trap.png")
    # bell
    m_b = QuickANFIS.for_regression(X, n_mfs=2, mf_type="bell")
    plot_membership_functions(m_b, save_path=tmp_path / "m_bell.png")
    assert (tmp_path / "m_tri.png").exists()
    assert (tmp_path / "m_trap.png").exists()
    assert (tmp_path / "m_bell.png").exists()

    # invalid input name
    with pytest.raises(ValueError):
        plot_membership_functions(m_b, input_name="does_not_exist")


def test_quick_helpers_show_flag(tmp_path):
    # show=True path; using Agg backend, this is safe
    quick_plot_training([1.0, 0.8, 0.5], save_path=tmp_path / "loss2.png", show=True)
    assert (tmp_path / "loss2.png").exists()

    rng = np.random.RandomState(4)
    X = rng.randn(12, 2)
    y = rng.randn(12)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    quick_plot_results(X, y, model, save_path=tmp_path / "res2.png", show=True)
    assert (tmp_path / "res2.png").exists()


def test_invalid_ndim_and_backend_show(monkeypatch, tmp_path):
    rng = np.random.RandomState(5)
    X = rng.randn(10)
    y = rng.randn(10)
    # invalid ndim for feature_vs_target and correlation_heatmap
    with pytest.raises(ValueError):
        plot_feature_vs_target(X, y)
    with pytest.raises(ValueError):
        plot_correlation_heatmap(X, y)

    # simulate non-Agg backend and stub plt.show to avoid GUI
    monkeypatch.setattr("matplotlib.get_backend", lambda: "MacOSX", raising=False)
    called = {"count": 0}

    def _fake_show():
        called["count"] += 1

    monkeypatch.setattr(plt, "show", _fake_show)

    # quick helpers should call show now
    quick_plot_training([1.0, 0.5], save_path=tmp_path / "loss3.png", show=True)
    model = QuickANFIS.for_regression(rng.randn(12, 2), n_mfs=2)
    quick_plot_results(rng.randn(12, 2), rng.randn(12), model, save_path=tmp_path / "res3.png", show=True)
    assert called["count"] >= 2
    assert (tmp_path / "loss3.png").exists()
    assert (tmp_path / "res3.png").exists()


def test_infer_range_else_branch_sigmoidal(tmp_path):
    rng = np.random.RandomState(6)
    X = rng.randn(25, 1)
    m = QuickANFIS.for_regression(X, n_mfs=2, mf_type="sigmoidal")
    fig = plot_membership_functions(m, save_path=tmp_path / "m_sig.png")
    assert (tmp_path / "m_sig.png").exists()
    plt.close(fig)


def test_training_curve_save_branch(tmp_path):
    fig = plot_training_curve([1.0, 0.9, 0.7], save_path=tmp_path / "tc.png")
    assert (tmp_path / "tc.png").exists()
    plt.close(fig)


def test_visualizer_wrappers_cover_branches(tmp_path):
    import matplotlib as mpl

    mpl.rcParams["figure.max_open_warning"] = 0
    rng = np.random.RandomState(7)
    X = rng.randn(20, 2)
    y = rng.randn(20)
    model = QuickANFIS.for_regression(X, n_mfs=2)

    from anfis_toolbox.visualization import ANFISVisualizer

    viz = ANFISVisualizer(model)
    figs = []
    figs.append(viz.plot_feature_histograms(X))
    figs.append(viz.plot_feature_vs_target(X, y))
    figs.append(viz.plot_correlation_heatmap(X, y))
    figs.append(viz.plot_target_distribution(y))
    figs.append(viz.plot_training_curves([1.0, 0.8, 0.6]))
    figs.append(viz.plot_prediction_vs_target(X, y))
    figs.append(viz.plot_residuals(X, y))

    # 1D function approximation happy path
    X1 = rng.randn(15, 1)
    y1 = rng.randn(15)
    model1 = QuickANFIS.for_regression(X1, n_mfs=2)
    viz1 = ANFISVisualizer(model1)
    figs.append(viz1.plot_1d_function_approximation(X1, y1, n_points=50))
    # error path for 1D plot
    with pytest.raises(ValueError):
        viz.plot_1d_function_approximation(X, y)

    for f in figs:
        assert hasattr(f, "savefig")
        plt.close(f)


def test_anfis_internals_plots_smoke(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randn(20, 2)
    model = QuickANFIS.for_regression(X, n_mfs=2)

    f1 = plot_membership_functions(model, num_points=100)
    f2 = plot_membership_functions(model, input_name="x1", save_path=tmp_path / "mfs.png", figsize=(6, 3))
    f3 = plot_rule_activations(model, X, sample_idx=0, save_path=tmp_path / "rules.png", figsize=(8, 3))
    # also call without save_path to hit else branch
    f4 = plot_rule_activations(model, X, sample_idx=1)

    assert hasattr(f1, "savefig") and hasattr(f2, "savefig") and hasattr(f3, "savefig")
    assert (tmp_path / "mfs.png").exists()
    assert (tmp_path / "rules.png").exists()

    plt.close(f1)
    plt.close(f2)
    plt.close(f3)
    plt.close(f4)
    # error path: sample_idx out of range
    with pytest.raises(ValueError):
        plot_rule_activations(model, X, sample_idx=999)


def test_model_convenience_plots_smoke(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.randn(30, 3)
    y = rng.randn(30)
    model = QuickANFIS.for_regression(X, n_mfs=2)

    fp = tmp_path / "pred.png"
    fr = tmp_path / "resid.png"
    f1 = plot_predictions_for_model(model, X, y, save_path=fp, figsize=(6, 4))
    f2 = plot_residuals_for_model(model, X, y, save_path=fr, figsize=(7, 3))

    assert hasattr(f1, "savefig") and hasattr(f2, "savefig")
    assert fp.exists() and fr.exists()


def test_quick_helpers_save(tmp_path):
    f = quick_plot_training([1.0, 0.5, 0.25], save_path=tmp_path / "loss.png", show=True)
    assert hasattr(f, "savefig")
    assert (tmp_path / "loss.png").exists()

    # quick results
    rng = np.random.RandomState(2)
    X = rng.randn(15, 2)
    y = rng.randn(15)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    f2 = quick_plot_results(X, y, model, save_path=tmp_path / "res.png", show=True)
    assert hasattr(f2, "savefig")
    assert (tmp_path / "res.png").exists()
    plt.close(f)
    plt.close(f2)


def test_quick_helpers_no_save_no_show():
    # default backend is Agg from module-level setting
    f = quick_plot_training([1.0, 0.8], show=False)
    assert hasattr(f, "savefig")
    plt.close(f)
    rng = np.random.RandomState(8)
    X = rng.randn(10, 2)
    y = rng.randn(10)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    f2 = quick_plot_results(X, y, model, show=False)
    assert hasattr(f2, "savefig")
    plt.close(f2)
