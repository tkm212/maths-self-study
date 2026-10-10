"""Deep Learning textbook dashboard tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tests.dashboards.support import (
    CH2_DASHBOARD,
    CH3_DASHBOARD,
    CH4_DASHBOARD,
    CH5_DASHBOARD,
    CH6_DASHBOARD,
    CH7_DASHBOARD,
    CH8_DASHBOARD,
    CH9_DASHBOARD,
    load_dashboard_module,
    prepare_chapter_import,
)

DL_DASHBOARDS = [
    pytest.param(CH2_DASHBOARD, 6, id="ch2"),
    pytest.param(CH3_DASHBOARD, 5, id="ch3"),
    pytest.param(CH4_DASHBOARD, 6, id="ch4"),
    pytest.param(CH5_DASHBOARD, 6, id="ch5"),
    pytest.param(CH6_DASHBOARD, 5, id="ch6"),
    pytest.param(CH7_DASHBOARD, 9, id="ch7"),
    pytest.param(CH8_DASHBOARD, 4, id="ch8"),
    pytest.param(CH9_DASHBOARD, 5, id="ch9"),
]


@pytest.mark.parametrize(("dashboard_path", "page_count"), DL_DASHBOARDS)
def test_dashboard_app_layout(dashboard_path, page_count):
    module = load_dashboard_module(dashboard_path)
    app = module.create_app()
    assert app.layout is not None
    assert len(module.PAGES) == page_count


def test_vectors_page_builds_filters():
    ch2 = load_dashboard_module(CH2_DASHBOARD)
    page = ch2.PAGES[0]
    filters = page.build_filters()
    assert filters is not None
    assert page.body_id == "vectors-body"


def test_ch2_vectors_filters_via_dashboard():
    ch2_dir = CH2_DASHBOARD.parent
    spec = importlib.util.spec_from_file_location(
        "ch2_vectors_filters",
        ch2_dir / "dl_ch02_pages/vectors/filters.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build_filters() is not None


def test_create_deep_learning_dashboard():
    from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

    ch2 = load_dashboard_module(CH2_DASHBOARD)
    app = create_deep_learning_dashboard(
        "test_dl_dashboard",
        chapter_number=2,
        chapter_title="Linear Algebra",
        book_slug="linear_algebra.html",
        book_link_text="Deep Learning Book — Linear Algebra",
        pages=[ch2.PAGES[0]],
        default_page=ch2.PAGES[0].value,
    )
    assert app.layout is not None
    assert app.title == "Deep Learning Ch. 2 — Linear Algebra"


def test_capacity_page_updates():
    prepare_chapter_import(CH5_DASHBOARD.parent)
    from dl_ch05_pages.capacity.content import render_body

    low = render_body(2, 0.05)
    high = render_body(10, 0.3)
    assert low is not None and high is not None
    assert str(low.to_plotly_json()) != str(high.to_plotly_json())


def test_stability_softmax_updates():
    prepare_chapter_import(CH4_DASHBOARD.parent)
    from dl_ch04_pages.stability.content import render_body

    small = render_body(0.0, 1.0, 2.0)
    large = render_body(1000.0, 1001.0, 1002.0)
    assert small is not None and large is not None
    assert str(small.to_plotly_json()) != str(large.to_plotly_json())


def test_random_variables_moments_update():
    prepare_chapter_import(CH3_DASHBOARD.parent)
    from dl_ch03_pages.random_variables.content import render_body

    low = render_body(0.1, 0.15, 0.25, 0.5, 0.1, 0.2, 0.3, 0.4)
    high = render_body(0.1, 0.15, 0.25, 0.5, 0.4, 0.3, 0.2, 0.1)
    assert low is not None and high is not None
    assert str(low.to_plotly_json()) != str(high.to_plotly_json())


def test_suggest_grid_range_scales_with_stretch():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(2)
    small = helpers.suggest_grid_range(np.array([[0.5, 0.0], [0.0, 0.5]]))
    large = helpers.suggest_grid_range(np.array([[3.0, 0.0], [0.0, 3.0]]))
    assert small > large


def test_plot_tensor_3d_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(2)
    tensor = helpers.TENSOR_DEFAULT
    fig = helpers.plot_tensor_3d(tensor, axis=2, index=0)
    assert fig is not None
    assert len(fig.data) == 2
    assert fig.layout.scene is not None


def test_plot_lp_unit_ball_l1_is_diamond():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(2)
    xs, ys = helpers._lp_unit_ball_boundary(1.0)
    assert (xs[0], ys[0]) == (1.0, 0.0)
    assert (xs[1], ys[1]) == (0.0, 1.0)
    assert len(xs) == 5


def test_plot_markov_chain_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(3)
    demo = helpers.markov_chain_demo()
    fig = helpers.plot_markov_chain(demo.p_x1, demo.p_x2_given_x1, demo.p_x3_given_x2)
    assert fig is not None
    assert len(fig.layout.annotations) >= 4


def test_plot_softmax_comparison_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(4)
    fig = helpers.plot_softmax_comparison(helpers.SOFTMAX_LOGITS, labels=helpers.SOFTMAX_LABELS)
    assert fig is not None
    assert len(fig.data) >= 2


def test_plot_gradient_descent_path_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(4)
    fig = helpers.plot_gradient_descent_path(
        helpers.GD_HESSIAN,
        helpers.GD_LINEAR,
        helpers.GD_START,
        learning_rate=0.1,
    )
    assert fig is not None
    assert len(fig.data) >= 2


def test_plot_capacity_fit_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(5)
    fig = helpers.plot_capacity_fit(helpers.CAPACITY_DEGREE)
    assert fig is not None
    assert len(fig.data) >= 3


def test_plot_sgd_paths_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(5)
    fig = helpers.plot_sgd_paths(helpers.SGD_LEARNING_RATE, helpers.SGD_BATCH_SIZE)
    assert fig is not None
    assert len(fig.data) >= 2


def test_xor_page_updates():
    prepare_chapter_import(CH6_DASHBOARD.parent)
    from dl_ch06_pages.xor.content import render_body

    shallow = render_body(2, 0.5, 500, "tanh")
    deep = render_body(12, 0.5, 8000, "relu")
    assert shallow is not None and deep is not None
    assert str(shallow.to_plotly_json()) != str(deep.to_plotly_json())


def test_plot_xor_decision_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(6)
    fig, stats = helpers.plot_xor_decision(n_hidden=4, n_epochs=500)
    assert fig is not None
    assert len(fig.data) >= 2
    assert "mse" in stats


def test_plot_universal_approximation_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(6)
    fig, stats = helpers.plot_universal_approximation(n_hidden=8)
    assert fig is not None
    assert len(fig.data) >= 2
    assert "mse" in stats


def test_backprop_page_updates():
    prepare_chapter_import(CH6_DASHBOARD.parent)
    from dl_ch06_pages.backprop.content import render_body

    tight = render_body(4, 1e-5, 0, "tanh")
    loose = render_body(8, 1e-4, 3, "relu")
    assert tight is not None and loose is not None
    assert str(tight.to_plotly_json()) != str(loose.to_plotly_json())


def test_plot_backprop_gradient_check_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(6)
    fig, stats = helpers.plot_backprop_gradient_check(n_hidden=4, epsilon=1e-5, sample_index=0)
    assert fig is not None
    assert len(fig.data) >= 1
    assert "max_rel_error" in stats


def test_weight_decay_page_updates():
    prepare_chapter_import(CH7_DASHBOARD.parent)
    from dl_ch07_pages.weight_decay.content import render_body

    low = render_body(0.0)
    high = render_body(0.15)
    assert low is not None and high is not None
    assert str(low.to_plotly_json()) != str(high.to_plotly_json())


def test_plot_weight_decay_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(7)
    fig, stats = helpers.plot_weight_decay(0.01)
    assert fig is not None
    assert len(fig.data) >= 3
    assert "val_mse" in stats


def test_plot_early_stopping_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(7)
    fig, stats = helpers.plot_early_stopping(120)
    assert fig is not None
    assert len(fig.data) >= 2
    assert "best_epoch" in stats


def test_semi_supervised_multitask_page_updates():
    prepare_chapter_import(CH7_DASHBOARD.parent)
    from dl_ch07_pages.semi_supervised_multitask.content import render_body

    few = render_body(10)
    many = render_body(60)
    assert few is not None and many is not None
    assert str(few.to_plotly_json()) != str(many.to_plotly_json())


def test_plot_semi_supervised_multitask_builds_figures():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(7)
    fig_semi, fig_multi, stats = helpers.plot_semi_supervised_multitask(25)
    assert fig_semi is not None and fig_multi is not None
    assert len(fig_multi.data) >= 1
    assert "shared_val_mse" in stats


def test_plot_parameter_sharing_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(7)
    fig_bar, fig_curve, stats = helpers.plot_parameter_sharing(3)
    assert fig_bar is not None and fig_curve is not None
    assert len(fig_curve.data) >= 2
    assert stats["conv_params"] < stats["fc_params"]


def test_plot_bagging_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(7)
    fig_bar, fig_curve, stats = helpers.plot_bagging(3)
    assert fig_bar is not None and fig_curve is not None
    assert len(fig_curve.data) >= 2
    assert "bagged_val_mse" in stats


def test_plot_adversarial_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(7)
    fig, stats = helpers.plot_adversarial(0.2)
    assert fig is not None
    assert stats["mse_adv"] >= stats["mse_clean"]


def test_plot_tangent_distance_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(7)
    fig, stats = helpers.plot_tangent_distance(0.3)
    assert fig is not None
    assert stats["tangent"] <= stats["euclidean"]


def test_momentum_page_updates():
    prepare_chapter_import(CH8_DASHBOARD.parent)
    from dl_ch08_pages.momentum.content import render_body

    slow = render_body(0.02, 0.2)
    fast = render_body(0.08, 0.95)
    assert slow is not None and fast is not None
    assert str(slow.to_plotly_json()) != str(fast.to_plotly_json())


def test_plot_momentum_paths_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(8)
    fig_path, fig_loss, stats = helpers.plot_momentum_paths(0.04, 0.9)
    assert fig_path is not None and fig_loss is not None
    assert stats["momentum_final_loss"] <= stats["gd_final_loss"] or stats["condition_number"] > 1
    assert "nesterov_final_loss" in stats
    assert stats["eta_max_gd"] > 0


def test_plot_adaptive_optimizers_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(8)
    fig_val, fig_gap, fig_lr, fig_bar, stats = helpers.plot_adaptive_optimizers(0.03)
    assert fig_val is not None and fig_gap is not None
    assert fig_lr is not None and fig_bar is not None
    assert "adam_final" in stats


def test_plot_initialization_builds_figures():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(8)
    fig_epoch, fig_sweep, stats = helpers.plot_initialization(2.0)
    assert fig_epoch is not None and fig_sweep is not None
    assert "xavier_std" in stats


def test_plot_minibatch_noise_builds_figures():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(8)
    fig_path, fig_sweep, stats = helpers.plot_minibatch_noise(8)
    assert fig_path is not None and fig_sweep is not None
    assert stats["grad_variance"] >= 0


def test_convolution_page_updates():
    prepare_chapter_import(CH9_DASHBOARD.parent)
    from dl_ch09_pages.convolution.content import render_body

    tight = render_body("identity", 1, 0)
    wide = render_body("sobel_x", 2, 2)
    assert tight is not None and wide is not None
    assert str(tight.to_plotly_json()) != str(wide.to_plotly_json())


def test_plot_convolution_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(9)
    fig_maps, fig_bar, stats = helpers.plot_convolution(1, 1, "sobel_x")
    assert fig_maps is not None and fig_bar is not None
    assert stats["output_h"] >= 1
    assert len(fig_maps.data) >= 3


def test_plot_pooling_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(9)
    fig_maps, fig_diff, stats = helpers.plot_pooling(2, 2, "max")
    assert fig_maps is not None and fig_diff is not None
    assert stats["compression"] >= 1.0


def test_motivation_page_updates():
    prepare_chapter_import(CH9_DASHBOARD.parent)
    from dl_ch09_pages.motivation.content import render_body

    tight = render_body("identity", 0, 1)
    wide = render_body("sobel_x", 3, 4)
    assert tight is not None and wide is not None
    assert str(tight.to_plotly_json()) != str(wide.to_plotly_json())


def test_plot_receptive_field_builds_figures():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(9)
    fig_rf, fig_size, stats = helpers.plot_receptive_field(3, 3)
    assert fig_rf is not None and fig_size is not None
    assert stats["conv_params"] < stats["fc_params"]


def test_plot_translation_equivariance_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(9)
    fig = helpers.plot_translation_equivariance(2, 1)
    assert fig is not None


def test_plot_cnn_tower_builds_figures():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(9)
    fig_bar, fig_params, stats = helpers.plot_cnn_tower(4, 3)
    assert fig_bar is not None and fig_params is not None
    assert stats["output_size"] < stats["input_size"]


def test_variants_page_updates():
    prepare_chapter_import(CH9_DASHBOARD.parent)
    from dl_ch09_pages.variants.content import render_body

    small = render_body(1)
    large = render_body(4)
    assert small is not None and large is not None
    assert str(small.to_plotly_json()) != str(large.to_plotly_json())


def test_plot_variants_builds_figures():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(9)
    fig_dil, fig_1x1, fig_rf, stats = helpers.plot_variants(2)
    assert fig_dil is not None and fig_1x1 is not None and fig_rf is not None
    assert stats["expanded_kernel_size"] > 3


def test_applications_page_updates():
    prepare_chapter_import(CH9_DASHBOARD.parent)
    from dl_ch09_pages.applications.content import render_body

    a = render_body(0.2, 1)
    b = render_body(2.5, 15)
    assert a is not None and b is not None
    assert str(a.to_plotly_json()) != str(b.to_plotly_json())


def test_plot_gabor_and_random_builds_figure():
    from tests.dashboards.support import load_dl_helpers

    helpers = load_dl_helpers(9)
    fig_g, fig_bar, stats = helpers.plot_gabor_and_random(1.0, 5)
    assert fig_g is not None and fig_bar is not None
    assert stats["sobel_mean_abs"] >= 0
