from pathlib import Path
import io
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
import dashboard_data as data

ROOT = Path(__file__).resolve().parents[1]


def require_local_results():
    runs = data.discover_runs(ROOT)
    if len(runs) < 2:
        pytest.skip("Two local experiment bundles are needed for integration tests")
    return runs


def test_real_results_and_operating_point():
    runs = require_local_results()
    assert len(runs) == 2
    run = runs[0]
    assert run.name.endswith("205834")
    frame = data.predictions(run, "Georgia_selected_ensemble", "external")
    # Saved run 20261027_205834: the poster's Georgia confusion counts (SR, AF order).
    assert data.confusion(frame, float(frame.threshold.iloc[0])) == [[1709, 6], [59, 446]]
    metrics = data.operating_metrics(run)
    assert set(metrics.operating_point) == {"source_sensitivity", "youden"}
    assert set(data.operating_metrics(runs[1]).operating_point) == {"youden"}


def test_calibration_retains_every_record():
    frame = data.predictions(require_local_results()[0], "Georgia_selected_ensemble", "external")
    bins = data.calibration_bins(frame)
    assert bins.n.sum() == len(frame)
    assert bins.mean_probability.between(0, 1).all()
    assert bins.observed_af.between(0, 1).all()


@pytest.mark.parametrize("shape,fs", [((1000,12),100), ((12,999),100), ((12,1000),250)])
def test_rejects_wrong_shape_and_sampling(shape, fs):
    with pytest.raises(ValueError):
        data.validate_ecg(np.ones(shape), fs)


def test_rejects_bad_signals_and_pickle():
    x = np.random.default_rng(42).normal(0,.1,(12,1000))
    assert data.validate_ecg(x,100).shape == (12,1000)
    for bad in [np.zeros_like(x), x * np.nan, x * 10000, x.astype(complex)]:
        with pytest.raises(ValueError): data.validate_ecg(bad,100)
    payload=io.BytesIO(); np.save(payload,np.array([object()],dtype=object));payload.seek(0)
    with pytest.raises(ValueError): data.read_upload(payload,100)


@pytest.mark.parametrize("run_index", [0,1])
def test_dashboard_views(run_index):
    require_local_results()
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception
    if run_index:
        at.selectbox(key="run").select_index(run_index).run()
    for page in data.SECTIONS:
        at.radio(key="section").set_value(page).run()
        assert not at.exception, (page, list(at.exception))
    at.radio(key="section").set_value("Model comparison").run()
    assert len(at.dataframe) >= 1


def test_stress_uses_the_tested_member_not_the_ensemble():
    run=require_local_results()[0]
    frame=data.stress_comparison(run)
    assert frame.tag.nunique()==1
    assert frame.tag.iloc[0]==data.metadata(run)["members"][0]["tag"]
    clean=frame[frame.condition=="clean"].iloc[0]
    # Saved stress-comparison table for the selected member: 0.968018..., not ensemble 0.970720....
    assert clean.accuracy==pytest.approx(.968018018018018)
    assert clean.accuracy!=pytest.approx(.9707207207207207)


def test_uploaded_array_header_limits_allocation():
    payload=io.BytesIO()
    np.lib.format.write_array_header_1_0(payload,dict(descr="<f8",fortran_order=False,shape=(12,10**12)))
    payload.seek(0)
    with pytest.raises(ValueError): data.read_upload(payload,100)


@pytest.mark.parametrize("run_index",[0,1])
def test_frozen_weights_inference(run_index):
    require_local_results()
    torch=pytest.importorskip("torch")
    pytest.importorskip("wfdb")
    torch.set_num_threads(2)
    run=data.discover_runs(ROOT)[run_index]
    model=data.load_predictor(run)
    assert len(model.members)==len(data.metadata(run)["members"])
    for fs in (100,500):
        x=np.random.default_rng(42).normal(0,.1,(12,fs*10))
        result=model.predict_array(x,fs)
        assert 0<=result["p_af"]<=1
        assert np.isfinite(result["entropy"])
        assert result["label"]==("AF" if result["p_af"]>=data.metadata(run)["threshold"] else "SR")


def test_prediction_ui_with_real_weights(monkeypatch):
    require_local_results()
    torch=pytest.importorskip("torch")
    pytest.importorskip("wfdb")
    torch.set_num_threads(2)
    payload=io.BytesIO()
    np.save(payload,np.random.default_rng(42).normal(0,.1,(12,1000)))
    monkeypatch.setattr("streamlit.file_uploader",lambda *a,**kw:payload)
    at=AppTest.from_file(str(ROOT/"streamlit_app.py"),default_timeout=45).run()
    at.radio(key="section").set_value("ECG prediction").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    assert [m.label for m in at.metric]==["Estimated AF probability","Predicted rhythm","Uncertainty review"]
