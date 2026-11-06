"""Interactive research dashboard. Run with `streamlit run streamlit_app.py`."""
from pathlib import Path
import io
import json
import sys
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import dashboard_data as data

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="ECG | Cross-domain research",page_icon="??",layout="wide")
st.markdown("<style>"+(ROOT/"dashboard.css").read_text()+"</style>",unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_table(run,name): return data.table(Path(run),name)

@st.cache_data(show_spinner=False)
def read_predictions(run,tag,split): return data.predictions(Path(run),tag,split)

@st.cache_resource(show_spinner="Loading the trained ECG ensemble?")
def predictor(run): return data.load_predictor(Path(run))


def heading(kicker,title,description):
    st.markdown(f'<div class="eyebrow">{kicker}</div>',unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="deck">{description}</div>',unsafe_allow_html=True)


def show_table(frame,filename):
    st.dataframe(frame,hide_index=True,width="stretch")
    st.download_button("Download this table",frame.to_csv(index=False),file_name=filename,mime="text/csv",key=filename)


def chart(frame,x,y,color=None,title=None):
    enc=dict(x=alt.X(x),y=alt.Y(y),tooltip=list(frame.columns))
    if color: enc["color"]=alt.Color(color,scale=alt.Scale(scheme="tableau10"))
    c=alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(**enc).properties(height=300)
    if title:c=c.properties(title=title)
    st.altair_chart(c,width="stretch")


runs=data.discover_runs(ROOT)
if not runs:
    st.info("The public repository contains the study code and poster. The ECG recordings, record-level results, and trained weights are kept outside GitHub. To explore the dashboard, place a locally generated ecg_results_* export beside streamlit_app.py.")
    st.stop()
with st.sidebar:
    st.markdown("### ECG / Research lab")
    st.caption("CROSS-DOMAIN RHYTHM STUDY")
    selected=st.selectbox("Experiment run",[p.name for p in runs],key="run")
    section=st.radio("Explore the study",data.SECTIONS,key="section")
    st.divider()
    st.caption("AF versus explicit sinus rhythm. Research use only.")
run=next(p for p in runs if p.name==selected)
meta=data.metadata(run)
metrics=data.operating_metrics(run)
ensemble="Georgia_selected_ensemble"
active=meta.get("operating_point","youden")
external=metrics[(metrics.tag==ensemble)&(metrics.split=="external")&(metrics.operating_point==active)].iloc[0]

if section=="Study overview":
    heading("01 / Study overview","Can an ECG model travel?","Explore how an AF-versus-sinus-rhythm classifier performs when the evaluation dataset comes from a different hospital domain.")
    cols=st.columns(4)
    for c,label,value in zip(cols,["Georgia accuracy","AF sensitivity","Specificity","External recordings"],[f"{external.accuracy:.2%}",f"{external.sensitivity:.2%}",f"{external.specificity:.2%}",f"{int(external.n):,}"]): c.metric(label,value)
    st.caption("Selected ensemble ? Georgia external evaluation ? frozen operating point: "+active)
    left,right=st.columns([1.4,1])
    with left:
        st.subheader("One question, three dataset domains")
        st.write("Train and select models using PTB-XL and Shaoxing, calibrate using separate source partitions, then evaluate on Georgia. This completed pilot tests one external domain; the complete rotation study has not been executed.")
        counts=read_table(str(run),"class_counts_after_duplicate_audit.csv")
        long=counts.melt("domain",var_name="Rhythm",value_name="Recordings")
        chart(long,"domain:N","Recordings:Q","Rhythm:N")
    with right:
        st.subheader("What the experiment compares")
        st.write("ResNet and InceptionTime; empirical risk minimization, class-conditional alignment and CORAL; source calibration, uncertainty review and synthetic noise stress.")
        diag=meta.get("source_operating_diagnostics")
        if diag and not diag["target_met"]:
            st.warning(f"Source sensitivity goal unmet: {diag['achieved_min_source_sensitivity']:.2%} minimum achieved versus {diag['desired_sensitivity']:.0%} requested. Georgia sensitivity is {external.sensitivity:.2%}.")
        if "exploratory_revision" in json.loads((run/"config.json").read_text()): st.info("This revision followed inspection of previous Georgia results. Repeat Georgia evaluation is exploratory.")
        st.caption("Strict sinus rhythm does not imply a healthy control. Georgia and Shaoxing patient identifiers are record-level proxies; near duplicates may remain.")
    st.subheader("Experiment provenance")
    st.write(f"Run: `{run.name}` ? {len(meta['members'])} exported ensemble members")
    st.caption("Frozen protocol SHA256: "+meta["protocol_sha256"])
    with st.expander("Read the full experiment report"):
        st.markdown((run/"research_report.md").read_text(encoding="utf-8"))

elif section=="ECG prediction":
    heading("02 / ECG prediction","Inspect a rhythm prediction.","Upload a 10-second numerical ECG and run the trained ensemble. The study model distinguishes AF from explicit sinus rhythm.")
    st.info("Research demonstration. Other rhythms are outside the model's validated task; every accepted signal still receives an AF or SR label.")
    st.write("**Input:** a `.npy` array in millivolts, with rows in this order: "+", ".join(data.LEADS)+".")
    fs=st.selectbox("Sampling frequency (Hz)",[100,500],key="fs")
    st.caption(f"Required shape: 12 ? {fs*10:,} (leads ? samples). ECG images and PDFs are not supported. Uploads are processed in memory and are not saved by this app.")
    upload=st.file_uploader("Upload ECG array",type=["npy"])
    if upload is None:
        st.caption("The existing results remain available in the other sections without uploading a signal.")
    else:
        try:
            x=data.read_upload(upload,fs)
            chosen=st.multiselect("Waveform leads",data.LEADS,default=["II"])
            if chosen:
                stride=max(1,fs//100)
                wave=pd.DataFrame({lead:x[data.LEADS.index(lead),::stride] for lead in chosen})
                wave["Time (seconds)"]=np.arange(len(wave))*stride/fs
                st.line_chart(wave,x="Time (seconds)",y=chosen,x_label="Time (seconds)",y_label="Amplitude (mV)")
            if st.button("Analyze ECG",type="primary"):
                try:
                    result=predictor(str(run)).predict_array(x,fs)
                except ModuleNotFoundError as exc:
                    st.error(f"Inference dependency unavailable ({exc.name}) in the Python environment running this app. Stop this server and launch run_dashboard.cmd from the project folder.")
                    st.caption(f"Current Python: {sys.executable}")
                    st.code(r".\.venv\Scripts\python.exe -m streamlit run streamlit_app.py", language="powershell")
                else:
                    cols=st.columns(3)
                    cols[0].metric("Estimated AF probability",f"{result['p_af']:.2%}")
                    cols[1].metric("Predicted rhythm",result["label"])
                    cols[2].metric("Uncertainty review","Flagged" if result["review"] else "Not flagged")
                    st.caption(f"Frozen classification threshold: {meta['threshold']:.4f}. Entropy: {result['entropy']:.4f}; ensemble variance: {result['variance']:.6f}. A low uncertainty score does not establish correctness.")
                    st.download_button("Download prediction",json.dumps(dict(run=run.name,fs=fs,threshold=meta["threshold"],**result),indent=2),"ecg_prediction.json","application/json")
        except (ValueError,TypeError,EOFError,OSError) as exc: st.error(str(exc))

elif section=="Cross-domain performance":
    heading("03 / Generalization","Follow the domain shift.","Compare held-out source test recordings with the external Georgia evaluation. Dataset prevalence and cohort composition differ, so accuracy alone does not describe transfer.")
    tags=list(metrics.tag.unique())
    tag=st.selectbox("Model",tags,index=tags.index(ensemble),key="domain_model")
    point=st.selectbox("Operating point",list(metrics.operating_point.unique()),key="domain_point")
    view=metrics[(metrics.tag==tag)&(metrics.operating_point==point)].copy()
    measure=st.selectbox("Metric",["auroc","auprc","sensitivity","specificity","accuracy","f1","brier","ece"],key="domain_measure")
    chart(view,"domain:N",measure+":Q",title="Held-out performance by dataset domain")
    show_table(view,"cross_domain_metrics.csv")
    if tag==ensemble:
        st.subheader("95% patient bootstrap intervals")
        st.caption("Intervals below apply to the ensemble's frozen active operating point ("+active+"). Conditional on fitted models; Georgia/Shaoxing clusters use patient proxies.")
        show_table(read_table(str(run),"patient_bootstrap_CI.csv"),"ensemble_confidence_intervals.csv")
    st.caption("AUROC/AUPRC: higher is better. Brier/ECE: lower is better. Comparing domains is descriptive and does not identify the cause of performance changes.")

elif section=="Model comparison":
    heading("04 / Model comparison","Compare the approaches.","Inspect every completed model rather than assuming domain alignment or ensembling improves all performance measures.")
    point=st.selectbox("Operating point",list(metrics.operating_point.unique()),key="compare_point")
    measure=st.selectbox("Comparison metric",["auroc","auprc","sensitivity","specificity","accuracy","f1","brier","ece"],key="compare_measure")
    view=metrics[(metrics.split=="external")&(metrics.operating_point==point)].copy()
    chart(view,measure+":Q","tag:N",title="Georgia external evaluation")
    show_table(view,"model_comparison.csv")
    st.subheader("Paired ensemble minus ERM differences")
    st.caption("Saved 95% bootstrap intervals at the active frozen operating point. An interval crossing zero does not establish an improvement. Negative differences favor the ensemble only for Brier and ECE.")
    show_table(read_table(str(run),"paired_bootstrap_differences.csv"),"paired_differences.csv")
    with st.expander("Training specifications and stopping behavior"):
        show_table(read_table(str(run),"training_summary.csv"),"training_summary.csv")
    with st.expander("Selected specification: variation across training seeds"):
        st.code((run/"tables/seed_variability.csv").read_text(),language="csv")
        st.caption("Descriptive seed variation; the first seed also participated in specification selection.")

elif section=="Calibration & uncertainty":
    heading("05 / Reliability","Look beyond the label.","Inspect probability calibration, false negatives and uncertainty review using saved ensemble predictions at the frozen decision threshold.")
    split=st.selectbox("Evaluation split",["external","source_test"],key="reliability_split")
    pred=read_predictions(str(run),ensemble,split)
    domain=st.selectbox("Dataset domain",list(pred.domain.unique()),key="reliability_domain")
    pred=pred[pred.domain==domain].copy()
    threshold=float(pred.threshold.iloc[0])
    left,right=st.columns(2)
    with left:
        st.subheader("Calibration")
        bins=data.calibration_bins(pred)
        base=alt.Chart(bins).encode(x=alt.X("mean_probability:Q",scale=alt.Scale(domain=[0,1]),title="Mean estimated AF probability"),y=alt.Y("observed_af:Q",scale=alt.Scale(domain=[0,1]),title="Observed AF fraction"),tooltip=["mean_probability","observed_af","n"])
        ideal=alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_line(strokeDash=[5,5],color="#89969d").encode(x="x:Q",y="y:Q")
        st.altair_chart(ideal+base.mark_line(color="#087e83")+base.mark_circle(color="#087e83",size=65),width="stretch")
        st.caption("15 equal-width probability bins. Bin counts appear on hover; the dashed diagonal indicates ideal calibration.")
    with right:
        st.subheader("Confusion matrix")
        cm=data.confusion(pred,threshold)
        st.dataframe(pd.DataFrame(cm,index=["Actual SR","Actual AF"],columns=["Predicted SR","Predicted AF"]),width="stretch")
        st.metric("Missed AF cases",f"{cm[1][0]} / {cm[1][0]+cm[1][1]}")
        st.caption(f"Frozen threshold: {threshold:.4f}. Rows are actual rhythm, columns predicted rhythm.")
    st.subheader("Uncertainty review")
    review=pred.assign(error=(pred.p_af>=threshold)!=pred.label).groupby("review",as_index=False).agg(n=("error","size"),error_rate=("error","mean"))
    show_table(review,"uncertainty_review.csv")
    rc=read_table(str(run),f"risk_coverage_Georgia_selected_ensemble_{split}.csv")
    if split=="external" and not rc.empty:
        st.line_chart(rc,x="coverage",y="risk",x_label="Fraction retained (lowest entropy first)",y_label="Error rate among retained recordings")
        st.caption("Risk?coverage is retrospective: labels are needed to measure error. Source-calibrated review thresholds do not force a fixed external review rate.")
    cal=read_table(str(run),"source_calibration_operating_points.csv")
    if not cal.empty:
        with st.expander("Source operating-point constraints"):
            show_table(cal[cal.tag==ensemble],"source_calibration_constraints.csv")
    if st.checkbox("Show saved ROC / precision-recall figure"):
        st.image(str(run/"figures/Georgia_roc_pr_risk.png"),caption="Saved experiment figure at the frozen operating point.")

elif section=="Robustness":
    heading("06 / Robustness","Test the edges of performance.","Explore synthetic signal perturbations of the first selected model and descriptive subgroup results for the ensemble on Georgia.")
    stress=data.stress_comparison(run)
    st.caption("Noise stress model: "+stress.tag.iloc[0]+". The clean baseline uses the same individual model and frozen threshold.")
    metric=st.selectbox("Stress-test metric",["auroc","sensitivity","specificity","accuracy","f1","brier","ece"],key="stress_metric")
    chart(stress,"condition:N",metric+":Q")
    show_table(stress,"synthetic_noise_stress.csv")
    st.caption("Synthetic noise and baseline wander are sensitivity analyses, not real-device or prospective validation. Added noise can occasionally improve a point metric.")
    st.subheader("Ensemble subgroup performance")
    groups=read_table(str(run),"subgroups_Georgia.csv")
    group=st.selectbox("Subgroup variable",list(groups.group.unique()),key="subgroup")
    sub=groups[groups.group==group]
    chart(sub,"value:N","sensitivity:Q")
    show_table(sub,"subgroup_performance.csv")
    st.caption("High-frequency signal energy is a quality proxy, not an expert quality annotation. Small subgroups and differing prevalence limit comparisons.")

st.divider()
st.caption(f"{run.name} ? AF versus explicit SR ? Georgia external evaluation ? Research demonstration")
