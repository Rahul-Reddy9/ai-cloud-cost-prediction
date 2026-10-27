"""Streamlit dashboard (Part H):  streamlit run src/dashboard/app.py

Five sections: forecasting, anomaly monitoring, resource allocation, cost & performance, experiment
comparison. It only visualises files produced by the pipeline (results/, data/processed/); it never
trains or changes anything.  Simulated cost / latency are NOT cloud invoices or load-test measurements.
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

R = lambda p: ROOT / p

st.set_page_config(page_title="AI Cloud Cost Prediction", layout="wide")
st.title("AI Cloud Cost Prediction & Intelligent Resource Allocation")
st.caption("Research prototype on a SIMULATOR with SYNTHETIC workload data - results are not cloud invoices "
           "and say nothing about real HPA performance until validated on a testbed.")

PAGES = ["1. Workload forecasting", "2. Anomaly monitoring", "3. Resource allocation", "4. Cost and performance",
         "5. Experiment comparison"]
page = st.sidebar.radio("Section", PAGES)


@st.cache_data
def load_csv(p):
    return pd.read_csv(p, parse_dates=[c for c in ("timestamp",) if c in pd.read_csv(p, nrows=1).columns])


def need(path: str, hint: str):
    if not R(path).exists():
        st.warning(f"`{path}` not found. {hint}")
        return False
    return True


def steps_files():
    return sorted(Path(p).name[6:-8] for p in glob.glob(str(R("results/logs/steps_*.parquet"))))


if page == PAGES[0]:
    st.header("Workload forecasting (TFT)")
    if need("results/metrics/tft_test_predictions.csv", "Run `python -m src.forecasting.train_tft`."):
        d = load_csv(R("results/metrics/tft_test_predictions.csv"))
        c1, c2 = st.columns(2)
        g = c1.selectbox("Service segment", sorted(d["group_id"].unique()))
        h = c2.slider("Forecast horizon step (x5 min)", 1, int(d["horizon_step"].max()), 1)
        x = d[(d["group_id"] == g) & (d["horizon_step"] == h)].sort_values("timestamp")
        n = st.slider("Points to show", 100, len(x), min(600, len(x)))
        x = x.iloc[:n]
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(x["timestamp"], x["actual"], "k", lw=1, label="actual")
        ax.plot(x["timestamp"], x["median"], lw=1, label="TFT median")
        ax.fill_between(x["timestamp"], x["lower"], x["upper"], alpha=0.25, label="10-90% interval")
        ax.set_ylabel("requests/s"); ax.legend(ncol=3)
        st.pyplot(fig)
        err = (x["median"] - x["actual"])
        a, b, c = st.columns(3)
        a.metric("MAE (shown)", f"{err.abs().mean():.1f}"); b.metric("RMSE (shown)", f"{np.sqrt((err**2).mean()):.1f}")
        c.metric("Interval coverage", f"{((x['actual'] >= x['lower']) & (x['actual'] <= x['upper'])).mean():.1%}")
        fe = R("results/metrics/forecast_eval.json")
        if fe.exists():
            j = json.load(open(fe))
            st.subheader("Test-set forecast comparison")
            st.dataframe(pd.DataFrame({k: {"MAE": v["mae"], "RMSE": v["rmse"]} for k, v in j["methods"].items()}).T)

elif page == PAGES[1]:
    st.header("Anomaly monitoring (Isolation Forest)")

    if need("data/processed/anomaly_scores.parquet",
             "Run `python -m src.anomaly.train_detector`."):

        s = pd.read_parquet(R("data/processed/anomaly_scores.parquet"))

        c1, c2 = st.columns(2)
        sp = c1.selectbox("Split", ["test", "validation", "train"])
        g = c2.selectbox("Segment", sorted(s["group_id"].unique()))

        x = s[(s["split"] == sp) & (s["group_id"] == g)]
        x = x.sort_values("time_idx").iloc[:800].copy()

        if x.empty:
            st.warning("No data available for this split and segment.")
        else:
            # Summary
            st.subheader("Anomaly summary")

            if "anomaly_label" in x.columns:
                labels = x["anomaly_label"].fillna(0).astype(int)
                anomaly_count = int(labels.eq(1).sum())
                anomaly_share = labels.eq(1).mean()
            else:
                labels = None
                anomaly_count = 0
                anomaly_share = 0.0

            a, b, c = st.columns(3)
            a.metric("Data points analysed", f"{len(x):,}")
            b.metric("Synthetic anomaly labels", f"{anomaly_count:,}")
            c.metric("Labelled anomaly share", f"{anomaly_share:.1%}")

            # Chart
            st.subheader("Anomaly scores")

            fig, ax = plt.subplots(figsize=(11, 3.5))
            ax.plot(
                x["timestamp"], x["z_raw"],
                lw=0.8, label="Raw-metric score"
            )

            if x["z_resid"].notna().any():
                ax.plot(
                    x["timestamp"], x["z_resid"],
                    lw=0.8, label="Forecast-residual score"
                )

            if labels is not None:
                for t in x.loc[labels.eq(1), "timestamp"]:
                    ax.axvspan(
                        t,
                        t + pd.Timedelta(minutes=5),
                        color="red",
                        alpha=0.15,
                        lw=0
                    )

            ax.legend()
            ax.set_ylabel("Normalised score")
            ax.set_xlabel("Timestamp")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Red bands show SYNTHETIC injected anomaly labels. "
                "They do not establish real production incidents."
            )

            # Evaluation metrics
            st.subheader("Detector evaluation")

            eval_path = R("results/metrics/detector_eval.json")

            if eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    evaluation = json.load(f)

                rows = []
                for key, label in [
                    ("raw_metrics", "Raw metrics"),
                    ("forecast_residuals", "Forecast residuals"),
                    ("union_of_both", "Union of both"),
                ]:
                    values = evaluation.get(key, {})
                    rows.append({
                        "Detector": label,
                        "Precision": values.get("precision"),
                        "Recall": values.get("recall"),
                        "F1 Score": values.get("f1"),
                        "False Positive Rate": values.get("fpr"),
                        "AUROC": values.get("auroc"),
                        "True Positives": values.get("tp"),
                        "False Positives": values.get("fp"),
                        "False Negatives": values.get("fn"),
                    })

                evaluation_df = pd.DataFrame(rows)
                numeric_cols = [
                    "Precision", "Recall", "F1 Score",
                    "False Positive Rate", "AUROC"
                ]
                evaluation_df[numeric_cols] = evaluation_df[numeric_cols].round(3)

                st.dataframe(evaluation_df, hide_index=True, width="stretch")

                st.caption(
                    f"Evaluation split: {evaluation.get('split', 'unknown')} | "
                    f"Samples: {evaluation.get('n', 0):,} | "
                    f"Synthetic anomalies: {evaluation.get('n_anomalous', 0):,}"
                )

                st.info(
                    "These metrics use synthetic anomaly labels and do not "
                    "establish real-world production detection performance."
                )
            else:
                st.warning("Detector evaluation file not found.")


elif page == PAGES[2]:
    st.header("Resource allocation (replicas, capacity, scaling actions, provisioning delay)")
    files = steps_files()
    if not files:
        st.warning("No step logs yet. Run `python scripts/run_all_experiments.py`.")
    else:
        rid = st.selectbox("Run", files, index=0)
        s = pd.read_parquet(R(f"results/logs/steps_{rid}.parquet"))
        tr = st.selectbox("Trace", sorted(s["trace"].unique()))
        x = s[s["trace"] == tr].iloc[:700]
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        ax[0].plot(x["timestamp"], x["demand"] / 100, "k", lw=1, label="demand / per-replica capacity")
        ax[0].step(x["timestamp"], x["ready"], where="post", label="ready replicas")
        ax[0].step(x["timestamp"], x["ready"] + x["pending"], where="post", ls="--", label="ready + pending")
        ax[0].legend(ncol=3, fontsize=8); ax[0].set_ylabel("replicas")
        ax[1].bar(x["timestamp"], x["applied_delta"], width=0.003, color="tab:red"); ax[1].set_ylabel("applied scaling action")
        st.pyplot(fig)
        a, b, c = st.columns(3)
        a.metric("Scaling events", int((x["applied_delta"] != 0).sum())); b.metric("Max pending", int(x["pending"].max()))
        c.metric("Safety interventions", int(x["safety_intervened"].sum()))

elif page == PAGES[3]:
    st.header("Cost and performance")
    if need("results/metrics/all_runs.csv", "Run `python -m src.evaluation.compare_results`."):
        r = pd.read_csv(R("results/metrics/all_runs.csv"))
        delay = st.selectbox("Provisioning delay (steps)", sorted(r["delay_steps"].unique()), index=list(sorted(r["delay_steps"].unique())).index(3) if 3 in set(r["delay_steps"]) else 0)
        x = r[(r["split"] == "test") & (r["delay_steps"] == delay)].groupby("policy").mean(numeric_only=True)
        cols = ["total_cost", "cost_reduction_pct", "p95_latency_ms", "p99_latency_ms", "slo_violation_rate_intervals", "scaling_events_per_day", "underprovisioning_fraction"]
        st.dataframe(x[cols].style.format("{:.3f}"))
        fig, ax = plt.subplots(figsize=(6, 4))
        for p, row in x.iterrows():
            ax.scatter(row["total_cost"], row["slo_violation_rate_intervals"] * 100); ax.annotate(p, (row["total_cost"], row["slo_violation_rate_intervals"] * 100), fontsize=8)
        ax.set_xlabel("simulated cost per trace (USD)"); ax.set_ylabel("SLO-violating intervals (%)"); st.pyplot(fig)
        st.caption("Lower cost must be read together with latency and SLO violations.")

else:
    st.header("Experiment comparison")
    tabs = {"Exp 2 - predictive vs reactive": "table_exp2_baselines", "Exp 3 - uncertainty": "table_exp3_uncertainty",
            "Exp 4 - pending capacity": "table_exp4_pending", "Exp 5 - anomaly signal": "table_exp5_anomaly"}
    for title, name in tabs.items():
        p = R(f"results/metrics/{name}.csv")
        if p.exists():
            st.subheader(title)
            df = pd.read_csv(p, index_col=0)
            st.dataframe(df)
            st.download_button(f"Download {name}.csv", df.to_csv().encode(), file_name=f"{name}.csv")
    p = R("results/metrics/table_exp4_delay_sensitivity.csv")
    if p.exists():
        st.subheader("Exp 4 - delay sensitivity"); st.dataframe(pd.read_csv(p))
    m = R("results/metrics/experiments_manifest.json")
m = R("results/metrics/experiments_manifest.json")
if m.exists():
    with open(m, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    st.subheader("Experiment configuration")

    c1, c2, c3 = st.columns(3)
    c1.metric("Default delay", f"{manifest.get('default_delay', 'N/A')} steps")
    c2.metric("PPO timesteps", f"{manifest.get('ppo_total_timesteps', 0):,}")
    c3.metric("Profile", manifest.get("profile", "N/A"))

    st.write("**Provisioning delays:**", manifest.get("delays_evaluated", []))
    st.write("**Random seeds:**", manifest.get("seeds", []))

    if manifest.get("wall_seconds") is not None:
        st.write(f"**Runtime:** {manifest['wall_seconds'] / 60:.1f} minutes")

st.caption(
    "PPO results report mean ± standard deviation over random seeds where available. "
    "All seeds are retained in all_runs.csv."
)
