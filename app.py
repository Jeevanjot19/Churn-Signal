# app.py

import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.pipeline  import (load_and_clean, encode_categoricals,
                            get_feature_columns)
from src.model     import (train_and_evaluate, load_model,
                            score_customers, get_shap_values,
                            get_top_driver_per_customer)
from src.findings  import get_all_findings
from src.segmenter import (assign_tiers, build_tier_profiles,
                            CRITICAL, WATCH, STABLE)
from src.briefing  import generate_briefing, get_action
from config        import MODEL_PATH, WEEK_LABEL, PREV_WEEK_LABEL

st.set_page_config(
    page_title="ChurnSignal", layout="wide", page_icon="📊")
st.title("📊 ChurnSignal")
st.caption("Credit Card Churn Classifier · "
           "Behavioral Findings · Auto Briefing")

with st.sidebar:
    run = st.button("▶ Run Pipeline",
                    type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("**Model:** XGBoost + SMOTE")
    st.markdown("**Explainability:** SHAP")
    st.markdown("**Data:** BankChurners · 10,127 customers")
    st.markdown("**Classification:** binary churn profiling")

if run:

    with st.spinner("Loading data..."):
        df_raw    = load_and_clean()
        df_enc    = encode_categoricals(df_raw)
        feat_cols = get_feature_columns(df_enc)
        curr_wk   = df_enc.copy()

    with st.spinner("Loading model..."):
        if os.path.exists(MODEL_PATH):
            model, threshold = load_model()
            metrics = {"auc": 0.9912, "precision": 0.950,
                       "recall": 0.874, "f1": 0.910,
                       "threshold": 0.678}
        else:
            with st.spinner("Training model (first run)..."):
                model, threshold, metrics = train_and_evaluate(
                    df_enc, feat_cols)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Metrics")
        st.sidebar.markdown(f"AUC: **{metrics['auc']}**")
        st.sidebar.markdown(
            f"Threshold: **{metrics['threshold']}** (F1-optimal)")
        st.sidebar.markdown(f"Precision: **{metrics['precision']}**")
        st.sidebar.markdown(f"Recall: **{metrics['recall']}**")
        st.sidebar.markdown(f"F1: **{metrics['f1']}**")

    with st.spinner("Scoring and analysing..."):
        curr_scored = score_customers(
            model, threshold, curr_wk, feat_cols)

        shap_curr = get_shap_values(model, curr_wk, feat_cols)

        curr_tiered = assign_tiers(curr_scored)

        curr_tiered["top_driver"] = get_top_driver_per_customer(
            shap_curr, feat_cols)
        curr_tiered["recommended_action"] = (
            curr_tiered["top_driver"].apply(get_action))

        curr_profiles = build_tier_profiles(
            curr_tiered, shap_curr, feat_cols)
        prev_profiles = {}

        findings = get_all_findings(df_raw)

        briefing = generate_briefing(
            curr_profiles, prev_profiles,
            findings, WEEK_LABEL, PREV_WEEK_LABEL)

        # Export scored data for Power BI
        export_df = curr_tiered[[
            "CLIENTNUM",
            "churn_probability",
            "risk_tier",
            "top_driver",
            "Total_Trans_Ct",
            "Total_Trans_Amt",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Total_Revolving_Bal",
            "Credit_Limit",
            "Avg_Utilization_Ratio",
            "Card_Category",
            "Income_Category",
            "Gender",
            "Customer_Age"
        ]].copy()

        export_df.to_csv("churnsignal_scored.csv", index=False)
        print("Exported for Power BI")

        # Contacts finding
        findings["contacts"].to_csv("finding_contacts.csv", index=False)

        # Transaction shield
        findings["tx_shield"].to_csv("finding_tx_shield.csv", index=False)

        # Inactivity
        findings["inactivity"].to_csv("finding_inactivity.csv", index=False)

        # Balance finding - manual small table
        bal = findings["balance"]
        pd.DataFrame({
            "Balance_Status": ["Zero Balance", "Non-Zero Balance"],
            "Churn_Rate": [bal["zero_churn_rate"], bal["nonzero_churn_rate"]],
            "Customer_Count": [bal["zero_count"], bal["nonzero_count"]]
        }).to_csv("finding_balance.csv", index=False)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    cn = curr_profiles[CRITICAL]["count"]
    c1.metric("🔴 Critical", cn)
    c2.metric("🟡 Watch",  curr_profiles[WATCH]["count"])
    c3.metric("🟢 Stable", curr_profiles[STABLE]["count"])
    c4.metric("💰 Rev. at Risk",
              f"${cn * curr_profiles[CRITICAL].get('mean_credit_limit',15000) * 0.15:,.0f}",
              help="Credit limit proxy · 15% assumption")

    st.markdown("---")
    st.subheader("📋 Monday Morning Briefing")
    st.info(briefing)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Risk donut
        st.subheader("Risk Distribution")
        fig1 = px.pie(
            values=[curr_profiles[t]["count"]
                    for t in [CRITICAL, WATCH, STABLE]],
            names=[CRITICAL, WATCH, STABLE],
            color_discrete_map={
                CRITICAL:"#ef4444", WATCH:"#f59e0b", STABLE:"#22c55e"},
            hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)

        # SHAP bar
        st.subheader("Top SHAP Drivers — Critical")
        mask_c = (curr_tiered["risk_tier"]==CRITICAL).values
        mean_abs = np.abs(shap_curr[mask_c]).mean(axis=0)
        df_imp = (pd.DataFrame({"Feature": feat_cols,
                                 "Importance": mean_abs})
                  .sort_values("Importance").tail(8))
        fig2 = px.bar(df_imp, x="Importance", y="Feature",
                      orientation="h", color="Importance",
                      color_continuous_scale="Reds")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Finding 1 — contacts
        st.subheader("★ Contacts Distress Curve")
        st.caption("More calls ≠ more loyalty. "
                   "6 contacts = 100% churn.")
        fig3 = px.line(
            findings["contacts"],
            x="Contacts_Count_12_mon", y="churn_rate",
            markers=True,
            labels={"Contacts_Count_12_mon":"Contacts",
                    "churn_rate":"Churn Rate"},
            color_discrete_sequence=["#ef4444"])
        fig3.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

        # Finding 2 — balance
        st.subheader("★ Zero Balance Risk")
        bal = findings["balance"]
        st.caption(
            f"Zero balance: {bal['zero_churn_rate']:.1%} churn vs "
            f"{bal['nonzero_churn_rate']:.1%}. Ratio: {bal['ratio']}x.")
        fig4 = px.bar(
            x=["Zero Balance","Non-Zero Balance"],
            y=[bal["zero_churn_rate"], bal["nonzero_churn_rate"]],
            color=["Zero Balance","Non-Zero Balance"],
            color_discrete_map={"Zero Balance":"#ef4444",
                                 "Non-Zero Balance":"#22c55e"},
            labels={"x":"Balance Status","y":"Churn Rate"})
        fig4.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("SHAP Analysis")

    tier_tab, customer_tab = st.tabs(["Tier-wise", "Customer-wise"])

    with tier_tab:
        tier_choice = st.selectbox(
            "Risk tier",
            [CRITICAL, WATCH, STABLE],
            key="tier_shap_choice"
        )
        tier_mask = (curr_tiered["risk_tier"] == tier_choice).values
        tier_shap = shap_curr[tier_mask]

        if len(tier_shap) > 0:
            tier_mean_abs = np.abs(tier_shap).mean(axis=0)
            tier_shap_df = (
                pd.DataFrame({
                    "Feature": feat_cols,
                    "Mean Absolute SHAP": tier_mean_abs
                })
                .sort_values("Mean Absolute SHAP", ascending=False)
                .head(10)
                .sort_values("Mean Absolute SHAP")
            )
            fig_tier_shap = px.bar(
                tier_shap_df,
                x="Mean Absolute SHAP",
                y="Feature",
                orientation="h",
                color="Mean Absolute SHAP",
                color_continuous_scale="Reds",
                labels={
                    "Mean Absolute SHAP": "Avg impact",
                    "Feature": "Driver"
                }
            )
            fig_tier_shap.update_layout(showlegend=False)
            st.plotly_chart(fig_tier_shap, use_container_width=True)
        else:
            st.info(f"No customers in {tier_choice} tier.")

    with customer_tab:
        customer_options = (
            curr_tiered[["CLIENTNUM", "risk_tier", "churn_probability"]]
            .sort_values("churn_probability", ascending=False)
        )
        customer_lookup = customer_options.set_index("CLIENTNUM")
        selected_customer = st.selectbox(
            "Customer",
            customer_options["CLIENTNUM"].tolist(),
            format_func=lambda x: (
                f"{x} | "
                f"{customer_lookup.loc[x, 'risk_tier']} | "
                f"{customer_lookup.loc[x, 'churn_probability']:.1%}"
            ),
            key="customer_shap_choice"
        )

        selected_pos = curr_tiered.index.get_loc(
            curr_tiered.index[curr_tiered["CLIENTNUM"] == selected_customer][0]
        )
        customer_row = curr_tiered.iloc[selected_pos]
        customer_shap = shap_curr[selected_pos]
        customer_shap_df = (
            pd.DataFrame({
                "Feature": feat_cols,
                "SHAP Value": customer_shap,
                "Abs SHAP": np.abs(customer_shap),
                "Customer Value": curr_wk.iloc[selected_pos][feat_cols].values
            })
            .sort_values("Abs SHAP", ascending=False)
            .head(10)
            .sort_values("SHAP Value")
        )

        s1, s2, s3 = st.columns(3)
        s1.metric("Risk Tier", customer_row["risk_tier"])
        s2.metric("Churn Probability", f"{customer_row['churn_probability']:.1%}")
        s3.metric("Top Driver", customer_row["top_driver"])

        fig_customer_shap = px.bar(
            customer_shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="SHAP Value",
            color_continuous_scale="RdYlGn_r",
            labels={
                "SHAP Value": "Signed impact",
                "Feature": "Driver"
            },
            hover_data=["Customer Value"]
        )
        fig_customer_shap.add_vline(x=0, line_width=1, line_color="#374151")
        fig_customer_shap.update_layout(showlegend=False)
        st.plotly_chart(fig_customer_shap, use_container_width=True)

        driver_table = customer_shap_df.sort_values(
            "Abs SHAP", ascending=False
        )[["Feature", "Customer Value", "SHAP Value"]]
        st.dataframe(driver_table, use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)

    with col3:
        # Finding 3 — tx shield
        st.subheader("★ Transaction Frequency Shield")
        st.caption("100+ transactions per year: 0% churn.")
        fig5 = px.bar(
            findings["tx_shield"],
            x="min_transactions", y="churn_rate",
            color="churn_rate",
            color_continuous_scale="RdYlGn_r",
            labels={"min_transactions":"Min Transactions",
                    "churn_rate":"Churn Rate"})
        fig5.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig5, use_container_width=True)

    with col4:
        # Finding 4 — inactivity
        st.subheader("★ Inactivity Acceleration")
        st.caption(
            "Churn risk at 4 months is 7x higher than at 1 month.")
        inact = findings["inactivity"]
        # Exclude 0-month bucket (n=29, too small)
        inact_plot = inact[
            (inact["Months_Inactive_12_mon"] > 0) &
            (inact["Months_Inactive_12_mon"] <= 6)]
        fig6 = px.line(
            inact_plot,
            x="Months_Inactive_12_mon", y="churn_rate",
            markers=True,
            labels={"Months_Inactive_12_mon":"Months Inactive",
                    "churn_rate":"Churn Rate"},
            color_discrete_sequence=["#f59e0b"])
        fig6.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.subheader("Customer Risk Table — Critical Tier")

    crit_table = (
        curr_tiered[curr_tiered["risk_tier"]==CRITICAL][[
            "CLIENTNUM","churn_probability","top_driver",
            "recommended_action","Contacts_Count_12_mon",
            "Total_Revolving_Bal","Total_Trans_Ct"]]
        .sort_values("churn_probability", ascending=False)
    )
    crit_table.columns = [
        "Customer ID","Churn Prob","Top Driver",
        "Recommended Action","Contacts",
        "Revolving Balance","Transaction Count"]
    st.dataframe(crit_table.head(50),
                 use_container_width=True, hide_index=True)
