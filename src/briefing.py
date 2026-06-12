# src/briefing.py

DRIVER_ACTION = {
    "Total_Trans_Ct":
        "Frequency reactivation — reward next 3 transactions within 30 days",
    "Total_Trans_Amt":
        "Spend acceleration — double points for 30 days",
    "Total_Revolving_Bal":
        "Balance transfer offer or low-APR promotion",
    "Total_Relationship_Count":
        "Cross-sell review — complementary product offer",
    "Total_Ct_Chng_Q4_Q1":
        "Trend alert — frequency declining, outreach before further drop",
    "Total_Amt_Chng_Q4_Q1":
        "Spend trend alert — premium category bonus offer",
    "Months_Inactive_12_mon":
        "Re-engagement campaign — personalised outreach",
    "Contacts_Count_12_mon":
        "Distress flag — relationship manager review immediately",
    "Avg_Utilization_Ratio":
        "Credit limit review — proactive increase",
    "Credit_Limit":
        "Credit line upgrade eligibility assessment",
}


def get_action(driver: str) -> str:
    return DRIVER_ACTION.get(driver, "General engagement review")


def generate_briefing(curr_profiles: dict,
                       prev_profiles: dict,
                       findings: dict,
                       week: str,
                       prev_week: str) -> str:

    crit   = curr_profiles.get("Critical", {})
    watch  = curr_profiles.get("Watch",    {})
    stable = curr_profiles.get("Stable",   {})
    total  = sum(p.get("count", 0) for p in curr_profiles.values())

    # Week-over-week delta
    wow_line = ""
    if prev_profiles and "Critical" in prev_profiles:
        prev_n = prev_profiles["Critical"].get("count", 0)
        curr_n = crit.get("count", 0)
        delta  = curr_n - prev_n
        if prev_n > 0:
            pct = round(abs(delta / prev_n) * 100, 1)
            direction = "increase" if delta > 0 else "decrease"
            wow_line = (
                f" This is a {pct}% {direction} of "
                f"{abs(delta)} customers vs {prev_week}."
            )

    # Revenue at risk — stated assumption, not a fact
    rev = round(
        crit.get("count", 0) *
        crit.get("mean_credit_limit", 15000) * 0.15, 0
    )

    top_driver   = crit.get("top_drivers",  ["Total_Trans_Ct"])[0]
    watch_driver = watch.get("top_drivers", ["Months_Inactive_12_mon"])[0]

    # Contacts warning if high contacts in critical tier
    contact_warning = ""
    if crit.get("mean_contacts", 0) >= 3.0:
        contact_warning = (
            f" Note: critical tier averages "
            f"{crit.get('mean_contacts',0):.1f} bank contacts — "
            f"data shows 6 contacts correlates with 100% "
            f"historical churn. Flag for immediate relationship "
            f"manager review."
        )

    p1 = (
        f"Weekly Retention Briefing — {week}. "
        f"ChurnSignal scored {total:,} customers: "
        f"{crit.get('count',0)} Critical, "
        f"{watch.get('count',0)} Watch, "
        f"{stable.get('count',0)} Stable.{wow_line} "
        f"Estimated revenue at risk: ${rev:,.0f} "
        f"(credit limit proxy, 15% annual revenue assumption)."
    )

    bal = findings.get("balance", {})
    p2 = (
        f"Critical tier: avg {crit.get('mean_trans_count',0):.0f} "
        f"transactions over 12 months, "
        f"{crit.get('mean_inactive_mo',0):.1f} months avg inactivity, "
        f"{crit.get('zero_balance_pct',0):.0%} with zero revolving "
        f"balance ({bal.get('ratio','3.8')}x higher historical churn "
        f"risk vs non-zero balance customers). "
        f"Primary driver: {top_driver.replace('_',' ')}. "
        f"Action: {get_action(top_driver)}.{contact_warning}"
    )

    p3 = (
        f"Watch tier: {watch.get('count',0)} customers with early "
        f"warning signals. Primary driver: "
        f"{watch_driver.replace('_',' ')}. "
        f"Action: {get_action(watch_driver)}. "
        f"Intervene before month 3 of inactivity — churn risk at "
        f"4 months inactive (29.9%) is 7x higher than at 1 month "
        f"(4.5%). The window closes fast."
    )

    return f"{p1}\n\n{p2}\n\n{p3}"