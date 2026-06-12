from src.findings import get_all_findings
from src.pipeline import load_and_clean


def main():
    df_raw = load_and_clean()
    findings = get_all_findings(df_raw)

    c = findings["contacts"]
    assert c[c["Contacts_Count_12_mon"] == 6]["churn_rate"].values[0] == 1.0
    print("Finding 1 OK")

    b = findings["balance"]
    assert b["ratio"] == 3.8
    print(f"Finding 2 OK - zero={b['zero_churn_rate']}, ratio={b['ratio']}")

    t = findings["tx_shield"]
    assert t[t["min_transactions"] == 100]["churn_rate"].values[0] == 0.0
    print("Finding 3 OK")

    i = findings["inactivity"]
    row1 = i[i["Months_Inactive_12_mon"] == 1]["churn_rate"].values[0]
    row4 = i[i["Months_Inactive_12_mon"] == 4]["churn_rate"].values[0]
    print(f"Finding 4 OK - 1mo={row1}, 4mo={row4}")


if __name__ == "__main__":
    main()
