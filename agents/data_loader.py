import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/ats_mock_data.csv")


def load_ats_data() -> pd.DataFrame:
    """Load and clean the ATS dataset."""
    df = pd.read_csv(DATA_PATH)
    df["apply_date"] = pd.to_datetime(df["apply_date"])
    df["last_activity_date"] = pd.to_datetime(df["last_activity_date"])
    df["interview_score"] = pd.to_numeric(df["interview_score"], errors="coerce")
    df["offer_amount"] = pd.to_numeric(df["offer_amount"], errors="coerce")
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Generate summary statistics for dashboard and agent context."""
    return {
        "total_candidates": len(df),
        "stage_distribution": df["current_stage"].value_counts().to_dict(),
        "source_distribution": df["source"].value_counts().to_dict(),
        "role_distribution": df["role"].value_counts().to_dict(),
        "avg_days_in_pipeline": round(df["days_in_pipeline"].mean(), 1),
        "sla_breach_rate": round(df["sla_breached"].eq("Yes").mean() * 100, 1),
        "hired_count": len(df[df["current_stage"] == "Hired"]),
        "offer_count": len(df[df["current_stage"] == "Offer"]),
        "rejection_reasons": df[df["rejection_reason"] != ""]["rejection_reason"].value_counts().to_dict(),
        "offer_decline_reasons": df[df["offer_decline_reason"] != ""]["offer_decline_reason"].value_counts().to_dict(),
        "top_sources_by_hire": df[df["current_stage"] == "Hired"]["source"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    df = load_ats_data()
    stats = get_summary_stats(df)
    print(f"Loaded {len(df)} candidates")
    print(f"Stage distribution: {stats['stage_distribution']}")
    print(f"SLA breach rate: {stats['sla_breach_rate']}%")
    print(f"Top hire sources: {stats['top_sources_by_hire']}")
