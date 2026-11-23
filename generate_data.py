"""
generate_data.py
Creates the SIMULATED dataset (data.csv) used by the Smart Study Planner.

Each row describes one student-subject situation. The target column
`Recommended_Hours_Per_Day` is produced from a transparent workload rule
(topics, difficulty, confidence, past score, subject type, days left)
plus random variation, so a machine learning model can learn the pattern.
"""
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 6000

TYPE_MULTIPLIER = {"Theory": 1.00, "Practical": 0.90, "Mathematical": 1.25, "Programming": 1.15}


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    subject_type = rng.choice(list(TYPE_MULTIPLIER), size=n_rows, p=[0.30, 0.20, 0.25, 0.25])
    difficulty = rng.choice([1, 2, 3, 4, 5], size=n_rows, p=[0.10, 0.20, 0.35, 0.25, 0.10])

    # Harder subjects tend to give lower confidence
    confidence = np.clip(np.round(rng.normal(3.2 - 0.25 * (difficulty - 3), 1.0)), 1, 5).astype(int)

    # Last test score is related to confidence
    last_score = np.clip(np.round(rng.normal(55 + 8 * (confidence - 3), 14)), 5, 98).astype(int)

    topics_remaining = rng.integers(2, 31, size=n_rows)
    days_left = np.clip((rng.gamma(2.0, 15.0, size=n_rows)).astype(int) + 2, 2, 90)
    daily_free_hours = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10], size=n_rows,
                                  p=[0.08, 0.12, 0.18, 0.18, 0.16, 0.12, 0.08, 0.05, 0.03])

    # --- Workload rule -------------------------------------------------
    hours_per_topic = 0.6 + 0.3 * difficulty
    type_mult = np.array([TYPE_MULTIPLIER[t] for t in subject_type])
    confidence_factor = 1.5 - 0.15 * confidence      # low confidence -> more work
    score_factor = 1.3 - 0.006 * last_score          # low score -> more work
    total_hours = topics_remaining * hours_per_topic * type_mult * confidence_factor * score_factor

    per_day = 1.4 * total_hours / days_left
    per_day = per_day * rng.normal(1.0, 0.06, size=n_rows)          # natural variation

    # One subject can take at most 60% of the free time (and never more than 6 h)
    cap = np.minimum(0.6 * daily_free_hours, 6.0)
    per_day = np.clip(per_day, 0.25, cap)
    per_day = np.round(per_day * 4) / 4                              # nearest 15 minutes

    return pd.DataFrame({
        "Subject_Type": subject_type,
        "Subject_Difficulty": difficulty,
        "Confidence": confidence,
        "Last_Test_Score": last_score,
        "Topics_Remaining": topics_remaining,
        "Days_Left": days_left,
        "Daily_Free_Hours": daily_free_hours,
        "Recommended_Hours_Per_Day": per_day,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("data.csv", index=False)
    print(f"data.csv created with {len(df)} rows and {df.shape[1]} columns")
    print(df.describe().round(2))
