"""
Smart Study Planner - Streamlit app
Run with:  streamlit run app.py
"""
import os
from datetime import date, datetime, time, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

SUBJECT_TYPES = ["Theory", "Practical", "Mathematical", "Programming"]
FEATURES = ["Subject_Type", "Subject_Difficulty", "Confidence", "Last_Test_Score",
            "Topics_Remaining", "Days_Left", "Daily_Free_Hours"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def _train_from_csv():
    """Fallback: train the model again from data.csv (used if model.joblib cannot be loaded)."""
    from sklearn.ensemble import GradientBoostingRegressor

    df = pd.read_csv(DATA_PATH)
    type_map = {name: i for i, name in enumerate(sorted(df["Subject_Type"].unique()))}
    data = df.copy()
    data["Subject_Type"] = data["Subject_Type"].map(type_map)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=4, random_state=42)
    model.fit(data[FEATURES], data["Recommended_Hours_Per_Day"])
    return {"model": model, "model_name": "Gradient Boosting", "features": FEATURES,
            "subject_type_map": type_map, "metrics": {}}


@st.cache_resource(show_spinner=False)
def load_bundle():
    """Load model.joblib; if that fails (e.g. different scikit-learn version) retrain from data.csv."""
    try:
        bundle = joblib.load(MODEL_PATH)
        for key in ("model", "features", "subject_type_map"):
            if key not in bundle:
                raise KeyError(key)
        return bundle
    except Exception:
        bundle = _train_from_csv()
        try:
            joblib.dump(bundle, MODEL_PATH)
        except Exception:
            pass
        return bundle


# ----------------------------------------------------------------------------
# Planner logic
# ----------------------------------------------------------------------------
def priority_label(days_left):
    if days_left <= 7:
        return "High"
    if days_left <= 21:
        return "Medium"
    return "Low"


def predict_hours(bundle, subjects, free_hours):
    """Predict recommended study hours per day for every subject (rounded to 15 minutes)."""
    rows = []
    for s in subjects:
        rows.append({
            "Subject_Type": bundle["subject_type_map"].get(s["type"], 0),
            "Subject_Difficulty": s["difficulty"],
            "Confidence": s["confidence"],
            "Last_Test_Score": s["score"],
            "Topics_Remaining": s["topics"],
            "Days_Left": s["days_left"],
            "Daily_Free_Hours": free_hours,
        })
    X = pd.DataFrame(rows)[bundle["features"]]
    pred = np.asarray(bundle["model"].predict(X), dtype=float)
    pred = np.clip(pred, 0.25, 6.0)
    return np.round(pred * 4) / 4


def allocate_minutes(pred_hours, free_hours):
    """Scale the predicted hours so the total fits in the daily free time (15-minute steps)."""
    pred_hours = np.asarray(pred_hours, dtype=float)
    total = pred_hours.sum()
    scale = min(1.0, free_hours / total) if total > 0 else 1.0
    minutes = np.maximum(15, np.round(pred_hours * scale * 60 / 15) * 15).astype(int)
    limit = int(free_hours * 60)
    while minutes.sum() > limit and (minutes > 15).any():
        minutes[int(np.argmax(minutes))] -= 15
    return minutes, scale


def build_timetable(plan, start_date, start_time, rest_day, break_min=10, max_block=90):
    """
    Build a 7-day timetable.
    plan: DataFrame with Subject, Minutes, Days_Left, Exam_Date (date object)
    Returns (timetable DataFrame, crosses_midnight flag)
    """
    ordered = plan.sort_values(["Days_Left", "Minutes"], ascending=[True, False])
    rows = []
    crosses_midnight = False

    for offset in range(7):
        day = start_date + timedelta(days=offset)
        day_name = day.strftime("%A")
        date_text = day.strftime("%d %b %Y")

        if day_name == rest_day:
            rows.append({"Date": date_text, "Day": day_name[:3], "Start": "-", "End": "-",
                         "Subject": "Rest day", "Minutes": 0})
            continue

        current = datetime.combine(day, start_time)
        scheduled = 0
        for _, item in ordered.iterrows():
            if day >= item["Exam_Date"]:          # exam day or later: no study session
                continue
            remaining = int(item["Minutes"])
            while remaining > 0:
                block = min(remaining, max_block)
                end = current + timedelta(minutes=block)
                if end.date() != day:
                    crosses_midnight = True
                rows.append({"Date": date_text, "Day": day_name[:3],
                             "Start": current.strftime("%H:%M"), "End": end.strftime("%H:%M"),
                             "Subject": item["Subject"], "Minutes": block})
                scheduled += 1
                remaining -= block
                current = end + timedelta(minutes=break_min)
        if scheduled == 0:
            rows.append({"Date": date_text, "Day": day_name[:3], "Start": "-", "End": "-",
                         "Subject": "No study needed", "Minutes": 0})

    table = pd.DataFrame(rows, columns=["Date", "Day", "Start", "End", "Subject", "Minutes"])
    return table, crosses_midnight


def weekly_summary(timetable):
    """Hours per subject for each day of the week."""
    study = timetable[timetable["Minutes"] > 0]
    if study.empty:
        return pd.DataFrame()
    pivot = study.pivot_table(index="Subject", columns="Day", values="Minutes",
                              aggfunc="sum", fill_value=0)
    order = [d for d in timetable["Day"].unique() if d in pivot.columns]
    pivot = pivot[order]
    pivot["Total hours"] = (pivot.sum(axis=1) / 60).round(2)
    pivot[order] = (pivot[order] / 60).round(2)
    return pivot.reset_index()


def make_plan(bundle, subjects, free_hours):
    """Predict and allocate: returns the plan DataFrame and the scaling factor."""
    pred = predict_hours(bundle, subjects, free_hours)
    minutes, scale = allocate_minutes(pred, free_hours)
    plan = pd.DataFrame({
        "Subject": [s["name"] for s in subjects],
        "Type": [s["type"] for s in subjects],
        "Exam_Date": [s["exam_date"] for s in subjects],
        "Days_Left": [s["days_left"] for s in subjects],
        "Priority": [priority_label(s["days_left"]) for s in subjects],
        "Model_Hours": pred,
        "Minutes": minutes,
    })
    return plan, scale


# ----------------------------------------------------------------------------
# Streamlit user interface
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Smart Study Planner", page_icon="📚", layout="wide")
    st.title("📚 Smart Study Planner")
    st.write("Enter your subjects and exam dates. A machine learning model predicts how many "
             "hours you need to study each subject every day, and the app builds a weekly timetable.")

    bundle = load_bundle()
    today = date.today()

    num_subjects = st.number_input("How many subjects do you want to plan?",
                                   min_value=1, max_value=8, value=3, step=1)

    with st.form("planner_form"):
        st.subheader("Your daily routine")
        c1, c2, c3 = st.columns(3)
        free_hours = c1.slider("Free study hours per day", 1.0, 12.0, 5.0, 0.5)
        start_time = c2.time_input("Study start time", value=time(17, 0))
        rest_day = c3.selectbox("Rest day", ["None"] + WEEKDAYS, index=0)

        st.subheader("Your subjects")
        subjects = []
        for i in range(int(num_subjects)):
            st.markdown(f"**Subject {i + 1}**")
            a1, a2, a3, a4 = st.columns(4)
            name = a1.text_input("Subject name", value=f"Subject {i + 1}", key=f"name_{i}")
            s_type = a2.selectbox("Subject type", SUBJECT_TYPES, key=f"type_{i}")
            difficulty = a3.slider("Difficulty (1 = easy, 5 = hard)", 1, 5, 3, key=f"diff_{i}")
            confidence = a4.slider("Your confidence (1 = low, 5 = high)", 1, 5, 3, key=f"conf_{i}")
            b1, b2, b3 = st.columns(3)
            score = b1.number_input("Last test score (%)", min_value=0, max_value=100,
                                    value=60, step=1, key=f"score_{i}")
            topics = b2.number_input("Topics remaining", min_value=1, max_value=60,
                                     value=10, step=1, key=f"topics_{i}")
            exam_date = b3.date_input("Exam date", value=today + timedelta(days=15 + 5 * i),
                                      min_value=today + timedelta(days=1), key=f"exam_{i}")
            subjects.append({
                "name": (name or "").strip() or f"Subject {i + 1}",
                "type": s_type,
                "difficulty": int(difficulty),
                "confidence": int(confidence),
                "score": int(score),
                "topics": int(topics),
                "exam_date": exam_date,
                "days_left": max(1, (exam_date - today).days),
            })

        submitted = st.form_submit_button("Generate Study Plan")

    if not submitted:
        st.info("Fill in the details above and click **Generate Study Plan**.")
        return

    # Make subject names unique
    seen = {}
    for s in subjects:
        count = seen.get(s["name"], 0) + 1
        seen[s["name"]] = count
        if count > 1:
            s["name"] = f"{s['name']} ({count})"

    plan, scale = make_plan(bundle, subjects, float(free_hours))
    timetable, crosses_midnight = build_timetable(
        plan, today, start_time, rest_day if rest_day != "None" else "")

    st.divider()
    st.header("Your Study Plan")

    total_planned = plan["Minutes"].sum() / 60
    urgent = plan.sort_values(["Days_Left", "Minutes"], ascending=[True, False]).iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Planned study time per day", f"{total_planned:.2f} h")
    m2.metric("Subjects", len(plan))
    m3.metric("Most urgent subject", urgent["Subject"], f"{int(urgent['Days_Left'])} days left")

    if scale < 1.0:
        st.warning(f"The subjects need about {plan['Model_Hours'].sum():.2f} hours per day, but you "
                   f"have {free_hours:.1f} free hours. Study time was reduced proportionally. "
                   "Try increasing free time or starting earlier.")
    if crosses_midnight:
        st.warning("Some sessions run past midnight. Choose an earlier study start time.")

    st.subheader("Subject priorities")
    show = pd.DataFrame({
        "Subject": plan["Subject"],
        "Type": plan["Type"],
        "Exam date": [d.strftime("%d %b %Y") for d in plan["Exam_Date"]],
        "Days left": plan["Days_Left"],
        "Priority": plan["Priority"].map({"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}),
        "Model suggestion (h/day)": plan["Model_Hours"].round(2),
        "Planned (h/day)": (plan["Minutes"] / 60).round(2),
    })
    st.dataframe(show, hide_index=True)

    st.subheader("Planned hours per day by subject")
    chart_df = pd.DataFrame({"Hours": (plan["Minutes"] / 60).values}, index=plan["Subject"].values)
    st.bar_chart(chart_df)

    st.subheader("Weekly timetable")
    st.dataframe(timetable, hide_index=True)

    summary = weekly_summary(timetable)
    if not summary.empty:
        st.subheader("Hours per subject each day")
        st.dataframe(summary, hide_index=True)

    st.download_button("Download timetable (CSV)",
                       data=timetable.to_csv(index=False).encode("utf-8"),
                       file_name="study_timetable.csv", mime="text/csv")

    st.caption(f"Prediction model: {bundle.get('model_name', 'Gradient Boosting')} trained on a "
               "simulated dataset of 6000 records. Plans are suggestions - adjust them to your needs.")


main()
