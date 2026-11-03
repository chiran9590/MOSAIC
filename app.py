from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MOSAIC | Quantitative Research Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORIGINAL_RESULTS_PATH = Path("reports/final_test_results.csv")
ORIGINAL_PREDICTIONS_PATH = Path("reports/final_test_predictions.npz")

EXPANDED_RESULTS_PATH = Path("reports/final_test_results_v2.csv")
EXPANDED_PREDICTIONS_PATH = Path("reports/final_test_predictions_v2.npz")

CLASS_NAMES = ["DOWN", "STATIONARY", "UP"]
CLASS_IDS = [0, 1, 2]

COLORS = {
    "blue": "#38BDF8",
    "cyan": "#22D3EE",
    "green": "#34D399",
    "orange": "#FBBF24",
    "red": "#F87171",
    "purple": "#A78BFA",
    "muted": "#94A3B8",
    "grid": "#263449",
}

# ============================================================
# 2. PROFESSIONAL DARK THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1120;
        color: #E5E7EB;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263449;
    }
    h1 {
        color: #F8FAFC !important;
        font-size: 42px !important;
        font-weight: 800 !important;
        letter-spacing: 3px;
    }
    h2, h3 {
        color: #E2E8F0 !important;
    }
    [data-testid="stMetric"] {
        background: #172033;
        border: 1px solid #263449;
        padding: 16px;
        border-radius: 12px;
    }
    [data-testid="stMetricValue"] {
        color: #38BDF8;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #263449;
        border-radius: 10px;
    }
    hr {
        border-color: #263449;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA LOADING AND VALIDATION
# ============================================================

@st.cache_data
def load_results(path_string):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_predictions(path_string):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def prediction_keys(archive, fold):
    """Support both historical and current prediction key conventions."""
    true_candidates = [
        f"y_true_fold_{fold}",
        f"y_true_{fold}",
    ]
    pred_candidates = [
        f"y_pred_fold_{fold}",
        f"predictions_fold_{fold}",
        f"y_pred_{fold}",
    ]

    true_key = next((k for k in true_candidates if k in archive), None)
    pred_key = next((k for k in pred_candidates if k in archive), None)
    return true_key, pred_key


def validate_results(df, label):
    required = {
        "fold", "accuracy", "macro_f1",
        "class_1_f1", "class_2_f1", "class_3_f1",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} results missing columns: {sorted(missing)}")


def get_fold_predictions(archive, fold):
    true_key, pred_key = prediction_keys(archive, fold)
    if true_key is None or pred_key is None:
        raise KeyError(
            f"Could not locate actual/predicted arrays for fold {fold}. "
            f"Available keys: {list(archive.keys())}"
        )

    y_true = np.asarray(archive[true_key]).astype(int)
    y_pred = np.asarray(archive[pred_key]).astype(int)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Prediction length mismatch in fold {fold}.")
    if not np.isin(y_true, CLASS_IDS).all():
        raise ValueError(f"Unexpected actual labels in fold {fold}.")
    if not np.isin(y_pred, CLASS_IDS).all():
        raise ValueError(f"Unexpected predicted labels in fold {fold}.")
    return y_true, y_pred


def get_all_predictions(archive):
    actual, predicted, fold_ids, sequence_indices = [], [], [], []

    for fold in range(1, 10):
        y_true, y_pred = get_fold_predictions(archive, fold)
        actual.append(y_true)
        predicted.append(y_pred)
        fold_ids.append(np.full(len(y_true), fold, dtype=int))

        index_key = f"test_indices_fold_{fold}"
        if index_key in archive:
            idx = np.asarray(archive[index_key]).astype(int)
        else:
            idx = np.arange(len(y_true))
        sequence_indices.append(idx)

    return (
        np.concatenate(actual),
        np.concatenate(predicted),
        np.concatenate(fold_ids),
        np.concatenate(sequence_indices),
    )


def style_axis(ax):
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#CBD5E1")
    ax.xaxis.label.set_color("#CBD5E1")
    ax.yaxis.label.set_color("#CBD5E1")
    ax.title.set_color("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#263449")
    ax.grid(True, color="#263449", alpha=0.7, linewidth=0.7)
    ax.set_axisbelow(True)


def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def make_confusion_heatmap(cm, normalized=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0B1120")
    ax.set_facecolor("#111827")

    if normalized:
        row_totals = cm.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            cm,
            row_totals,
            out=np.zeros_like(cm, dtype=float),
            where=row_totals != 0,
        )
    else:
        display_matrix = cm

    image = ax.imshow(display_matrix, cmap="Blues", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, color="#E2E8F0")
    ax.set_yticklabels(CLASS_NAMES, color="#E2E8F0")
    ax.set_xlabel("Predicted Class", labelpad=12)
    ax.set_ylabel("Actual Class", labelpad=12)
    ax.set_title("Confusion Matrix", pad=18, fontsize=15)

    threshold = display_matrix.max() / 2 if display_matrix.size else 0
    for i in range(3):
        for j in range(3):
            text = (
                f"{cm[i, j]:,}\n{display_matrix[i, j]:.1%}"
                if normalized
                else f"{cm[i, j]:,}"
            )
            color = "#FFFFFF" if display_matrix[i, j] > threshold else "#E2E8F0"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.tick_params(length=0)
    plt.tight_layout()
    return fig


# Load the expanded experiment, which is the current primary model.
try:
    expanded_results = load_results(str(EXPANDED_RESULTS_PATH))
    expanded_predictions = load_predictions(str(EXPANDED_PREDICTIONS_PATH))
    validate_results(expanded_results, "Expanded")
except Exception as error:
    st.error(f"Could not load expanded-feature results: {error}")
    st.info(
        "Confirm that final_test_results_v2.csv and "
        "final_test_predictions_v2.npz exist inside MOSAIC/reports."
    )
    st.stop()

# Original results are loaded for side-by-side comparison when available.
original_available = (
    ORIGINAL_RESULTS_PATH.exists() and ORIGINAL_PREDICTIONS_PATH.exists()
)
original_results = None
original_predictions = None

if original_available:
    try:
        original_results = load_results(str(ORIGINAL_RESULTS_PATH))
        original_predictions = load_predictions(str(ORIGINAL_PREDICTIONS_PATH))
        validate_results(original_results, "Original")
    except Exception as error:
        original_available = False
        st.sidebar.warning(f"Original comparison unavailable: {error}")


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.markdown("# MOSAIC")
st.sidebar.markdown("**QUANTITATIVE RESEARCH TERMINAL**")
st.sidebar.markdown("---")

selected_fold = st.sidebar.selectbox(
    "Evaluation fold",
    ["All folds"] + [f"Fold {i}" for i in range(1, 10)],
)

model_choice = st.sidebar.radio(
    "Prediction dataset",
    ["Expanded — 1,440 features", "Original — 576 features"],
    disabled=not original_available,
)

use_expanded = model_choice.startswith("Expanded")

if use_expanded:
    results = expanded_results
    predictions = expanded_predictions
    feature_count = 1440
else:
    results = original_results
    predictions = original_predictions
    feature_count = 576

st.sidebar.markdown("---")
st.sidebar.markdown("### Model configuration")
st.sidebar.write("**Model:** HistGradientBoosting")
st.sidebar.write("**Dataset:** FI-2010")
st.sidebar.write("**Subset:** NoAuction Z-score")
st.sidebar.write("**Sequence length:** 100")
st.sidebar.write("**Sequence stride:** 10")
st.sidebar.write(f"**Summary features:** {feature_count:,}")
st.sidebar.write("**Prediction horizon:** Fifth label")
st.sidebar.markdown("---")
st.sidebar.caption("Classification research prototype. No trading execution.")


# ============================================================
# 5. CURRENT SELECTION
# ============================================================

if selected_fold == "All folds":
    selected_results = results.sort_values("fold").copy()
    y_true, y_pred, fold_ids, sequence_indices = get_all_predictions(predictions)
    fold_label = "All nine official test folds"
else:
    fold_number = int(selected_fold.split()[1])
    selected_results = results[results["fold"] == fold_number].copy()
    y_true, y_pred = get_fold_predictions(predictions, fold_number)
    fold_ids = np.full(len(y_true), fold_number, dtype=int)

    index_key = f"test_indices_fold_{fold_number}"
    sequence_indices = (
        np.asarray(predictions[index_key]).astype(int)
        if index_key in predictions
        else np.arange(len(y_true))
    )
    fold_label = f"Fold {fold_number}"

if len(selected_results) == 0:
    st.error("No result row found for the selected fold.")
    st.stop()


# ============================================================
# 6. HEADER
# ============================================================

st.title("MOSAIC")
st.markdown(
    "### Market Order-flow Sequence Intelligence & Cross-Asset Learning"
)
st.markdown(
    "Research platform for short-horizon Limit Order Book movement classification."
)
st.caption(
    "FI-2010  |  NoAuction Z-score  |  HistGradientBoosting  |  "
    "Official test evaluation"
)
st.markdown("---")


# ============================================================
# 7. PERFORMANCE OVERVIEW
# ============================================================

section_title(
    "Performance Overview",
    "Metrics are means of the nine fold-level test metrics when All folds is selected.",
)

if selected_fold == "All folds":
    accuracy = selected_results["accuracy"].mean()
    macro_f1 = selected_results["macro_f1"].mean()
    accuracy_std = selected_results["accuracy"].std()
    macro_std = selected_results["macro_f1"].std()
else:
    accuracy = selected_results["accuracy"].iloc[0]
    macro_f1 = selected_results["macro_f1"].iloc[0]
    accuracy_std = None
    macro_std = None

correct_predictions = int(np.sum(y_true == y_pred))
total_predictions = len(y_true)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Mean Accuracy" if selected_fold == "All folds" else "Accuracy",
          f"{accuracy:.2%}")
k2.metric("Mean Macro F1" if selected_fold == "All folds" else "Macro F1",
          f"{macro_f1:.2%}")

if accuracy_std is not None:
    k3.metric("Accuracy Std. Dev.", f"{accuracy_std:.2%}")
    k4.metric("Macro F1 Std. Dev.", f"{macro_std:.2%}")
else:
    k3.metric("Test Sequences", f"{total_predictions:,}")
    k4.metric("Correct Predictions", f"{correct_predictions:,}")

st.caption(f"Current selection: {fold_label} | Feature representation: {feature_count:,}")
st.markdown("---")


# ============================================================
# 8. ORIGINAL VS EXPANDED COMPARISON
# ============================================================

section_title(
    "Feature Engineering Impact",
    "Side-by-side official test results for the original 576-feature and expanded 1,440-feature representations.",
)

if original_available:
    comparison = pd.DataFrame({
        "Feature representation": ["Original (576)", "Expanded (1,440)"],
        "Mean accuracy": [
            original_results["accuracy"].mean(),
            expanded_results["accuracy"].mean(),
        ],
        "Mean macro F1": [
            original_results["macro_f1"].mean(),
            expanded_results["macro_f1"].mean(),
        ],
        "Mean DOWN F1": [
            original_results["class_1_f1"].mean(),
            expanded_results["class_1_f1"].mean(),
        ],
        "Mean STATIONARY F1": [
            original_results["class_2_f1"].mean(),
            expanded_results["class_2_f1"].mean(),
        ],
        "Mean UP F1": [
            original_results["class_3_f1"].mean(),
            expanded_results["class_3_f1"].mean(),
        ],
    })

    delta_accuracy = comparison.loc[1, "Mean accuracy"] - comparison.loc[0, "Mean accuracy"]
    delta_macro = comparison.loc[1, "Mean macro F1"] - comparison.loc[0, "Mean macro F1"]

    m1, m2 = st.columns(2)
    m1.metric("Accuracy improvement", f"{delta_accuracy:+.2%}")
    m2.metric("Macro F1 improvement", f"{delta_macro:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0B1120")

    metric_specs = [
        ("Mean accuracy", "Accuracy"),
        ("Mean macro F1", "Macro F1"),
    ]
    for ax, (column, title) in zip(axes, metric_specs):
        values = comparison[column].to_numpy()
        bars = ax.bar(
            ["Original\n576", "Expanded\n1,440"],
            values,
            color=[COLORS["muted"], COLORS["green"]],
            width=0.55,
        )
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("Score")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{value:.2%}", ha="center",
                    color="#F8FAFC", fontweight="bold")
        style_axis(ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        comparison.style.format({
            "Mean accuracy": "{:.2%}",
            "Mean macro F1": "{:.2%}",
            "Mean DOWN F1": "{:.2%}",
            "Mean STATIONARY F1": "{:.2%}",
            "Mean UP F1": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if delta_macro > 0:
        st.success(
            f"The expanded representation improved mean official-test "
            f"macro F1 by {delta_macro:.4f} compared with the original representation."
        )
else:
    st.warning(
        "Original result files were not found. The expanded model dashboard "
        "is available, but side-by-side comparison is disabled."
    )

st.markdown("---")


# ============================================================
# 9. FOLD PERFORMANCE VISUALISATION
# ============================================================

section_title(
    "Performance Across Folds",
    "Accuracy and macro F1 on each official test fold.",
)

fig, ax = plt.subplots(figsize=(11, 4.5))
fig.patch.set_facecolor("#0B1120")

if selected_fold == "All folds":
    x = selected_results["fold"].to_numpy()
    ax.plot(x, selected_results["accuracy"], marker="o",
            linewidth=2.5, markersize=7, color=COLORS["blue"], label="Accuracy")
    ax.plot(x, selected_results["macro_f1"], marker="s",
            linewidth=2.5, markersize=7, color=COLORS["green"], label="Macro F1")
    ax.set_xticks(x)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(facecolor="#172033", edgecolor="#263449", labelcolor="#E2E8F0")
else:
    row = selected_results.iloc[0]
    ax.bar(["Accuracy", "Macro F1"], [row["accuracy"], row["macro_f1"]],
           color=[COLORS["blue"], COLORS["green"]], width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")

style_axis(ax)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ============================================================
# 10. FOLD-LEVEL COMPARISON
# ============================================================

if original_available and selected_fold == "All folds":
    section_title(
        "Fold-by-Fold Feature Comparison",
        "Macro F1 on each official test fold.",
    )
    merged = original_results[["fold", "macro_f1"]].merge(
        expanded_results[["fold", "macro_f1"]],
        on="fold",
        suffixes=("_original", "_expanded"),
    ).sort_values("fold")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#0B1120")
    x = merged["fold"].to_numpy()
    ax.plot(x, merged["macro_f1_original"], marker="o", linewidth=2.2,
            color=COLORS["muted"], label="Original 576")
    ax.plot(x, merged["macro_f1_expanded"], marker="s", linewidth=2.2,
            color=COLORS["green"], label="Expanded 1,440")
    ax.set_xticks(x)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.legend(facecolor="#172033", edgecolor="#263449", labelcolor="#E2E8F0")
    style_axis(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    merged["Improvement"] = merged["macro_f1_expanded"] - merged["macro_f1_original"]
    st.dataframe(
        merged.rename(columns={
            "fold": "Fold",
            "macro_f1_original": "Original Macro F1",
            "macro_f1_expanded": "Expanded Macro F1",
        }).style.format({
            "Original Macro F1": "{:.2%}",
            "Expanded Macro F1": "{:.2%}",
            "Improvement": "{:+.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("---")


# ============================================================
# 11. FOLD RESULTS TABLE
# ============================================================

section_title("Detailed Fold Results")
table = selected_results[
    ["fold", "accuracy", "macro_f1", "class_1_f1", "class_2_f1", "class_3_f1"]
].copy()
table.columns = [
    "Fold", "Accuracy", "Macro F1",
    "DOWN F1", "STATIONARY F1", "UP F1",
]
st.dataframe(
    table.style.format({
        "Accuracy": "{:.2%}",
        "Macro F1": "{:.2%}",
        "DOWN F1": "{:.2%}",
        "STATIONARY F1": "{:.2%}",
        "UP F1": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)
st.markdown("---")


# ============================================================
# 12. CLASS-WISE F1 ANALYSIS
# ============================================================

section_title(
    "Class-wise Performance",
    "Mean class F1 across folds or selected-fold results.",
)

class_columns = ["class_1_f1", "class_2_f1", "class_3_f1"]
if selected_fold == "All folds":
    class_scores = [selected_results[col].mean() for col in class_columns]
else:
    class_scores = [selected_results[col].iloc[0] for col in class_columns]

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("#0B1120")
bars = ax.bar(
    CLASS_NAMES,
    class_scores,
    color=[COLORS["blue"], COLORS["orange"], COLORS["green"]],
    width=0.55,
)
for bar, score in zip(bars, class_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{score:.1%}", ha="center", color="#F8FAFC", fontweight="bold")
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.set_title("Class-wise F1 Comparison")
style_axis(ax)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

c1, c2, c3 = st.columns(3)
for column, name, score in zip([c1, c2, c3], CLASS_NAMES, class_scores):
    column.metric(f"{name} F1", f"{score:.2%}")


# ============================================================
# 13. CLASS PERFORMANCE VARIABILITY
# ============================================================

if selected_fold == "All folds":
    section_title(
        "Class Stability Across Folds",
        "Distribution of class-wise F1 scores across nine folds.",
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0B1120")
    data = [
        selected_results["class_1_f1"].to_numpy(),
        selected_results["class_2_f1"].to_numpy(),
        selected_results["class_3_f1"].to_numpy(),
    ]
    box = ax.boxplot(data, tick_labels=CLASS_NAMES, patch_artist=True,
                     widths=0.5, showmeans=True)
    for patch, color in zip(
        box["boxes"], [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for element in ["whiskers", "caps", "medians"]:
        for item in box[element]:
            item.set_color("#E2E8F0")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Distribution by Class")
    style_axis(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

section_title(
    "Confusion Matrix",
    "Actual versus predicted class assignments for the selected prediction set.",
)

cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
matrix_choice = st.radio(
    "Select matrix representation",
    ["Counts", "Normalised percentages"],
    horizontal=True,
)
fig = make_confusion_heatmap(cm, normalized=(matrix_choice != "Counts"))
st.pyplot(fig)
plt.close(fig)
st.caption(
    "Rows are actual classes; columns are predicted classes. "
    "Normalisation is performed separately for each actual class."
)


# ============================================================
# 15. PRECISION AND RECALL
# ============================================================

section_title(
    "Precision and Recall Analysis",
    "Metrics calculated from the selected stored test predictions.",
)

precision, recall, pooled_f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=CLASS_IDS, zero_division=0
)

pr_df = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": pooled_f1,
    "Support": support,
})

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor("#0B1120")
x = np.arange(3)
width = 0.24
ax.bar(x - width, precision, width, label="Precision", color=COLORS["blue"])
ax.bar(x, recall, width, label="Recall", color=COLORS["green"])
ax.bar(x + width, pooled_f1, width, label="F1", color=COLORS["orange"])
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.legend(facecolor="#172033", edgecolor="#263449", labelcolor="#E2E8F0")
style_axis(ax)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.dataframe(
    pr_df.style.format({
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}",
        "Support": "{:,.0f}",
    }),
    use_container_width=True,
    hide_index=True,
)
st.markdown("---")


# ============================================================
# 16. PREDICTION EXPLORER
# ============================================================

section_title(
    "Prediction Explorer",
    "Interactive investigation of model predictions and classification errors.",
)

explorer_df = pd.DataFrame({
    "Fold": fold_ids,
    "Sequence Index": sequence_indices,
    "Actual": [CLASS_NAMES[int(x)] for x in y_true],
    "Predicted": [CLASS_NAMES[int(x)] for x in y_pred],
    "Correct": y_true == y_pred,
})
explorer_df["Result"] = np.where(explorer_df["Correct"], "Correct", "Incorrect")

st.markdown("### Filter predictions")
f1, f2, f3 = st.columns(3)
with f1:
    result_filter = st.selectbox("Prediction result", ["All", "Correct", "Incorrect"])
with f2:
    actual_filter = st.selectbox("Actual class", ["All"] + CLASS_NAMES)
with f3:
    predicted_filter = st.selectbox("Predicted class", ["All"] + CLASS_NAMES)

filtered_df = explorer_df.copy()
if result_filter != "All":
    filtered_df = filtered_df[filtered_df["Result"] == result_filter]
if actual_filter != "All":
    filtered_df = filtered_df[filtered_df["Actual"] == actual_filter]
if predicted_filter != "All":
    filtered_df = filtered_df[filtered_df["Predicted"] == predicted_filter]

st.markdown("### Filtered prediction statistics")
if len(filtered_df) > 0:
    filtered_accuracy = filtered_df["Correct"].mean()
    correct_count = int(filtered_df["Correct"].sum())
    incorrect_count = len(filtered_df) - correct_count
    a, b, c, d = st.columns(4)
    a.metric("Matching predictions", f"{len(filtered_df):,}")
    b.metric("Correct", f"{correct_count:,}")
    c.metric("Incorrect", f"{incorrect_count:,}")
    d.metric("Filtered accuracy", f"{filtered_accuracy:.2%}")
else:
    st.warning("No predictions match these filters.")

st.markdown("---")
st.markdown("### Individual prediction inspector")
if len(filtered_df) > 0:
    selected_position = st.slider(
        "Select prediction from filtered results",
        min_value=0,
        max_value=len(filtered_df) - 1,
        value=0,
        step=1,
        key="prediction_position",
    )
    selected_record = filtered_df.iloc[selected_position]
    actual_label = selected_record["Actual"]
    predicted_label = selected_record["Predicted"]
    prediction_correct = bool(selected_record["Correct"])

    st.caption(
        f"Fold: {int(selected_record['Fold'])} | "
        f"Sequence index: {int(selected_record['Sequence Index']):,}"
    )

    left, right = st.columns(2)
    with left:
        st.metric("Actual market direction", actual_label)
    with right:
        st.metric("Model prediction", predicted_label)

    if prediction_correct:
        st.success("Correct classification: the model predicted the actual FI-2010 direction label.")
    else:
        st.error(
            f"Misclassification: actual direction was {actual_label}, "
            f"but the model predicted {predicted_label}."
        )

st.markdown("---")
st.markdown("### Prediction records")
if len(filtered_df) > 0:
    page_size = st.selectbox("Records per page", [10, 25, 50, 100], index=1)
    total_pages = max(1, int(np.ceil(len(filtered_df) / page_size)))
    page_number = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1
    )
    start = (page_number - 1) * page_size
    end = start + page_size
    display_df = filtered_df.iloc[start:end].copy()
    display_df["Correct"] = display_df["Correct"].map({True: "Yes", False: "No"})
    st.dataframe(
        display_df[
            ["Fold", "Sequence Index", "Actual", "Predicted", "Result", "Correct"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Showing {start + 1:,}–{min(end, len(filtered_df)):,} "
        f"of {len(filtered_df):,} matching predictions."
    )
else:
    st.info("Adjust the filters to display prediction records.")

st.markdown("---")
st.markdown("### Classification error analysis")
errors = explorer_df[explorer_df["Correct"] == False]
if len(errors) > 0:
    error_matrix = pd.crosstab(errors["Actual"], errors["Predicted"])
    error_matrix = error_matrix.reindex(
        index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0
    )
    st.write("This table shows the direction of incorrect predictions.")
    st.dataframe(error_matrix, use_container_width=True)
    most_common_error = errors.groupby(["Actual", "Predicted"]).size().idxmax()
    st.info(
        f"Most frequent error: actual {most_common_error[0]} "
        f"predicted as {most_common_error[1]}."
    )
else:
    st.success("No classification errors in the selected data.")

st.markdown("---")
st.markdown("### Export filtered predictions")
st.download_button(
    "Download filtered prediction records",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="mosaic_filtered_predictions.csv",
    mime="text/csv",
    use_container_width=True,
)


# ============================================================
# 17. DOWNLOAD CENTRE
# ============================================================

st.markdown("---")
section_title(
    "Research Data Export",
    "Download saved evaluation metrics and prediction records.",
)

download_col1, download_col2 = st.columns(2)
with download_col1:
    st.download_button(
        "Download selected model fold results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=(
            "mosaic_expanded_test_results.csv"
            if use_expanded else "mosaic_original_test_results.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

with download_col2:
    export_df = explorer_df.copy()
    st.download_button(
        "Download selected model prediction records",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=(
            "mosaic_expanded_prediction_records.csv"
            if use_expanded else "mosaic_original_prediction_records.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

if original_available:
    st.download_button(
        "Download feature comparison table",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="mosaic_feature_comparison.csv",
        mime="text/csv",
    )


# ============================================================
# 18. RESEARCH METHODOLOGY
# ============================================================

st.markdown("---")
section_title("Research Methodology")

with st.expander("Dataset and experimental setup"):
    st.markdown(
        """
        **Dataset**
        - FI-2010 benchmark.
        - NoAuction Z-score subset.
        - Nine paired training and official test folds.
        - 144 input features per observation.

        **Sequence construction**
        - Sequence length: 100 observations.
        - Stride: 10 observations.
        - The first 99 observations of each test fold are used as sequence
          warm-up and are not scored.

        **Target**
        - Fifth FI-2010 label.
        - Three classes: DOWN, STATIONARY and UP.

        **Model**
        - HistGradientBoostingClassifier
        - max_iter = 150
        - learning_rate = 0.08
        - max_leaf_nodes = 31
        - l2_regularization = 1.0
        - random_state = 42

        **Feature representations**
        - Original: latest values, mean, standard deviation and linear trend
          (576 features).
        - Expanded: original groups plus minimum, maximum, median, IQR,
          recent-window mean and recent-window standard deviation
          (1,440 features).
        """
    )

with st.expander("Research findings and limitations"):
    st.markdown(
        """
        **Current expanded-feature official test results**
        - Mean fold accuracy: 65.41%.
        - Mean fold macro F1: 64.65%.
        - Mean DOWN F1: 64.99%.
        - Mean STATIONARY F1: 64.03%.
        - Mean UP F1: 64.92%.

        **Interpretation and limitations**
        - These are fold-level means across the FI-2010 official test folds.
        - The nine folds are not nine independent markets.
        - The results do not establish cross-asset generalisation.
        - Classification metrics do not demonstrate trading profitability.
        - Transaction costs, slippage, market impact and execution are not
          modelled.
        - No causal relationship is established by the feature experiments.
        - The dashboard's pooled prediction-level metrics may differ slightly
          from the unweighted mean of fold-level metrics because test-fold
          sizes differ.
        """
    )


# ============================================================
# 19. FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px;
        color:#94A3B8;
        font-size:13px;
    ">
        <strong style="color:#38BDF8;">MOSAIC</strong><br>
        Market Order-flow Sequence Intelligence
        & Cross-Asset Learning<br>
        Quantitative Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)