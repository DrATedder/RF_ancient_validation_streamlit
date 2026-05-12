# RF_ML_streamlit.py
# ============================================================
# STREAMLIT APP FOR ANCIENT DNA RANDOM FOREST CLASSIFICATION
# ============================================================

import os
import glob
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Ancient DNA RF Classifier",
    layout="wide"
)

st.title("Ancient DNA Random Forest Classifier")

st.markdown("""
This app trains and evaluates a Random Forest model for
Ancient vs Modern DNA classification.
""")

# ============================================================
# SIDEBAR CONFIG
# ============================================================

st.sidebar.header("Configuration")

data_dir = st.sidebar.text_input(
    "Dataset Directory",
    value="./data_sets"
)

random_state = st.sidebar.number_input(
    "Random State",
    value=42
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

n_estimators = st.sidebar.slider(
    "Number of Trees",
    min_value=50,
    max_value=1000,
    value=300,
    step=50
)

top_n = st.sidebar.slider(
    "Top Features",
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

run_button = st.sidebar.button("Run Analysis")

# ============================================================
# RESULTS DIRECTORY
# ============================================================

results_dir = os.path.join(
    data_dir,
    "RF_RESULTS"
)

os.makedirs(results_dir, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def find_datasets(data_dir):

    X_files = sorted(
        glob.glob(
            os.path.join(data_dir, "X_data_*.npy")
        )
    )

    datasets = []

    for x_file in X_files:

        dataset_name = (
            os.path.basename(x_file)
            .replace("X_data_", "")
            .replace(".npy", "")
        )

        y_file = os.path.join(
            data_dir,
            f"y_labels_{dataset_name}.npy"
        )

        if os.path.exists(y_file):

            datasets.append({
                "name": dataset_name,
                "X": x_file,
                "y": y_file
            })

    return datasets


@st.cache_data
def load_all_data(datasets):

    X_all = []
    y_all = []
    dataset_labels = []

    for ds in datasets:

        X = np.load(ds["X"])
        y = np.load(ds["y"])

        X = X.reshape(X.shape[0], -1)

        X_all.append(X)
        y_all.append(y)

        dataset_labels.extend(
            [ds["name"]] * len(y)
        )

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    dataset_labels = np.array(dataset_labels)

    return X_all, y_all, dataset_labels


def plot_confusion_matrix(y_test, y_pred):

    fig, ax = plt.subplots(figsize=(6, 6))

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Modern", "Ancient"]
    )

    disp.plot(ax=ax)

    plt.title("Confusion Matrix")

    return fig


def plot_roc_curve(y_test, y_prob):

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {roc_auc:.4f}"
    )

    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    ax.legend()

    return fig


def plot_pr_curve(y_test, y_prob):

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    ap = average_precision_score(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"AP = {ap:.4f}"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")

    ax.legend()

    return fig


def calculate_feature_importance(
    rf,
    X_all
):

    importances = rf.feature_importances_

    base_names = ["A", "C", "G", "T", "N"]

    total_features = X_all.shape[1]

    sequence_length = total_features // 5

    half_length = sequence_length // 2

    feature_rows = []

    for idx, importance in enumerate(importances):

        seq_position = idx // 5
        base_channel = idx % 5

        base = base_names[base_channel]

        if seq_position < half_length:

            read = "R1"
            read_position = seq_position

        else:

            read = "R2"
            read_position = seq_position - half_length

        feature_rows.append({
            "Feature_Index": idx,
            "Importance": importance,
            "Read": read,
            "Position": read_position,
            "Base": base
        })

    return pd.DataFrame(feature_rows)


def plot_feature_importance(top_features):

    fig, ax = plt.subplots(figsize=(16, 6))

    labels = [
        f"{r.Read}:{r.Position}:{r.Base}"
        for _, r in top_features.iterrows()
    ]

    ax.bar(
        range(len(top_features)),
        top_features["Importance"]
    )

    ax.set_xticks(range(len(top_features)))

    ax.set_xticklabels(
        labels,
        rotation=90
    )

    ax.set_xlabel("Read : Position : Base")
    ax.set_ylabel("Importance")
    ax.set_title("Top Feature Importances")

    plt.tight_layout()

    return fig


def plot_positional_importance(position_summary):

    fig, ax = plt.subplots(figsize=(12, 5))

    for read in ["R1", "R2"]:

        subset = (
            position_summary[
                position_summary["Read"] == read
            ]
            .sort_values("Position")
        )

        ax.plot(
            subset["Position"],
            subset["Importance"],
            linewidth=2,
            label=read
        )

    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Summed Importance")
    ax.set_title("Positional Importance Profile")

    ax.legend()

    return fig


# ============================================================
# MAIN APP
# ============================================================

if run_button:

    st.header("Dataset Discovery")

    datasets = find_datasets(data_dir)

    if len(datasets) == 0:

        st.error("No datasets found.")
        st.stop()

    st.success(
        f"Found {len(datasets)} datasets."
    )

    dataset_df = pd.DataFrame(datasets)

    st.dataframe(
        dataset_df[["name"]]
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    with st.spinner("Loading datasets..."):

        X_all, y_all, dataset_labels = load_all_data(
            datasets
        )

    st.header("Combined Dataset")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Samples",
            X_all.shape[0]
        )

    with col2:

        st.metric(
            "Features",
            X_all.shape[1]
        )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
        ds_train,
        ds_test
    ) = train_test_split(
        X_all,
        y_all,
        dataset_labels,
        test_size=test_size,
        stratify=y_all,
        random_state=random_state
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    st.header("Training Random Forest")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state
    )

    with st.spinner("Training model..."):

        rf.fit(X_train, y_train)

    st.success("Model training complete.")

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        results_dir,
        "random_forest_combined.joblib"
    )

    joblib.dump(rf, model_path)

    st.info(f"Model saved to:\n{model_path}")

    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred = rf.predict(X_test)

    y_prob = rf.predict_proba(X_test)[:, 1]

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    st.header("Overall Performance")

    overall_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Average Precision": average_precision_score(
            y_test,
            y_prob
        )
    }

    metrics_df = pd.DataFrame(
        overall_metrics,
        index=["Combined Model"]
    )

    st.dataframe(
        metrics_df.round(4)
    )

    metrics_df.to_csv(
        os.path.join(
            results_dir,
            "overall_metrics.csv"
        )
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.header("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Modern", "Ancient"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4)
    )

    report_df.to_csv(
        os.path.join(
            results_dir,
            "classification_report.csv"
        )
    )

    # ========================================================
    # EVALUATION PLOTS
    # ========================================================

    st.header("Evaluation Plots")

    col1, col2 = st.columns(2)

    with col1:

        fig_cm = plot_confusion_matrix(
            y_test,
            y_pred
        )

        fig_cm.savefig(
            os.path.join(
                results_dir,
                "confusion_matrix.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        st.pyplot(fig_cm)

    with col2:

        fig_roc = plot_roc_curve(
            y_test,
            y_prob
        )

        fig_roc.savefig(
            os.path.join(
                results_dir,
                "roc_curve.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        st.pyplot(fig_roc)

    fig_pr = plot_pr_curve(
        y_test,
        y_prob
    )

    fig_pr.savefig(
        os.path.join(
            results_dir,
            "precision_recall_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    st.pyplot(fig_pr)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.header("Feature Importance")

    feature_df = calculate_feature_importance(
        rf,
        X_all
    )

    top_features = (
        feature_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_n)
    )

    st.dataframe(
        top_features.round(6)
    )

    top_features.to_csv(
        os.path.join(
            results_dir,
            "top_feature_importances.csv"
        ),
        index=False
    )

    fig_feat = plot_feature_importance(
        top_features
    )

    fig_feat.savefig(
        os.path.join(
            results_dir,
            "feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    st.pyplot(fig_feat)

    # ========================================================
    # POSITIONAL IMPORTANCE
    # ========================================================

    st.header("Positional Importance")

    position_summary = (
        feature_df
        .groupby(["Read", "Position"])["Importance"]
        .sum()
        .reset_index()
    )

    st.dataframe(
        position_summary.round(6)
    )

    position_summary.to_csv(
        os.path.join(
            results_dir,
            "positional_importance.csv"
        ),
        index=False
    )

    fig_pos = plot_positional_importance(
        position_summary
    )

    fig_pos.savefig(
        os.path.join(
            results_dir,
            "positional_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    st.pyplot(fig_pos)

    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    st.header("Cross Validation")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    with st.spinner("Running cross-validation..."):

        cv_scores = cross_val_score(
            rf,
            X_all,
            y_all,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

    cv_df = pd.DataFrame({
        "Fold": np.arange(
            1,
            len(cv_scores) + 1
        ),
        "Accuracy": cv_scores
    })

    st.dataframe(
        cv_df.round(4)
    )

    cv_df.to_csv(
        os.path.join(
            results_dir,
            "cross_validation_scores.csv"
        ),
        index=False
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Mean Accuracy",
            f"{cv_scores.mean():.4f}"
        )

    with col2:

        st.metric(
            "Std Accuracy",
            f"{cv_scores.std():.4f}"
        )

    # ========================================================
    # PER-DATASET PERFORMANCE
    # ========================================================

    st.header("Per-Dataset Performance")

    dataset_results = []

    for dataset in np.unique(ds_test):

        mask = ds_test == dataset

        y_true_ds = y_test[mask]
        y_pred_ds = y_pred[mask]

        dataset_results.append({
            "Dataset": dataset,
            "Accuracy": accuracy_score(
                y_true_ds,
                y_pred_ds
            ),
            "Precision": precision_score(
                y_true_ds,
                y_pred_ds
            ),
            "Recall": recall_score(
                y_true_ds,
                y_pred_ds
            ),
            "F1": f1_score(
                y_true_ds,
                y_pred_ds
            ),
            "N_Reads": len(y_true_ds)
        })

    dataset_df = pd.DataFrame(dataset_results)

    dataset_df = dataset_df.sort_values(
        "Accuracy",
        ascending=False
    )

    st.dataframe(
        dataset_df.round(4)
    )

    dataset_df.to_csv(
        os.path.join(
            results_dir,
            "per_dataset_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # DATASET ACCURACY PLOT
    # ========================================================

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        dataset_df["Dataset"],
        dataset_df["Accuracy"]
    )

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Dataset")
    ax.set_title("Per-Dataset Accuracy")

    plt.xticks(rotation=90)

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            results_dir,
            "dataset_accuracy.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    st.pyplot(fig)

    # ========================================================
    # SUMMARY FILE
    # ========================================================

    summary_path = os.path.join(
        results_dir,
        "summary.txt"
    )

    with open(summary_path, "w") as f:

        f.write("===================================\n")
        f.write("RANDOM FOREST SUMMARY\n")
        f.write("===================================\n\n")

        f.write(
            f"Samples: {X_all.shape[0]}\n"
        )

        f.write(
            f"Features: {X_all.shape[1]}\n\n"
        )

        for k, v in overall_metrics.items():

            f.write(
                f"{k}: {v:.4f}\n"
            )

        f.write("\n")

        f.write(
            f"CV Mean Accuracy: "
            f"{cv_scores.mean():.4f}\n"
        )

        f.write(
            f"CV Std Accuracy: "
            f"{cv_scores.std():.4f}\n"
        )

    # ========================================================
    # DOWNLOADS
    # ========================================================

    st.header("Download Results")

    csv = metrics_df.to_csv().encode("utf-8")

    st.download_button(
        label="Download Overall Metrics CSV",
        data=csv,
        file_name="overall_metrics.csv",
        mime="text/csv"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    st.success(
        f"Analysis complete.\n\n"
        f"Results saved to:\n{results_dir}"
    )
