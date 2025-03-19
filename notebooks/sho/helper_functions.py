import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

# Constants
POOL_COLUMNS = [
    "GBBFS", "GBBGH", "GBCDF", "GBDCT", "GBDPT", "GBFXS", "GBGMO",
    "GBLDS", "GBLGP", "GBLPL", "GBMNC", "GBSFD", "GBSOU", "GBSSH",
    "GBTEE", "GBWID",
]

TEST_COLUMNS = POOL_COLUMNS + ["horizon", "pred_date"]

PREDICTION_TYPES = {
    "week": str,
    "horizon": float,
    "pred_date": str,
    "pool": str,
}


def mpe(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Percentage Error (MPE)."""
    mask = y > 0
    return np.mean((y[mask] - y_pred[mask]) / y[mask]) * 100


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((y - y_pred) ** 2))


def wmape(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error (WMAPE)."""
    return np.mean(np.abs(y - y_pred)) / np.mean(y) * 100


def _build_weekly_df(
    data_array: np.ndarray,
    base_date: pd.Timestamp,
    offset: int,
    pool_columns: list,
    pred_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a DataFrame for weekly predictions or actuals.
    
    Parameters:
        data_array (np.ndarray): Array of prediction or actual values.
        base_date (pd.Timestamp): Base start date.
        offset (int): Offset (in weeks) added to the base date.
        pool_columns (list): List of pool column names.
        pred_date (pd.Timestamp): Prediction date.
    
    Returns:
        pd.DataFrame: DataFrame with weekly data.
    """
    start_date = base_date + pd.Timedelta(offset, unit="W")
    weeks = pd.date_range(start=start_date, periods=13, freq="W")
    horizons = np.arange(1, 14).reshape(13, 1)
    
    df = pd.DataFrame(data=data_array.T, index=weeks, columns=pool_columns)
    df.index.name = "week"
    df["horizon"] = horizons
    df["pred_date"] = pred_date
    return df


def create_predictions(all_preds: np.ndarray, all_targets: np.ndarray) -> pd.DataFrame:
    """
    Create a merged DataFrame of predictions and actuals.
    
    Parameters:
        all_preds (np.ndarray): Array of predictions.
        all_targets (np.ndarray): Array of actual values.
    
    Returns:
        pd.DataFrame: Merged DataFrame with predictions and actuals.
    """
    base_date = pd.to_datetime("2023-08-06")
    pred_date = base_date - pd.Timedelta(1, unit="W")
    
    # Build DataFrames for actuals
    actuals_dfs = [
        _build_weekly_df(all_targets[i], base_date, i, POOL_COLUMNS, pred_date)
        for i in range(all_targets.shape[0])
    ]
    actuals = pd.concat(actuals_dfs).reset_index()
    actuals = pd.melt(
        actuals,
        id_vars=["week", "horizon", "pred_date"],
        value_vars=POOL_COLUMNS,
        var_name="pool",
        value_name="actuals",
    )
    actuals["actuals"] = actuals["actuals"].astype(float)
    actuals = actuals.astype(PREDICTION_TYPES)
    
    # Build DataFrames for predictions
    predictions_dfs = [
        _build_weekly_df(all_preds[i], base_date, i, POOL_COLUMNS, pred_date)
        for i in range(all_preds.shape[0])
    ]
    predictions_df = pd.concat(predictions_dfs).reset_index()
    predictions_df = pd.melt(
        predictions_df,
        id_vars=["week", "horizon", "pred_date"],
        value_vars=POOL_COLUMNS,
        var_name="pool",
        value_name="prediction",
    )
    predictions_df["prediction"] = predictions_df["prediction"].astype(float)
    predictions_df = predictions_df.astype(PREDICTION_TYPES)
    
    # Merge predictions and actuals
    merged = pd.merge(
        actuals, predictions_df, on=["week", "horizon", "pred_date", "pool"]
    )
    
    # Filter by week range
    merged = merged.loc[
        (merged.week >= "2023-10-29") & (merged.week <= "2024-07-28")
    ]
    return merged


def create_results_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate error metrics grouped by horizon.
    
    Parameters:
        predictions (pd.DataFrame): DataFrame with predictions and actuals.
    
    Returns:
        pd.DataFrame: Error metrics (RMSE, MPE, WMAPE) per horizon.
    """
    rmse_df = predictions.groupby("horizon").apply(
        lambda x: np.sqrt(np.mean((x["actuals"] - x["prediction"]) ** 2))
    )
    mpe_df = predictions.groupby("horizon").apply(
        lambda x: np.mean((x["actuals"] - x["prediction"]) / x["actuals"]) * 100
    )
    wmape_df = predictions.groupby("horizon").apply(
        lambda x: np.mean(np.abs(x["actuals"] - x["prediction"])) / np.mean(x["actuals"]) * 100
    )
    results = pd.concat([rmse_df, mpe_df, wmape_df], axis=1)
    results.columns = ["RMSE", "MPE", "WMAPE"]
    return results


def create_results_pool(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate error metrics grouped by pool.
    
    Parameters:
        predictions (pd.DataFrame): DataFrame with predictions and actuals.
    
    Returns:
        pd.DataFrame: Error metrics (RMSE, MPE, WMAPE) per pool.
    """
    rmse_df = predictions.groupby("pool").apply(
        lambda x: np.sqrt(np.mean((x["actuals"] - x["prediction"]) ** 2))
    )
    mpe_df = predictions.groupby("pool").apply(
        lambda x: np.mean((x["actuals"] - x["prediction"]) / x["actuals"]) * 100
    )
    wmape_df = predictions.groupby("pool").apply(
        lambda x: np.mean(np.abs(x["actuals"] - x["prediction"])) / np.mean(x["actuals"]) * 100
    )
    results = pd.concat([rmse_df, mpe_df, wmape_df], axis=1)
    results.columns = ["RMSE", "MPE", "WMAPE"]
    return results


def visualize_errors(predictions: pd.DataFrame) -> None:
    """
    Visualize prediction errors using a scatter plot and print error metrics.
    
    Parameters:
        predictions (pd.DataFrame): DataFrame with predictions and actuals.
    """
    pred_flat = predictions["prediction"].values
    true_flat = predictions["actuals"].values

    # Calculate metrics over the entire test set.
    error_rmse = rmse(true_flat, pred_flat)
    error_mpe = mpe(true_flat, pred_flat)
    error_wmape = wmape(true_flat, pred_flat)

    print(f"Test RMSE: {error_rmse:.4f}")
    print(f"Test MPE: {error_mpe:.4f}")
    print(f"Test WMAPE: {error_wmape:.4f}")

    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(x=true_flat, y=pred_flat, mode="markers", name="Test Data")
    )
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal",
            line=dict(dash="dash"),
        )
    )
    fig_scatter.update_layout(
        title="Predicted vs. True Values on Test Set",
        xaxis_title="True Value",
        yaxis_title="Predicted Value",
    )
    fig_scatter.show()


def visualize_results(predictions: pd.DataFrame, pool: str) -> None:
    """
    Visualize predictions and actuals for a specified pool.
    
    Parameters:
        predictions (pd.DataFrame): DataFrame with predictions and actuals.
        pool (str): Pool identifier to filter the data.
    """
    pred = predictions[predictions.pool == pool]
    act = pred[pred.horizon == 1][["week", "actuals"]]

    fig = px.line(pred, x="week", y="prediction", color="horizon")
    fig.add_scatter(x=act["week"], y=act["actuals"], mode="lines", name="actuals")
    fig.show()
