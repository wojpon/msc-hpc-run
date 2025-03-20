import os
import pickle
import sys
import logging
import traceback
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from time import time
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import optuna

MAIN_PATH = "/dtu/3d-imaging-center/courses/02509/groups/group10/msc-hpc-run/"

# ---------------------------
# Setup Logging
# ---------------------------
LOG_DIR = os.path.join(MAIN_PATH, "output/mho/logs_gat")
OUTPUT_DIR = os.path.join(MAIN_PATH, "output/mho/study_results_gat")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "run.log"))
    ]
)

# ---------------------------
# Global settings and seeding
# ---------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
pl.seed_everything(42)

# Global paths and constants
IMPORTS_PATH = os.path.join(MAIN_PATH, "data/import-volume.csv")
ADJACENCY_MATRIX = pd.read_parquet(os.path.join(MAIN_PATH, "data/adjacency-matrix.parquet"))
BOOKINGS_PATH = os.path.join(MAIN_PATH, "data/bookings_data.pkl")

# Global hyperparameters and settings
use_validation = False  
NODE_FEATURES = 19        # 6 from volume/week encoding + 13 booking features
TIME_WINDOW_SIZE = 13
LOADERS_WOKRES = 4

MIN_HORIZON = 1
MAX_HORIZON = 14

NUM_TRIALS = 100
MAX_EPOCHS = 300

EARLY_STOP_PATIENCE = 7
EARLY_STOP_DELTA = 0.001

# ---------------------------
# Data and Graph Preparation
# ---------------------------
def get_import_data():
    """Load and pivot the import volume data."""
    try:
        import_data = pd.read_csv(IMPORTS_PATH)
        import_data["week"] = pd.to_datetime(import_data["week"])
        import_data["pool"] = import_data["pool"].astype("str")
        import_data["volume"] = import_data["import"].astype("float")
        import_data.drop(columns=["import"], inplace=True)
        import_data = import_data.sort_values("week")
        import_data = import_data.loc[
            (import_data.week >= pd.to_datetime("2017-05-01")) &
            (import_data.week <= pd.to_datetime("2024-10-20"))
        ]
        logging.info(f"Imports data range from {import_data['week'].min().strftime('%Y-%m-%d')} "
                     f"to {import_data['week'].max().strftime('%Y-%m-%d')}")
        import_data = import_data.pivot(index='week', columns='pool', values='volume').T
        import_data = import_data.fillna(0)
        return import_data
    except Exception as e:
        logging.error("Error in get_import_data(): " + str(e))
        raise

def get_graph_structure(threshold, a):
    """Construct the graph structure (edge_index and edge_weights) from the adjacency matrix."""
    try:
        a = a.reset_index(drop=True)
        a.columns = range(a.shape[1])
        a = a.where(pd.notnull(a), a.T)
        a = a.to_numpy()
        a_filtered = (a < threshold).astype(np.int32)
        edge_index = torch.nonzero(torch.tensor(a_filtered, dtype=torch.long), as_tuple=False).t()
        edge_weights = []
        for e in edge_index.numpy().T:
            distance = a[e[0], e[1]]
            edge_weights.append(distance)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        return edge_index, edge_weights
    except Exception as e:
        logging.error("Error in get_graph_structure(): " + str(e))
        raise

def prepare_data(use_validation, prediction_horizon, batch_size):
    """Preprocess the data, create features, and return DataLoaders."""
    try:
        data = get_import_data()
        bookings = pickle.load(open(BOOKINGS_PATH, "rb"))
        logging.info(f"Bookings data range from {min(bookings.keys())} to {max(bookings.keys())}")

        if use_validation:
            val_weeks = 52
            test_weeks = 77
            train_weeks = data.shape[1] - val_weeks - test_weeks
        else:
            test_weeks = 77
            train_weeks = data.shape[1] - test_weeks

        volume_tensor = torch.tensor(data.values, dtype=torch.float32)
        week_numbers = data.columns.to_series().dt.isocalendar().week.astype(np.float32).values

        if use_validation:
            train_volume = volume_tensor[:, :train_weeks]
            val_volume = volume_tensor[:, train_weeks: train_weeks + val_weeks]
            test_volume = volume_tensor[:, train_weeks + val_weeks: train_weeks + val_weeks + test_weeks]
        else:
            train_volume = volume_tensor[:, :train_weeks]
            test_volume = volume_tensor[:, train_weeks: train_weeks + test_weeks]

        # Normalize volume using training statistics
        train_mean = train_volume.mean(dim=1, keepdim=True)
        train_std = train_volume.std(dim=1, keepdim=True) + 1e-6
        train_volume_norm = (train_volume - train_mean) / train_std
        if use_validation:
            val_volume_norm = (val_volume - train_mean) / train_std
        test_volume_norm = (test_volume - train_mean) / train_std

        # Create week encoding features
        week_numbers_tensor = torch.tensor(week_numbers, dtype=torch.float32)
        sin_week = torch.sin(2 * np.pi * week_numbers_tensor / 52)
        cos_week = torch.cos(2 * np.pi * week_numbers_tensor / 52)
        holiday_ohe = (week_numbers_tensor == 52).long()
        weeks_to_holiday = (52 - week_numbers_tensor) % 52
        weeks_from_holiday = (week_numbers_tensor - 52) % 52

        def split_features(feature, n_train, n_val, n_test):
            train_feat = feature[:n_train]
            val_feat = feature[n_train: n_train + n_val]
            test_feat = feature[n_train + n_val: n_train + n_val + n_test]
            return train_feat, val_feat, test_feat

        if use_validation:
            train_sin, val_sin, test_sin = split_features(sin_week, train_weeks, val_weeks, test_weeks)
            train_cos, val_cos, test_cos = split_features(cos_week, train_weeks, val_weeks, test_weeks)
            train_holiday, val_holiday, test_holiday = split_features(holiday_ohe, train_weeks, val_weeks, test_weeks)
            train_to, val_to, test_to = split_features(weeks_to_holiday, train_weeks, val_weeks, test_weeks)
            train_from, val_from, test_from = split_features(weeks_from_holiday, train_weeks, val_weeks, test_weeks)
        else:
            train_sin = sin_week[:train_weeks]
            test_sin = sin_week[train_weeks: train_weeks + test_weeks]
            train_cos = cos_week[:train_weeks]
            test_cos = cos_week[train_weeks: train_weeks + test_weeks]
            train_holiday = holiday_ohe[:train_weeks]
            test_holiday = holiday_ohe[train_weeks: train_weeks + test_weeks]
            train_to = weeks_to_holiday[:train_weeks]
            test_to = weeks_to_holiday[train_weeks: train_weeks + test_weeks]
            train_from = weeks_from_holiday[:train_weeks]
            test_from = weeks_from_holiday[train_weeks: train_weeks + test_weeks]

        def create_week_feature_tensor(week_feat, num_nodes):
            if week_feat.dim() == 1:
                week_feat = week_feat.unsqueeze(1)
            return week_feat.unsqueeze(0).repeat(num_nodes, 1, 1)

        num_nodes = 16  # Hard-coded based on the data
        train_sin_feat = create_week_feature_tensor(train_sin, num_nodes)
        train_cos_feat = create_week_feature_tensor(train_cos, num_nodes)
        train_holiday_feat = create_week_feature_tensor(train_holiday, num_nodes)
        train_to_feat = create_week_feature_tensor(train_to, num_nodes)
        train_from_feat = create_week_feature_tensor(train_from, num_nodes)

        if use_validation:
            val_sin_feat = create_week_feature_tensor(val_sin, num_nodes)
            val_cos_feat = create_week_feature_tensor(val_cos, num_nodes)
            val_holiday_feat = create_week_feature_tensor(val_holiday, num_nodes)
            val_to_feat = create_week_feature_tensor(val_to, num_nodes)
            val_from_feat = create_week_feature_tensor(val_from, num_nodes)
            test_sin_feat = create_week_feature_tensor(test_sin, num_nodes)
            test_cos_feat = create_week_feature_tensor(test_cos, num_nodes)
            test_holiday_feat = create_week_feature_tensor(test_holiday, num_nodes)
            test_to_feat = create_week_feature_tensor(test_to, num_nodes)
            test_from_feat = create_week_feature_tensor(test_from, num_nodes)
        else:
            test_sin_feat = create_week_feature_tensor(test_sin, num_nodes)
            test_cos_feat = create_week_feature_tensor(test_cos, num_nodes)
            test_holiday_feat = create_week_feature_tensor(test_holiday, num_nodes)
            test_to_feat = create_week_feature_tensor(test_to, num_nodes)
            test_from_feat = create_week_feature_tensor(test_from, num_nodes)

        train_volume_feat = train_volume_norm.unsqueeze(2)
        if use_validation:
            val_volume_feat = val_volume_norm.unsqueeze(2)
        test_volume_feat = test_volume_norm.unsqueeze(2)

        if use_validation:
            train_data_combined = torch.cat([
                train_volume_feat, train_sin_feat, train_cos_feat,
                train_holiday_feat, train_to_feat, train_from_feat
            ], dim=2)
            val_data_combined = torch.cat([
                val_volume_feat, val_sin_feat, val_cos_feat,
                val_holiday_feat, val_to_feat, val_from_feat
            ], dim=2)
            test_data_combined = torch.cat([
                test_volume_feat, test_sin_feat, test_cos_feat,
                test_holiday_feat, test_to_feat, test_from_feat
            ], dim=2)
        else:
            train_data_combined = torch.cat([
                train_volume_feat, train_sin_feat, train_cos_feat,
                train_holiday_feat, train_to_feat, train_from_feat
            ], dim=2)
            test_data_combined = torch.cat([
                test_volume_feat, test_sin_feat, test_cos_feat,
                test_holiday_feat, test_to_feat, test_from_feat
            ], dim=2)

        # Process booking features
        booking_list = []
        for date in data.columns:
            booking_df = bookings[date.strftime("%Y-%m-%d")]
            booking_list.append(booking_df.values)
        booking_array = np.stack(booking_list, axis=1)
        bookings_tensor = torch.tensor(booking_array, dtype=torch.float32)

        if use_validation:
            train_bookings = bookings_tensor[:, :train_weeks, :]
            val_bookings = bookings_tensor[:, train_weeks: train_weeks + val_weeks, :]
            test_bookings = bookings_tensor[:, train_weeks + val_weeks: train_weeks + val_weeks + test_weeks, :]
        else:
            train_bookings = bookings_tensor[:, :train_weeks, :]
            test_bookings = bookings_tensor[:, train_weeks: train_weeks + test_weeks, :]

        booking_mean = train_bookings.mean(dim=(0, 1), keepdim=True)
        booking_std = train_bookings.std(dim=(0, 1), keepdim=True) + 1e-6
        train_bookings = (train_bookings - booking_mean) / booking_std
        if use_validation:
            val_bookings = (val_bookings - booking_mean) / booking_std
        test_bookings = (test_bookings - booking_mean) / booking_std

        if use_validation:
            train_data_combined = torch.cat([train_data_combined, train_bookings], dim=2)
            val_data_combined = torch.cat([val_data_combined, val_bookings], dim=2)
            test_data_combined = torch.cat([test_data_combined, test_bookings], dim=2)
        else:
            train_data_combined = torch.cat([train_data_combined, train_bookings], dim=2)
            test_data_combined = torch.cat([test_data_combined, test_bookings], dim=2)

        class TimeSeriesDataset(Dataset):
            def __init__(self, data, window_size, horizon):
                self.data = data
                self.window_size = window_size
                self.horizon = horizon  # Uses the provided prediction horizon
                self.num_samples = data.shape[1] - window_size - (horizon - 1)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                x = self.data[:, idx: idx + self.window_size, :].transpose(0, 1)
                y = self.data[:, idx + self.window_size + self.horizon - 1, 0]
                return x, y

        window_size = TIME_WINDOW_SIZE
        train_dataset = TimeSeriesDataset(train_data_combined, window_size, horizon=prediction_horizon)
        test_dataset = TimeSeriesDataset(test_data_combined, window_size, horizon=prediction_horizon)
        if use_validation:
            val_dataset = TimeSeriesDataset(val_data_combined, window_size, horizon=prediction_horizon)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  pin_memory=True, num_workers=LOADERS_WOKRES)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=LOADERS_WOKRES)
        if use_validation:
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                    pin_memory=True)
        else:
            val_loader = None

        return train_loader, val_loader, test_loader
    except Exception as e:
        logging.error("Error in prepare_data(): " + str(e))
        traceback.print_exc()
        raise

def prepare_graph(threshold):
    """Prepare the graph structure (edge_index, edge_weights) using the given threshold."""
    try:
        edge_index, edge_weights = get_graph_structure(threshold, ADJACENCY_MATRIX)
        return edge_index, edge_weights
    except Exception as e:
        logging.error("Error in prepare_graph(): " + str(e))
        raise

# ---------------------------
# Model Definitions
# ---------------------------
class GNNLSTM(nn.Module):
    def __init__(self, in_channels, gnn_hidden, gat_heads, gat_dropout, gnn_dropout,
                 lstm_hidden, lstm_layers, lstm_dropout):
        super(GNNLSTM, self).__init__()
        self.gnn1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=gnn_hidden,
            heads=gat_heads,
            concat=True,
            dropout=gat_dropout,
        )
        self.gnn2 = GATv2Conv(
            in_channels=gnn_hidden * gat_heads,
            out_channels=gnn_hidden,
            heads=1,
            concat=False,
        )
        self.lstm = nn.LSTM(
            input_size=gnn_hidden,
            hidden_size=lstm_hidden,
            batch_first=True,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

        self.norm1 = nn.LayerNorm(gnn_hidden * gat_heads)
        self.norm2 = nn.LayerNorm(gnn_hidden)
        self.relu = nn.ReLU()
        self.gnn_dropout = gnn_dropout

    def forward(self, x, edge_index):
        # x: (batch_size, seq_len, num_nodes, in_channels)
        batch_size, seq_len, num_nodes, _ = x.shape
        device = x.device

        total_graphs = batch_size * seq_len
        x_reshaped = x.reshape(total_graphs, num_nodes, -1)

        E = edge_index.size(1)
        batched_edge_index = edge_index.unsqueeze(0).repeat(total_graphs, 1, 1)
        offsets = (torch.arange(total_graphs, device=device) * num_nodes).view(total_graphs, 1, 1)
        batched_edge_index = batched_edge_index + offsets

        if E != 0:
            batched_edge_index = batched_edge_index.cpu().numpy()
            edge_index_final = []
            for l in range(batched_edge_index.shape[0]):
                for k in range(E):
                    edge_index_final.append(np.array([batched_edge_index[l, 0, k], batched_edge_index[l, 1, k]]))
            
            batched_edge_index = torch.tensor(np.vstack(edge_index_final), device=device).t().contiguous()
        
        else:
            batched_edge_index = batched_edge_index.reshape(2,0)

        x_flat = x_reshaped.reshape(total_graphs * num_nodes, -1)

        gnn_out = self.gnn1(x_flat, batched_edge_index)
        gnn_out = self.norm1(gnn_out)
        gnn_out = self.relu(gnn_out)
        gnn_out = F.dropout(gnn_out, p=self.gnn_dropout, training=self.training)
        gnn_out = self.gnn2(gnn_out, batched_edge_index)
        gnn_out = self.norm2(gnn_out)
        gnn_out = gnn_out.reshape(total_graphs, num_nodes, -1)
        gnn_out = gnn_out.reshape(batch_size, seq_len, num_nodes, -1)

        lstm_input = gnn_out.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_input)
        last_out = lstm_out[:, -1, :]
        pred = self.fc(last_out)
        pred = pred.reshape(batch_size, num_nodes)
        return pred

class LitGNNLSTM(pl.LightningModule):
    def __init__(self, in_channels, gnn_hidden, gat_heads, gat_dropout, gnn_dropout,
                 lstm_hidden, lstm_layers, lstm_dropout, learning_rate, edge_index):
        super(LitGNNLSTM, self).__init__()
        self.model = GNNLSTM(in_channels, gnn_hidden, gat_heads, gat_dropout, gnn_dropout,
                             lstm_hidden, lstm_layers, lstm_dropout)
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.register_buffer("edge_index", edge_index)

    def forward(self, x):
        return self.model(x, self.edge_index)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

def create_model(edge_index, params):
    """Create the Lightning model using hyperparameters from Optuna."""
    try:
        model = LitGNNLSTM(
            in_channels=NODE_FEATURES,
            gnn_hidden=params["gnn_hidden"],
            gat_heads=params["gat_heads"],
            gat_dropout=params["gat_dropout"],
            gnn_dropout=params["gnn_dropout"],
            lstm_hidden=params["lstm_hidden"],
            lstm_layers=params["lstm_layers"],
            lstm_dropout=params["lstm_dropout"],
            learning_rate=params["learning_rate"],
            edge_index=edge_index,
        )
        return model
    except Exception as e:
        logging.error("Error in create_model(): " + str(e))
        raise

# ---------------------------
# Trainer Setup
# ---------------------------
def create_trainer(max_epochs):
    """Create a PyTorch Lightning Trainer with early stopping and AMP enabled."""
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=EARLY_STOP_DELTA,
        patience=EARLY_STOP_PATIENCE,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[early_stop_callback],
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    return trainer

# ---------------------------
# Optuna Objective Function
# ---------------------------
def objective(trial: optuna.Trial):
    try:
        # Sample hyperparameters using trial suggestions.
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "gnn_hidden": trial.suggest_int("gnn_hidden", 16, 512, step=16),
            "gnn_dropout": trial.suggest_float("gnn_dropout", 0.0, 0.7, step=0.1),
            "gat_heads": trial.suggest_int("gat_heads", 1, 16, step=1),
            "gat_dropout": trial.suggest_float("gat_dropout", 0.0, 0.7, step=0.1),
            "lstm_hidden": trial.suggest_int("lstm_hidden", 16, 512, step=16),
            "lstm_dropout": trial.suggest_float("lstm_dropout", 0.0, 0.7, step=0.1),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 2, step=1),
            "graph_threshold": trial.suggest_int("graph_threshold", 0, 800, step=50),
            "batch_size": trial.suggest_int("batch_size", 8, 32, step=8),
        }

        # Prepare data and graph for tuning.
        train_loader, _, test_loader = prepare_data(use_validation=use_validation, prediction_horizon=PREDICTION_HORIZON, batch_size=params["batch_size"])
        edge_index, _ = prepare_graph(params["graph_threshold"])

        model = create_model(edge_index, params)
        trainer = create_trainer(max_epochs=MAX_EPOCHS)

        # Run training.
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        trial.set_user_attr("epochs", str(trainer.current_epoch + 1))

        # Get validation loss from the trainer callback metrics.
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return float("inf")
        return val_loss.item()
    except Exception as e:
        logging.error("Error in objective(): " + str(e))
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        return float("inf")

# ---------------------------
# Main Routine for Optuna Tuning
# ---------------------------
def main():
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=NUM_TRIALS)

        trials_df = study.trials_dataframe()
        output_file = os.path.join(OUTPUT_DIR, f"HORIZON_{PREDICTION_HORIZON}_trials_df.parquet")
        trials_df.to_parquet(output_file)

        with open(OUTPUT_DIR + f'/HORIZON_{PREDICTION_HORIZON}_study.pickle', 'wb') as handle:
            pickle.dump(study, handle)

        logging.info(f"Trials dataframe saved to {output_file}")

        logging.info("Best trial:")
        best_trial = study.best_trial
        logging.info(f"  Value: {best_trial.value}")
        logging.info("  Params: ")
        for key, value in best_trial.params.items():
            logging.info(f"    {key}: {value}")

    except Exception as e:
        logging.error("Error in main(): " + str(e))
        traceback.print_exc()
        raise

# ---------------------------
# Run Over Multiple Prediction Horizons
# ---------------------------
if __name__ == "__main__":
    horizons = range(MIN_HORIZON, MAX_HORIZON)
    for h in horizons:
        try:
            PREDICTION_HORIZON = h
            print(f"\nTUNING FOR HORIZON {PREDICTION_HORIZON}\n")
            logging.info(f"Optimizing for prediction horizon {PREDICTION_HORIZON}...")
            main()
        except Exception as e:
            logging.error(f"Error while optimizing for horizon {h}: {e}")
            continue
