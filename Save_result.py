import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import MAE, MAPE, RMSE, correlation


LOAD_NAMES = ("electricity", "cooling", "heating")


def _write_first_step_csv(path, y_true, y_pred):
    header = ['test_y', 'predicted_values']
    values = np.concatenate((y_true, y_pred), axis=1)
    with open(path, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(values)


def _plot_first_month(path, title, ylabel, all_y_true, all_predict_values, node_id):
    time = 24 * 31
    plt.figure(figsize=(20, 10))
    plt.title(title)
    plt.xlabel("time/one_hour")
    plt.ylabel(ylabel)
    plt.plot(all_y_true[:time, 0, node_id], linewidth=6.0, label='true')
    plt.plot(all_predict_values[:time, 0, node_id], linewidth=1.0, label='pred')
    plt.legend()
    plt.savefig(path)
    plt.close()


def show_pred(all_y_true, all_predict_values, horizon, result_dir='./result', assets_dir=None, write_plots=True):
    os.makedirs(result_dir, exist_ok=True)
    if assets_dir is None:
        assets_dir = os.path.join(result_dir, 'assets')
    if write_plots:
        os.makedirs(assets_dir, exist_ok=True)

    for load_idx, load_name in enumerate(LOAD_NAMES):
        _write_first_step_csv(
            os.path.join(result_dir, f'{load_name}.csv'),
            all_y_true[:, 0, load_idx:load_idx + 1],
            all_predict_values[:, 0, load_idx:load_idx + 1],
        )

    report_path = os.path.join(result_dir, 'prediction_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        for load_idx, load_name in enumerate(LOAD_NAMES):
            y_true = all_y_true[:, 0, load_idx:load_idx + 1]
            y_pred = all_predict_values[:, 0, load_idx:load_idx + 1]
            mae = MAE(y_true, y_pred)
            mape = MAPE(y_true, y_pred)
            rmse = RMSE(y_true, y_pred)
            corr = correlation(y_true, y_pred)
            print(f"============{load_name} first-step===================")
            print("MAE = " + str(mae))
            print("MAPE = " + str(mape))
            print("RMSE = " + str(rmse))
            print("corr = " + str(corr))
            print(f"============{load_name} first-step===================", file=f, flush=True)
            print("MAE = " + str(mae), file=f, flush=True)
            print("MAPE = " + str(mape), file=f, flush=True)
            print("RMSE = " + str(rmse), file=f, flush=True)
            print("corr = " + str(corr), file=f, flush=True)

        mae = MAE(all_y_true, all_predict_values)
        rmse = RMSE(all_y_true, all_predict_values)
        mape = MAPE(all_y_true, all_predict_values)
        print("Cfc full-horizon original-value metrics  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))
        print("Cfc full-horizon original-value metrics  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape), file=f)

    if write_plots:
        _plot_first_month(os.path.join(assets_dir, "the first month pred electricity.png"), "electricity", "electricity", all_y_true, all_predict_values, 0)
        _plot_first_month(os.path.join(assets_dir, "the first month pred cooling.png"), "cooling", "cooling", all_y_true, all_predict_values, 1)
        _plot_first_month(os.path.join(assets_dir, "the first month pred heating.png"), "heating", "heating", all_y_true, all_predict_values, 2)
