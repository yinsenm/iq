"""
Name     : portfolio_analyzer.py
Author   : Yinsen Miao
Contact  : yinsenm@gmail.com
Time     : July 2024
Desc     : assess the portfolio performance
"""
import os.path

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import stats
from math import sqrt
import pandas_datareader.data as web  # download 3m t-bill from FRED as risk-free rate
rf_cache_path = "../data"  # path to save for risk free rate data

class portfolio_analyzer:
    def __init__(self):
        self.rf_benchmark = "DGS3MO"

    def get_df_rf(self, begin_date: str, end_date: str) -> pd.DataFrame:
        rf_cache_file = ("%s/%s_%s_%s.csv" % (
            rf_cache_path, self.rf_benchmark.lower(), begin_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        ))
        if os.path.isfile(rf_cache_file):
            df_rf = pd.read_csv(rf_cache_file, parse_dates=["date"], index_col="date")
        else:
            df_rf = web.DataReader(self.rf_benchmark, "fred", begin_date, end_date).\
                resample("M", label="right").last()
            df_rf.index.name = "date"
            df_rf.to_csv(rf_cache_file)
        df_rf = (1 + (df_rf / 100)) ** (1 / 12.) - 1.0
        return df_rf

    @staticmethod
    def get_metrics(prcs, rets, excess_rets):
        metrics = dict()
        metrics["Sharpe Ratio"] = excess_rets.mean() / rets.std()
        metrics["Annualized Sharpe Ratio"] = metrics["Sharpe Ratio"] * sqrt(12)
        metrics["Skewness"] = rets.skew()
        metrics["Kurtosis"] = stats.kurtosis(rets.to_list(), fisher=False)
        metrics["Adjusted Sharpe Ratio"] = metrics["Sharpe Ratio"] * (
                1.0 + (metrics["Skewness"] / 6.0) * metrics["Sharpe Ratio"] -
                ((metrics["Kurtosis"] - 3) / 24.) * metrics["Sharpe Ratio"] ** 2
        )

        metrics["Annualized STD (%)"] = sqrt(12) * rets.std() * 100.
        metrics["Annualized Kurtosis"] = stats.kurtosis((12 * rets).to_list(), fisher=False)
        metrics["Annualized Skewness"] = (12 * rets).skew()
        metrics["Cumulative Return (%)"] = ((1. + rets).prod() - 1.0) * 100.
        metrics["Annual Return (%)"] = (1.0 + rets).groupby(rets.index.year).prod() - 1.0

        n = len(metrics["Annual Return (%)"])
        metrics["Arithmetic Return (%)"] = metrics["Annual Return (%)"].mean() * 100.
        metrics["Geometric Return (%)"] = (((metrics["Annual Return (%)"] + 1.).prod()) ** (1.0 / n) - 1.0) * 100.

        # compute VaR
        metrics["Monthly 95% VaR (%)"] = rets.quantile(0.05) * 100.
        metrics["Alt Monthly 95% VaR (%)"] = norm.ppf(0.05, rets.mean(), rets.std()) * 100.

        # compute maximum drawdown
        rolling_max = prcs.expanding().max()
        drawdown = prcs / rolling_max - 1.0
        metrics["Maximum Drawdown (%)"] = drawdown.min() * 100.
        metrics["MDD / VOL"] = metrics["Maximum Drawdown (%)"] / metrics["Annualized STD (%)"]

        dd = np.sqrt(np.sum(np.minimum(excess_rets, 0) ** 2) / len(excess_rets))
        metrics["Sortino Ratio"] = excess_rets.mean() / (dd + 1e-8)
        metrics["Annualized Sortino Ratio"] = sqrt(12) * metrics["Sortino Ratio"]

        metrics["Calmar Ratio"] = metrics["Geometric Return (%)"] / (metrics["Maximum Drawdown (%)"] + 1e-8)
        return metrics

    def get_portfolio_metrics(self, df_ports, used_metrics=None) -> pd.DataFrame:
        begin_date, end_date = df_ports["date"].min(), df_ports["date"].max()
        df_rf = self.get_df_rf(begin_date=begin_date, end_date=end_date)  # get risk free rate

        if used_metrics is None:
            used_metrics = [
                "Arithmetic Return (%)",
                "Geometric Return (%)",
                "Annualized STD (%)",
                "Cumulative Return (%)",
                "Maximum Drawdown (%)",
                "MDD / VOL",
                "Monthly 95% VaR (%)",
                "Sharpe Ratio",
                "Adjusted Sharpe Ratio",
                "Annualized Sharpe Ratio",
                "Annualized Sortino Ratio",
                "Calmar Ratio",
                "Turnover (%)",
                "Transaction Cost ($)",
                "Effective Holdings"
            ]

        df_port_values = df_ports.pivot(index="date", columns="port_name", values="port_value").\
            resample("ME", label="right").last()

        df_port_transaction_fee = df_ports.pivot(index="date", columns="port_name", values="port_trans").sum(axis=0)
        df_port_turnover = df_ports.pivot(index="date", columns="port_name", values="port_turnover")[2:].mean()
        df_port_hhi = df_ports.pivot(index="date", columns="port_name", values="port_hhi")[1:].mean()
        df_port_eff = 1.0 / df_port_hhi  # portfolio effective number of asset holding

        port_metrics = {}
        for port_name in df_port_values.columns:
            prcs = df_port_values[port_name]
            rets = prcs.pct_change().dropna()
            excess_rets = rets - df_rf.loc[rets.index, self.rf_benchmark]
            port_metrics[port_name] = self.get_metrics(prcs=prcs, rets=rets, excess_rets=excess_rets)
            port_metrics[port_name]["Turnover (%)"] = df_port_turnover.loc[port_name] * 100.
            port_metrics[port_name]["Transaction Cost ($)"] = df_port_transaction_fee.loc[port_name]
            port_metrics[port_name]["Effective Holdings"] = df_port_eff.loc[port_name]

        df_port_performance = (
            pd.DataFrame(port_metrics).loc[used_metrics].
                infer_objects(copy=False).fillna(0.)
        )
        df_port_performance.rename_axis("metrics", inplace=True)
        return df_port_performance


if __name__ == "__main__":
    _df_ports = pd.read_csv("../results/prcs/portfolio/values/hc.csv", parse_dates=["date"])
    pm = portfolio_analyzer()
    df_performance = pm.get_portfolio_metrics(df_ports=_df_ports)
    print(df_performance)