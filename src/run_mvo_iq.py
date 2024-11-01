import argparse
import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from util import get_mean_variance_space
from portfolio_analyzer import portfolio_analyzer
from lib_cov_func import gerber_cov_stat1, ledoit
from stcs import IQ
from glob import glob
import plotly.express as px
DEBUG = 0

# set global variables
# collective of risk targets, risk strategies and covariance statistics to be used
target_volatility_array = np.arange(2, 16) / 100.
obj_function_list = ['minVariance', 'maxSharpe', 'maxReturn', 'riskParity']

# portfolio setting  set initial cash
cash_start = 100000.

# setting upper limit on lookback window (in months)
lookback_win_in_year = 2
lookback_win_size = 12 * lookback_win_in_year
optimization_cost = 10  # penalty for excessive transaction
transaction_cost = 10   # actual transaction fee in trading simulation

dict_cov_funcs = {}

# add iq methods with selected parameters
dict_cov_funcs["IQ1"] = {
    "cov_function": IQ, "cov_params": {
        "center_method": "median", "scale_method": "vols_max",
        "dCplus": 2, "dCminus": 2, "dDplus": 2, "dDminus": 2,
        "eta": 0.1, "gamma": 0.075
    }
}


dict_cov_funcs["IQ2"] = {
        "cov_function": IQ, "cov_params": {
        "center_method": "median", "scale_method": "vols_max",
        "dCplus": 2, "dCminus": 2, "dDplus": 2, "dDminus": 2,
        "eta": 0.1, "gamma": 0
    }
}

# add gerber1 statistics
dict_cov_funcs["GS"] = {
    "cov_function": gerber_cov_stat1, "cov_params": {"threshold": 0.5, "center_method": "zero"}
}

# add hc for historical covariance matrix and the ledoit as the shrinkage method
dict_cov_funcs["SM"] = {"cov_function": ledoit, "cov_params": {}}
dict_cov_funcs["HC"] = {"cov_function": lambda x: x.cov(), "cov_params": {}}
list_cov_funcs = [dict_cov_funcs]

# read data and create folder to save results
filename = "../data/prcs.csv"
savepath = "../results/%s" % os.path.basename(filename).split(".")[0]
os.makedirs(savepath, exist_ok=True)
os.makedirs("%s/covs" % savepath, exist_ok=True)
os.makedirs("%s/portfolio/allocations" % savepath, exist_ok=True)
os.makedirs("%s/portfolio/values" % savepath, exist_ok=True)
os.makedirs("%s/portfolio/weights" % savepath, exist_ok=True)
os.makedirs("%s/portfolio/metrics" % savepath, exist_ok=True)


def run_covs(dict_cov_funcs: dict, filename: str, verbose: bool = False):
    prcs = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
    rets = prcs.pct_change().dropna(axis=0)
    prcs = prcs.iloc[1:]  # drop first row
    nT, p = prcs.shape
    symbols = prcs.columns.to_list()

    port_names = obj_function_list + ['%02dpct' % (tgt * 100) for tgt in target_volatility_array]
    account_dict = {}
    cov_dict = {}
    for cov_name, dict_cov_func in dict_cov_funcs.items():
        account_dict[cov_name] = {}
        cov_dict[cov_name] = {}
        for port_name in port_names:
            account_dict[cov_name][port_name] = []
            account_dict[cov_name][port_name].append(
                {
                    "date": prcs.index[lookback_win_size - 1],
                    "weights": np.array([0] * p),  # portfolio weight for each asset
                    "shares": np.array([0] * p),   # portfolio shares for each asset
                    "values": np.array([0] * p),   # portfolio dollar value for each asset
                    "portReturn": np.nan,
                    "transCost": np.nan,
                    "weightDelta": np.nan,  # compute portfolio turnover for current rebalancing period
                    "portValue": cash_start,
                    "port_vol": np.nan,
                    "port_ret": np.nan,
                    "port_hhi": np.nan
                }
            )

    # keep track of previous optimal weights to penalize extensive turnover
    prev_port_weights_dict = {key: None for key in dict_cov_funcs.keys()}
    for t in tqdm(range(lookback_win_size, nT), disable=not verbose):
        bgn_date = rets.index[t - lookback_win_size]
        end_date = rets.index[t - 1]
        end_date_p1 = rets.index[t]

        bgn_date_str = bgn_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        end_date_p1_str = end_date_p1.strftime("%Y-%m-%d")

        # subset the data accordingly
        sub_rets = rets.iloc[t - lookback_win_size: t]
        _nT, _ = sub_rets.shape
        prcs_t = prcs.iloc[t-1: t].values[0]  # price at time t
        prcs_tp1 = prcs.iloc[t: t + 1].values[0]  # price at time t + 1
        rets_tp1 = rets.iloc[t: t + 1].values[0]  # return at time t + 1

        if DEBUG:
            print("MVO optimimize from [%s, %s] (n=%d) and applied to rets at %s" % \
                  (bgn_date_str, end_date_str, _nT, end_date_p1_str))

        # get portfolio weight for a given cov_function
        opt_ports_dict = {}
        for cov_name, dict_cov_func in dict_cov_funcs.items():
            if DEBUG:
                print("Processing %s ..." % cov_name)
            opt_ports_dict[cov_name] = get_mean_variance_space(
                returns_df=sub_rets,
                target_risks_array=target_volatility_array,
                obj_function_list=obj_function_list,
                cov_function=dict_cov_func["cov_function"],
                cov_params=dict_cov_func["cov_params"],
                prev_port_weights=prev_port_weights_dict[cov_name],
                cost=optimization_cost
            )

            # save the covariance matrix for this date
            cov_dict[cov_name][end_date_p1_str] = {
                "cov": opt_ports_dict[cov_name]["cov"],
                "is_psd": opt_ports_dict[cov_name]["is_psd"],
                "non_psd_cov": opt_ports_dict[cov_name]["non_psd_cov"]
            }

            # save optimal portfolio weights and invest accordingly
            prev_port_weights_dict[cov_name] = opt_ports_dict[cov_name]["port_opt"]
            for port_name in port_names:
                port_tm1 = account_dict[cov_name][port_name][-1]
                # updated portfolio
                port_t = {
                    "date": end_date_p1,
                    "weights": opt_ports_dict[cov_name]['port_opt'][port_name]['weights'],
                    "shares": None,
                    "values": None,
                    "portReturn": None,
                    "transCost": None,
                    "weightDelta": None,  # for portfolio turnover for current period
                    "portValue": None,
                    "port_vol": None,
                    "port_ret": None,
                    "port_hhi": sum([w ** 2.0 for w in opt_ports_dict[cov_name]['port_opt'][port_name]['weights']]),  # portfolio Herfindahl-Hirschman Index
                }

                # calculate shares given new weight
                port_t["shares"] = port_tm1['portValue'] * port_t["weights"] / prcs_t
                port_t["portReturn"] = (port_t["weights"] * rets_tp1).sum()
                port_t['values'] = port_t['weights'] * port_tm1['portValue']

                # compute transaction by trading volume
                # redistribute money according to the new weight
                volume_buy  = np.maximum(port_t["values"] - port_tm1["values"], 0)
                volume_sell = np.maximum(port_tm1["values"] - port_t["values"], 0)
                port_t["transCost"] = (volume_buy + volume_sell).sum() * transaction_cost / 10000  # pay transaction_cost bps trans. cost
                # calculate portfolio turnover for this period
                port_t["weightDelta"] = np.sum(np.abs(port_t["weights"] - port_tm1["weights"]))

                # calculate updated portfolio at time t
                port_t["portValue"] = (port_tm1['portValue'] - port_t["transCost"]) * (1 + port_t['portReturn'])
                port_t["port_ret"] = opt_ports_dict[cov_name]['port_opt'][port_name]['ret_std'][0]
                port_t["port_vol"] = opt_ports_dict[cov_name]['port_opt'][port_name]['ret_std'][1]
                account_dict[cov_name][port_name].append(port_t)

    # loop through different covariance functions
    for cov_name in dict_cov_funcs.keys():
        # save portfolio to pickle file
        with open("%s/portfolio/allocations/%s.pickle" % (savepath, cov_name), "wb") as f:
            pickle.dump(account_dict[cov_name], f)

        # save covariance matrix
        with open("%s/covs/%s.pickle" % (savepath, cov_name), "wb") as f:
            pickle.dump(cov_dict[cov_name], f)

        # extract and save portfolio performance
        portAccountDF = pd.DataFrame.from_dict({
            (port_name, account['date']): {
                "port_value":    account['portValue'],
                "port_return":   account['portReturn'],
                "port_trans":    account['transCost'],
                "port_turnover": account['weightDelta'],
                "port_vol":      account["port_vol"],
                "port_ret":      account["port_ret"],
                "port_hhi":      account["port_hhi"],
            }
            for port_name in account_dict[cov_name].keys()
            for account in account_dict[cov_name][port_name]
        }, orient='index')

        portWeightsDF = pd.DataFrame.from_dict({
            (port_name, account['date']): {
                s: w for s, w in zip(symbols, account["weights"])
            }
            for port_name in account_dict[cov_name].keys()
            for account in account_dict[cov_name][port_name][1:]
        }, orient='index')

        # export portfolio growth curves
        portAccountDF.rename_axis(("port_name", "date"), inplace=True)
        portAccountDF.to_csv("%s/portfolio/values/%s.csv" % (
            savepath,
            cov_name
        ), float_format="%.4f")

        # export weights
        portWeightsDF.rename_axis(("port_name", "date"), inplace=True)
        portWeightsDF.to_csv("%s/portfolio/weights/%s.csv" % (
            savepath,
            cov_name
        ), float_format="%.4f")

        # compute and then export portfolio performance
        pa = portfolio_analyzer()
        portAccountDF.reset_index(inplace=True)
        df_port_metrics = pa.get_portfolio_metrics(df_ports=portAccountDF)
        df_port_metrics.to_csv("%s/portfolio/metrics/%s.csv" % (
            savepath,
            cov_name
        ), float_format="%.3f")
        df_port_metrics.to_latex("%s/portfolio/metrics/%s.tex" % (
            savepath,
            cov_name
        ), float_format="%.3f")


if __name__ == "__main__":
    # run portfolio optimization using the selected covariance matrix estimators
    for idx, dict_cov_func in enumerate(list_cov_funcs):
        run_covs(dict_cov_funcs=dict_cov_func, filename=filename, verbose=True)

    # aggregate and plot the portfolio performance
    input_folder = f"{savepath}/portfolio/metrics/"
    csv_files = glob("%s/*.csv" % input_folder)
    df_performance = pd.concat([pd.read_csv(csv_file, index_col=0) for csv_file in csv_files], axis=1)

    list_performs = []
    for csv_file in csv_files:
        cov_name = os.path.basename(csv_file).replace(".csv", "")
        df_perform = pd.read_csv(csv_file, index_col=0)
        df_perform["cov_name"] = cov_name
        list_performs.append(df_perform)

    df_performs = pd.concat(list_performs, axis=0).reset_index()
    df_performs_melt = df_performs.melt(
        id_vars=["cov_name", "metrics"], var_name="strategy"
    )

    df_performs_pivot = df_performs_melt.pivot(index="metrics", columns=["strategy", "cov_name"])
    cov_names = [cov_name for cov_dicts in list_cov_funcs for cov_name in cov_dicts.keys()]
    selec_cols = [
        ('value', strategy, cov_name)
        for strategy in ('minVariance', '03pct', '06pct', '09pct', '12pct', '15pct', 'maxSharpe', 'maxReturn', 'riskParity') for cov_name in cov_names]
    df_performs_pivot.loc[:, selec_cols].to_csv(f"{savepath}/performance_table.csv", float_format="%.3f")

    df_perfs_plt = df_performs_melt[
        df_performs_melt["metrics"].isin(["Annualized STD (%)", "Geometric Return (%)"]) &
        df_performs_melt["strategy"].isin(["03pct", "05pct", "07pct", "09pct", "11pct", "13pct", "15pct"])
    ].pivot(columns=["metrics"], index=["cov_name", "strategy"], values="value").reset_index()
    df_perfs_plt["cov_name"] = pd.Categorical(df_perfs_plt["cov_name"], cov_names, ordered=True)

    # plot the ex-post efficient frontier
    fig = px.line(
        df_perfs_plt,
        x="Annualized STD (%)", y="Geometric Return (%)", color="cov_name", template="plotly_white", markers=True,
        labels={"GR": "Annualized Return (%)", "VOL": "Annualized Volatility (%)", "strategy": "Cov Func"},
        category_orders={"cov_name": cov_names, }
    )
    fig.update_layout(legend_title="", )
    fig.update_traces(textposition='top center')
    fig.write_image(f"{savepath}/performance_frontiers.pdf", width=600, height=400)

