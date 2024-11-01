"""
Name : portfolio_optimizer.py
Author   : Yinsen Miao/William Smyth
Contact : yinsenm@gmail.com/drwss.academy@gmail.com
Time     : 1/12/2022
Desc: solve mean-variance optimization
"""

import numpy as np
import pandas as pd
from nearest_correlation import nearcorr
from lib_cov_func import check_symmetric, is_psd_def, correlation_from_covariance
from scipy.optimize import minimize
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='debug.log', level=logging.DEBUG)


"""
NNNN = 1
combs = pd.read_csv("collection_200_modified.csv".format(NNNN)).set_index('IQ')
combs = combs.iloc[20*(NNNN-1):20*NNNN,:]

##############################################################################################
########## CHECK THE RELATIONSHIPS BETWEEN THE DC'S ##########################################
DCplus   = {"IQ_%s" % ((NNNN-1)*20+i+1): combs['dCplus'][(NNNN-1)*20+i+1]  for i in range(20)}
DDplus   = {"IQ_%s" % ((NNNN-1)*20+i+1): combs['dCplus'][(NNNN-1)*20+i+1]  for i in range(20)}
DCminus  = {"IQ_%s" % ((NNNN-1)*20+i+1): combs['dCminus'][(NNNN-1)*20+i+1] for i in range(20)}
DDminus  = {"IQ_%s" % ((NNNN-1)*20+i+1): combs['dCminus'][(NNNN-1)*20+i+1]  for i in range(20)}
##############################################################################################
Eta      = {"IQ_%s" % ((NNNN-1)*20+i+1): combs['eta'][(NNNN-1)*20+i+1]     for i in range(20)}
Gamma    = {"IQ_%s" % ((NNNN-1)*20+i+1): combs['gamma'][(NNNN-1)*20+i+1]   for i in range(20)}
"""
    
def set_eps_wgt_to_zeros(in_array, eps=1e-4):
    # set small weights to 0 and return a list
    out_array = np.array(in_array)
    out_array[np.abs(in_array) < eps] = 0
    out_array = np.array(out_array) / np.sum(out_array)
    return out_array


class portfolio_optimizer:
    def __init__(
            self,
            cov_function,
            cov_params: dict={},
            freq: str = "monthly",
            min_weight: float = 0., max_weight: float = 1.0,
    ):
        """
        :param cov_func: covariance function
        :param cov_param: parameter of the covariance function
        :param freq: frequency of the returns series either daily or monthly
        :param min_weight: minimum weight for portfolio
        :param max_weight: maximum weight for portfolio
        """
        # check arguments
        assert freq in ['daily', 'monthly'], "The return series can only be either daily or monthly"
        assert 1 > min_weight >= 0, "The minimal weight shall be in [0, 1)"
        assert 1 >= max_weight > 0, "The maximum weight shall be in (0, 1]"

        self.min_weight = min_weight
        self.max_weight = max_weight

        self.factor = 252 if freq == "daily" else 12  # annual converter
        self.cov_function = cov_function  # handle for covariance function
        self.cov_params = cov_params  # parameters of covariance functions
        self.freq = freq  # freq of return series can be either daily or monthly
        self.init_weights = None  # initial portfolio weights
        self.covariance = None
        self.non_psd_covariance = None
        self.is_psd = False
        self.returns_df = None
        self.obj_function = None
        self.by_risk = None

    def set_returns(self, returns_df: pd.DataFrame):
        """
        pass the return series to the class
        :param returns_df: pd.DataFrame of historical daily or monthly returns
        """
        self.returns_df = returns_df.copy(deep=True)

        # compute covariance matrix once the return is given
        self.covariance = self.cov_function(self.returns_df, **self.cov_params)

        if not check_symmetric(self.covariance):
            logging.debug((
                "%s - %s" % (
                self.returns_df.index[0].strftime("%Y%m%d"),
                self.returns_df.index[-1].strftime("%Y%m%d"),
            ), self.cov_params))

        # check if the covariance is psd or not
        self.is_psd = is_psd_def(self.covariance)

        if not self.is_psd:
            self.non_psd_covariance = self.covariance.copy()
            # correct the correlation matrix
            cor_mat = correlation_from_covariance(self.covariance).values
            # find the nearest correlation matrix that is psd
            cor_mat_corrected = nearcorr(cor_mat)
            sd = np.diag(np.sqrt(np.diag(self.covariance.values)))
            self.covariance = sd @ cor_mat_corrected @ sd

    def optimize(
            self, obj_function: str,
            target_std: float = None,
            target_return: float = None,
            prev_weights: np.ndarray = None,
            init_weights: np.ndarray = None,
            cost: float = 0
    ) -> np.array:
        """
        Perform portfolio optimization given a series of returns
        :param obj_function:
        :param target_std: targeted annualized portfolio standard deviation (std)
        :param target_return: targeted annualized portfolio return deviation
        :param prev_weights: previous weights
        :param init_weights: set the initial weight for the MVO optimization
        :param cost: cost of transaction fee and slippage in bps or 0.01%
        :return: an array of portfolio weights p x 1
        """
        n, p = self.returns_df.shape  # n is number of observations, p is number of assets

        if init_weights is None:
            self.init_weights = np.array(p * [1. / p])  # initialize weights: equal weighting
        else:
            self.init_weights = init_weights  # otherwise use the nearby weights as hot start for MVO

        self.obj_function = obj_function

        """

        # get covariance matrix
        if self.cov_function == "HC":
            self.covariance = self.returns_df.cov().to_numpy()  # convert to numpy
        elif self.cov_function == "SM":
            self.covariance, _ = ledoit(self.returns_df.values)
        elif self.cov_function == "GS":
            self.covariance, _ = gerber_cov_stat1(self.returns_df.values)
        elif self.cov_function == "Ktau":
            self.covariance = Ktau(self.returns_df.values)
        
        elif self.cov_function == "LS1":
            self.covariance = cov1Para(self.returns_df)
        elif self.cov_function == "LS2":
            self.covariance = cov2Para(self.returns_df) 
        elif self.cov_function == "LS3":
            self.covariance = covCor(self.returns_df)
        elif self.cov_function == "LS4":
            self.covariance = covDiag(self.returns_df) 
        elif self.cov_function == "LS5":
            self.covariance = covMarket(self.returns_df)
        elif self.cov_function == "NLS6":
            self.covariance = GIS(self.returns_df) 
        elif self.cov_function == "NLS7":
            self.covariance = LIS(self.returns_df)
        elif self.cov_function == "NLS8":
            self.covariance = QIS(self.returns_df) 
        
        elif self.cov_function in ["IQ_%s" % ((NNNN-1)*20+i+1) for i in range(20)]:
            self.covariance, _ =  IQ(self.returns_df.values,
                                    dCplus  = DCplus[self.cov_function],
                                    dCminus = DCminus[self.cov_function],
                                    dDplus  = DDplus[self.cov_function],
                                    dDminus = DDminus[self.cov_function],
                                    eta     = Eta[self.cov_function],
                                    gamma   = Gamma[self.cov_function])
        """
        
        # set objective function
        if obj_function == "equalWeighting":
            self.init_weights = np.array(p * [1. / p])  # initialize weights: equal weighting
            return self.init_weights

        # set the bounds of each asset holding from 0 to 1
        bounds = tuple((self.min_weight, self.max_weight) for k in range(p))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]  # fully invest

        if obj_function == 'meanVariance':
            if target_std is not None:
                self.by_risk = True
                # optimize under risk constraint
                constraints.append({'type': 'eq', 'fun': lambda weights: \
                    self.calc_annualized_portfolio_std(weights) - target_std})
            else:
                # optimize under return constraint
                self.by_risk = False
                constraints.append({'type': 'eq', 'fun': lambda weights: \
                    self.calc_annualized_portfolio_return(weights) - target_return})

        if prev_weights is not None and cost is not None:
            # cost function with transaction fee
            cost_fun = lambda weights: self.object_function(weights) +\
                                       np.abs(weights - prev_weights).sum() * cost / 10000.
        else:
            # cost function without any transaction fee
            cost_fun = lambda weights: self.object_function(weights)

        # trust-constr, SLSQP, L-BFGS-B
        try:
            opt = minimize(cost_fun, x0=self.init_weights, bounds=bounds, constraints=constraints, method="SLSQP")
        except:
            # if SLSQP fails then switch to trust-constr
            opt = minimize(cost_fun, x0=self.init_weights, bounds=bounds, constraints=constraints, method="trust-constr")

        return set_eps_wgt_to_zeros(opt['x'])   # pull small values to zeros

    def object_function(self, weights: np.array) -> float:
        """
        :param weights: current weights to be optimized
        """

        if self.obj_function == "maxReturn":
            f = self.calc_annualized_portfolio_return(weights)
            return -f
        elif self.obj_function == "minVariance":
            f = self.calc_annualized_portfolio_std(weights)
            return f
        elif self.obj_function == "meanVariance" and self.by_risk:
            f = self.calc_annualized_portfolio_return(weights)  # maximize target return level
            return -f
        elif self.obj_function == "meanVariance" and not self.by_risk:
            f = self.calc_annualized_portfolio_std(weights)  # minimize target risk or std level
            return f
        elif self.obj_function == "maxSharpe":
            f = self.calc_annualized_portfolio_sharpe_ratio(weights)
            return -f
        elif self.obj_function == "maxSortino":
            f = self.calc_annualized_sortino_ratio(weights)
            return -f
        elif self.obj_function == 'riskParity':
            f = self.calc_risk_parity_func(weights)
            return f
        else:
            raise ValueError("Object function shall be one of the equalWeighting, maxReturn, minVariance, " +
                             "meanVariance, maxSharpe, maxSortino or riskParity")

    def calc_annualized_portfolio_return(self, weights: np.array) -> float:
        # calculate the annualized standard returns
        annualized_portfolio_return = float(np.sum(self.returns_df.mean() * self.factor * weights))
        #float(np.sum(((1 + self.returns_df.mean()) ** self.factor - 1) * weights))
        return annualized_portfolio_return

    def calc_annualized_portfolio_std(self, weights: np.array) -> float:
        if self.obj_function == "equalWeighting":
            # if equal weight then set the off diagonal of covariance matrix to zero
            annualized_portfolio_std = np.sqrt(np.dot(weights.T, np.dot(np.diag(self.covariance.diagonal()) * self.factor, weights)))
        else:
            temp = np.dot(weights.T, np.dot(self.covariance * self.factor, weights))
            if temp <= 0:
                temp = 1e-20  # set std to a tiny number
            annualized_portfolio_std = np.sqrt(temp)
        if annualized_portfolio_std <= 0:
            raise ValueError('annualized_portfolio_std cannot be zero. Weights: {weights}')
        return annualized_portfolio_std

    def calc_annualized_portfolio_moments(self, weights: np.array) -> tuple:
        # calculate the annualized portfolio returns as well as its standard deviation
        return self.calc_annualized_portfolio_return(weights), self.calc_annualized_portfolio_std(weights)

    def calc_annualized_portfolio_sharpe_ratio(self, weights: np.array) -> float:
        # calculate the annualized Sharpe Ratio
        return self.calc_annualized_portfolio_return(weights) / self.calc_annualized_portfolio_std(weights)

    def calc_risk_parity_func(self, weights):
        # Spinu formulation of risk parity portfolio
        assets_risk_budget = self.init_weights
        portfolio_volatility = self.calc_annualized_portfolio_std(weights)

        x = weights / portfolio_volatility
        risk_parity = (np.dot(x.T, np.dot(self.covariance * self.factor, x)) / 2.) - np.dot(assets_risk_budget.T, np.log(x + 1e-10))
        return risk_parity

    def calc_relative_risk_contributions(self, weights):
        # calculate the relative risk contributions for each asset given returns and weights
        rrc = weights * np.dot(weights.T, self.covariance) / np.dot(weights.T, np.dot(self.covariance, weights))
        return rrc