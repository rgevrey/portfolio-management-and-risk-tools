import pandas as pd
import numpy as np
from scipy.stats import norm
import pdb as pdb
import os

# Last updated on 5/01/2022 19:08

#######
# Data Extraction
#######

def path_data():
    # Path parsing to make it work across computers
    path_directory = os.path.dirname(os.getcwd())
    path_data = path_directory + "/data/"
    return path_data

def get_ffme_returns(get_all = False):
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by
    Market Cap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                  header=0, index_col=0, parse_dates=True, na_values=-99.99)
    if get_all == False:
        rets = me_m[['Lo 10', 'Hi 10']]
        rets.columns = ['SmallCap', 'LargeCap']
    else:
        rets = me_m
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format = "%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fun Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                  header=0, index_col=0, parse_dates=True, na_values=-99.99)
    hfi = hfi/100
    hfi.index = pd.to_datetime(hfi.index, format = "%Y%m").to_period('M')
    return hfi

def get_ind_returns():
    """
    Load and format the Ken French industry returns
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col =0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format ="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolio Value Weighted size data
    TO MERGE INTO A SINGLE RETRIEVAL FUNCTION
    """
    ind = pd.read_csv("data/ind30_m_size.csv", header=0, index_col =0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format ="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
    Load and format the Ken French 30 Industry Portfolio Value Weighted number of firms
    TO MERGE INTO A SINGLE RETRIEVAL FUNCTION
    """
    ind = pd.read_csv("data/ind30_m_nfirms.csv", header=0, index_col =0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format ="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_total_market_index_returns():
    """
    Retrieves the return from the market-weighted index we created
    """
    ind_return = get_ind_returns()
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    
    if ind_return.shape == ind_size.shape == ind_return.shape == ind_nfirms.shape:
        ind_mktcap = ind_nfirms * ind_size
        
        # Summing across columns rather than lines
        total_mktcap = ind_mktcap.sum(axis="columns")
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        total_market_return = (ind_capweight*ind_return).sum(axis="columns")
        
    else: 
        raise ValueError("Arrays must have the same size")
   
    return total_market_return

#######
# Distribution statistics
#######

def skewness(r):
    """
    Alternative to scipy.stats.skewness()
    Computes the skewness of a series
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of a series
    Returns a float or a series
    ALTERNATIVELY YOU CAN MAKE A PARAMETER RATHER THAT COPYING THE CODE AGAIN
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is acccepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r) # capturing a tupple here
    return p_value > level

#######
# RISK METRICS
#######

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def var_historic(r, level =5):
    """
    Returns the historical Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level) # weird, will be called on all columns of dataframe first
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        
def var_gaussian(r, level =5, modified = False):
        """
        Returns the parametric gaussian VaR of a series or DataFrame
        If "modified" is True, then the modified VaR is returned
        using the Cornish-Fisher modification
        """
        # Compute the Z score assuming it was Gaussian
        z = norm.ppf(level/100)
        if modified:
            # modify the Z score base on observed skewness and kurtosis
            s = skewness(r)
            k = kurtosis(r)
            z = (z +
                    (z**2 - 1)*s/6 +
                    (z**3 -3*z)*(k-3)/24 - 
                    (2*z**3 -5*z)*(s**2)/36
                )
        return -(r.mean() + z*r.std(ddof=0))
    
def cvar_historic(r, level =5):
    """
    Computes the Conditional VaR of series or Dataframe
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a series or DataFrame")
        
def drawdown(returns_series: pd.Series): 
    """
    Takes a time series of returns
    Computes and returns a DataFrame that contains:
    - the wealth index
    - the previous peaks
    - percent drawdowns (in decimals
    """
    wealth_index = 1000*(1+returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks" : previous_peaks,
        "Drawdown": drawdowns})
#######
# PORTFOLIO STATISTICS        
#######

def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())

def period_returns(r,output="All"):
    """
    Hopefully correctly computes the total period money return, monthly and annual
    provided monthly data is supplied
    returns a df in %
    Output = "All" for full report  
    Output = "Annual" for annual only
    Output = "Monthly" for monthly only
    Output = "Period" for matching the period only
    Output = "Cumprod" for cumprod
    """
    r_prod = (np.prod(r+1) - 1)
    r_cumprod = (np.cumprod(r+1) - 1)
    r_monthly = (((1+r_prod)**(1/r_cumprod.shape[0])) - 1)
    r_annual = (((1 + r_monthly)**12) - 1)
    if output == "All":
        r = pd.DataFrame({"Period": r_prod.values*100,
                        "Monthly": r_monthly.values*100, 
                        "Annual" : r_annual.values*100})
    elif output == "Annual":
        r = r_annual
    elif output == "Monthly":
        r = r_monthly
    elif output == "Period":
        r = r.pct_change().dropna()
    elif output == "Cumprod":
        r = r_cumprod
    return r
    
def period_volatility(r, output="All"):
    """
    Hopefully correctly computes the monthly and annual vol
    provided monthly data is supplised
    returns a df in %
    PENDING = ANY PERIODS
    Output = "All" for full report  
    Output = "Annual" for annual only
    """
    volatility_monthly = r.std()
    volatility_annual = volatility_monthly * np.sqrt(12)
    if output == "All":
        v = pd.DataFrame({"Monthly": volatility_monthly.values*100, 
                            "Annual" : volatility_annual.values*100})
    elif output == "Annual":
        v = volatility_annual
    return v
    
def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = period_returns(excess_ret, output="Annual")
    ann_vol = period_volatility(r, output="Annual")
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Weights to Returns
    """
    # W transpose R in matrix notation, @ covmat weights
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weights to vol
    Turns lists,arrays into float volatility figure
    
    Last updated 24/12/2020
    """
    if isinstance(weights, list):
        weights = np.array(weights)
    
    return (weights.T @ covmat @ weights) ** 0.5

def terminal_values(rets):
    """
    Returns the final value of a dollar at the end of the return period for each scenario
    Nothing more than the compounded return
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce summary statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N dataframe of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor # how often I end up below my floor, gives booleans
    reach = terminal_wealth >= cap 
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan # expected shortfall
    e_surplus = (cap-terminal_wealth[reach].mean()) if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std": terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short": e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

#######
# PORTFOLIO ALLOCATION
#######

def plot_ef2(n_points, er, cov, style = ".-"):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"R": rets, "Vol": vols})
    return ef.plot.line(x="Vol", y="R", style=style)

from scipy.optimize import minimize
def minimize_vol(target_r, er, cov):
    """
    From target return to a weight vector
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    #constraints
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_r - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    # running the optimizer
    results = minimize(portfolio_vol, init_guess,
                      args=(cov,), method="SLSQP",
                      #options={'disp':False},
                      constraints = (return_is_target, weights_sum_to_1),
                      bounds=bounds
                      )
    return results.x

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two set of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step and N is
    the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters,
    and produces an allocation to the first portfolio (the rest of the money is invested in the GHP)
    as a T x 1 Dataframe. 
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")
    
    # Function as an object
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that don't match r1")
        
    # Now computing the returns of the mix
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and the GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and the GHP such that:
        each column is a scenario
        each row is the price for a timestep
    Returns a T x N DataFrame of PSP weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate betwen PSP and GHP with the goal to provide exposure to the upside of the PSP without
    violating the floor. 
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cushion in the PSP
    Returns a dataframe with the same shape as the PSP/GHP representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape 
    account_value = np.repeat(1, n_scenarios) # computing a vector of n scenarios, initialised at 1 for each scenario
    floor_value = np.repeat(1, n_scenarios) # set the floor value at 1, doesn't matter because we update later
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns) # about to return a sequence of weight
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC, because ZC = factor
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) # same as applying min and max, more compact
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history # sequence of weight over time

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate betwen PSP and GHP with the goal to provide exposure to the upside of the PSP without
    violating the drawdown constrainst.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cushion in the PSP
    Returns a dataframe with the same shape as the PSP/GHP representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape 
    account_value = np.repeat(1, n_scenarios) # computing a vector of n scenarios, initialised at 1 for each scenario
    floor_value = np.repeat(1, n_scenarios) # set the floor value at 1, doesn't matter because we update later
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns) # about to return a sequence of weight
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ## Floor is based on previous peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) # same as applying min and max, more compact
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history # sequence of weight over time


def glidepath_allocator(r1, r2, start_glide=1, end_glide=0): # 100% to 0% is the most extreme version
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1) # replicating n of those in a dataframe, multiplying a list replicates it
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def optimal_weights(n_points, er, cov):
    '''
    Generates a list of weights to run the optimiser on to minimise the vol
    '''
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def gmv(cov):
    """
    Returns the weights of the Global Minimum Vol portfolio given the covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1,n), cov)

def plot_ef(n_points, er, cov, style = ".-", show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the N-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "R": rets,
        "Vol": vols})
    ax = ef.plot.line(x="Vol", y="R", style=style)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        #display EW
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=12)
    if show_gmv:
        w_gmv = gmv(cov) #only depends on covariance matrix
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        #display gmv
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize=12)
    if show_cml:
        rf = 0.1
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add CML
        cml_x = [0,vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize=12, linewidth=2)
    return ax

def msr(riskfree_rate, er, cov):
    """
    From riskfree_rate to a weight vector
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    #constraints
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    # running the optimizer, we want to maximize the Sharpe Ratio
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio, init_guess,
                      args=(riskfree_rate, er, cov,), method="SLSQP",
                      #options={'disp':False},
                      constraints = (weights_sum_to_1),
                      bounds=bounds
                      )
    return results.x

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Runs a backtest of the CPPI strategy, given a set of return for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    
    # CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns = ["R"])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # Fast way to set all values to a number

    # Tracking back test values
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1) # We don't want leverage
        risky_w = np.maximum(risky_w, 0) # We don't want to go short
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w

        # update the account value for this time step
        # import pdb; pdb.set_trace()
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the values so we can look at the history and plot it etc
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        
    risky_wealth = start*(1+risky_r).cumprod()
    
    backtest_result = {
        
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risky Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start" : start,
        "floor" : floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(period_returns, output="Annual")
    ann_vol = r.aggregate(period_volatility, output="Annual")
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(var_gaussian, modified=True)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown (5%)": dd
    })

#######
# SIMULATIONS
#######

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val


def show_cppi(n_scenarios = 50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100, steps_per_year=12):
    """
    Plots the results of a Monte Carlo Simulation of CPPI
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, steps_per_year=steps_per_year)
    risky_r = pd.DataFrame(sim_rets)
    risky_r = period_returns(risky_r,output="Period")
    
    # Run the backtest
    btr = run_cppi(risky_r=pd.DataFrame(risky_r), riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
    wealth = btr["Wealth"]
    
    # Calculate terminal wealth stats
    y_max = wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1] # picking up the last row via -1
    
    # Boolean mask = an array of booleans
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    tw_min = terminal_wealth.min()
    tw_max = terminal_wealth.max()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios
    
    # If there's a failure, what's the average failure extent?
    # Dot is the dot product
    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0
    
    # Plotting
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={"width_ratios":[3,2]},figsize=(24,9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax = wealth_ax, legend=False, alpha=0.3, color="indianred")
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls="--", color='red')
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc="indianred", orientation='horizontal')
    hist_ax.axhline(y=start, ls=":", color="black")
    hist_ax.axhline(y=tw_mean, ls=":", color="blue")
    hist_ax.axhline(y=tw_median, ls=":", color="purple")
    hist_ax.annotate(f"Range: ${int(tw_min)} - ${int(tw_max)} ({int(tw_max-tw_min)})", xy=(.5,.95),
                     xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.5, .9), xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.5, .85), xycoords='axes fraction', fontsize=24)
    if (floor > 0.01):
        hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:.2}",
                         xy=(.5,.7), xycoords='axes fraction', fontsize=24)
    
    cppi_controls = widgets.interactive(show_cppi,
                                       n_scenarios=widgets.IntSlider(min=1, max=1000, step=5, value=50),
                                       mu=(0., +.2, .01),
                                        sigma=(0,.5,.05),
                                        floor=(0,2,.1),
                                        m=(1,5,.5),
                                        riskfree_rate=(0, .05, .01),
                                        steps_per_year=widgets.IntSlider(min=1,max=12, step=1, value=12,
                                                                         description="Rebals/Year"),
                                        y_max=widgets.IntSlider(min=0,max=100,step=1,value=100,
                                                               description="Zoom Y Axis")
                                       )
    display(cppi_controls)

#######
# ASSET LIABILITY MANAGEMENT AND FIXED INCOME
#######

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of some assets given liabilities and interest rate
    """
    return float(pv(assets, r)/pv(liabilities, r))

def pv(cashflows, r):
    """
    Computes the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = cashflows.index
    discounts = discount(dates, r)
    return discounts.multiply(cashflows, axis='rows').sum()

def discount(time, rate):
    """
    Compute the price of a pure discount bond that pays a dollar at time t given interest rate r
    Rate is the per period interest rate
    Returns a |t| x |r| Series or Dataframe
    Rate can be a float, Series or Dataframe
    Returns a Dataframe indexed by time
    """
    discounts = pd.DataFrame([(rate+1)**-i for i in time])
    discounts.index = time
    return discounts

def macaulay_duration(cashflows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows
    """
    discounted_flows = discount(cashflows.index, discount_rate)*cashflows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(cashflows.index, weights=weights.squeeze())

def show_funding_ratio(assets,r):
    fr = funding_ratio(assets, liabilities, r)
    print(f'{fr*100:.2f}')
    
    controls = widgets.interactive(show_funding_ratio,
                              assets = widgets.IntSlider(min=1,max=10, step=1, value=5),
                              r = (0, .20, .01)
                              )
    display(controls)

def inst_to_ann(r):
    """
    Converts short rate to an annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Converts annualized to a short rate
    """
    return np.log1p(r)

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns a dataframe of cash flows generated by a bond,
    indexed by a coupon number (I think)
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal # For the last cash flow, also add the principal
    return pd.DataFrame(data=cash_flows)

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather it is to
    illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time
    i.e. the index of the discount_rate DataFrame is assumed to be the coupon number
    """
    
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    
    else: # Base case of a single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def match_duration(cf_target, cf_short_duration_bond, cf_long_duration_bond, discount_rate):
    """
    Takes cash flows, calculates durations and returns the weight W in cf_short_bond that, along wiht (1-W) in cf_l will have an effective
    duration that matches cf_target
    """
    d_target = macaulay_duration(cf_target, discount_rate)
    d_short = macaulay_duration(cf_short_duration_bond, discount_rate)
    d_long = macaulay_duration(cf_long_duration_bond, discount_rate)
    return (d_long - d_target)/(d_long - d_short)


def cir(n_years = 10, n_scenarios = 1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Implements Cox Ingersoll Ross Model for interest rate
    ## $$ dr_t = a(b - r_i)dt + \sigma\sqrt{r_t}dW_t $$
    b and r_0 are assumed to be the annualized rates, not the short rate
    The returned values are the annualized rates as well
    Returns dataframe of simulated changes in interest rates over time and prices!
    """
    import math
    if r_0 is None: r_0 = b
        
    # Need to convert the rate to an instantaneous rate, but the difference isn't that problematic
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    
    # We need random numbers, that's dWt
    num_steps = int(n_years*steps_per_year) + 1 # +1 because we initialise an array of rates that starts at row 0, so need one more
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    
    # Array of rates
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    # For price generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    
    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    
    # Now time to simulate the changes in the rates
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        prices[step] = price(n_years-step*dt, rates[step])
        
    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    
    return rates, prices
    
def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices (big assumption) and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data=0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    
    # Spreads out coupon payments over the period
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    # The shift approach to calculating returns
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()
