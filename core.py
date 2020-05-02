# general imports
import numpy as np
import pandas as pd
import datetime as dt

# matplotlib
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# scipy specifics
from scipy import stats as sps
from scipy.stats import dirichlet
from scipy.interpolate import interp1d

# GAMs and sklearn
from pygam import GammaGAM, PoissonGAM, s, l
from sklearn.utils import resample

# urls
CITY_DATA_URL = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv'
STATE_DATA_URL = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv'

# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

# best sigma for Brazil (prior hyperparameters)
#OPTIMAL_SIGMA = 0.35 # through Kevin's Optimization
OPTIMAL_SIGMA = 0.01

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
GAMMA = 1/7

# recovery rate (1 / recovery time)
RECOVERY_RATE = 1 / 14

def load_data():

    """
    Loads state and city data from wcota repository

    Returns
    ----------
    city_df: city data
    state_df: state data (Brazil)

    """

    city_df = (
                pd.read_csv(CITY_DATA_URL, parse_dates=['date'])
                .rename(columns={'totalCases':'confirmed_total',
                                 'newCases': 'confirmed_new',
                                 'deaths': 'deaths_total',
                                 'newDeaths': 'deaths_new'})
                .drop(['ibgeID','country','state'], axis=1)
                .assign(city = lambda x: x['city'].replace('TOTAL', 'Brazil'))
                .groupby(['city','date']).sum()
            )

    state_df = (
                pd.read_csv(STATE_DATA_URL, parse_dates=['date'])
                .rename(columns={'totalCases':'confirmed_total',
                                 'newCases': 'confirmed_new',
                                 'deaths': 'deaths_total',
                                 'newDeaths': 'deaths_new'})
                .drop(['ibgeID','country','city',
                       'deaths_per_100k_inhabitants',
                       'totalCases_per_100k_inhabitants',
                       'deaths_by_totalCases'], axis=1)
                .assign(state = lambda x: x['state'].replace('TOTAL', 'Brazil'))
                .groupby(['state','date']).sum()
                )

    return city_df, state_df

def tidy_raw_time_series_data(df_raw, index_str):
  
  # setting index for time series
  df = (
         df_raw
         .set_index(['Province/State','Country/Region','Lat','Long'])
        )

  # creating multi index for slicing
  # also converting dates to datetime
  df.columns = pd.MultiIndex.from_product([[index_str], pd.to_datetime(df.columns)])
  df = df.sort_index(axis=1).stack()
  
  return df

def load_johns_hopkins_data():
  
  # downloading data for confirmed, deaths and recoveries
  confirmed_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
  deaths_raw=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
  
  # tidying the data
  confirmed = tidy_raw_time_series_data(confirmed_raw, 'confirmed')
  deaths = tidy_raw_time_series_data(deaths_raw, 'deaths')
  
  # let us concat these dfs and then we're ready
  df = (
        pd.concat([confirmed, deaths], axis=1)
        .reset_index()
        .rename(columns={'level_4':'date', 
                         'Province/State':'province',
                         'Country/Region':'country',
                         'confirmed':'confirmed_total',
                         'deaths':'deaths_total'})
        .drop(['Lat', 'Long', 'province'], axis=1)
        .query('confirmed_total != 0')
        .groupby(['country','date']).sum()
       )
  
  # calculating new cases
  df['confirmed_new'] = df['confirmed_total'].groupby('country').diff()
  df['deaths_new'] = df['deaths_total'].groupby('country').diff()
  
  return df

def load_data_us():
    url = 'https://covidtracking.com/api/v1/states/daily.csv'
    state_df = pd.read_csv(url,
                        usecols=['date', 'state', 'positive'],
                        parse_dates=['date'],
                        index_col=['state', 'date'],
                        squeeze=True).sort_index()

    state_df = (state_df
                .to_frame()
                .rename(columns={'positive':'confirmed_total'})
                .loc[lambda x: x['confirmed_total'] > 0])

    state_df['confirmed_new'] = state_df['confirmed_total'].groupby(level='state').diff()
    state_df = state_df.dropna().clip(lower=0)

    return state_df


def smooth_new_cases(new_cases):
    
    """
    Function to apply gaussian smoothing to cases

    Arguments
    ----------
    new_cases: time series of new cases

    Returns 
    ----------
    smoothed_cases: cases after gaussian smoothing

    See also
    ----------
    This code is heavily based on Realtime R0
    by Kevin Systrom
    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb
    """

    smoothed_cases = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    zeros = smoothed_cases.index[smoothed_cases.eq(0)]
    if len(zeros) == 0:
        idx_start = 0
    else:
        last_zero = zeros.max()
        idx_start = smoothed_cases.index.get_loc(last_zero) + 1
    smoothed_cases = smoothed_cases.iloc[idx_start:]
    original = new_cases.loc[smoothed_cases.index]
    
    return original, smoothed_cases


def calculate_posteriors(sr, sigma=0.15):

    """
    Function to calculate posteriors of Rt over time

    Arguments
    ----------
    sr: smoothed time series of new cases

    sigma: gaussian noise applied to prior so we can "forget" past observations
           works like exponential weighting

    Returns 
    ----------
    posteriors: posterior distributions
    log_likelihood: log likelihood given data

    See also
    ----------
    This code is heavily based on Realtime R0
    by Kevin Systrom
    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb
    """

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood


def highest_density_interval(pmf, p=.9):

    """
    Function to calculate highest density interval 
    from posteriors of Rt over time

    Arguments
    ----------
    pmf: posterior distribution of Rt

    p: mass of high density interval

    Returns 
    ----------
    interval: expected value and density interval

    See also
    ----------
    This code is heavily based on Realtime R0
    by Kevin Systrom
    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb
    """

    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    most_likely = pmf.idxmax(axis=0)

    interval = pd.Series([most_likely, low, high], index=['ML',f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])

    return interval


def plot_rt(result, ax, state_name):
    
    """
    Function to plot Rt

    Arguments
    ----------
    result: expected value and HDI of posterior

    ax: matplotlib axes 

    state_name: state to be considered

    See also
    ----------
    This code is heavily based on Realtime R0
    by Kevin Systrom
    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb
    """

    ax.set_title(f"{state_name}")
    
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low_90'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      result['High_90'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1)) 


def plot_standings(mr, figsize=None, title='Most Recent $R_t$ by State'):
    
    """
    Function to plot standings

    Arguments
    ----------
    mr: results by state

    See also
    ----------
    This code is heavily based on Realtime R0
    by Kevin Systrom
    https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb
    """


    if not figsize:
        figsize = ((15.9/50)*len(mr)+.1,2.5)
        
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    ax.set_title(title)
    err = mr[['Low_90', 'High_90']].sub(mr['ML'], axis=0).abs()
    bars = ax.bar(mr.index,
                  mr['ML'],
                  width=.825,
                  color=[.7,.7,.7],
                  ecolor=[.3,.3,.3],
                  capsize=2,
                  error_kw={'alpha':.5, 'lw':1},
                  yerr=err.values.T)

     #for bar, state_name in zip(bars, mr.index):
     #   if state_name in no_lockdown:
     #       bar.set_color(NONE_COLOR)
     #   if state_name in partial_lockdown:
     #       bar.set_color(PARTIAL_COLOR)

    labels = mr.index.to_series().replace({'District of Columbia':'DC'})
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.margins(0)
    ax.set_ylim(0,4.)
    ax.axhline(1.0, linestyle=':', color='k', lw=1)

    #leg = ax.legend(handles=[
    #                    Patch(label='Full', color=FULL_COLOR),
    #                    Patch(label='Partial', color=PARTIAL_COLOR),
    #                    Patch(label='None', color=NONE_COLOR)
    #                ],
    #                title='Lockdown',
    #                ncol=3,
    #                loc='upper left',
    #                columnspacing=.75,
    #                handletextpad=.5,
    #                handlelength=1)

    #leg._legend_box.align = "left"
    fig.set_facecolor('w')
    return fig, ax

def run_full_model(cases, sigma=OPTIMAL_SIGMA):

    # initializing result dict
    result = {''}

    # smoothing series
    new, smoothed = smooth_new_cases(cases)

    # calculating posteriors
    posteriors, log_likelihood = calculate_posteriors(smoothed, sigma=sigma)

    # calculating HDI
    result = highest_density_interval(posteriors, p=.9)

    return result


def estimate_gam(series, n_splines=25, algo=PoissonGAM, n_bootstrap=100):
    
    X = np.arange(series.shape[0])
    y = series.values

    # running GAM in bootstrap
    bootstrap = []
    for _ in range(n_bootstrap):

        weights = dirichlet([1] * series.shape[0]).rvs(1)

        gam = algo(s(0, n_splines) + l(0))
        gam.fit(X, y, weights=weights[0])

        bootstrap.append(gam)
    
    preds = pd.DataFrame([m.predict(X) for m in bootstrap]).T

    return preds

def fit_gam(series, n_splines=25, algo=PoissonGAM, n_bootstrap=100):
    
    X = np.arange(series.shape[0])
    y = series.values

    # running GAM in bootstrap
    bootstrap = []
    for _ in range(n_bootstrap):

        weights = dirichlet([1] * series.shape[0]).rvs(1)

        gam = algo(s(0, n_splines) + l(0))
        gam.fit(X, y, weights=weights[0])

        bootstrap.append(gam)

    return bootstrap

def run_gam_effective_r_from_counts(state_data, n_splines=25, algo=PoissonGAM, n_bootstrap=100):

    estimate_total = estimate_gam(state_data['confirmed_total'], n_splines, algo, n_bootstrap)
    estimate_new = estimate_gam(state_data['confirmed_new'], n_splines, algo, n_bootstrap)

    Rt_samples = estimate_new / estimate_total.shift(1) * (1/RECOVERY_RATE)
    estimate_rt = pd.DataFrame(index = state_data.index)
    estimate_rt['ML'] = Rt_samples.mean(axis=1).values
    estimate_rt['Low_90'] = Rt_samples.quantile(0.05, axis=1).values
    estimate_rt['High_90'] = Rt_samples.quantile(0.95, axis=1).values

    return estimate_rt.dropna()

def run_gam_effective_r_from_empirical(state_data, n_splines=25, algo=GammaGAM, n_bootstrap=100):

    # for numerical stability
    epsilon = 1

    R_series = (state_data['confirmed_new'] / state_data['confirmed_total'].shift(1)).dropna() * 1/RECOVERY_RATE

    X = np.arange(R_series.shape[0])
    y = R_series.values + epsilon

    # running GAM in bootstrap
    bootstrap = []
    for _ in range(n_bootstrap):

        weights = dirichlet([1] * R_series.shape[0]).rvs(1)

        gam = algo(s(0, n_splines) + l(0))
        gam.fit(X, y, weights=weights[0])

        bootstrap.append(gam)

    preds = pd.DataFrame([m.predict(X) - epsilon for m in bootstrap]).T

    estimate_rt = pd.DataFrame(index = R_series.index)
    estimate_rt['ML'] = preds.mean(axis=1).values
    estimate_rt['Low_90'] = preds.quantile(0.05, axis=1).values
    estimate_rt['High_90'] = preds.quantile(0.95, axis=1).values

    return estimate_rt.dropna()

