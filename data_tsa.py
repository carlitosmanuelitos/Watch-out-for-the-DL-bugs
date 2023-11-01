import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import jarque_bera, kstest
from data_fetcher import btc_data


# Other settings
from IPython.display import display, HTML
import os, warnings, logging
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.3f}'.format)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesAnalysis:
    """
    A class to perform various time series analysis tasks such as stationarity checks, volatility modeling, and decomposition.

    Attributes:
        data (pd.DataFrame): Time series data.
        target (str): Target column for time series analysis.
    """

    def __init__(self, data, target):
        """
        Initialize the TimeSeriesAnalysis class.

        Parameters:
            data (pd.DataFrame): Time series data.
            target (str): Target column for time series analysis.
        """
        if target not in data.columns:
            raise ValueError(f"'{target}' is not a column in the provided data.")
        self.original_data = data.copy()
        self.data = data
        self.target = target
        self.alpha = 0.05  

    def check_autocorrelation(self, show_plot=True):
        """
        Check the autocorrelation of the time series using ACF and PACF plots.

        Returns:
            tuple: ACF and PACF figures.
        """
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,6))
        plot_acf(self.data[self.target], lags=50, alpha=0.05, ax=ax1)
        ax1.set_title("ACF for {}".format(self.target))
        plot_pacf(self.data[self.target], lags=50, alpha=0.05, method='ols', ax=ax2)
        ax2.set_title("PACF for {}".format(self.target))
        self.save_and_show_plot(fig, 'autocorrelation.png', show=show_plot)
        return fig
    
    def decompose_time_series(self, model='additive', period=None, show=True):
        """
        Decompose the time series data into trend, seasonal, and residual components.
    
        Parameters:
            model (str): The type of decomposition model ('additive' or 'multiplicative').
            period (int): The period for seasonal decomposition. If None, it will be inferred.
            show (bool): Whether to display the plot.
        Returns:
            dict: A dictionary containing decomposition components and guidance.
        """
        logger.info("Decomposing the time series")
        if period is None:
            # Attempt to infer the seasonal period
            period = self.infer_seasonal_period()
            
        result = seasonal_decompose(self.data[self.target], model=model, period=period)
    
        # Adjusting the figsize here
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
        result.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        result.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        result.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        result.resid.plot(ax=ax4)
        ax4.set_title('Residual')
    
        self.save_and_show_plot(fig, 'decompose.png', show=show)
        
        # Analyzing the decomposition results
        guidance = self.analyze_decomposition(result)
        
        # Returning the decomposition components and guidance
        return {
            'Observed': result.observed,
            'Trend': result.trend,
            'Seasonal': result.seasonal,
            'Residual': result.resid,
            'Guidance': guidance
        }

    def analyze_decomposition(self, decomposition_result):
        """
        Analyze the time series decomposition results and provide guidance.
    
        Parameters:
            decomposition_result (DecomposeResult): The result from seasonal decomposition.
    
        Returns:
            str: Guidance based on the decomposition results.
        """
        guidance = ""
        
        # Analyzing the trend component
        if decomposition_result.trend.isnull().sum() > 0:
            guidance += "The trend component has missing values at the boundaries. Consider using a different decomposition method or filling the missing values.\n"
        else:
            trend_strength = np.nanmean(np.abs(decomposition_result.trend - np.nanmean(decomposition_result.trend)))
            if trend_strength > 0.05 * np.nanmean(decomposition_result.observed):
                guidance += "The trend component is strong. Consider detrending the series if you intend to use models that assume stationarity.\n"
            else:
                guidance += "The trend component is weak, indicating a relatively stable mean over time.\n"
                
        # Analyzing the seasonal component
        seasonal_strength = np.nanmean(np.abs(decomposition_result.seasonal - np.nanmean(decomposition_result.seasonal)))
        if seasonal_strength > 0.05 * np.nanmean(decomposition_result.observed):
            guidance += "The seasonal component is strong. Consider seasonal adjustment or using models that can handle seasonality.\n"
        else:
            guidance += "The seasonal component is weak, indicating that seasonality may not be a significant factor.\n"
            
        # Analyzing the residual component
        if decomposition_result.resid.isnull().sum() > 0:
            guidance += "The residual component has missing values at the boundaries. Consider using a different decomposition method or filling the missing values.\n"
        else:
            if np.nanstd(decomposition_result.resid) > 0.05 * np.nanmean(decomposition_result.observed):
                guidance += "The residual component shows variability. Consider further analysis to identify any remaining patterns or anomalies.\n"
            else:
                guidance += "The residual component is relatively stable, indicating that most of the patterns have been captured by the trend and seasonal components.\n"
                
        return guidance
        
    def save_and_show_plot(self, fig, filename, show=True):
        """
        Utility method to save and display the plot.

        Parameters:
            fig (matplotlib.figure.Figure): The plot figure.
            filename (str): Filename to save the plot.
            show (bool, optional): Whether to display the plot. Default is True.
        """
        if not os.path.exists('ts_plots_assets'):
            os.makedirs('ts_plots_assets')
        path = os.path.join('ts_plots_assets', filename)
        fig.savefig(path)
        if show:
            plt.show()

    def diagnostic_check(self, alpha=None, return_stationarity=False):
        alpha = alpha if alpha is not None else self.alpha
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(self.data[self.target])
        if adf_result[1] <= alpha:
            adf_conclusion = "The time series appears to be stationary based on the ADF test."
            adf_guidance = "You might not need to difference the time series. However, consider checking other diagnostics and plots to confirm."
        else:
            adf_conclusion = "The time series appears to be non-stationary based on the ADF test. Differencing might be needed to make it stationary."
            adf_guidance = "Consider applying differencing or transformation to achieve stationarity."
        
        # Jarque-Bera test
        jb_value, p_value = jarque_bera(self.data[self.target])
        if p_value > alpha:
            jb_conclusion = "The time series seems to follow a normal distribution based on the Jarque-Bera test."
            jb_guidance = "The normality assumption holds. This is good if you plan to use models that assume normally distributed residuals."
        else:
            jb_conclusion = "The time series does not appear to be normally distributed based on the Jarque-Bera test."
            jb_guidance = "Consider transforming the series or using models that do not assume normality."
        
        # KPSS test
        kpss_value, kpss_p_value, _, kpss_crit = kpss(self.data[self.target])
        if kpss_p_value > alpha:
            kpss_conclusion = "The time series appears to be stationary around a constant or trend based on the KPSS test."
            kpss_guidance = "The series might be stationary. However, consider other diagnostics to confirm."
        else:
            kpss_conclusion = "The time series appears to be non-stationary based on the KPSS test. It might have a unit root."
            kpss_guidance = "Consider differencing or detrending the series to achieve stationarity."
        
        # Kolmogorov-Smirnov test
        ks_value, ks_p_value = kstest(self.data[self.target], 'norm')
        if ks_p_value > alpha:
            ks_conclusion = "The time series appears to follow a normal distribution based on the Kolmogorov-Smirnov test."
            ks_guidance = "The normality assumption holds, which is beneficial for certain statistical models."
        else:
            ks_conclusion = "The time series does not seem to follow a normal distribution based on the Kolmogorov-Smirnov test."
            ks_guidance = "Consider transforming the series or using models that do not assume normality."
    
        # Create the results dictionary
        results = {
            'ADF': {
                'Statistic': adf_result[0],
                'p-value': adf_result[1],
                'Critical Values': adf_result[4],
                'Conclusion': adf_conclusion,
                'Guidance': adf_guidance
            },
            'Jarque-Bera': {
                'Statistic': jb_value,
                'p-value': p_value,
                'Conclusion': jb_conclusion,
                'Guidance': jb_guidance
            },
            'KPSS': {
                'Statistic': kpss_value,
                'p-value': kpss_p_value,
                'Critical Values': kpss_crit,
                'Conclusion': kpss_conclusion,
                'Guidance': kpss_guidance
            },
            'Kolmogorov-Smirnov': {
                'Statistic': ks_value,
                'p-value': ks_p_value,
                'Conclusion': ks_conclusion,
                'Guidance': ks_guidance
            }
        }
    
        # Check for Seasonality using ACF
        lag_acf = acf(self.data[self.target], nlags=40)
        lag_pacf = pacf(self.data[self.target], nlags=40, method='ols')
    
        # If there are significant peaks at regular intervals in ACF, we can suspect seasonality
        seasonality_conclusion = "Seasonality is likely present in the time series." if any(lag_acf > 0.2) else "Seasonality is likely not present in the time series."
    
        # Now, update the results dictionary with the seasonality check
        results['Seasonality'] = {
            'Conclusion': seasonality_conclusion
        }
    
        # Determine stationarity based on the tests
        is_stationary = adf_result[1] <= alpha and kpss_p_value > alpha
    
        # Output or return results based on return_stationarity
        if return_stationarity:
            return is_stationary
        else:
            # Output to console
            for test, result in results.items():
                print(f"--- {test} ---")
                for key, value in result.items():
                    print(f"{key}: {value}")
                print("\n")
            return None

    def auto_stationary(self, seasonal_period=None):
        """
        Automatically choose the best method to make the series stationary based on its characteristics.
    
        Parameters:
            seasonal_period (int, optional): The period of the seasonality, used for seasonal differencing.
    
        Returns:
            pd.Series: The transformed time series.
        """
        original_series = self.data[self.target]
        
        # Check for stationarity
        if self.diagnostic_check(return_stationarity=True):
            print("The series is already stationary.")
            return original_series
        
        # Try differencing
        diff_series = original_series.diff().dropna()
        if self.diagnostic_check(alpha=self.alpha, return_stationarity=True):
            print("Differencing made the series stationary.")
            return diff_series
        
        # Try log transformation
        log_series = np.log(original_series).dropna()
        if self.diagnostic_check(alpha=self.alpha, return_stationarity=True):
            print("Log transformation made the series stationary.")
            return log_series
        
        # Try square root transformation
        sqrt_series = np.sqrt(original_series).dropna()
        if self.diagnostic_check(alpha=self.alpha, return_stationarity=True):
            print("Square root transformation made the series stationary.")
            return sqrt_series
        
        # If seasonal_period is provided, try seasonal differencing
        if seasonal_period is not None:
            seasonal_diff_series = original_series.diff(seasonal_period).dropna()
            if self.diagnostic_check(alpha=self.alpha, return_stationarity=True):
                print("Seasonal differencing made the series stationary.")
                return seasonal_diff_series
        
        print("Could not make the series stationary with the tried transformations.")
        return None

    def make_stationary(self, method='auto', seasonal_period=None):
        """
        Make the time series stationary.
    
        Parameters:
            method (str): The method used to make the series stationary. Options are 'diff' (differencing), 
                          'seasonal_diff' (seasonal differencing), 'log' (log transformation), 'sqrt' (square root transformation),
                          and 'auto' (automatically choose the best method).
            seasonal_period (int, optional): The period of the seasonality, used for seasonal differencing.
    
        Returns:
            pd.Series: The transformed time series.
        """
        if method == 'auto':
            # Providing recommendations based on the characteristics of the time series
            return self.auto_stationary(seasonal_period)
        elif method == 'diff':
            return self.data[self.target].diff().dropna()
        elif method == 'seasonal_diff' and seasonal_period:
            return self.data[self.target].diff(seasonal_period).dropna()
        elif method == 'log':
            return np.log(self.data[self.target]).dropna()
        elif method == 'sqrt':
            return np.sqrt(self.data[self.target]).dropna()
        else:
            raise ValueError("Invalid method or missing seasonal_period for seasonal differencing.")
    
    def visualize_stationarity(self, show_plot=True):
        """
        Visualize the series before and after making it stationary.

        Parameters:
            show_plot (bool): Whether to display the plot.
        
        Returns:
            matplotlib.Figure: The generated figure.
        """
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))

        # Plot original data
        self.original_data[self.target].plot(ax=ax[0], title='Original Series', color='blue')
        
        # Plot transformed data
        self.data[self.target].plot(ax=ax[1], title='Transformed Series', color='green')
        
        plt.tight_layout()
        self.save_and_show_plot(fig, 'stationarity_comparison.png', show=show_plot)
        
        return fig
    
    def test_granger_causality(self, other_column, maxlag=30, verbose=False):
        """Test Granger Causality between target and another time series column.
        
        Parameters:
            other_column (str): The name of the other column to test for Granger Causality.
            maxlag (int): The maximum number of lags to consider for the test.
            verbose (bool): Whether to display detailed output.
        
        Returns:
            dict: A dictionary containing the Granger Causality test results.
        """
        logger.info("Testing Granger causality")
        if other_column not in self.data.columns:
            raise ValueError(f"'{other_column}' is not a column in the provided data.")
        other_data = self.data[other_column].values
        target_data = self.data[self.target].values
        data = np.column_stack((target_data, other_data))
        result = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)
        return result
    
    def concise_granger_output_table(self, granger_results):
        """Generate a concise report from the Granger Causality test results in a table format."""
        table_content = ['<table border="1" style="border-collapse:collapse;">']
        lags = list(granger_results.keys())
        for i in range(0, len(lags), 6):
            table_content.append('<tr>')
            for j in range(6):
                if i + j < len(lags):
                    lag = lags[i + j]
                    test_statistics = granger_results[lag][0]
                    cell_content = (f"<b>Lag: {lag}</b><br>"
                                    f"ssr_ftest: F={test_statistics['ssr_ftest'][0]:.4f}, p={test_statistics['ssr_ftest'][1]:.4f}<br>"
                                    f"ssr_chi2test: chi2={test_statistics['ssr_chi2test'][0]:.4f}, p={test_statistics['ssr_chi2test'][1]:.4f}<br>"
                                    f"lrtest: chi2={test_statistics['lrtest'][0]:.4f}, p={test_statistics['lrtest'][1]:.4f}<br>"
                                    f"params_ftest: F={test_statistics['params_ftest'][0]:.4f}, p={test_statistics['params_ftest'][1]:.4f}")
                    table_content.append(f'<td style="padding: 8px; text-align: left;">{cell_content}</td>')
            table_content.append('</tr>')
        table_content.append('</table>')
        return "\n".join(table_content)
    
tsa = TimeSeriesAnalysis(btc_data, target='Close')
print("Diagnostics for Original Series:")
original_diagnostics = tsa.diagnostic_check()
    # Make the series stationary and update the data attribute
stationary_series = tsa.make_stationary(method='log')
tsa.data[tsa.target] = stationary_series

tsa.visualize_stationarity(show_plot=True)
    # Create a new instance with the stationary series
tsa_stationary = TimeSeriesAnalysis(pd.DataFrame({tsa.target: stationary_series}), target=tsa.target)
print("\nDiagnostics for Stationary Series:")
stationary_diagnostics = tsa_stationary.diagnostic_check()
