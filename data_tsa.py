import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox  # Import the correct function
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import jarque_bera, kstest
from data_fetcher import btc_data
import matplotlib.dates as mdates


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

    def check_autocorrelation(self, lags=25, alpha=0.05, show_plot=True):
        """
        Check the autocorrelation of the time series using ACF and PACF plots.
        Parameters:
            lags (int): Number of lags to consider. Default is 25.
            alpha (float): Significance level for the confidence interval. Default is 0.05.
            show_plot (bool): Whether to display the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        plot_acf(self.data[self.target], lags=lags, alpha=alpha, ax=ax1)
        ax1.set_title("ACF for {}".format(self.target))
        plot_pacf(self.data[self.target], lags=lags, alpha=alpha, method='ols', ax=ax2)
        ax2.set_title("PACF for {}".format(self.target))
        self.save_and_show_plot(fig, 'autocorrelation.png', show=show_plot)
        return fig

    def decompose_time_series(self, model='additive', filter_last='all', period=12):
        """
        Decomposes the time series data into trend, seasonal, and residual components.

        Parameters:
            model (str): Type of decomposition model - 'additive' or 'multiplicative'.
            filter_last (str): Timeframe to filter the data. Options are 'all', '1Y' (last year), '5Y' (last 5 years).
            period (int): The number of observations per cycle (e.g., 12 for monthly data with an annual cycle).
        """
        # Filter data based on the timeframe
        if filter_last != 'all':
            if filter_last == '1Y':
                start_date = self.data.index.max() - pd.DateOffset(years=1)
            elif filter_last == '3Y':
                start_date = self.data.index.max() - pd.DateOffset(years=3)
            elif filter_last == '5Y':
                start_date = self.data.index.max() - pd.DateOffset(years=5)
            filtered_data = self.data[self.data.index >= start_date]
        else:
            filtered_data = self.data

        # Decomposition
        decomposed = seasonal_decompose(filtered_data[self.target], model=model, period=period)
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axs[0].plot(decomposed.observed)
        axs[0].set_title('Observed')
        axs[1].plot(decomposed.trend)
        axs[1].set_title('Trend')
        axs[2].plot(decomposed.seasonal)
        axs[2].set_title('Seasonality')
        axs[3].plot(decomposed.resid)
        axs[3].set_title('Residuals')

        # Formatting the x-axis for better readability
        axs[-1].xaxis.set_major_locator(mdates.YearLocator())
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        self.save_and_show_plot(fig, f'decomposition_{filter_last}.png')
        return decomposed

    def diagnostics(self, lags=40, alpha=0.05):
        """
        Performs various statistical tests for diagnostics.

        Parameters:
            lags (int): The number of lags to be used in the Ljung-Box test.
            alpha (float): Significance level for the tests.
        """
        try:
            logger.info("Performing diagnostics on the time series data.")
            results = {}

            # Augmented Dickey-Fuller test
            adf_test = adfuller(self.data[self.target])
            adf_conclusion = ("Stationary, no differencing needed." if adf_test[1] <= alpha else
                              "Non-stationary, consider differencing.")
            results['ADF Test'] = {'Statistic': adf_test[0], 'p-value': adf_test[1], 'Conclusion': adf_conclusion}

            # KPSS test
            kpss_test = kpss(self.data[self.target], 'ct')
            kpss_conclusion = ("Non-stationary, consider differencing." if kpss_test[1] <= alpha else
                               "Stationary, no differencing needed.")
            results['KPSS Test'] = {'Statistic': kpss_test[0], 'p-value': kpss_test[1], 'Conclusion': kpss_conclusion}

            # # Ljung-Box test
            # lb_test = acorr_ljungbox(self.data[self.target], lags=lags, return_df=True)
            # lb_conclusion = "Significant autocorrelation." if (lb_test['lb_pvalue'] < alpha).any() else "No significant autocorrelation."
            # results['Ljung-Box Test'] = {'p-values': lb_test['lb_pvalue'].to_dict(), 'Conclusion': lb_conclusion}

            # Jarque-Bera test
            jb_test = jarque_bera(self.data[self.target])
            jb_conclusion = ("Non-normal distribution." if jb_test[1] <= alpha else
                             "Normal distribution.")
            results['Jarque-Bera Test'] = {'Statistic': jb_test[0], 'p-value': jb_test[1], 'Conclusion': jb_conclusion}

            # Kolmogorov-Smirnov test
            ks_test = kstest(self.data[self.target], 'norm')
            ks_conclusion = ("Distribution differs significantly from normal." if ks_test[1] <= alpha else
                             "No significant difference from normal distribution.")
            results['Kolmogorov-Smirnov Test'] = {'Statistic': ks_test[0], 'p-value': ks_test[1], 'Conclusion': ks_conclusion}

            # Output to console
            for test, result in results.items():
                print(f"--- {test} ---")
                for key, value in result.items():
                    if key == 'p-values':
                        for lag, p_val in value.items():
                            print(f"    Lag {lag}: p-value = {p_val:.4f}")
                    else:
                        print(f"    {key}: {value}")
                print("\n")
                
            logger.info("Diagnostics tests completed successfully.")
            return None

        except Exception as e:
            logger.error(f"Error during diagnostics: {e}")
            return None

    def transform_series(self, method='difference', param=None, plot=True):
        """
        Transforms the time series to achieve stationarity.

        Parameters:
            method (str): The method of transformation ('difference', 'log', 'boxcox').
            param (float, optional): Parameter for the transformation (e.g., lambda for Box-Cox).
            plot (bool): Whether to plot the original and transformed series.
        """
        if method not in ['difference', 'log', 'boxcox']:
            raise ValueError("Invalid method. Choose from 'difference', 'log', 'boxcox'.")

        self.original_data = self.data.copy()  # Store the original data

        if method == 'difference':
            # Differencing the series
            self.transformed_data = self.data[self.target].diff().dropna()
            transform_title = 'Differenced Series'

        elif method == 'log':
            # Log transformation
            self.transformed_data = np.log(self.data[self.target])
            transform_title = 'Log Transformed Series'

        elif method == 'boxcox':
            # Box-Cox transformation
            from scipy.stats import boxcox
            self.transformed_data, fitted_lambda = boxcox(self.data[self.target])
            transform_title = f'Box-Cox Transformed Series (Lambda={fitted_lambda:.2f})'
            if param is not None:
                self.transformed_data, _ = boxcox(self.data[self.target], lmbda=param)

        # Plotting the original and transformed series
        if plot:
            fig, ax = plt.subplots(2, 1, figsize=(12, 8))
            ax[0].plot(self.original_data[self.target], label='Original Series')
            ax[0].set_title('Original Time Series')
            ax[0].legend()

            ax[1].plot(self.transformed_data, label=transform_title, color='orange')
            ax[1].set_title(transform_title)
            ax[1].legend()

            plt.tight_layout()
            plt.show()

        return self.transformed_data



tsa = TimeSeriesAnalysis(btc_data, target='Close')
#granger_results = tsa.test_granger_causality('Open', maxlag=30, verbose=False)
#display(HTML(tsa.concise_granger_output_table(granger_results)))
#autocorr_fig = tsa.check_autocorrelation(lags=35)
#decomposed = tsa.decompose_time_series(model='additive', filter_last='3Y', period=12)
#diagnostic_results = tsa.diagnostics(lags=40)
#transformed_data = tsa.transform_series(method='difference', plot=True)
