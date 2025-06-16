import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# Assuming you have your fitted model 'res' and returns data
# This code shows practical applications of your GARCH model


class GARCHAnalysis:
    def __init__(self, fitted_model, returns_data):
        self.model = fitted_model
        self.returns = returns_data
        self.forecasts = None

    def generate_volatility_forecasts(self, horizon=30):
        """Generate multi-step ahead volatility forecasts"""
        # In-sample conditional volatility (what model estimated for historical data)
        self.conditional_vol = self.model.conditional_volatility

        # Out-of-sample forecasts
        self.forecasts = self.model.forecast(horizon=horizon)

        return self.forecasts

    def plot_volatility_evolution(self):
        """Plot historical volatility and forecasts with FIXED date alignment"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Returns and Conditional Volatility (this one was working fine)
        dates = self.returns.index
        ax1.plot(
            dates, self.returns, alpha=0.7, color="blue", label="Returns", linewidth=0.5
        )
        ax1.plot(
            dates,
            self.conditional_vol,
            color="red",
            linewidth=2,
            label="GARCH Volatility",
        )
        ax1.set_title("Apple Returns vs GARCH Conditional Volatility", fontsize=14)
        ax1.set_ylabel("Returns (%)", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Historical volatility + Forecasts (FIXED VERSION)
        if self.forecasts is not None:
            forecast_vol = np.sqrt(self.forecasts.variance.values[-1, :])

            # FIXED: Create proper date index for forecasts
            last_date = dates[-1]

            # Use business days (skip weekends) - change to 'D' if your data includes weekends
            forecast_dates = pd.bdate_range(
                start=last_date + timedelta(days=1), periods=len(forecast_vol), freq="B"
            )

            # Show last 60 days of historical volatility for context
            hist_window = 60
            hist_dates = dates[-hist_window:]
            hist_vol = self.conditional_vol[-hist_window:]

            # Plot historical volatility
            ax2.plot(
                hist_dates,
                hist_vol,
                color="red",
                linewidth=2,
                label="Historical GARCH Vol",
            )

            # FIXED: Plot forecasted volatility with proper connectivity
            # Create extended series that connects historical to forecast
            connection_date = [hist_dates.iloc[-1]]  # Last historical date
            connection_vol = [hist_vol.iloc[-1]]  # Last historical volatility

            # Combine connection point with forecasts for seamless line
            all_forecast_dates = connection_date + list(forecast_dates)
            all_forecast_vol = connection_vol + list(forecast_vol)

            ax2.plot(
                all_forecast_dates,
                all_forecast_vol,
                color="orange",
                linewidth=2,
                linestyle="--",
                label="GARCH Forecast",
                marker="o",
                markersize=3,
            )

            ax2.set_title("GARCH Volatility: Historical vs Forecasted", fontsize=14)
            ax2.set_ylabel("Volatility (%)", fontsize=12)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis="x", rotation=45)

            # Print diagnostic information
            print("=== FORECAST DIAGNOSTICS ===")
            print(f"Last historical date: {last_date}")
            print(f"First forecast date: {forecast_dates[0]}")
            print(f"Last forecast date: {forecast_dates[-1]}")
            print(f"Last historical volatility: {hist_vol.iloc[-1]:.3f}%")
            print(f"First forecast volatility: {forecast_vol[0]:.3f}%")
            print(f"Forecast horizon: {len(forecast_vol)} days")

        plt.tight_layout()
        plt.show()

    def compare_volatility_models(self, returns_data, test_size=252):
        """Compare GARCH against other volatility models"""
        # Split data for out-of-sample testing
        train_data = returns_data[:-test_size]
        test_data = returns_data[-test_size:]

        # Realized volatility (rolling window)
        realized_vol = test_data.rolling(window=20).std() * np.sqrt(252)

        # Model 1: GARCH(1,1) - Refit on training data
        garch_model = arch_model(
            train_data, vol="Garch", p=1, q=1, mean="Constant", dist="normal"
        )
        garch_fit = garch_model.fit(disp="off")

        # Model 2: EWMA (Exponentially Weighted Moving Average)
        ewma_vol = self.calculate_ewma_volatility(returns_data, lambda_=0.94)

        # Model 3: Simple Rolling Standard Deviation
        rolling_vol = returns_data.rolling(window=30).std() * np.sqrt(252)

        # Model 4: Historical Volatility (expanding window)
        expanding_vol = returns_data.expanding().std() * np.sqrt(252)

        # Generate GARCH forecasts for test period
        garch_forecasts = []
        for i in range(len(test_data)):
            if i == 0:
                forecast = garch_fit.forecast(horizon=1)
                garch_forecasts.append(np.sqrt(forecast.variance.values[0, 0]))
            else:
                # Refit model with one more observation (rolling window approach)
                current_data = returns_data[: -(test_size - i)]
                temp_model = arch_model(
                    current_data, vol="Garch", p=1, q=1, mean="Constant", dist="normal"
                )
                temp_fit = temp_model.fit(disp="off")
                forecast = temp_fit.forecast(horizon=1)
                garch_forecasts.append(np.sqrt(forecast.variance.values[0, 0]))

        garch_forecasts = pd.Series(garch_forecasts, index=test_data.index)

        # Align all series for comparison
        comparison_data = pd.DataFrame(
            {
                "Realized_Vol": realized_vol,
                "GARCH_Forecast": garch_forecasts,
                "EWMA": ewma_vol[-test_size:],
                "Rolling_30d": rolling_vol[-test_size:],
                "Expanding": expanding_vol[-test_size:],
            }
        ).dropna()

        return comparison_data

    def calculate_ewma_volatility(self, returns, lambda_=0.94):
        """Calculate EWMA volatility"""
        ewma_var = np.zeros(len(returns))
        ewma_var[0] = returns.iloc[0] ** 2

        for i in range(1, len(returns)):
            ewma_var[i] = (
                lambda_ * ewma_var[i - 1] + (1 - lambda_) * returns.iloc[i] ** 2
            )

        return pd.Series(np.sqrt(ewma_var * 252), index=returns.index)

    def evaluate_forecast_accuracy(self, comparison_data):
        """Evaluate different volatility models"""
        results = {}
        realized = comparison_data["Realized_Vol"].dropna()

        for model_name in ["GARCH_Forecast", "EWMA", "Rolling_30d", "Expanding"]:
            forecasts = comparison_data[model_name].dropna()
            common_idx = realized.index.intersection(forecasts.index)

            if len(common_idx) > 0:
                real_aligned = realized[common_idx]
                forecast_aligned = forecasts[common_idx]

                mse = mean_squared_error(real_aligned, forecast_aligned)
                mae = mean_absolute_error(real_aligned, forecast_aligned)
                rmse = np.sqrt(mse)

                # Mean Absolute Percentage Error
                mape = (
                    np.mean(np.abs((real_aligned - forecast_aligned) / real_aligned))
                    * 100
                )

                results[model_name] = {
                    "RMSE": rmse,
                    "MAE": mae,
                    "MAPE": mape,
                    "R²": np.corrcoef(real_aligned, forecast_aligned)[0, 1] ** 2,
                }

        return pd.DataFrame(results).T

    def risk_management_applications(self):
        """Show practical risk management uses"""
        current_vol = self.conditional_vol.iloc[-1]

        # Value at Risk (VaR) calculation
        portfolio_value = 100000  # $100k portfolio
        confidence_levels = [0.95, 0.99]

        print("=== RISK MANAGEMENT APPLICATIONS ===\n")
        print(f"Current GARCH Volatility Estimate: {current_vol:.2f}%")
        print(f"Annualized Volatility: {current_vol * np.sqrt(252):.1f}%\n")

        print("Value at Risk (VaR) Estimates:")
        for conf in confidence_levels:
            z_score = np.abs(
                np.percentile(np.random.normal(0, 1, 10000), (1 - conf) * 100)
            )
            var_1day = portfolio_value * z_score * (current_vol / 100)
            var_10day = portfolio_value * z_score * (current_vol / 100) * np.sqrt(10)

            print(f"{conf*100}% VaR (1-day): ${var_1day:,.0f}")
            print(f"{conf*100}% VaR (10-day): ${var_10day:,.0f}")

        # Position sizing based on volatility
        print(f"\nPOSITION SIZING:")
        target_vol = 2.0  # Target 2% daily volatility
        position_size = target_vol / current_vol
        print(f"To achieve {target_vol}% daily vol, use {position_size:.1f}x leverage")

        # Volatility timing strategy
        long_run_vol = np.sqrt(0.3556 / (1 - 0.1034 - 0.7348))  # From your model
        vol_ratio = current_vol / long_run_vol

        print(f"\nVOLATILITY TIMING:")
        print(f"Current vol vs long-run avg: {vol_ratio:.2f}x")
        if vol_ratio > 1.2:
            print("→ Consider reducing position size (high vol period)")
        elif vol_ratio < 0.8:
            print("→ Consider increasing position size (low vol period)")
        else:
            print("→ Volatility near normal levels")


# Example usage:
# Assuming you have your fitted model 'res' and returns data
# analysis = GARCHAnalysis(res, returns)
# forecasts = analysis.generate_volatility_forecasts(horizon=30)
# analysis.plot_volatility_evolution()
#
# # Compare models
# comparison = analysis.compare_volatility_models(returns, test_size=252)
# accuracy_results = analysis.evaluate_forecast_accuracy(comparison)
# print(accuracy_results)
#
# # Risk management applications
# analysis.risk_management_applications()

print("GARCH Model Applications:")
print("1. Volatility Forecasting - Generate multi-step ahead volatility predictions")
print("2. Model Comparison - Test against EWMA, rolling vol, expanding vol")
print("3. Risk Management - VaR calculations, position sizing, volatility timing")
print("4. Options Pricing - Use forecasted volatility in Black-Scholes models")
print(
    "5. Portfolio Optimization - Dynamic risk budgeting based on volatility forecasts"
)
