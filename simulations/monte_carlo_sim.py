import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import logging
import os
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_inputs(params: Dict[str, Any]) -> None:
    """Validate input parameters for the Monte Carlo simulation."""
    if params['initial_capital'] <= 0:
        raise ValueError("Initial capital must be positive")
    if not 0 <= params['winrate'] <= 100:
        raise ValueError("Win rate must be between 0 and 100")
    if params['trades_per_day'] <= 0:
        raise ValueError("Trades per day must be positive")
    if params['trading_days'] <= 0:
        raise ValueError("Trading days must be positive")
    if params['risk_per_trade'] <= 0:
        raise ValueError("Risk per trade must be positive")
    if params['risk_reward_ratio'] <= 0:
        raise ValueError("Risk reward ratio must be positive")
    if params['commission'] < 0:
        raise ValueError("Commission cannot be negative")
    if params['num_simulations'] <= 0:
        raise ValueError("Number of simulations must be positive")
    if params['breakeven_rate'] < 0:
        raise ValueError("Breakeven rate cannot be negative")

def generate_correlated_outcomes(num_trades: int, win_rate: float, correlation: float = 0.2) -> np.ndarray:
    """Generate correlated trade outcomes."""
    mean = [0.5] * num_trades
    cov = [[correlation if i != j else 1 for j in range(num_trades)] for i in range(num_trades)]
    correlated_random = np.random.multivariate_normal(mean=mean, cov=cov)
    return correlated_random < win_rate/100

def get_dynamic_winrate(market_condition: str, base_winrate: float) -> float:
    """Adjust win rate based on market conditions."""
    condition_adjustment = {
        'bull': 1.1,
        'bear': 0.9,
        'sideways': 1.0
    }
    return base_winrate * condition_adjustment[market_condition]

def calculate_position_size(capital: float, volatility: float, max_risk_percent: float) -> float:
    """Dynamic position sizing based on volatility."""
    return min(capital * max_risk_percent / 100, capital * 0.02 / volatility)

def add_market_volatility(returns: np.ndarray, volatility_factor: float = 1.0) -> np.ndarray:
    """Adjust returns based on market volatility."""
    market_volatility = np.random.normal(0, volatility_factor, len(returns))
    return returns * (1 + market_volatility)

def calculate_risk_metrics(returns: np.ndarray, risk_free_rate: float, trading_days: int) -> Dict[str, float]:
    """Calculate various risk metrics."""
    excess_returns = returns - (risk_free_rate / 252 * trading_days)
    std_dev = np.std(returns)
    downside_returns = returns[returns < 0]
    
    metrics = {
        'sharpe_ratio': np.sqrt(252) * np.mean(excess_returns) / std_dev if std_dev > 0 else 0,
        'sortino_ratio': np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns) if len(downside_returns) > 0 else 0,
        'var_95': np.percentile(returns, 5),  # 95% VaR
        'expected_shortfall': np.mean(returns[returns <= np.percentile(returns, 5)]) if len(returns[returns <= np.percentile(returns, 5)]) > 0 else 0,
        'max_drawdown': abs(np.min(returns)) if len(returns) > 0 else 0,
        'calmar_ratio': np.mean(returns) / abs(np.min(returns)) if len(returns) > 0 and np.min(returns) < 0 else 0
    }
    return metrics

# Monte Carlo simulation function for trading strategy
def monte_carlo_simulation(
    initial_capital=10000,
    winrate=55,
    trades_per_day=3.0,
    trading_days=65,
    risk_per_trade=0.5,
    is_risk_percent=True,
    risk_reward_ratio=1.5,
    commission=0.01,  # Roundtrip commission in percentage
    num_simulations=10000,
    correlation=0.2,
    volatility_factor=1.0,
    market_condition='sideways',
    breakeven_rate=3.0  # Default 3% of trades move to breakeven
):
    try:
        # Create plots directory if it doesn't exist
        script_dir = Path(__file__).parent
        plots_dir = script_dir / 'plots_simulations'
        plots_dir.mkdir(exist_ok=True)

        # Validate inputs
        params = locals()
        validate_inputs(params)
        logger.info(f"Starting Monte Carlo simulation with parameters: {params}")

        # Initialize arrays
        final_capitals = np.zeros(num_simulations)
        max_drawdowns = np.zeros(num_simulations)
        daily_returns_all = []
        capital_histories = []
        selected_sim_indices = [0, 1, 2]

        # Adjust winrate based on market condition
        adjusted_winrate = get_dynamic_winrate(market_condition, winrate)

        # Run simulations
        for sim in range(num_simulations):
            capital = initial_capital
            peak_capital = initial_capital
            max_drawdown = 0
            capital_history = [capital]
            daily_returns = []
            
            for day in range(trading_days):
                num_trades = max(0, int(np.random.normal(trades_per_day, trades_per_day * 0.2)))
                
                # Generate correlated outcomes for the day
                if num_trades > 0:
                    trade_outcomes = generate_correlated_outcomes(num_trades, adjusted_winrate, correlation)
                    
                    for trade_won in trade_outcomes:
                        # Calculate dynamic position size based on recent volatility
                        volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else risk_per_trade
                        if is_risk_percent:
                            risk = calculate_position_size(capital, volatility, risk_per_trade)
                        else:
                            risk = risk_per_trade

                        commission_cost = risk * (commission / 100)
                        
                        # Check if trade moves to breakeven
                        if np.random.random() * 100 < breakeven_rate:
                            # Breakeven trade (only pay commission)
                            capital -= commission_cost
                            daily_returns.append(-commission_cost / capital)
                        else:
                            if trade_won:
                                profit = risk * risk_reward_ratio - commission_cost
                                capital += profit
                                daily_returns.append(profit / capital)
                            else:
                                loss = risk + commission_cost
                                capital -= loss
                                daily_returns.append(-loss / capital)

                        # Update peak capital and max drawdown
                        peak_capital = max(peak_capital, capital)
                        current_drawdown = (peak_capital - capital) / peak_capital * 100
                        max_drawdown = max(max_drawdown, current_drawdown)

                        if capital <= 0:
                            capital = 0
                            break

                        capital_history.append(capital)

                if capital <= 0:
                    break

            final_capitals[sim] = capital
            max_drawdowns[sim] = max_drawdown
            daily_returns_all.extend(daily_returns)
            
            if sim in selected_sim_indices:
                capital_histories.append(capital_history)

        # Calculate all statistics
        returns = (final_capitals - initial_capital) / initial_capital
        returns_with_volatility = add_market_volatility(returns, volatility_factor)
        
        stats = {
            'mean_capital': np.mean(final_capitals),
            'median_capital': np.median(final_capitals),
            'percentile_5': np.percentile(final_capitals, 5),
            'percentile_95': np.percentile(final_capitals, 95),
            'prob_loss': np.sum(final_capitals < initial_capital) / num_simulations * 100,
            'avg_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': np.max(max_drawdowns)
        }
        
        # Calculate additional risk metrics
        risk_metrics = calculate_risk_metrics(returns_with_volatility, 0.02, trading_days)
        stats.update(risk_metrics)

        # Create visualizations with all metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = plots_dir / f'monte_carlo_sim_{timestamp}.png'
        create_visualizations(
            final_capitals=final_capitals,
            initial_capital=initial_capital,
            stats=stats,
            capital_histories=capital_histories,
            params=params,
            plot_filename=str(plot_filename)
        )

        logger.info("Monte Carlo simulation completed successfully")
        return stats

    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {str(e)}")
        raise

def create_visualizations(
    final_capitals: np.ndarray,
    initial_capital: float,
    stats: Dict[str, float],
    capital_histories: List[List[float]],
    params: Dict[str, Any],
    plot_filename: str
):
    """Create and save visualization plots with all parameters and results."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')  # Using a built-in style
        # Create figure with three subplots
        fig = plt.figure(figsize=(16, 24))
        gs = plt.GridSpec(3, 1, height_ratios=[1, 1.2, 1.2])
        
        # Parameters text subplot
        ax0 = fig.add_subplot(gs[0])
        ax0.axis('off')
        
        # Create parameter text with larger font
        param_text = "Monte Carlo Simulation Parameters:\n\n"
        param_text += f"Initial Capital: ${params['initial_capital']:,.2f}\n"
        param_text += f"Win Rate: {params['winrate']}%\n"
        param_text += f"Trades per Day: {params['trades_per_day']}\n"
        param_text += f"Trading Days: {params['trading_days']}\n"
        param_text += f"Risk per Trade: {'$' + str(params['risk_per_trade']) if not params['is_risk_percent'] else str(params['risk_per_trade']) + '%'}\n"
        param_text += f"Risk/Reward Ratio: {params['risk_reward_ratio']}\n"
        param_text += f"Commission: {params['commission']}%\n"
        param_text += f"Breakeven Rate: {params['breakeven_rate']}%\n"
        param_text += f"Market Condition: {params['market_condition'].title()}\n"
        param_text += f"Trade Correlation: {params['correlation']}\n\n"
        
        # Add results text
        param_text += "Simulation Results:\n\n"
        param_text += f"Final Median Capital: ${stats['median_capital']:,.2f}\n"
        param_text += f"5th Percentile: ${stats['percentile_5']:,.2f}\n"
        param_text += f"95th Percentile: ${stats['percentile_95']:,.2f}\n"
        param_text += f"Probability of Loss: {stats['prob_loss']:.2f}%\n"
        param_text += f"Average Max Drawdown: {stats['avg_max_drawdown']:.2f}%\n"
        param_text += f"Worst Drawdown: {stats['worst_drawdown']:.2f}%\n\n"
        param_text += f"Risk Metrics:\n"
        param_text += f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n"
        param_text += f"Sortino Ratio: {stats['sortino_ratio']:.2f}\n"
        param_text += f"Calmar Ratio: {stats['calmar_ratio']:.2f}\n"
        param_text += f"Value at Risk (95%): {stats['var_95']*100:.2f}%\n"
        param_text += f"Expected Shortfall: {stats['expected_shortfall']*100:.2f}%"
        
        # Add text with larger font and better positioning
        ax0.text(0.02, 0.98, param_text, transform=ax0.transAxes,
                verticalalignment='top', fontfamily='monospace',
                fontsize=14, bbox=dict(facecolor='white', alpha=0.9,
                                     edgecolor='gray', boxstyle='round,pad=1'))

        # Histogram subplot
        ax1 = fig.add_subplot(gs[1])
        ax1.hist(final_capitals, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax1.axvline(initial_capital, color='red', linestyle='dashed', label='Initial Capital')
        ax1.axvline(stats['median_capital'], color='green', linestyle='dashed', label='Median')
        ax1.axvline(stats['percentile_5'], color='orange', linestyle='dashed', label='5th %')
        ax1.axvline(stats['percentile_95'], color='purple', linestyle='dashed', label='95th %')
        ax1.set_title('Final Capital Distribution', fontsize=16, pad=20)
        ax1.set_xlabel('Final Capital ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=12)

        # Capital evolution subplot
        ax2 = fig.add_subplot(gs[2])
        labels = ['Median Outcome', '5th Percentile', '95th Percentile']
        colors = ['green', 'orange', 'purple']
        for i, history in enumerate(capital_histories):
            ax2.plot(history, color=colors[i], label=labels[i], alpha=0.7, linewidth=2)
        ax2.axhline(initial_capital, color='red', linestyle='dashed', label='Initial Capital')
        ax2.set_title('Capital Evolution Over Trades', fontsize=16, pad=20)
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Capital ($)', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=12)

        # Add timestamp and watermark
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.99, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', 
                alpha=0.5, fontsize=10, fontfamily='monospace')

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Visualization saved to {plot_filename}")

    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise

# Run simulation with default parameters
if __name__ == "__main__":
    # Run with specified parameters
    monte_carlo_simulation(
        initial_capital=10000,  # Starting capital
        winrate=63,            # 63% win rate
        trades_per_day=5.0,    # 5 trades per day
        trading_days=65,       # ~3 months (21 trading days per month)
        risk_per_trade=100,    # $100 per trade
        is_risk_percent=False, # Fixed dollar amount
        risk_reward_ratio=1.8, # Average RR 1.8
        commission=0,          # No commission
        breakeven_rate=3.0,    # 3% trades move to breakeven
        market_condition='sideways'  # Neutral market condition
    )