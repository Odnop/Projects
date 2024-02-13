import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def simulate_trades(win_rate, risk_reward_ratio, position_size, num_trades):
    wins = 0
    losses = 0
    total_profit = 10000  # Initial Amount
    profit_per_trade = []
    drawdown_per_trade = []
    max_profit = 0
    
    for _ in range(num_trades):
        if random.random() < win_rate:
            profit = (1 + risk_reward_ratio) * position_size
            wins += 1
        else:
            profit = -position_size
            losses += 1
        
        total_profit += profit
        profit_per_trade.append(total_profit)
        max_profit = max(max_profit, total_profit)
        drawdown_per_trade.append(total_profit - max_profit)
    
    return wins, losses, profit_per_trade, drawdown_per_trade


def dollar_formatter(x, pos):
    return '${:,.0f}'.format(x)

def main():
    win_rate = 0.3
    risk_reward_ratio = 2
    num_simulations = 30
    
    all_profits = []
    all_drawdowns = []  # Added to save drawdowns
    labels = []
    for i in range(num_simulations):
        _, _, profit_per_trade, drawdown_per_trade = simulate_trades(win_rate, risk_reward_ratio, 100, 1000)
        all_profits.append(profit_per_trade)
        all_drawdowns.append(drawdown_per_trade)  # Save drawdowns
        labels.append(f"Simulation {i+1}")
    
    avg_profits = [sum(profits) / len(profits) for profits in zip(*all_profits)]
    
    plt.figure(figsize=(10, 6))
    
    for i, profit_per_trade in enumerate(all_profits):
        plt.plot(profit_per_trade, color='blue', alpha=0.2)  # Reduced blue color intensity

    # Plot the best simulation
    best_simulation = max(all_profits, key=lambda x: x[-1])
    plt.plot(best_simulation, label="Best Simulation", color='green')

    # Plot the worst simulation
    worst_simulation = min(all_profits, key=lambda x: x[-1])
    plt.plot(worst_simulation, label="Worst Simulation", color='red')

    plt.plot(avg_profits, label='Average', linestyle='--', color='black')
    
    plt.title('Monte Carlo Simulations of Profit')
    plt.xlabel('Number of Trades')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.grid(True)

    # Setting custom formatter for y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    
    plt.show()

    # New figure for drawdown and profit distributions
    plt.figure(figsize=(14, 6))
    
    # Plotting Drawdowns
    plt.subplot(1, 2, 1)
    for drawdowns in all_drawdowns:
        n, bins, _ = plt.hist(drawdowns, bins=50, alpha=0.8, color='orange', density=True)
    plt.title('Empirical Distribution of Drawdowns')
    plt.xlabel('Drawdown')
    plt.ylabel('Density')
    plt.grid(True)

    # Plotting Profits
    plt.subplot(1, 2, 2)
    for profits in all_profits:
        n, bins, _ = plt.hist(profits, bins=50, alpha=0.8, color='orange', density=True)
    plt.title('Empirical Distribution of Profits')
    plt.xlabel('Cumulative Profit')
    plt.ylabel('Density')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

main()



#https://python.plainenglish.io/trading-systems-drawdown-analysis-using-monte-carlo-672c9806ed0e
#https://www.forexsignals.com/monte-carlo-simulation#use-tool
