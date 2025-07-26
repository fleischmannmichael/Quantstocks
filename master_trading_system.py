"""
Master Trading System Integration - FIXED VERSION
Combines fundamentals, trends, and valuation analysis for maximum profitability
"""

import pandas as pd
import numpy as np
import warnings
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import our analysis systems
from scripts.fundamental import ProfessionalFundamentalsSystem
from scripts.trends import ProfessionalTrendsSystem
from scripts.valuation import ProfessionalValuationSystem
from scripts.nasdaq_tickers import get_nasdaq_tickers, get_high_volume_nasdaq_tickers

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterTradingSystem:
    """
    Master trading system that integrates all analysis components.
    Designed for maximum profitability through systematic analysis.
    """
    
    def __init__(self):
        """Initialize the master trading system"""
        
        # Create necessary directories
        self.setup_directories()
        
        # System configuration
        self.config = {
            'weights': {
                'fundamentals_weight': 0.35,  # 35% weight to fundamentals
                'trends_weight': 0.35,        # 35% weight to technical trends  
                'valuation_weight': 0.30      # 30% weight to valuation
            },
            'scoring_thresholds': {
                'strong_buy': 85,
                'buy': 70,
                'hold': 45,
                'sell': 30,
                'strong_sell': 0
            },
            'position_sizing': {
                'max_position': 0.08,         # Max 8% per position
                'high_conviction': 0.06,      # 6% for strong signals
                'medium_conviction': 0.04,    # 4% for good signals
                'low_conviction': 0.02        # 2% for weak signals
            },
            'risk_management': {
                'max_portfolio_risk': 0.75,  # Max 75% invested
                'cash_reserve': 0.25,        # Min 25% cash
                'stop_loss': 0.15,           # 15% stop loss
                'take_profit': 0.30,         # 30% take profit
                'max_positions': 20          # Max 20 positions
            }
        }
        
        # Initialize analysis systems
        self.fundamentals_system = ProfessionalFundamentalsSystem()
        self.trends_system = ProfessionalTrendsSystem()
        self.valuation_system = ProfessionalValuationSystem()
        
        # Results storage
        self.fundamentals_results = None
        self.trends_results = None
        self.valuation_results = None
        self.master_results = None
        
        logger.info("Master Trading System initialized with all 3 analysis systems")
    
    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            'scripts',
            'scripts/other_outputs',
            'outputs',
            'outputs/masters'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("Directory structure created successfully")
        
    def get_target_tickers(self, target_count: int = 500, use_high_volume: bool = True) -> List[str]:
        """Get target tickers for analysis"""
        logger.info(f"Retrieving target tickers (target: {target_count})")
        
        if use_high_volume:
            # Get high-volume tickers for better liquidity
            tickers = get_high_volume_nasdaq_tickers(min_volume=500_000)
            if len(tickers) < target_count:
                # Fallback to all NASDAQ tickers
                tickers = get_nasdaq_tickers()
        else:
            tickers = get_nasdaq_tickers()
        
        # Limit to target count
        if len(tickers) > target_count:
            tickers = tickers[:target_count]
        
        logger.info(f"Selected {len(tickers)} tickers for analysis")
        return tickers
    
    def run_fundamentals_analysis(self, tickers: List[str]) -> pd.DataFrame:
        """Run fundamentals analysis on all tickers"""
        logger.info("Starting fundamentals analysis")
        start_time = time.time()
        
        self.fundamentals_results = self.fundamentals_system.analyze_multiple_stocks(tickers)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Fundamentals analysis completed in {elapsed_time:.1f} seconds")
        
        return self.fundamentals_results
    
    def run_trends_analysis(self, tickers: List[str]) -> pd.DataFrame:
        """Run trends/technical analysis on all tickers"""
        logger.info("Starting trends analysis")
        start_time = time.time()
        
        self.trends_results = self.trends_system.analyze_multiple_stocks(tickers)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Trends analysis completed in {elapsed_time:.1f} seconds")
        
        return self.trends_results
    
    def run_valuation_analysis(self, tickers: List[str]) -> pd.DataFrame:
        """Run valuation analysis on all tickers"""
        logger.info("Starting valuation analysis")
        start_time = time.time()
        
        self.valuation_results = self.valuation_system.analyze_multiple_stocks(tickers)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Valuation analysis completed in {elapsed_time:.1f} seconds")
        
        return self.valuation_results
    
    def merge_all_results(self) -> pd.DataFrame:
        """Merge fundamentals, trends, and valuation results"""
        logger.info("Merging all analysis results")
        
        if (self.fundamentals_results is None or 
            self.trends_results is None or 
            self.valuation_results is None):
            raise ValueError("All three analysis systems must be completed first")
        
        # Start with fundamentals
        merged_df = self.fundamentals_results.copy()
        
        # Merge trends
        merged_df = pd.merge(
            merged_df, 
            self.trends_results, 
            on='ticker', 
            how='inner',
            suffixes=('_fund', '_trend')
        )
        
        # Merge valuation
        merged_df = pd.merge(
            merged_df,
            self.valuation_results,
            on='ticker',
            how='inner'
        )
        
        logger.info(f"Merged {len(merged_df)} stocks with complete fundamental, technical, and valuation data")
        
        return merged_df
    
    def calculate_composite_scores(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite scores from all three analysis systems"""
        logger.info("Calculating composite scores from all three systems")
        
        df = merged_df.copy()
        
        # Extract scores with error handling
        def safe_score(row, column, default=50.0):
            try:
                value = row.get(column, default)
                if pd.isna(value) or value is None:
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default
        
        # Get individual scores
        df['fundamentals_score_clean'] = df.apply(lambda row: safe_score(row, 'fundamentals_score'), axis=1)
        df['trends_score_clean'] = df.apply(lambda row: safe_score(row, 'trends_score'), axis=1)
        df['valuation_score_clean'] = df.apply(lambda row: safe_score(row, 'valuation_score'), axis=1)
        
        # Calculate composite score using weights
        weights = self.config['weights']
        
        df['composite_score'] = (
            df['fundamentals_score_clean'] * weights['fundamentals_weight'] +
            df['trends_score_clean'] * weights['trends_weight'] +
            df['valuation_score_clean'] * weights['valuation_weight']
        )
        
        # Round to 1 decimal place
        df['composite_score'] = df['composite_score'].round(1)
        
        logger.info(f"Calculated composite scores (avg: {df['composite_score'].mean():.1f})")
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on composite scores"""
        logger.info("Generating trading signals")
        
        result_df = df.copy()
        thresholds = self.config['scoring_thresholds']
        
        # Generate action signals
        def get_action(score):
            if score >= thresholds['strong_buy']:
                return 'STRONG_BUY'
            elif score >= thresholds['buy']:
                return 'BUY'
            elif score >= thresholds['hold']:
                return 'HOLD'
            elif score >= thresholds['sell']:
                return 'SELL'
            else:
                return 'STRONG_SELL'
        
        # Generate confidence levels
        def get_confidence(fund_score, trend_score, val_score, composite_score):
            # Calculate agreement between systems
            scores = [fund_score, trend_score, val_score]
            score_std = np.std(scores)
            
            # High confidence if all systems agree and scores are extreme
            high_agreement = score_std <= 10
            extreme_score = composite_score >= 80 or composite_score <= 25
            
            if high_agreement and extreme_score:
                return 'HIGH'
            elif high_agreement or extreme_score:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        # Generate position sizing
        def get_position_size(score, confidence):
            position_sizes = self.config['position_sizing']
            
            if score >= 85 and confidence == 'HIGH':
                return position_sizes['high_conviction']
            elif score >= 75 or (score >= 70 and confidence == 'HIGH'):
                return position_sizes['medium_conviction']
            elif score >= 65:
                return position_sizes['low_conviction']
            else:
                return 0.0
        
        # Apply signal generation
        result_df['action'] = result_df['composite_score'].apply(get_action)
        result_df['confidence'] = result_df.apply(
            lambda row: get_confidence(
                row['fundamentals_score_clean'], 
                row['trends_score_clean'], 
                row['valuation_score_clean'],
                row['composite_score']
            ), axis=1
        )
        result_df['position_size'] = result_df.apply(
            lambda row: get_position_size(row['composite_score'], row['confidence']), axis=1
        )
        
        # Generate priority ranking
        result_df['priority_rank'] = result_df['composite_score'].rank(ascending=False, method='min')
        
        # Count signals
        signal_counts = result_df['action'].value_counts()
        logger.info(f"Trading signals generated: {dict(signal_counts)}")
        
        return result_df
    
    def calculate_portfolio_allocation(self, df: pd.DataFrame, total_capital: float = 100000) -> pd.DataFrame:
        """Calculate optimal portfolio allocation"""
        logger.info(f"Calculating portfolio allocation for ${total_capital:,.0f}")
        
        # Filter for buy signals only
        buy_signals = df[df['action'].isin(['STRONG_BUY', 'BUY'])].copy()
        buy_signals = buy_signals.sort_values('composite_score', ascending=False)
        
        if buy_signals.empty:
            logger.warning("No buy signals found for portfolio allocation")
            return pd.DataFrame()
        
        # Calculate allocations
        max_positions = self.config['risk_management']['max_positions']
        max_portfolio_risk = self.config['risk_management']['max_portfolio_risk']
        
        allocation_data = []
        total_allocated = 0.0
        positions_count = 0
        
        for _, stock in buy_signals.iterrows():
            if positions_count >= max_positions:
                break
            
            if total_allocated >= max_portfolio_risk:
                break
            
            position_size = stock['position_size']
            allocation_amount = total_capital * position_size
            
            if total_allocated + position_size <= max_portfolio_risk:
                allocation_data.append({
                    'ticker': stock['ticker'],
                    'composite_score': stock['composite_score'],
                    'action': stock['action'],
                    'confidence': stock['confidence'],
                    'position_size_pct': position_size * 100,
                    'allocation_amount': allocation_amount,
                    'fundamentals_score': stock['fundamentals_score_clean'],
                    'trends_score': stock['trends_score_clean'],
                    'valuation_score': stock['valuation_score_clean'],
                    'sector': stock.get('sector', 'Unknown'),
                    'priority_rank': stock['priority_rank']
                })
                
                total_allocated += position_size
                positions_count += 1
        
        allocation_df = pd.DataFrame(allocation_data)
        
        if not allocation_df.empty:
            cash_reserve = (1 - total_allocated) * 100
            logger.info(f"Portfolio allocation: {positions_count} positions, "
                       f"{total_allocated*100:.1f}% invested, {cash_reserve:.1f}% cash")
        
        return allocation_df
    
    def save_results(self, results_df: pd.DataFrame, allocation_df: pd.DataFrame) -> str:
        """Save all results to files"""
        logger.info("Saving results to files")
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save master results in outputs/masters
        master_filename = f'outputs/masters/master_trading_analysis_{timestamp}.csv'
        results_df.to_csv(master_filename, index=False)
        
        # Save portfolio allocation in outputs/masters
        allocation_filename = f'outputs/masters/portfolio_allocation_{timestamp}.csv'
        allocation_df.to_csv(allocation_filename, index=False)
        
        # Save individual system results in scripts/other_outputs
        if self.fundamentals_results is not None:
            fund_filename = f'scripts/other_outputs/fundamentals_results_{timestamp}.csv'
            self.fundamentals_results.to_csv(fund_filename, index=False)
        
        if self.trends_results is not None:
            trends_filename = f'scripts/other_outputs/trends_results_{timestamp}.csv'
            self.trends_results.to_csv(trends_filename, index=False)
        
        if self.valuation_results is not None:
            val_filename = f'scripts/other_outputs/valuation_results_{timestamp}.csv'
            self.valuation_results.to_csv(val_filename, index=False)
        
        # Save summary report in outputs/masters
        summary_filename = f'outputs/masters/trading_summary_{timestamp}.txt'
        self.generate_summary_report(results_df, allocation_df, summary_filename)
        
        logger.info(f"Results saved with timestamp {timestamp}")
        logger.info(f"Master results: {master_filename}")
        logger.info(f"Other outputs: scripts/other_outputs/")
        
        return master_filename
    
    def generate_summary_report(self, results_df: pd.DataFrame, allocation_df: pd.DataFrame, filename: str):
        """Generate a text summary report"""
        
        with open(filename, 'w') as f:
            f.write("MASTER TRADING SYSTEM ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Stocks Analyzed: {len(results_df)}\n\n")
            
            # System weights used
            f.write("SYSTEM WEIGHTS:\n")
            f.write("-" * 15 + "\n")
            weights = self.config['weights']
            f.write(f"Fundamentals: {weights['fundamentals_weight']:.0%}\n")
            f.write(f"Trends/Technical: {weights['trends_weight']:.0%}\n")
            f.write(f"Valuation: {weights['valuation_weight']:.0%}\n\n")
            
            # Signal distribution
            signal_counts = results_df['action'].value_counts()
            f.write("SIGNAL DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            for signal, count in signal_counts.items():
                f.write(f"{signal}: {count}\n")
            f.write(f"\n")
            
            # Average scores by system
            f.write("AVERAGE SCORES BY SYSTEM:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Fundamentals: {results_df['fundamentals_score_clean'].mean():.1f}\n")
            f.write(f"Trends: {results_df['trends_score_clean'].mean():.1f}\n")
            f.write(f"Valuation: {results_df['valuation_score_clean'].mean():.1f}\n")
            f.write(f"Composite: {results_df['composite_score'].mean():.1f}\n\n")
            
            # Top buy recommendations
            buy_signals = results_df[results_df['action'].isin(['STRONG_BUY', 'BUY'])]
            if not buy_signals.empty:
                f.write("TOP 10 BUY RECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                for i, (_, stock) in enumerate(buy_signals.head(10).iterrows(), 1):
                    f.write(f"{i:2d}. {stock['ticker']:>6} | Composite: {stock['composite_score']:>5.1f} | "
                           f"Fund: {stock['fundamentals_score_clean']:>4.0f} | "
                           f"Tech: {stock['trends_score_clean']:>4.0f} | "
                           f"Val: {stock['valuation_score_clean']:>4.0f} | "
                           f"{stock['action']:>10} | {stock['confidence']:>6}\n")
                f.write(f"\n")
            
            # Portfolio allocation summary
            if not allocation_df.empty:
                f.write("PORTFOLIO ALLOCATION SUMMARY:\n")
                f.write("-" * 30 + "\n")
                total_allocation = allocation_df['position_size_pct'].sum()
                f.write(f"Total Positions: {len(allocation_df)}\n")
                f.write(f"Total Allocation: {total_allocation:.1f}%\n")
                f.write(f"Cash Reserve: {100 - total_allocation:.1f}%\n")
                f.write(f"Average Composite Score: {allocation_df['composite_score'].mean():.1f}\n\n")
                
                f.write("RECOMMENDED POSITIONS:\n")
                f.write("-" * 25 + "\n")
                for _, pos in allocation_df.iterrows():
                    f.write(f"{pos['ticker']:>6} | {pos['position_size_pct']:>5.1f}% | "
                           f"${pos['allocation_amount']:>8,.0f} | Score: {pos['composite_score']:>5.1f} | "
                           f"{pos['action']:>10}\n")
            
            # Configuration used
            f.write(f"\nRISK MANAGEMENT CONFIGURATION:\n")
            f.write("-" * 35 + "\n")
            risk_config = self.config['risk_management']
            f.write(f"Max Portfolio Risk: {risk_config['max_portfolio_risk']:.0%}\n")
            f.write(f"Cash Reserve: {risk_config['cash_reserve']:.0%}\n")
            f.write(f"Stop Loss: {risk_config['stop_loss']:.0%}\n")
            f.write(f"Take Profit: {risk_config['take_profit']:.0%}\n")
            f.write(f"Max Positions: {risk_config['max_positions']}\n")
    
    def run_complete_analysis(self, target_tickers: int = 500, total_capital: float = 100000) -> Dict[str, Any]:
        """Run the complete master trading system analysis"""
        
        logger.info("Starting Master Trading System Analysis")
        logger.info("=" * 60)
        
        overall_start_time = time.time()
        
        try:
            # Step 1: Get target tickers
            logger.info("STEP 1: Retrieving target tickers")
            tickers = self.get_target_tickers(target_count=target_tickers)
            
            if not tickers:
                raise ValueError("No tickers retrieved for analysis")
            
            # Step 2: Run fundamentals analysis
            logger.info("STEP 2: Running fundamentals analysis")
            fundamentals_df = self.run_fundamentals_analysis(tickers)
            
            # Step 3: Run trends analysis  
            logger.info("STEP 3: Running trends analysis")
            trends_df = self.run_trends_analysis(tickers)
            
            # Step 4: Run valuation analysis
            logger.info("STEP 4: Running valuation analysis")
            valuation_df = self.run_valuation_analysis(tickers)
            
            # Step 5: Merge all results
            logger.info("STEP 5: Merging all analysis results")
            merged_df = self.merge_all_results()
            
            if merged_df.empty:
                raise ValueError("No merged results - check individual analysis outputs")
            
            # Step 6: Calculate composite scores
            logger.info("STEP 6: Calculating composite scores")
            scored_df = self.calculate_composite_scores(merged_df)
            
            # Step 7: Generate trading signals
            logger.info("STEP 7: Generating trading signals")
            signals_df = self.generate_trading_signals(scored_df)
            
            # Step 8: Calculate portfolio allocation
            logger.info("STEP 8: Calculating portfolio allocation")
            allocation_df = self.calculate_portfolio_allocation(signals_df, total_capital)
            
            # Step 9: Save results
            logger.info("STEP 9: Saving results")
            master_filename = self.save_results(signals_df, allocation_df)
            
            # Calculate final statistics
            total_time = time.time() - overall_start_time
            
            # Analysis summary
            buy_count = len(signals_df[signals_df['action'].isin(['STRONG_BUY', 'BUY'])])
            strong_buy_count = len(signals_df[signals_df['action'] == 'STRONG_BUY'])
            avg_composite_score = signals_df['composite_score'].mean()
            avg_fund_score = signals_df['fundamentals_score_clean'].mean()
            avg_trends_score = signals_df['trends_score_clean'].mean()
            avg_val_score = signals_df['valuation_score_clean'].mean()
            
            # Store master results
            self.master_results = signals_df
            
            results_summary = {
                'total_stocks_analyzed': len(signals_df),
                'total_tickers_requested': target_tickers,
                'buy_signals': buy_count,
                'strong_buy_signals': strong_buy_count,
                'average_composite_score': round(avg_composite_score, 1),
                'average_fundamentals_score': round(avg_fund_score, 1),
                'average_trends_score': round(avg_trends_score, 1),
                'average_valuation_score': round(avg_val_score, 1),
                'portfolio_positions': len(allocation_df),
                'total_capital': total_capital,
                'analysis_time_seconds': round(total_time, 1),
                'master_filename': master_filename,
                'success': True
            }
            
            logger.info("MASTER TRADING SYSTEM ANALYSIS COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Total time: {total_time:.1f} seconds")
            logger.info(f"Stocks analyzed: {len(signals_df)}")
            logger.info(f"Buy signals: {buy_count}")
            logger.info(f"Strong buy signals: {strong_buy_count}")
            logger.info(f"Average composite score: {avg_composite_score:.1f}")
            logger.info(f"Portfolio positions: {len(allocation_df)}")
            logger.info(f"Results saved: {master_filename}")
            
            return results_summary
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - overall_start_time
            }
    
    def get_top_opportunities(self, n: int = 10) -> pd.DataFrame:
        """Get top N trading opportunities from latest analysis"""
        if self.master_results is None:
            logger.warning("No analysis results available. Run complete analysis first.")
            return pd.DataFrame()
        
        buy_signals = self.master_results[
            self.master_results['action'].isin(['STRONG_BUY', 'BUY'])
        ].copy()
        
        return buy_signals.head(n)[
            ['ticker', 'composite_score', 'action', 'confidence', 
             'fundamentals_score_clean', 'trends_score_clean', 'valuation_score_clean', 'sector']
        ]
    
    def get_risk_warnings(self) -> List[str]:
        """Get risk warnings based on current analysis"""
        warnings = []
        
        if self.master_results is None:
            warnings.append("No analysis results available")
            return warnings
        
        # Check for concentration risk
        buy_signals = self.master_results[
            self.master_results['action'].isin(['STRONG_BUY', 'BUY'])
        ]
        
        if not buy_signals.empty:
            # Sector concentration
            sector_counts = buy_signals['sector'].value_counts()
            max_sector_concentration = sector_counts.iloc[0] / len(buy_signals)
            
            if max_sector_concentration > 0.4:
                warnings.append(f"High sector concentration: {max_sector_concentration:.0%} in {sector_counts.index[0]}")
            
            # Score distribution
            high_scores = len(buy_signals[buy_signals['composite_score'] >= 85])
            if high_scores < 3:
                warnings.append("Limited high-conviction opportunities available")
            
            # Agreement between systems
            disagreement_count = 0
            for _, row in buy_signals.iterrows():
                scores = [row['fundamentals_score_clean'], row['trends_score_clean'], row['valuation_score_clean']]
                if np.std(scores) > 20:
                    disagreement_count += 1
            
            if disagreement_count > len(buy_signals) * 0.3:
                warnings.append("High disagreement between analysis systems")
        
        return warnings


def run_quick_analysis(target_stocks: int = 200, capital: float = 100000) -> Dict[str, Any]:
    """Quick analysis function for testing"""
    
    print("Quick Master Trading System Analysis")
    print("=" * 50)
    
    system = MasterTradingSystem()
    results = system.run_complete_analysis(target_tickers=target_stocks, total_capital=capital)
    
    if results.get('success'):
        print(f"\nAnalysis completed successfully!")
        print(f"Time: {results['analysis_time_seconds']} seconds")
        print(f"Stocks: {results['total_stocks_analyzed']}")
        print(f"Buy signals: {results['buy_signals']}")
        print(f"Strong buys: {results['strong_buy_signals']}")
        print(f"Average composite score: {results['average_composite_score']}")
        print(f"Average fundamentals: {results['average_fundamentals_score']}")
        print(f"Average trends: {results['average_trends_score']}")
        print(f"Average valuation: {results['average_valuation_score']}")
        
        # Show top opportunities
        top_ops = system.get_top_opportunities(5)
        if not top_ops.empty:
            print(f"\nTop 5 Opportunities:")
            for _, stock in top_ops.iterrows():
                print(f"  {stock['ticker']}: {stock['composite_score']:.1f} "
                      f"(F:{stock['fundamentals_score_clean']:.0f} "
                      f"T:{stock['trends_score_clean']:.0f} "
                      f"V:{stock['valuation_score_clean']:.0f}) - {stock['action']}")
        
        # Show risk warnings
        warnings = system.get_risk_warnings()
        if warnings:
            print(f"\nRisk Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    
    else:
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")
    
    return results


def run_full_analysis(target_stocks: int = 500, capital: float = 100000) -> Dict[str, Any]:
    """Full analysis with maximum stock coverage"""
    
    print("Full Master Trading System Analysis")
    print("=" * 50)
    print(f"Target stocks: {target_stocks}")
    print(f"Capital: ${capital:,.0f}")
    print("This may take 20-40 minutes depending on your system...")
    
    system = MasterTradingSystem()
    results = system.run_complete_analysis(target_tickers=target_stocks, total_capital=capital)
    
    return results


def main():
    """Main execution function"""
    
    print("Master Trading System with Full Analysis Integration")
    print("Choose analysis type:")
    print("1. Quick Analysis (200 stocks, ~8-15 minutes)")
    print("2. Standard Analysis (350 stocks, ~15-25 minutes)")
    print("3. Full Analysis (500 stocks, ~20-40 minutes)")
    print("4. Custom Analysis")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == '1':
        results = run_quick_analysis()
        
    elif choice == '2':
        results = run_full_analysis(350)
        
    elif choice == '3':
        results = run_full_analysis()
        
    elif choice == '4':
        try:
            target_stocks = int(input("Number of stocks to analyze (50-800): ") or "300")
            capital = float(input("Portfolio capital ($): ") or "100000")
            
            system = MasterTradingSystem()
            results = system.run_complete_analysis(target_tickers=target_stocks, total_capital=capital)
            
        except ValueError:
            print("Invalid input. Using defaults.")
            results = run_quick_analysis()
    
    else:
        print("Invalid choice. Running quick analysis.")
        results = run_quick_analysis()
    
    print(f"\nMaster Trading System analysis complete!")
    if results.get('success'):
        print(f"Check the outputs/masters folder for detailed results.")
    
    return results


if __name__ == "__main__":
    main()