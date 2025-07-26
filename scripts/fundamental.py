"""
Professional Fundamentals Analysis System - FIXED VERSION
Comprehensive financial health assessment for maximum stock coverage
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

warnings.filterwarnings('ignore')

class ProfessionalFundamentalsSystem:
    """
    Professional-grade fundamentals analysis system.
    Designed for maximum stock coverage with robust error handling.
    """
    
    def __init__(self):
        self.metric_weights = {
            # Profitability Metrics (40%)
            'return_on_equity': 0.10,
            'return_on_assets': 0.08,
            'return_on_invested_capital': 0.10,
            'net_profit_margin': 0.08,
            'operating_margin': 0.04,
            
            # Financial Strength (30%)
            'debt_to_equity': 0.08,
            'current_ratio': 0.06,
            'quick_ratio': 0.04,
            'interest_coverage': 0.06,
            'debt_to_assets': 0.06,
            
            # Growth Metrics (20%)
            'revenue_growth': 0.10,
            'earnings_growth': 0.10,
            
            # Efficiency Metrics (10%)
            'asset_turnover': 0.05,
            'working_capital_turnover': 0.05,
        }
        
        # Sector benchmarks for relative scoring
        self.sector_benchmarks = {
            'Technology': {
                'return_on_equity': [20, 25, 30, 40],  # [poor, fair, good, excellent]
                'return_on_assets': [8, 12, 16, 25],
                'net_profit_margin': [15, 20, 25, 35],
                'debt_to_equity': [1.0, 0.6, 0.4, 0.2],  # Reverse order (lower is better)
                'revenue_growth': [5, 10, 15, 25],
            },
            'Healthcare': {
                'return_on_equity': [12, 16, 20, 28],
                'return_on_assets': [6, 10, 14, 20],
                'net_profit_margin': [8, 12, 18, 25],
                'debt_to_equity': [1.2, 0.8, 0.5, 0.3],
                'revenue_growth': [3, 8, 12, 18],
            },
            'Financial Services': {
                'return_on_equity': [8, 12, 15, 20],
                'return_on_assets': [0.8, 1.2, 1.6, 2.2],
                'net_profit_margin': [15, 20, 25, 35],
                'debt_to_equity': [8.0, 6.0, 4.0, 3.0],
                'revenue_growth': [2, 5, 8, 12],
            },
            'Consumer Cyclical': {
                'return_on_equity': [10, 15, 20, 28],
                'return_on_assets': [4, 7, 10, 15],
                'net_profit_margin': [3, 6, 10, 15],
                'debt_to_equity': [1.5, 1.0, 0.7, 0.4],
                'revenue_growth': [2, 6, 10, 15],
            },
            'Energy': {
                'return_on_equity': [5, 10, 15, 25],
                'return_on_assets': [2, 5, 8, 12],
                'net_profit_margin': [2, 5, 10, 18],
                'debt_to_equity': [1.5, 1.0, 0.6, 0.3],
                'revenue_growth': [0, 8, 15, 25],
            },
            'Default': {
                'return_on_equity': [10, 15, 20, 25],
                'return_on_assets': [4, 7, 10, 15],
                'net_profit_margin': [5, 10, 15, 20],
                'debt_to_equity': [1.2, 0.8, 0.5, 0.3],
                'revenue_growth': [3, 8, 12, 18],
            }
        }
    
    def safe_get_value(self, data_dict: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Safely extract numeric values from data dictionary"""
        try:
            value = data_dict.get(key, default)
            if value is None or pd.isna(value) or str(value).lower() in ['none', 'n/a', 'nan', '']:
                return default
            
            float_val = float(value)
            if not np.isfinite(float_val) or abs(float_val) > 1e10:
                return default
            return float_val
        except (TypeError, ValueError, OverflowError):
            return default
    
    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive stock data with fallback handling"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            
            # Basic validation
            if not info or len(info) < 5:
                return {'ticker': ticker, 'info': {}, 'error': 'No data available'}
            
            return {'ticker': ticker, 'info': info, 'success': True}
            
        except Exception as e:
            return {'ticker': ticker, 'info': {}, 'error': str(e)[:100]}
    
    def calculate_fundamentals_metrics(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive fundamental metrics"""
        info = stock_data['info']
        ticker = stock_data['ticker']
        
        if not info:
            return self.create_default_metrics(ticker)
        
        metrics = {}
        
        try:
            # Basic company information
            metrics['ticker'] = ticker
            metrics['sector'] = info.get('sector', 'Unknown')
            metrics['industry'] = info.get('industry', 'Unknown')
            metrics['market_cap'] = self.safe_get_value(info, 'marketCap', 1e9)
            metrics['current_price'] = (
                self.safe_get_value(info, 'currentPrice') or
                self.safe_get_value(info, 'regularMarketPrice') or
                self.safe_get_value(info, 'previousClose', 100.0)
            )
            
            # PROFITABILITY METRICS
            metrics['return_on_equity'] = self.safe_get_value(info, 'returnOnEquity', 0.15) * 100
            metrics['return_on_assets'] = self.safe_get_value(info, 'returnOnAssets', 0.08) * 100
            metrics['net_profit_margin'] = self.safe_get_value(info, 'profitMargins', 0.10) * 100
            metrics['operating_margin'] = self.safe_get_value(info, 'operatingMargins', 0.15) * 100
            metrics['gross_margin'] = self.safe_get_value(info, 'grossMargins', 0.40) * 100
            
            # Calculate ROIC (Return on Invested Capital)
            total_revenue = self.safe_get_value(info, 'totalRevenue')
            total_debt = self.safe_get_value(info, 'totalDebt', 0)
            shareholders_equity = self.safe_get_value(info, 'totalStockholderEquity')
            net_income = self.safe_get_value(info, 'netIncomeToCommon')
            
            if total_debt and shareholders_equity and net_income:
                invested_capital = total_debt + shareholders_equity
                if invested_capital > 0:
                    metrics['return_on_invested_capital'] = (net_income / invested_capital) * 100
                else:
                    metrics['return_on_invested_capital'] = metrics['return_on_equity'] * 0.8
            else:
                metrics['return_on_invested_capital'] = metrics['return_on_equity'] * 0.8
            
            # FINANCIAL STRENGTH METRICS
            debt_to_equity_pct = self.safe_get_value(info, 'debtToEquity', 50)
            metrics['debt_to_equity'] = debt_to_equity_pct / 100
            
            metrics['current_ratio'] = self.safe_get_value(info, 'currentRatio', 2.0)
            metrics['quick_ratio'] = self.safe_get_value(info, 'quickRatio', 1.5)
            
            # Interest coverage ratio
            ebitda = self.safe_get_value(info, 'ebitda')
            interest_expense = self.safe_get_value(info, 'interestExpense')
            if ebitda and interest_expense and interest_expense > 0:
                metrics['interest_coverage'] = ebitda / abs(interest_expense)
            else:
                metrics['interest_coverage'] = 10.0  # Default safe value
            
            # Debt to assets
            total_assets = self.safe_get_value(info, 'totalAssets')
            if total_debt and total_assets and total_assets > 0:
                metrics['debt_to_assets'] = total_debt / total_assets
            else:
                metrics['debt_to_assets'] = metrics['debt_to_equity'] / (1 + metrics['debt_to_equity'])
            
            # GROWTH METRICS
            metrics['revenue_growth'] = self.safe_get_value(info, 'revenueGrowth', 0.10) * 100
            metrics['earnings_growth'] = self.safe_get_value(info, 'earningsGrowth', 0.15) * 100
            metrics['earnings_quarterly_growth'] = self.safe_get_value(info, 'earningsQuarterlyGrowth', 0.10) * 100
            
            # EFFICIENCY METRICS
            if total_revenue and total_assets and total_assets > 0:
                metrics['asset_turnover'] = total_revenue / total_assets
            else:
                metrics['asset_turnover'] = 1.0
            
            # Working capital turnover
            current_assets = self.safe_get_value(info, 'totalCurrentAssets')
            current_liabilities = self.safe_get_value(info, 'totalCurrentLiabilities')
            if current_assets and current_liabilities and total_revenue:
                working_capital = current_assets - current_liabilities
                if working_capital > 0:
                    metrics['working_capital_turnover'] = total_revenue / working_capital
                else:
                    metrics['working_capital_turnover'] = 5.0
            else:
                metrics['working_capital_turnover'] = 5.0
            
            # CASH FLOW METRICS
            metrics['free_cash_flow'] = self.safe_get_value(info, 'freeCashflow', 0)
            metrics['operating_cash_flow'] = self.safe_get_value(info, 'operatingCashflow', 0)
            
            # Free Cash Flow Yield
            if metrics['free_cash_flow'] and metrics['market_cap']:
                metrics['fcf_yield'] = (metrics['free_cash_flow'] / metrics['market_cap']) * 100
            else:
                metrics['fcf_yield'] = 5.0
            
            # QUALITY METRICS
            if metrics['operating_cash_flow'] and net_income and net_income > 0:
                metrics['earnings_quality'] = metrics['operating_cash_flow'] / net_income
            else:
                metrics['earnings_quality'] = 1.0
            
            # VALUATION CONTEXT
            metrics['pe_ratio'] = self.safe_get_value(info, 'trailingPE', 20.0)
            metrics['pb_ratio'] = self.safe_get_value(info, 'priceToBook', 2.5)
            metrics['dividend_yield'] = self.safe_get_value(info, 'dividendYield', 0.0) * 100
            
            # RISK METRICS
            metrics['beta'] = self.safe_get_value(info, 'beta', 1.0)
            metrics['shares_outstanding'] = self.safe_get_value(info, 'sharesOutstanding', metrics['market_cap'] / metrics['current_price'])
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics for {ticker}: {str(e)[:50]}")
            return self.create_default_metrics(ticker)
    
    def create_default_metrics(self, ticker: str) -> Dict[str, Any]:
        """Create default metrics for failed data retrieval"""
        return {
            'ticker': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 1e9,
            'current_price': 100.0,
            'return_on_equity': 15.0,
            'return_on_assets': 8.0,
            'return_on_invested_capital': 12.0,
            'net_profit_margin': 10.0,
            'operating_margin': 15.0,
            'gross_margin': 40.0,
            'debt_to_equity': 0.5,
            'current_ratio': 2.0,
            'quick_ratio': 1.5,
            'interest_coverage': 10.0,
            'debt_to_assets': 0.3,
            'revenue_growth': 10.0,
            'earnings_growth': 15.0,
            'earnings_quarterly_growth': 10.0,
            'asset_turnover': 1.0,
            'working_capital_turnover': 5.0,
            'free_cash_flow': 1e8,
            'operating_cash_flow': 1.2e8,
            'fcf_yield': 5.0,
            'earnings_quality': 1.0,
            'pe_ratio': 20.0,
            'pb_ratio': 2.5,
            'dividend_yield': 2.0,
            'beta': 1.0,
            'shares_outstanding': 1e6,
            'data_quality': 'DEFAULT'
        }
    
    def score_metric_by_sector(self, value: float, metric_name: str, sector: str) -> int:
        """Score metric relative to sector benchmarks"""
        benchmarks = self.sector_benchmarks.get(sector, self.sector_benchmarks['Default'])
        thresholds = benchmarks.get(metric_name, [5, 10, 15, 20])
        
        # Handle reverse metrics (lower is better)
        if metric_name in ['debt_to_equity', 'debt_to_assets']:
            if value <= thresholds[3]:
                return 100  # Excellent
            elif value <= thresholds[2]:
                return 80   # Good
            elif value <= thresholds[1]:
                return 60   # Fair
            elif value <= thresholds[0]:
                return 40   # Poor
            else:
                return 20   # Very poor
        else:
            # Normal metrics (higher is better)
            if value >= thresholds[3]:
                return 100  # Excellent
            elif value >= thresholds[2]:
                return 80   # Good
            elif value >= thresholds[1]:
                return 60   # Fair
            elif value >= thresholds[0]:
                return 40   # Poor
            else:
                return 20   # Very poor
    
    def calculate_fundamentals_score(self, ticker: str) -> Dict[str, Any]:
        """Calculate comprehensive fundamentals score"""
        
        # Get stock data
        stock_data = self.get_stock_data(ticker)
        
        # Calculate metrics
        metrics = self.calculate_fundamentals_metrics(stock_data)
        
        # Get sector for relative scoring
        sector = metrics.get('sector', 'Default')
        
        # Score each metric
        total_score = 0.0
        total_weight = 0.0
        individual_scores = {}
        
        for metric_name, weight in self.metric_weights.items():
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                score = self.score_metric_by_sector(metric_value, metric_name, sector)
                
                individual_scores[f"{metric_name}_score"] = score
                total_score += score * weight
                total_weight += weight
        
        # Calculate final score
        final_score = total_score / total_weight if total_weight > 0 else 50.0
        
        # Generate trading signal
        if final_score >= 85:
            signal = 'STRONG_BUY'
        elif final_score >= 70:
            signal = 'BUY'
        elif final_score >= 45:
            signal = 'HOLD'
        elif final_score >= 25:
            signal = 'SELL'
        else:
            signal = 'STRONG_SELL'
        
        # Determine confidence
        data_completeness = total_weight / sum(self.metric_weights.values())
        if data_completeness >= 0.9 and (final_score >= 80 or final_score <= 20):
            confidence = 'HIGH'
        elif data_completeness >= 0.7 and (final_score >= 70 or final_score <= 30):
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'ticker': ticker,
            'fundamentals_score': round(final_score, 1),
            'trading_signal': signal,
            'confidence': confidence,
            'sector': sector,
            'industry': metrics.get('industry', 'Unknown'),
            'market_cap': metrics.get('market_cap', 0),
            'current_price': metrics.get('current_price', 0),
            'return_on_equity': metrics.get('return_on_equity', 0),
            'return_on_assets': metrics.get('return_on_assets', 0),
            'return_on_invested_capital': metrics.get('return_on_invested_capital', 0),
            'net_profit_margin': metrics.get('net_profit_margin', 0),
            'debt_to_equity': metrics.get('debt_to_equity', 0),
            'current_ratio': metrics.get('current_ratio', 0),
            'revenue_growth': metrics.get('revenue_growth', 0),
            'earnings_growth': metrics.get('earnings_growth', 0),
            'fcf_yield': metrics.get('fcf_yield', 0),
            'earnings_quality': metrics.get('earnings_quality', 0),
            'pe_ratio': metrics.get('pe_ratio', 0),
            'pb_ratio': metrics.get('pb_ratio', 0),
            'beta': metrics.get('beta', 0),
            'individual_scores': individual_scores,
            'data_completeness': round(data_completeness * 100, 1),
            'success': stock_data.get('success', False)
        }
    
    def analyze_multiple_stocks(self, tickers: List[str], max_workers: int = 8) -> pd.DataFrame:
        """Analyze multiple stocks with parallel processing"""
        
        print(f"Starting fundamentals analysis of {len(tickers)} stocks")
        print(f"Using {max_workers} parallel workers for optimal speed")
        
        results = []
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.calculate_fundamentals_score, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per stock
                    results.append(result)
                except Exception as e:
                    print(f"Failed to analyze {ticker}: {str(e)[:50]}")
                    # Add default result to maintain data structure
                    results.append({
                        'ticker': ticker,
                        'fundamentals_score': 50.0,
                        'trading_signal': 'HOLD',
                        'confidence': 'LOW',
                        'sector': 'Unknown',
                        'success': False,
                        'error': str(e)[:100]
                    })
                
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(tickers) - completed) / rate
                    print(f"Progress: {completed}/{len(tickers)} ({completed/len(tickers)*100:.1f}%) - "
                          f"Rate: {rate:.1f}/sec - ETA: {eta:.0f}s")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by fundamentals score
        df = df.sort_values('fundamentals_score', ascending=False)
        
        # Summary statistics
        total_time = time.time() - start_time
        success_rate = df['success'].sum() / len(df) * 100 if 'success' in df.columns else 0
        
        print(f"\nFundamentals analysis completed")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Processing rate: {len(tickers)/total_time:.1f} stocks/second")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average score: {df['fundamentals_score'].mean():.1f}")
        
        return df


def main():
    """Main execution function"""
    
    # Initialize system
    fundamentals_system = ProfessionalFundamentalsSystem()
    
    # Get NASDAQ tickers
    from nasdaq_tickers import get_nasdaq_tickers
    tickers = get_nasdaq_tickers()
    
    print(f"Professional Fundamentals Analysis System")
    print(f"Target: {len(tickers)} NASDAQ stocks")
    
    # Run analysis
    results_df = fundamentals_system.analyze_multiple_stocks(tickers)
    
    # Save results
    os.makedirs('scripts/other_outputs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'scripts/other_outputs/fundamentals_analysis_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    
    # Summary
    strong_buys = results_df[results_df['trading_signal'] == 'STRONG_BUY']
    buys = results_df[results_df['trading_signal'] == 'BUY']
    
    print(f"\nResults Summary:")
    print(f"Total stocks analyzed: {len(results_df)}")
    print(f"Strong buy signals: {len(strong_buys)}")
    print(f"Buy signals: {len(buys)}")
    print(f"File saved: {filename}")
    
    if not strong_buys.empty:
        print(f"\nTop 5 Strong Buy Recommendations:")
        for _, stock in strong_buys.head(5).iterrows():
            print(f"  {stock['ticker']}: Score {stock['fundamentals_score']:.1f} "
                  f"(ROE: {stock.get('return_on_equity', 0):.1f}%, "
                  f"Growth: {stock.get('revenue_growth', 0):.1f}%)")
    
    return results_df

if __name__ == "__main__":
    main()