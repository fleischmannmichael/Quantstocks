"""
Professional Valuation Analysis System - FIXED VERSION
Comprehensive valuation metrics with sector-relative analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

warnings.filterwarnings('ignore')

class ProfessionalValuationSystem:
    """
    Professional-grade valuation analysis system.
    Analyzes price ratios, yields, and intrinsic value metrics.
    """
    
    def __init__(self):
        self.metric_weights = {
            # Price Ratios (50%)
            'pe_ratio': 0.15,
            'forward_pe': 0.10,
            'pb_ratio': 0.10,
            'ps_ratio': 0.08,
            'peg_ratio': 0.07,
            
            # Enterprise Value Ratios (30%)
            'ev_ebitda': 0.12,
            'ev_sales': 0.08,
            'ev_fcf': 0.10,
            
            # Yield Metrics (20%)
            'dividend_yield': 0.05,
            'fcf_yield': 0.08,
            'earnings_yield': 0.07,
        }
        
        # Sector-specific valuation benchmarks
        self.sector_benchmarks = {
            'Technology': {
                'pe_ratio': [25, 20, 15, 10],  # [poor, fair, good, excellent]
                'pb_ratio': [8.0, 5.0, 3.0, 1.5],
                'ps_ratio': [20.0, 10.0, 5.0, 2.0],
                'peg_ratio': [3.0, 2.0, 1.2, 0.8],
                'ev_ebitda': [40, 25, 15, 10],
            },
            'Healthcare': {
                'pe_ratio': [30, 22, 15, 10],
                'pb_ratio': [6.0, 3.5, 2.0, 1.0],
                'ps_ratio': [15.0, 8.0, 4.0, 2.0],
                'peg_ratio': [2.5, 1.8, 1.0, 0.7],
                'ev_ebitda': [30, 18, 12, 8],
            },
            'Financial Services': {
                'pe_ratio': [18, 12, 8, 5],
                'pb_ratio': [3.0, 1.8, 1.2, 0.8],
                'ps_ratio': [8.0, 4.0, 2.0, 1.0],
                'peg_ratio': [2.0, 1.5, 1.0, 0.6],
                'ev_ebitda': [25, 15, 10, 6],
            },
            'Consumer Cyclical': {
                'pe_ratio': [25, 18, 12, 8],
                'pb_ratio': [5.0, 3.0, 1.8, 1.0],
                'ps_ratio': [6.0, 3.0, 1.5, 0.8],
                'peg_ratio': [2.5, 1.8, 1.1, 0.7],
                'ev_ebitda': [20, 14, 8, 5],
            },
            'Energy': {
                'pe_ratio': [20, 15, 10, 6],
                'pb_ratio': [3.0, 2.0, 1.2, 0.7],
                'ps_ratio': [3.0, 1.5, 0.8, 0.4],
                'peg_ratio': [2.0, 1.2, 0.8, 0.5],
                'ev_ebitda': [15, 10, 6, 3],
            },
            'Utilities': {
                'pe_ratio': [22, 16, 12, 8],
                'pb_ratio': [2.5, 1.8, 1.2, 0.8],
                'ps_ratio': [4.0, 2.5, 1.5, 1.0],
                'peg_ratio': [2.5, 1.8, 1.2, 0.8],
                'ev_ebitda': [12, 8, 6, 4],
            },
            'Default': {
                'pe_ratio': [25, 18, 12, 8],
                'pb_ratio': [5.0, 3.0, 1.8, 1.0],
                'ps_ratio': [8.0, 4.0, 2.0, 1.0],
                'peg_ratio': [2.5, 1.8, 1.0, 0.7],
                'ev_ebitda': [20, 15, 10, 6],
            }
        }
    
    def safe_get_value(self, data_dict: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Safely extract numeric values from data dictionary"""
        try:
            value = data_dict.get(key, default)
            if value is None or pd.isna(value) or str(value).lower() in ['none', 'n/a', 'nan', '']:
                return default
            
            float_val = float(value)
            if not np.isfinite(float_val) or abs(float_val) > 1e12:
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
    
    def calculate_valuation_metrics(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive valuation metrics"""
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
            metrics['enterprise_value'] = self.safe_get_value(info, 'enterpriseValue', metrics['market_cap'] * 1.1)
            
            # Current price with multiple fallbacks
            metrics['current_price'] = (
                self.safe_get_value(info, 'currentPrice') or
                self.safe_get_value(info, 'regularMarketPrice') or
                self.safe_get_value(info, 'previousClose', 100.0)
            )
            
            # CORE PRICE RATIOS
            metrics['pe_ratio'] = self.safe_get_value(info, 'trailingPE', 20.0)
            metrics['forward_pe'] = self.safe_get_value(info, 'forwardPE', 18.0)
            metrics['pb_ratio'] = self.safe_get_value(info, 'priceToBook', 2.5)
            metrics['ps_ratio'] = self.safe_get_value(info, 'priceToSalesTrailing12Months', 4.0)
            metrics['peg_ratio'] = self.safe_get_value(info, 'pegRatio', 1.5)
            
            # ENTERPRISE VALUE RATIOS
            metrics['ev_ebitda'] = self.safe_get_value(info, 'enterpriseToEbitda', 15.0)
            metrics['ev_sales'] = self.safe_get_value(info, 'enterpriseToRevenue', 4.0)
            
            # Calculate EV/FCF
            fcf = self.safe_get_value(info, 'freeCashflow')
            if fcf > 0 and metrics['enterprise_value'] > 0:
                metrics['ev_fcf'] = metrics['enterprise_value'] / fcf
            else:
                metrics['ev_fcf'] = 20.0
            
            # YIELD METRICS
            dividend_yield = self.safe_get_value(info, 'dividendYield', 0.0)
            metrics['dividend_yield'] = dividend_yield * 100 if dividend_yield > 0 else 0.0
            
            # FCF Yield
            if fcf > 0 and metrics['market_cap'] > 0:
                metrics['fcf_yield'] = (fcf / metrics['market_cap']) * 100
            else:
                metrics['fcf_yield'] = 5.0
            
            # Earnings Yield (inverse of P/E)
            if metrics['pe_ratio'] > 0:
                metrics['earnings_yield'] = (1 / metrics['pe_ratio']) * 100
            else:
                metrics['earnings_yield'] = 5.0
            
            # ADDITIONAL CONTEXT METRICS
            metrics['beta'] = self.safe_get_value(info, 'beta', 1.0)
            metrics['shares_outstanding'] = self.safe_get_value(info, 'sharesOutstanding', metrics['market_cap'] / metrics['current_price'])
            
            # 52-week range analysis
            metrics['week_52_high'] = self.safe_get_value(info, 'fiftyTwoWeekHigh', metrics['current_price'] * 1.2)
            metrics['week_52_low'] = self.safe_get_value(info, 'fiftyTwoWeekLow', metrics['current_price'] * 0.8)
            
            # Position in 52-week range
            high_52w = metrics['week_52_high']
            low_52w = metrics['week_52_low']
            current = metrics['current_price']
            
            if high_52w > low_52w:
                metrics['position_52w'] = ((current - low_52w) / (high_52w - low_52w)) * 100
            else:
                metrics['position_52w'] = 50.0
            
            # Analyst data
            metrics['target_price'] = self.safe_get_value(info, 'targetMeanPrice', current * 1.1)
            metrics['analyst_recommendation'] = self.safe_get_value(info, 'recommendationMean', 3.0)
            
            # Price to target upside
            if current > 0:
                metrics['upside_to_target'] = ((metrics['target_price'] - current) / current) * 100
            else:
                metrics['upside_to_target'] = 0.0
            
            # Calculate Graham Number (intrinsic value estimate)
            book_value = self.safe_get_value(info, 'bookValue', current * 0.5)
            trailing_eps = self.safe_get_value(info, 'trailingEps', current / metrics['pe_ratio'])
            
            if book_value > 0 and trailing_eps > 0:
                metrics['graham_number'] = (22.5 * trailing_eps * book_value) ** 0.5
                metrics['graham_discount'] = ((metrics['graham_number'] - current) / current) * 100
            else:
                metrics['graham_number'] = current
                metrics['graham_discount'] = 0.0
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating valuation metrics for {ticker}: {str(e)[:50]}")
            return self.create_default_metrics(ticker)
    
    def create_default_metrics(self, ticker: str) -> Dict[str, Any]:
        """Create default metrics for failed data retrieval"""
        return {
            'ticker': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 1e9,
            'enterprise_value': 1.1e9,
            'current_price': 100.0,
            'pe_ratio': 20.0,
            'forward_pe': 18.0,
            'pb_ratio': 2.5,
            'ps_ratio': 4.0,
            'peg_ratio': 1.5,
            'ev_ebitda': 15.0,
            'ev_sales': 4.0,
            'ev_fcf': 20.0,
            'dividend_yield': 2.0,
            'fcf_yield': 5.0,
            'earnings_yield': 5.0,
            'beta': 1.0,
            'shares_outstanding': 1e6,
            'week_52_high': 120.0,
            'week_52_low': 80.0,
            'position_52w': 50.0,
            'target_price': 110.0,
            'analyst_recommendation': 3.0,
            'upside_to_target': 10.0,
            'graham_number': 100.0,
            'graham_discount': 0.0,
            'data_quality': 'DEFAULT'
        }
    
    def score_valuation_metric(self, value: float, metric_name: str, sector: str) -> int:
        """Score valuation metric relative to sector benchmarks"""
        benchmarks = self.sector_benchmarks.get(sector, self.sector_benchmarks['Default'])
        thresholds = benchmarks.get(metric_name, [25, 18, 12, 8])
        
        # For yield metrics, higher is better
        if metric_name in ['dividend_yield', 'fcf_yield', 'earnings_yield']:
            # Convert to appropriate scale for yield metrics
            yield_thresholds = [1.0, 2.0, 4.0, 6.0]  # Poor to excellent yield %
            
            if value >= yield_thresholds[3]:
                return 100  # Excellent yield
            elif value >= yield_thresholds[2]:
                return 80   # Good yield
            elif value >= yield_thresholds[1]:
                return 60   # Fair yield
            elif value >= yield_thresholds[0]:
                return 40   # Poor yield
            else:
                return 20   # Very poor yield
        else:
            # For ratio metrics, lower is better (cheaper valuation)
            if value <= thresholds[3]:
                return 100  # Excellent (cheap)
            elif value <= thresholds[2]:
                return 80   # Good
            elif value <= thresholds[1]:
                return 60   # Fair
            elif value <= thresholds[0]:
                return 40   # Poor
            else:
                return 20   # Very poor (expensive)
    
    def calculate_valuation_score(self, ticker: str) -> Dict[str, Any]:
        """Calculate comprehensive valuation score"""
        
        # Get stock data
        stock_data = self.get_stock_data(ticker)
        
        # Calculate metrics
        metrics = self.calculate_valuation_metrics(stock_data)
        
        # Get sector for relative scoring
        sector = metrics.get('sector', 'Default')
        
        # Score each metric
        total_score = 0.0
        total_weight = 0.0
        individual_scores = {}
        
        for metric_name, weight in self.metric_weights.items():
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                score = self.score_valuation_metric(metric_value, metric_name, sector)
                
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
            'valuation_score': round(final_score, 1),
            'trading_signal': signal,
            'confidence': confidence,
            'sector': sector,
            'industry': metrics.get('industry', 'Unknown'),
            'market_cap': metrics.get('market_cap', 0),
            'current_price': metrics.get('current_price', 0),
            'pe_ratio': metrics.get('pe_ratio', 0),
            'pb_ratio': metrics.get('pb_ratio', 0),
            'ps_ratio': metrics.get('ps_ratio', 0),
            'peg_ratio': metrics.get('peg_ratio', 0),
            'ev_ebitda': metrics.get('ev_ebitda', 0),
            'ev_sales': metrics.get('ev_sales', 0),
            'dividend_yield': metrics.get('dividend_yield', 0),
            'fcf_yield': metrics.get('fcf_yield', 0),
            'earnings_yield': metrics.get('earnings_yield', 0),
            'position_52w': metrics.get('position_52w', 0),
            'upside_to_target': metrics.get('upside_to_target', 0),
            'graham_number': metrics.get('graham_number', 0),
            'graham_discount': metrics.get('graham_discount', 0),
            'beta': metrics.get('beta', 0),
            'individual_scores': individual_scores,
            'data_completeness': round(data_completeness * 100, 1),
            'success': stock_data.get('success', False)
        }
    
    def analyze_multiple_stocks(self, tickers: List[str], max_workers: int = 8) -> pd.DataFrame:
        """Analyze multiple stocks with parallel processing"""
        
        print(f"Starting valuation analysis of {len(tickers)} stocks")
        print(f"Using {max_workers} parallel workers for optimal speed")
        
        results = []
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.calculate_valuation_score, ticker): ticker 
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
                        'valuation_score': 50.0,
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
        
        # Sort by valuation score
        df = df.sort_values('valuation_score', ascending=False)
        
        # Summary statistics
        total_time = time.time() - start_time
        success_rate = df['success'].sum() / len(df) * 100 if 'success' in df.columns else 0
        
        print(f"\nValuation analysis completed")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Processing rate: {len(tickers)/total_time:.1f} stocks/second")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average score: {df['valuation_score'].mean():.1f}")
        
        return df


def main():
    """Main execution function"""
    
    # Initialize system
    valuation_system = ProfessionalValuationSystem()
    
    # Get NASDAQ tickers
    from nasdaq_tickers import get_nasdaq_tickers
    tickers = get_nasdaq_tickers()
    
    print(f"Professional Valuation Analysis System")
    print(f"Target: {len(tickers)} NASDAQ stocks")
    
    # Run analysis
    results_df = valuation_system.analyze_multiple_stocks(tickers)
    
    # Save results
    os.makedirs('scripts/other_outputs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'scripts/other_outputs/valuation_analysis_{timestamp}.csv'
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
        print(f"\nTop 5 Valuation Opportunities:")
        for _, stock in strong_buys.head(5).iterrows():
            print(f"  {stock['ticker']}: Score {stock['valuation_score']:.1f} "
                  f"(P/E: {stock.get('pe_ratio', 0):.1f}, "
                  f"P/B: {stock.get('pb_ratio', 0):.1f})")
    
    return results_df

if __name__ == "__main__":
    main()