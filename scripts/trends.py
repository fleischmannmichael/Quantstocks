"""
Professional Trends/Technical Analysis System - FIXED VERSION
Comprehensive price action and momentum analysis for maximum stock coverage
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

class ProfessionalTrendsSystem:
    """
    Professional-grade technical analysis system.
    Analyzes price trends, momentum, and technical indicators.
    """
    
    def __init__(self):
        self.metric_weights = {
            # Trend Analysis (40%)
            'price_trend': 0.15,
            'moving_average_signal': 0.15,
            'trend_strength': 0.10,
            
            # Momentum Analysis (35%)
            'rsi_signal': 0.12,
            'macd_signal': 0.12,
            'momentum_score': 0.11,
            
            # Volume Analysis (15%)
            'volume_trend': 0.08,
            'volume_price_trend': 0.07,
            
            # Volatility Analysis (10%)
            'volatility_signal': 0.05,
            'bollinger_position': 0.05,
        }
        
        # Technical thresholds
        self.thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_low': 40,
            'rsi_neutral_high': 60,
            'trend_strength_strong': 5.0,
            'trend_strength_weak': -5.0,
            'volume_surge': 2.0,
            'volatility_high': 80,
            'volatility_low': 20,
        }
    
    def get_price_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Get historical price data with error handling"""
        try:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period=period)
            
            if hist_data.empty or len(hist_data) < 50:
                return {'ticker': ticker, 'data': pd.DataFrame(), 'error': 'Insufficient data'}
            
            # Get basic info
            try:
                info = stock.info or {}
            except:
                info = {}
            
            return {
                'ticker': ticker,
                'data': hist_data,
                'info': info,
                'success': True
            }
            
        except Exception as e:
            return {'ticker': ticker, 'data': pd.DataFrame(), 'error': str(e)[:100]}
    
    def calculate_moving_averages(self, price_data: pd.Series) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages"""
        mas = {}
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if len(price_data) >= period:
                mas[f'ma_{period}'] = price_data.rolling(window=period).mean()
            else:
                mas[f'ma_{period}'] = pd.Series([price_data.iloc[-1]] * len(price_data), 
                                              index=price_data.index)
        
        # Exponential moving averages
        if len(price_data) >= 12:
            mas['ema_12'] = price_data.ewm(span=12).mean()
            mas['ema_26'] = price_data.ewm(span=26).mean()
        
        return mas
    
    def calculate_rsi(self, price_data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        if len(price_data) < period + 1:
            return pd.Series([50] * len(price_data), index=price_data.index)
        
        delta = price_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def calculate_macd(self, price_data: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        if len(price_data) < 26:
            neutral_series = pd.Series([0] * len(price_data), index=price_data.index)
            return {
                'macd': neutral_series,
                'signal': neutral_series,
                'histogram': neutral_series
            }
        
        ema_12 = price_data.ewm(span=12).mean()
        ema_26 = price_data.ewm(span=26).mean()
        
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, price_data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        if len(price_data) < period:
            middle = pd.Series([price_data.mean()] * len(price_data), index=price_data.index)
            return {
                'upper': middle * 1.02,
                'middle': middle,
                'lower': middle * 0.98
            }
        
        middle = price_data.rolling(window=period).mean()
        std = price_data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def calculate_volume_indicators(self, price_data: pd.Series, volume_data: pd.Series) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        if len(volume_data) < 20:
            return {
                'volume_sma': volume_data.mean(),
                'volume_ratio': 1.0,
                'volume_trend': 0.0,
                'price_volume_correlation': 0.0
            }
        
        volume_sma = volume_data.rolling(window=20).mean()
        current_volume = volume_data.iloc[-1]
        avg_volume = volume_sma.iloc[-1]
        
        # Volume ratio (current vs average)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume trend (recent 10 days vs previous 10 days)
        recent_volume = volume_data.tail(10).mean()
        previous_volume = volume_data.iloc[-20:-10].mean()
        volume_trend = ((recent_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0.0
        
        # Price-volume correlation
        if len(price_data) >= 20 and len(volume_data) >= 20:
            price_returns = price_data.pct_change().tail(20)
            volume_changes = volume_data.pct_change().tail(20)
            correlation = price_returns.corr(volume_changes)
            price_volume_correlation = correlation if not pd.isna(correlation) else 0.0
        else:
            price_volume_correlation = 0.0
        
        return {
            'volume_sma': avg_volume,
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'price_volume_correlation': price_volume_correlation
        }
    
    def calculate_volatility_metrics(self, price_data: pd.Series) -> Dict[str, float]:
        """Calculate volatility metrics"""
        if len(price_data) < 20:
            return {
                'volatility': 0.2,
                'volatility_percentile': 50.0,
                'atr': price_data.std() if len(price_data) > 1 else 1.0
            }
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Volatility percentile (current vs 1-year history)
        volatility_series = returns.rolling(window=20).std() * np.sqrt(252)
        if len(volatility_series) >= 252:
            current_vol = volatility_series.iloc[-1]
            historical_vols = volatility_series.iloc[-252:]
            volatility_percentile = (current_vol > historical_vols).sum() / len(historical_vols) * 100
        else:
            volatility_percentile = 50.0
        
        # Average True Range (simplified)
        high_low = (price_data.rolling(window=20).max() - price_data.rolling(window=20).min()).iloc[-1]
        atr = high_low / 20 if high_low > 0 else price_data.std()
        
        return {
            'volatility': volatility,
            'volatility_percentile': volatility_percentile,
            'atr': atr
        }
    
    def calculate_trend_metrics(self, price_data: pd.Series, moving_averages: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate trend strength and direction metrics"""
        current_price = price_data.iloc[-1]
        
        # Price vs moving averages
        ma_scores = []
        ma_positions = {}
        
        for ma_name, ma_series in moving_averages.items():
            if not ma_series.empty:
                ma_value = ma_series.iloc[-1]
                if ma_value > 0:
                    position = (current_price - ma_value) / ma_value * 100
                    ma_positions[ma_name] = position
                    
                    # Score based on position relative to MA
                    if position > 5:
                        ma_scores.append(100)
                    elif position > 2:
                        ma_scores.append(80)
                    elif position > -2:
                        ma_scores.append(60)
                    elif position > -5:
                        ma_scores.append(40)
                    else:
                        ma_scores.append(20)
        
        # Overall MA signal
        ma_signal_score = np.mean(ma_scores) if ma_scores else 50
        
        # Trend strength (slope of MA20)
        if 'ma_20' in moving_averages and len(moving_averages['ma_20']) >= 10:
            ma_20 = moving_averages['ma_20']
            recent_ma = ma_20.iloc[-1]
            past_ma = ma_20.iloc[-10]
            if past_ma > 0:
                trend_strength = (recent_ma - past_ma) / past_ma * 100
            else:
                trend_strength = 0.0
        else:
            trend_strength = 0.0
        
        # Price momentum (20-day and 50-day)
        momentum_20d = 0.0
        momentum_50d = 0.0
        
        if len(price_data) >= 21:
            price_20d_ago = price_data.iloc[-21]
            momentum_20d = (current_price - price_20d_ago) / price_20d_ago * 100
        
        if len(price_data) >= 51:
            price_50d_ago = price_data.iloc[-51]
            momentum_50d = (current_price - price_50d_ago) / price_50d_ago * 100
        
        return {
            'ma_signal_score': ma_signal_score,
            'trend_strength': trend_strength,
            'momentum_20d': momentum_20d,
            'momentum_50d': momentum_50d,
            'price_vs_ma_20': ma_positions.get('ma_20', 0),
            'price_vs_ma_50': ma_positions.get('ma_50', 0),
            'price_vs_ma_200': ma_positions.get('ma_200', 0)
        }
    
    def calculate_technical_indicators(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        price_data = stock_data['data']
        ticker = stock_data['ticker']
        
        if price_data.empty:
            return self.create_default_technical_metrics(ticker)
        
        try:
            close_prices = price_data['Close']
            volume_data = price_data['Volume']
            
            # Moving averages
            moving_averages = self.calculate_moving_averages(close_prices)
            
            # Technical indicators
            rsi = self.calculate_rsi(close_prices)
            macd_data = self.calculate_macd(close_prices)
            bollinger_bands = self.calculate_bollinger_bands(close_prices)
            
            # Volume indicators
            volume_indicators = self.calculate_volume_indicators(close_prices, volume_data)
            
            # Volatility metrics
            volatility_metrics = self.calculate_volatility_metrics(close_prices)
            
            # Trend metrics
            trend_metrics = self.calculate_trend_metrics(close_prices, moving_averages)
            
            # Current values
            current_price = close_prices.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_macd = macd_data['macd'].iloc[-1]
            current_macd_signal = macd_data['signal'].iloc[-1]
            current_macd_histogram = macd_data['histogram'].iloc[-1]
            
            # Bollinger Band position
            bb_upper = bollinger_bands['upper'].iloc[-1]
            bb_lower = bollinger_bands['lower'].iloc[-1]
            bb_middle = bollinger_bands['middle'].iloc[-1]
            
            if bb_upper > bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
            else:
                bb_position = 50.0
            
            # 52-week high/low analysis
            week_52_high = close_prices.max()
            week_52_low = close_prices.min()
            position_52w = ((current_price - week_52_low) / (week_52_high - week_52_low) * 100) if week_52_high > week_52_low else 50.0
            
            # Compile all metrics
            technical_metrics = {
                'ticker': ticker,
                'current_price': current_price,
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_macd_signal,
                'macd_histogram': current_macd_histogram,
                'bb_position': bb_position,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'week_52_high': week_52_high,
                'week_52_low': week_52_low,
                'position_52w': position_52w,
                'volatility': volatility_metrics['volatility'],
                'volatility_percentile': volatility_metrics['volatility_percentile'],
                'volume_ratio': volume_indicators['volume_ratio'],
                'volume_trend': volume_indicators['volume_trend'],
                'price_volume_correlation': volume_indicators['price_volume_correlation'],
                'data_quality': len(price_data)
            }
            
            # Add trend metrics
            technical_metrics.update(trend_metrics)
            
            # Add moving average values
            for ma_name, ma_series in moving_averages.items():
                if not ma_series.empty:
                    technical_metrics[f'current_{ma_name}'] = ma_series.iloc[-1]
            
            return technical_metrics
            
        except Exception as e:
            print(f"Error calculating technical indicators for {ticker}: {str(e)[:50]}")
            return self.create_default_technical_metrics(ticker)
    
    def create_default_technical_metrics(self, ticker: str) -> Dict[str, Any]:
        """Create default technical metrics for failed calculations"""
        return {
            'ticker': ticker,
            'current_price': 100.0,
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'bb_position': 50.0,
            'bb_upper': 102.0,
            'bb_lower': 98.0,
            'bb_middle': 100.0,
            'week_52_high': 120.0,
            'week_52_low': 80.0,
            'position_52w': 50.0,
            'volatility': 0.25,
            'volatility_percentile': 50.0,
            'volume_ratio': 1.0,
            'volume_trend': 0.0,
            'price_volume_correlation': 0.0,
            'ma_signal_score': 50.0,
            'trend_strength': 0.0,
            'momentum_20d': 0.0,
            'momentum_50d': 0.0,
            'price_vs_ma_20': 0.0,
            'price_vs_ma_50': 0.0,
            'price_vs_ma_200': 0.0,
            'current_ma_20': 100.0,
            'current_ma_50': 100.0,
            'current_ma_200': 100.0,
            'data_quality': 0
        }
    
    def score_technical_metric(self, value: float, metric_name: str) -> int:
        """Score individual technical metrics"""
        try:
            if metric_name == 'rsi':
                if value <= 20:
                    return 95  # Extremely oversold - bullish
                elif value <= 30:
                    return 85  # Oversold - bullish
                elif value <= 40:
                    return 70  # Slightly oversold
                elif value <= 60:
                    return 50  # Neutral
                elif value <= 70:
                    return 30  # Slightly overbought
                elif value <= 80:
                    return 15  # Overbought - bearish
                else:
                    return 5   # Extremely overbought - very bearish
            
            elif metric_name == 'macd_histogram':
                if value > 1.0:
                    return 90
                elif value > 0.5:
                    return 75
                elif value > 0:
                    return 60
                elif value > -0.5:
                    return 40
                elif value > -1.0:
                    return 25
                else:
                    return 10
            
            elif metric_name == 'trend_strength':
                if value > 5.0:
                    return 95
                elif value > 2.0:
                    return 80
                elif value > 0:
                    return 60
                elif value > -2.0:
                    return 40
                elif value > -5.0:
                    return 20
                else:
                    return 5
            
            elif metric_name == 'momentum_20d':
                if value > 15:
                    return 95
                elif value > 8:
                    return 80
                elif value > 0:
                    return 60
                elif value > -8:
                    return 40
                elif value > -15:
                    return 20
                else:
                    return 5
            
            elif metric_name == 'volume_ratio':
                if value > 3.0:
                    return 90  # High volume surge
                elif value > 2.0:
                    return 75
                elif value > 1.5:
                    return 65
                elif value > 0.8:
                    return 50  # Normal volume
                elif value > 0.5:
                    return 35
                else:
                    return 20  # Very low volume
            
            elif metric_name == 'volatility_percentile':
                if value > 90:
                    return 20  # Extremely high volatility - risky
                elif value > 75:
                    return 35
                elif value > 25:
                    return 60  # Normal volatility range
                elif value > 10:
                    return 75
                else:
                    return 85  # Low volatility - stable
            
            elif metric_name == 'bb_position':
                if value <= 10:
                    return 90  # Near lower band - oversold
                elif value <= 25:
                    return 75
                elif value <= 75:
                    return 50  # Middle range
                elif value <= 90:
                    return 25
                else:
                    return 10  # Near upper band - overbought
            
            elif metric_name == 'ma_signal_score':
                return int(value)  # Already scored 0-100
            
            else:
                return 50  # Default neutral score
                
        except (TypeError, ValueError):
            return 50
    
    def calculate_trends_score(self, ticker: str) -> Dict[str, Any]:
        """Calculate comprehensive trends/technical score"""
        
        # Get price data
        stock_data = self.get_price_data(ticker)
        
        # Calculate technical indicators
        technical_metrics = self.calculate_technical_indicators(stock_data)
        
        # Score each component
        total_score = 0.0
        total_weight = 0.0
        individual_scores = {}
        
        # Map metrics to scoring
        metric_mapping = {
            'price_trend': 'ma_signal_score',
            'moving_average_signal': 'ma_signal_score', 
            'trend_strength': 'trend_strength',
            'rsi_signal': 'rsi',
            'macd_signal': 'macd_histogram',
            'momentum_score': 'momentum_20d',
            'volume_trend': 'volume_ratio',
            'volume_price_trend': 'volume_trend',
            'volatility_signal': 'volatility_percentile',
            'bollinger_position': 'bb_position'
        }
        
        for component, weight in self.metric_weights.items():
            metric_name = metric_mapping.get(component, component)
            if metric_name in technical_metrics:
                metric_value = technical_metrics[metric_name]
                score = self.score_technical_metric(metric_value, metric_name)
                
                individual_scores[f"{component}_score"] = score
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
        
        # Determine confidence based on data quality and score extremes
        data_quality = technical_metrics.get('data_quality', 0)
        if data_quality >= 200 and (final_score >= 80 or final_score <= 20):
            confidence = 'HIGH'
        elif data_quality >= 100 and (final_score >= 70 or final_score <= 30):
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'ticker': ticker,
            'trends_score': round(final_score, 1),
            'trading_signal': signal,
            'confidence': confidence,
            'current_price': technical_metrics.get('current_price', 0),
            'rsi': technical_metrics.get('rsi', 50),
            'macd': technical_metrics.get('macd', 0),
            'macd_signal': technical_metrics.get('macd_signal', 0),
            'trend_strength': technical_metrics.get('trend_strength', 0),
            'momentum_20d': technical_metrics.get('momentum_20d', 0),
            'momentum_50d': technical_metrics.get('momentum_50d', 0),
            'volume_ratio': technical_metrics.get('volume_ratio', 1),
            'volatility': technical_metrics.get('volatility', 0.25),
            'position_52w': technical_metrics.get('position_52w', 50),
            'bb_position': technical_metrics.get('bb_position', 50),
            'week_52_high': technical_metrics.get('week_52_high', 0),
            'week_52_low': technical_metrics.get('week_52_low', 0),
            'current_ma_20': technical_metrics.get('current_ma_20', 0),
            'current_ma_50': technical_metrics.get('current_ma_50', 0),
            'current_ma_200': technical_metrics.get('current_ma_200', 0),
            'individual_scores': individual_scores,
            'data_quality': data_quality,
            'success': stock_data.get('success', False)
        }
    
    def analyze_multiple_stocks(self, tickers: List[str], max_workers: int = 8) -> pd.DataFrame:
        """Analyze multiple stocks with parallel processing"""
        
        print(f"Starting trends/technical analysis of {len(tickers)} stocks")
        print(f"Using {max_workers} parallel workers for optimal speed")
        
        results = []
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.calculate_trends_score, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=45)  # 45 second timeout per stock
                    results.append(result)
                except Exception as e:
                    print(f"Failed to analyze {ticker}: {str(e)[:50]}")
                    # Add default result
                    results.append({
                        'ticker': ticker,
                        'trends_score': 50.0,
                        'trading_signal': 'HOLD',
                        'confidence': 'LOW',
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
        
        # Sort by trends score
        df = df.sort_values('trends_score', ascending=False)
        
        # Summary statistics
        total_time = time.time() - start_time
        success_rate = df['success'].sum() / len(df) * 100 if 'success' in df.columns else 0
        
        print(f"\nTrends analysis completed")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Processing rate: {len(tickers)/total_time:.1f} stocks/second")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average score: {df['trends_score'].mean():.1f}")
        
        return df


def main():
    """Main execution function"""
    
    # Initialize system
    trends_system = ProfessionalTrendsSystem()
    
    # Get NASDAQ tickers
    from nasdaq_tickers import get_nasdaq_tickers
    tickers = get_nasdaq_tickers()
    
    print(f"Professional Trends Analysis System")
    print(f"Target: {len(tickers)} NASDAQ stocks")
    
    # Run analysis
    results_df = trends_system.analyze_multiple_stocks(tickers)
    
    # Save results
    os.makedirs('scripts/other_outputs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'scripts/other_outputs/trends_analysis_{timestamp}.csv'
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
        print(f"\nTop 5 Strong Buy Technical Signals:")
        for _, stock in strong_buys.head(5).iterrows():
            print(f"  {stock['ticker']}: Score {stock['trends_score']:.1f} "
                  f"(RSI: {stock.get('rsi', 0):.1f}, "
                  f"Momentum: {stock.get('momentum_20d', 0):.1f}%)")
    
    return results_df

if __name__ == "__main__":
    main()