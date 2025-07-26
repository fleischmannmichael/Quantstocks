"""
NASDAQ Tickers Retrieval System - FIXED VERSION
Get comprehensive list of NASDAQ-listed stocks for analysis
"""

import pandas as pd
import requests
import yfinance as yf
from typing import List, Dict, Set
import time
import warnings
import os

warnings.filterwarnings('ignore')

def get_nasdaq_100_tickers() -> List[str]:
    """Get NASDAQ 100 tickers"""
    nasdaq_100 = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
        'NFLX', 'TMUS', 'ADBE', 'PEP', 'ASML', 'CSCO', 'AMD', 'LIN', 'TXN', 'QCOM',
        'CMCSA', 'INTU', 'AMAT', 'AMGN', 'HON', 'ISRG', 'VRTX', 'PANW', 'ADP', 'SBUX',
        'GILD', 'MU', 'INTC', 'ADI', 'LRCX', 'PYPL', 'REGN', 'KLAC', 'SNPS', 'CDNS',
        'MRVL', 'CRWD', 'CSX', 'FTNT', 'ORLY', 'ABNB', 'DASH', 'ROP', 'NXPI', 'WDAY',
        'CPRT', 'PCAR', 'MNST', 'PAYX', 'FAST', 'ROST', 'ODFL', 'EA', 'BKR', 'DDOG',
        'XEL', 'KDP', 'GEHC', 'VRSK', 'EXC', 'CTSH', 'TEAM', 'CSGP', 'AEP', 'KHC',
        'CCEP', 'FANG', 'ON', 'DXCM', 'BIIB', 'IDXX', 'ANSS', 'ZS', 'TTWO', 'CDW',
        'GFS', 'WBD', 'ILMN', 'ARM', 'MDB', 'ZM', 'LULU', 'ALGN', 'LCID', 'MRNA',
        'CRSP', 'OKTA', 'DOCU', 'PTON', 'ROKU', 'ZI', 'BILL', 'SGEN', 'BMRN', 'EXPE'
    ]
    return nasdaq_100

def get_additional_nasdaq_tickers() -> List[str]:
    """Get additional high-volume NASDAQ tickers beyond the top 100"""
    additional_tickers = [
        # Large Cap Growth
        'CRM', 'NOW', 'SNOW', 'PLTR', 'RBLX', 'U', 'NET', 'TWLO', 'PINS', 'SNAP',
        'UBER', 'LYFT', 'HOOD', 'COIN', 'RIVN', 'NKLA', 'QS', 'CHPT', 'BLNK', 'LCID',
        
        # Healthcare & Biotech
        'VRTX', 'GILD', 'REGN', 'BIIB', 'AMGN', 'CELG', 'BMRN', 'SGEN', 'ALXN', 'INCY',
        'TECH', 'VCEL', 'SAGE', 'BLUE', 'EDIT', 'NTLA', 'BEAM', 'CRSP', 'FOLD', 'RARE',
        
        # Technology Hardware
        'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'MRVL', 'ADI', 'LRCX', 'KLAC', 'AMAT',
        'NXPI', 'SWKS', 'QRVO', 'MCHP', 'XLNX', 'ALTR', 'ON', 'MPWR', 'CRUS', 'SLAB',
        
        # Software & Cloud
        'MSFT', 'ORCL', 'CRM', 'NOW', 'WDAY', 'ADBE', 'INTU', 'CTXS', 'SPLK', 'OKTA',
        'ZS', 'CRWD', 'PANW', 'FTNT', 'CYBR', 'FEYE', 'PING', 'ESTC', 'DDOG', 'SUMO',
        
        # E-commerce & Digital
        'AMZN', 'EBAY', 'ETSY', 'CHWY', 'CVNA', 'OSTK', 'W', 'WAYFAIR', 'OVERSTOCK',
        'SHOP', 'BIGC', 'VERB', 'PRTS', 'APPS', 'ECOM', 'FLWS', 'GRUB', 'DASH', 'UBER',
        
        # Media & Entertainment
        'NFLX', 'ROKU', 'SPOT', 'RBLX', 'TTWO', 'EA', 'ATVI', 'ZNGA', 'TAKE', 'MTCH',
        'BMBL', 'YELP', 'TRIP', 'EXPEDIA', 'BKNG', 'PCLN', 'EXPE', 'TZOO', 'MMYT', 'TRVG',
        
        # Electric Vehicles & Clean Energy
        'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'NKLA', 'RIDE', 'GOEV', 'HYLN',
        'QS', 'BLNK', 'CHPT', 'EVGO', 'CLSK', 'RIOT', 'MARA', 'HUT', 'BITF', 'CAN',
        
        # Financial Technology
        'PYPL', 'SQ', 'SOFI', 'AFRM', 'UPST', 'LMND', 'ROOT', 'OPEN', 'MTTR', 'HOOD',
        'COIN', 'MSTR', 'GBTC', 'ETHE', 'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'PRNT',
        
        # Semiconductors Extended
        'TSM', 'ASML', 'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI', 'LRCX',
        'KLAC', 'AMAT', 'NXPI', 'MRVL', 'SWKS', 'QRVO', 'MCHP', 'ON', 'MPWR', 'CRUS',
        
        # Cannabis & Speculative
        'TLRY', 'CGC', 'CRON', 'ACB', 'HEXO', 'OGI', 'APHA', 'SNDL', 'GRWG', 'IIPR',
        
        # Meme & High Volatility
        'GME', 'AMC', 'BBBY', 'KOSS', 'EXPR', 'NAKD', 'NOK', 'BB', 'WISH', 'CLOV',
        
        # REITs
        'O', 'STOR', 'STAG', 'GOOD', 'LAND', 'SAFE', 'CUBE', 'REXR', 'EXR', 'PSA',
        
        # Utilities & Infrastructure
        'NEE', 'DUK', 'SO', 'AEP', 'EXC', 'XEL', 'WEC', 'ES', 'ED', 'ETR',
        
        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'LOW', 'NKE', 'SBUX', 'TGT', 'COST', 'WMT', 'DIS',
        
        # Healthcare Services
        'UNH', 'CVS', 'CI', 'ANTM', 'HUM', 'CNC', 'MOH', 'WCG', 'ELV', 'TDOC',
        
        # Industrial & Transportation
        'CAT', 'DE', 'BA', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'UNP', 'CSX'
    ]
    
    # Remove duplicates and return
    return list(set(additional_tickers))

def get_nasdaq_screener_tickers() -> List[str]:
    """Get tickers from NASDAQ screener (alternative method)"""
    try:
        # Try to get from NASDAQ's API or screener
        url = "https://api.nasdaq.com/api/screener/stocks"
        params = {
            'tableonly': 'true',
            'limit': '500',
            'offset': '0',
            'exchange': 'nasdaq'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'rows' in data['data']:
                tickers = [row['symbol'] for row in data['data']['rows']]
                return tickers[:500]  # Limit to 500
        
    except Exception as e:
        print(f"Failed to get NASDAQ screener data: {e}")
    
    return []

def validate_tickers(tickers: List[str], sample_size: int = 50) -> List[str]:
    """Validate a sample of tickers to ensure they're tradeable"""
    if not tickers:
        return []
    
    # Take a sample for validation
    import random
    sample_tickers = random.sample(tickers, min(sample_size, len(tickers)))
    valid_tickers = []
    
    print(f"Validating {len(sample_tickers)} sample tickers...")
    
    for ticker in sample_tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Basic validation criteria
            if (info and 
                info.get('regularMarketPrice') and 
                info.get('marketCap') and 
                info.get('marketCap', 0) > 100_000_000):  # Min $100M market cap
                valid_tickers.append(ticker)
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception:
            continue
    
    validation_rate = len(valid_tickers) / len(sample_tickers) * 100
    print(f"Validation rate: {validation_rate:.1f}% ({len(valid_tickers)}/{len(sample_tickers)})")
    
    # If validation rate is good, return all original tickers
    if validation_rate >= 70:
        return tickers
    else:
        # Return only validated tickers
        return valid_tickers

def get_nasdaq_tickers() -> List[str]:
    """
    Main function to get comprehensive NASDAQ tickers list
    Priority order: NASDAQ 100 + Additional curated + Screener data
    """
    
    print("Building comprehensive NASDAQ tickers list...")
    
    all_tickers = set()
    
    # 1. Start with NASDAQ 100 (most reliable)
    nasdaq_100 = get_nasdaq_100_tickers()
    all_tickers.update(nasdaq_100)
    print(f"Added {len(nasdaq_100)} NASDAQ 100 tickers")
    
    # 2. Add curated additional tickers
    additional = get_additional_nasdaq_tickers()
    all_tickers.update(additional)
    print(f"Added {len(additional)} additional curated tickers")
    
    # 3. Try to get from NASDAQ screener
    screener_tickers = get_nasdaq_screener_tickers()
    if screener_tickers:
        all_tickers.update(screener_tickers)
        print(f"Added {len(screener_tickers)} from NASDAQ screener")
    
    # Convert to sorted list
    final_tickers = sorted(list(all_tickers))
    
    print(f"Total unique tickers collected: {len(final_tickers)}")
    
    # Validate a sample
    if len(final_tickers) > 100:
        print("Validating ticker quality...")
        validated = validate_tickers(final_tickers, sample_size=50)
        if len(validated) >= 50:  # If validation went well
            final_tickers = final_tickers  # Keep all
        else:
            print("Low validation rate, using only core tickers")
            final_tickers = nasdaq_100 + additional[:200]  # Fallback to curated
    
    # Ensure we have a good number but not too many
    if len(final_tickers) > 800:
        final_tickers = final_tickers[:800]  # Limit to 800 for processing efficiency
    
    print(f"Final ticker count: {len(final_tickers)}")
    
    return final_tickers

def get_high_volume_nasdaq_tickers(min_volume: int = 1_000_000) -> List[str]:
    """Get NASDAQ tickers filtered by minimum daily volume"""
    base_tickers = get_nasdaq_tickers()
    high_volume_tickers = []
    
    print(f"Filtering for tickers with minimum volume of {min_volume:,}")
    
    for i, ticker in enumerate(base_tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            avg_volume = info.get('averageVolume', 0)
            if avg_volume >= min_volume:
                high_volume_tickers.append(ticker)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(base_tickers)} tickers...")
            
            time.sleep(0.05)  # Rate limiting
            
        except Exception:
            continue
    
    print(f"Found {len(high_volume_tickers)} high-volume tickers")
    return high_volume_tickers

def save_tickers_to_file(tickers: List[str], filename: str = "scripts/other_outputs/nasdaq_tickers.txt"):
    """Save tickers list to file for future use"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")
    print(f"Saved {len(tickers)} tickers to {filename}")

def load_tickers_from_file(filename: str = "scripts/other_outputs/nasdaq_tickers.txt") -> List[str]:
    """Load tickers from file"""
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from {filename}")
        return tickers
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []

def get_sector_specific_tickers() -> Dict[str, List[str]]:
    """Get tickers organized by sector for targeted analysis"""
    sectors = {
        'Technology': [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'ORCL', 'CSCO',
            'ADBE', 'CRM', 'NOW', 'INTU', 'QCOM', 'TXN', 'ADI', 'AVGO', 'LRCX', 'KLAC'
        ],
        'Healthcare': [
            'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'ABT', 'LLY', 'DHR', 'BMY',
            'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ISRG', 'MRNA', 'BNTX', 'NVAX'
        ],
        'Consumer_Discretionary': [
            'AMZN', 'TSLA', 'HD', 'LOW', 'NKE', 'SBUX', 'TGT', 'COST', 'WMT', 'DIS',
            'MCD', 'LULU', 'ROKU', 'NFLX', 'ABNB', 'UBER', 'LYFT', 'DASH'
        ],
        'Financial_Services': [
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SPGI', 'V', 'MA',
            'PYPL', 'SQ', 'AFRM', 'SOFI', 'HOOD', 'COIN'
        ],
        'Communication': [
            'META', 'GOOGL', 'GOOG', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'DIS'
        ],
        'Energy': [
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY', 'DVN', 'FANG'
        ]
    }
    
    return sectors

# Test function
def test_ticker_retrieval():
    """Test the ticker retrieval system"""
    print("Testing NASDAQ ticker retrieval system...")
    
    # Test basic retrieval
    tickers = get_nasdaq_tickers()
    print(f"Retrieved {len(tickers)} tickers")
    
    # Test validation
    if len(tickers) >= 10:
        sample_valid = validate_tickers(tickers[:10], sample_size=10)
        print(f"Validation test: {len(sample_valid)}/10 tickers valid")
    
    # Test sector breakdown
    sector_tickers = get_sector_specific_tickers()
    total_sector_tickers = sum(len(tickers) for tickers in sector_tickers.values())
    print(f"Sector-specific tickers: {total_sector_tickers} across {len(sector_tickers)} sectors")
    
    return tickers

if __name__ == "__main__":
    # Run test
    tickers = test_ticker_retrieval()
    
    # Save for future use
    save_tickers_to_file(tickers, "scripts/other_outputs/nasdaq_tickers.txt")
    
    print(f"\nNASDAQ ticker retrieval system ready!")
    print(f"Total tickers available: {len(tickers)}")
    print(f"Sample tickers: {tickers[:10]}")