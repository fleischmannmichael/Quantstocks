#!/usr/bin/env python3
"""
Master Trading System - Main Execution Script
Run this script to execute the complete trading analysis system
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the current directory to Python path so we can import from scripts folder
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Now import the master system
from master_trading_system import (
    MasterTradingSystem, 
    run_quick_analysis, 
    run_full_analysis,
    main as master_main
)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'yfinance',
        'pandas', 
        'numpy',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        
        try:
            install = input("\n🤔 Would you like to install them now? (y/n): ").lower().strip()
            if install in ['y', 'yes']:
                for package in missing_packages:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print("✅ All packages installed successfully!")
                return True
            else:
                print("❌ Cannot proceed without required packages.")
                return False
        except Exception as e:
            print(f"❌ Failed to install packages: {e}")
            return False
    
    print("✅ All required packages are installed!")
    return True

def setup_project_structure():
    """Create the required directory structure"""
    directories = [
        'scripts',
        'scripts/other_outputs',
        'outputs', 
        'outputs/masters'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directory structure created!")

def display_banner():
    """Display the system banner"""
    print("=" * 70)
    print("🚀 MASTER TRADING SYSTEM - QUANTITATIVE STOCK ANALYSIS 🚀")
    print("=" * 70)
    print("📊 6-System Integration: Fundamentals + Trends + Valuation")
    print("🎯 Professional-Grade Analysis for Maximum Profitability")
    print("💼 Portfolio Optimization with Risk Management")
    print("=" * 70)

def display_menu():
    """Display the main menu"""
    print("\n📋 SELECT ANALYSIS TYPE:")
    print("─" * 50)
    print("1️⃣  Quick Analysis     (200 stocks, ~8-15 minutes)")
    print("2️⃣  Standard Analysis  (350 stocks, ~15-25 minutes)")
    print("3️⃣  Full Analysis      (500 stocks, ~20-40 minutes)")
    print("4️⃣  Custom Analysis    (Choose your parameters)")
    print("5️⃣  System Information")
    print("0️⃣  Exit")
    print("─" * 50)

def show_system_info():
    """Show system information"""
    print("\n📖 SYSTEM INFORMATION:")
    print("─" * 50)
    print("🔧 ANALYSIS SYSTEMS:")
    print("   • Fundamentals Analysis (35% weight)")
    print("     - ROE, ROA, Margins, Growth, Debt ratios")
    print("   • Technical/Trends Analysis (35% weight)")
    print("     - RSI, MACD, Moving averages, Volume, Momentum")
    print("   • Valuation Analysis (30% weight)")
    print("     - P/E, P/B, EV/EBITDA, Yields, Fair value")
    print()
    print("📊 SCORING SYSTEM:")
    print("   • 85-100: STRONG BUY (High conviction)")
    print("   • 70-84:  BUY (Medium-high conviction)")
    print("   • 45-69:  HOLD (Neutral)")
    print("   • 25-44:  SELL (Bearish)")
    print("   • 0-24:   STRONG SELL (Very bearish)")
    print()
    print("💰 PORTFOLIO MANAGEMENT:")
    print("   • Max 20 positions")
    print("   • Max 75% invested (25% cash reserve)")
    print("   • Position sizing: 2-6% per stock")
    print("   • Sector diversification analysis")
    print()
    print("📁 OUTPUT STRUCTURE:")
    print("   • Master results: outputs/masters/")
    print("   • Individual systems: scripts/other_outputs/")
    print("   • Summary reports included")

def run_custom_analysis():
    """Run custom analysis with user parameters"""
    print("\n⚙️  CUSTOM ANALYSIS SETUP:")
    print("─" * 40)
    
    try:
        # Get target stocks
        while True:
            try:
                target_stocks = input("📊 Number of stocks to analyze (50-800) [300]: ").strip()
                if not target_stocks:
                    target_stocks = 300
                else:
                    target_stocks = int(target_stocks)
                
                if 50 <= target_stocks <= 800:
                    break
                else:
                    print("❌ Please enter a number between 50 and 800")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get capital amount
        while True:
            try:
                capital_input = input("💵 Portfolio capital in $ [100000]: ").strip()
                if not capital_input:
                    capital = 100000
                else:
                    # Remove commas and dollar signs
                    capital_input = capital_input.replace(',', '').replace('$', '')
                    capital = float(capital_input)
                
                if capital >= 1000:
                    break
                else:
                    print("❌ Please enter at least $1,000")
            except ValueError:
                print("❌ Please enter a valid amount")
        
        print(f"\n✅ Configuration:")
        print(f"   📊 Stocks to analyze: {target_stocks}")
        print(f"   💵 Portfolio capital: ${capital:,.0f}")
        
        # Estimate time
        estimated_time = target_stocks * 0.15  # Rough estimate
        if estimated_time < 60:
            time_str = f"{estimated_time:.0f} seconds"
        else:
            time_str = f"{estimated_time/60:.1f} minutes"
        
        print(f"   ⏱️  Estimated time: {time_str}")
        
        confirm = input("\n🚀 Proceed with analysis? (y/n): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("❌ Analysis cancelled")
            return
        
        # Run the analysis
        system = MasterTradingSystem()
        results = system.run_complete_analysis(target_tickers=target_stocks, total_capital=capital)
        
        # Display results
        display_results(results, system)
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error in custom analysis: {e}")

def display_results(results, system):
    """Display analysis results"""
    if results.get('success'):
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"⏱️  Analysis time: {results['analysis_time_seconds']:.1f} seconds")
        print(f"📊 Stocks analyzed: {results['total_stocks_analyzed']}")
        print(f"🟢 Buy signals: {results['buy_signals']}")
        print(f"🔥 Strong buy signals: {results['strong_buy_signals']}")
        print(f"📈 Average composite score: {results['average_composite_score']}")
        print(f"🏢 Portfolio positions: {results['portfolio_positions']}")
        print(f"💰 Total capital: ${results['total_capital']:,.0f}")
        
        print(f"\n📁 Results saved to: {results['master_filename']}")
        
        # Show top opportunities
        try:
            top_ops = system.get_top_opportunities(5)
            if not top_ops.empty:
                print(f"\n🏆 TOP 5 INVESTMENT OPPORTUNITIES:")
                print("─" * 50)
                for i, (_, stock) in enumerate(top_ops.iterrows(), 1):
                    print(f"{i}. {stock['ticker']:>6} | Score: {stock['composite_score']:>5.1f} | "
                          f"F:{stock['fundamentals_score_clean']:>3.0f} "
                          f"T:{stock['trends_score_clean']:>3.0f} "
                          f"V:{stock['valuation_score_clean']:>3.0f} | "
                          f"{stock['action']}")
        except Exception as e:
            print(f"Could not display top opportunities: {e}")
        
        # Show risk warnings
        try:
            warnings = system.get_risk_warnings()
            if warnings:
                print(f"\n⚠️  RISK WARNINGS:")
                print("─" * 30)
                for warning in warnings:
                    print(f"   • {warning}")
        except Exception as e:
            print(f"Could not display risk warnings: {e}")
        
        print(f"\n✅ Check outputs/masters/ folder for detailed results")
        print(f"✅ Individual system outputs in scripts/other_outputs/")
        
    else:
        print("\n" + "=" * 60)
        print("❌ ANALYSIS FAILED")
        print("=" * 60)
        print(f"Error: {results.get('error', 'Unknown error')}")
        print(f"Time elapsed: {results.get('total_time', 0):.1f} seconds")

def main():
    """Main execution function"""
    
    # Display banner
    display_banner()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup project structure
    setup_project_structure()
    
    # Main menu loop
    while True:
        display_menu()
        
        try:
            choice = input("\n🎯 Enter your choice (0-5): ").strip()
            
            if choice == '0':
                print("\n👋 Goodbye! Happy trading!")
                break
                
            elif choice == '1':
                print("\n🚀 Starting Quick Analysis...")
                results = run_quick_analysis()
                
            elif choice == '2':
                print("\n🚀 Starting Standard Analysis...")
                results = run_full_analysis(350)
                
            elif choice == '3':
                print("\n🚀 Starting Full Analysis...")
                results = run_full_analysis()
                
            elif choice == '4':
                run_custom_analysis()
                continue
                
            elif choice == '5':
                show_system_info()
                continue
                
            else:
                print("❌ Invalid choice. Please enter 0-5.")
                continue
            
            # For choices 1-3, display results
            if choice in ['1', '2', '3']:
                # Create a system instance to get additional info
                system = MasterTradingSystem()
                display_results(results, system)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy trading!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            continue
        
        # Ask if user wants to continue
        try:
            continue_choice = input("\n🔄 Run another analysis? (y/n): ").lower().strip()
            if continue_choice not in ['y', 'yes']:
                print("\n👋 Goodbye! Happy trading!")
                break
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy trading!")
            break

if __name__ == "__main__":
    main()