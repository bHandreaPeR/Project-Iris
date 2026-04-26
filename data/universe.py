"""
Stock universes for India (NSE) and US markets.
Tickers use yfinance conventions: .NS suffix for NSE-listed stocks.
Edit these lists to customise the screening universe.
"""

# Nifty 100 — NSE tickers (yfinance .NS suffix)
INDIA_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS',
    'INFOSYS.NS', 'SBIN.NS', 'HINDUNILVR.NS', 'ITC.NS', 'LT.NS',
    'KOTAKBANK.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'TITAN.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'HCLTECH.NS',
    'POWERGRID.NS', 'NTPC.NS', 'ONGC.NS', 'COALINDIA.NS', 'JSWSTEEL.NS',
    'TATASTEEL.NS', 'TECHM.NS', 'NESTLEIND.NS', 'BAJAJFINSV.NS', 'DIVISLAB.NS',
    'DRREDDY.NS', 'CIPLA.NS', 'APOLLOHOSP.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS',
    'BAJAJ-AUTO.NS', 'M&M.NS', 'TATACONSUM.NS', 'BRITANNIA.NS', 'PIDILITIND.NS',
    'ADANIENT.NS', 'ADANIPORTS.NS', 'GRASIM.NS', 'INDUSINDBK.NS', 'SBILIFE.NS',
    'HDFCLIFE.NS', 'ICICIGI.NS', 'HAVELLS.NS', 'DABUR.NS', 'MARICO.NS',
    'SHREECEM.NS', 'AMBUJACEM.NS', 'ACC.NS', 'GODREJCP.NS', 'COLPAL.NS',
    'BERGEPAINT.NS', 'MCDOWELL-N.NS', 'UBL.NS', 'TVSMOTOR.NS', 'BOSCHLTD.NS',
    'SIEMENS.NS', 'ABB.NS', 'VEDL.NS', 'HINDALCO.NS', 'APLAPOLLO.NS',
    'SAIL.NS', 'NMDC.NS', 'GAIL.NS', 'BPCL.NS', 'IOC.NS',
    'HINDPETRO.NS', 'PETRONET.NS', 'IOCL.NS', 'PFC.NS', 'RECLTD.NS',
    'BANKBARODA.NS', 'CANBK.NS', 'UNIONBANK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS',
    'BANDHANBNK.NS', 'MUTHOOTFIN.NS', 'CHOLAFIN.NS', 'BAJAJHLDNG.NS', 'SBICARD.NS',
    'MOTHERSON.NS', 'ASHOKLEY.NS', 'BALKRISIND.NS', 'MRF.NS', 'CEAT.NS',
    'DMART.NS', 'NYKAA.NS', 'ZOMATO.NS', 'PAYTM.NS', 'NAUKRI.NS',
    'INFY.NS', 'MPHASIS.NS', 'LTIM.NS', 'PERSISTENT.NS', 'COFORGE.NS',
]

# S&P 500 — a representative subset for daily screening
US_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM', 'UNH',
    'V', 'XOM', 'MA', 'LLY', 'HD', 'PG', 'AVGO', 'COST', 'MRK', 'ABBV',
    'CVX', 'CRM', 'NFLX', 'AMD', 'TMO', 'BAC', 'PEP', 'ADBE', 'WMT', 'ACN',
    'MCD', 'LIN', 'CSCO', 'ABT', 'TXN', 'DHR', 'PM', 'AMGN', 'NEE', 'VZ',
    'INTC', 'ORCL', 'IBM', 'QCOM', 'HON', 'RTX', 'UPS', 'CAT', 'BA', 'GE',
    'GS', 'MS', 'BLK', 'C', 'WFC', 'USB', 'AXP', 'SCHW', 'CB', 'MMC',
    'LOW', 'TGT', 'SBUX', 'NKE', 'LULU', 'TJX', 'AZO', 'ORLY', 'DG', 'DLTR',
    'UNP', 'CSX', 'NSC', 'FDX', 'DE', 'ETN', 'EMR', 'ITW', 'PH', 'ROK',
    'LMT', 'GD', 'NOC', 'HII', 'L3H', 'REGN', 'GILD', 'VRTX', 'BIIB', 'BMY',
    'PFE', 'JNJ', 'MDT', 'SYK', 'BSX', 'ELV', 'CI', 'HUM', 'CVS', 'MCK',
]
