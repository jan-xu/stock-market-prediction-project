import argparse
from datetime import datetime

import yfinance as yf

from data.database import DATABASE_PATH, StockDatabase
from ui import eda_plots


def save_to_database(ticker, data):
    """
    Save stock data to a SQLite database via the `data` library.
    """
    stock_database = StockDatabase(DATABASE_PATH)

    # Save to CSV and update metadata
    csv_file = stock_database.save_to_csv(ticker, data)
    stock_database.update_metadata(ticker, csv_file, data)

    # Retrieve metadata for the ticker and print
    metadata = stock_database.get_metadata(ticker, printable=True)
    print(f"Metadata for ticker {ticker}:")
    print(metadata)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download and analyze historical stock data"
    )
    parser.add_argument("-t", "--ticker", type=str, help="Stock ticker symbol")
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date for historical data (YYYY-MM-DD) (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for historical data (YYYY-MM-DD) (default: today)",
    )
    parser.add_argument("--plot", action="store_true", help="Plot the closing price")
    parser.add_argument(
        "--dbsave", action="store_true", help="Save the data to a SQLite database"
    )
    args = parser.parse_args()

    # Assign the ticker symbol from command line argument
    ticker = args.ticker

    # Unpack dates
    start = args.start
    if args.end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    else:
        end = args.end

    # Download historical data from Yahoo Finance
    print(f"Downloading historical stock data for {ticker} from {start} to {end}...\n")
    stock_data = yf.download(ticker, start=start, end=end, progress=False)
    if stock_data.empty:
        raise ValueError(f"No data available for the specified ticker symbol: {ticker}")

    # Display the first few rows of the data
    print("First 10 days:\n", stock_data.head(10), "\n")
    print("Last 10 days:\n", stock_data.tail(10), "\n")

    # Plot the closing price
    if args.plot:
        _ = eda_plots(
            stock_data,
            label=ticker,
            value_col="Adj Close",
            date_col="Date",
            volume_col="Volume",
            add_ma=True,
            show_fig=args.plot,
        )
    else:
        # Calculate statistics of fetched data
        print(f"Basic statistics of {ticker} data:\n")
        print(stock_data.describe(), "\n")

    # Save the data to the SQLite database
    if args.dbsave:
        print(f"Saving data to SQLite database: {DATABASE_PATH}...\n")
        save_to_database(ticker, stock_data)
