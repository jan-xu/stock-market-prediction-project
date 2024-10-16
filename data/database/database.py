import sqlite3
import struct
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


def bytes_to_int(byte_data):
    # Decode the 8-byte little-endian binary data into an integer
    return struct.unpack("<Q", byte_data)[0]


class StockDatabase:

    def __init__(self, db_name):
        self._db_path = Path(db_name)
        self._csv_path = self._db_path.parent / "stock_data"
        self._csv_path.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    @property
    def database_path(self):
        return self._db_path

    @property
    def csv_path(self):
        return self._csv_path

    def create_table(self):
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS stock_metadata (
            ticker_symbol VARCHAR(10) PRIMARY KEY,
            csv_data_path TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            min_value FLOAT NOT NULL,
            max_value FLOAT NOT NULL,
            min_volume BIGINT NOT NULL,
            max_volume BIGINT NOT NULL,
            value_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        self.conn.commit()

    def download_stock_data(self, ticker_symbol, start_date, end_date=None):
        """
        Download historical stock data from Yahoo Finance.
        """
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            raise ValueError(
                f"No data found for {ticker_symbol} between {start_date} and {end_date}"
            )

        return data

    def save_to_csv(self, ticker_symbol, data):
        csv_file = self.csv_path / f"{ticker_symbol}.csv"

        if csv_file.exists():
            existing_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            combined_data = pd.concat([existing_data, data])
            combined_data = combined_data[
                ~combined_data.index.duplicated(keep="first")
            ]  # remove duplicates
            combined_data = combined_data.sort_index()  # sort data by date
            combined_data.to_csv(csv_file)
        else:
            data.to_csv(csv_file)

        return str(csv_file)

    def update_metadata(self, ticker_symbol, csv_file, data):
        start_date = data.index.min().strftime("%Y-%m-%d")
        end_date = data.index.max().strftime("%Y-%m-%d")
        min_value = data["Close"].min()
        max_value = data["Close"].max()
        min_volume = data["Volume"].min()
        max_volume = data["Volume"].max()
        value_count = len(data)

        self.cursor.execute(
            "SELECT * FROM stock_metadata WHERE ticker_symbol = ?", (ticker_symbol,)
        )
        existing = self.cursor.fetchone()

        if existing:
            self.cursor.execute(
                """
            UPDATE stock_metadata
            SET csv_data_path = ?, start_date = ?, end_date = ?, min_value = ?, max_value = ?, 
                min_volume = ?, max_volume = ?, value_count = ?, modified_at = CURRENT_TIMESTAMP
            WHERE ticker_symbol = ?
            """,
                (
                    csv_file,
                    start_date,
                    end_date,
                    min_value,
                    max_value,
                    min_volume,
                    max_volume,
                    value_count,
                    ticker_symbol,
                ),
            )
        else:
            self.cursor.execute(
                """
            INSERT INTO stock_metadata (ticker_symbol, csv_data_path, start_date, end_date, min_value, max_value, 
                                        min_volume, max_volume, value_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    ticker_symbol,
                    csv_file,
                    start_date,
                    end_date,
                    min_value,
                    max_value,
                    min_volume,
                    max_volume,
                    value_count,
                ),
            )

        self.conn.commit()

    def get_metadata(self, ticker_symbol, printable=False):
        """
        Retrieve metadata for a given ticker symbol.

        Parameters
        ----------
        ticker_symbol : str
            The ticker symbol for the stock.
        printable : bool, optional
            Whether to return metadata as a printable string. Default: False.
        """
        self.cursor.execute(
            "SELECT * FROM stock_metadata WHERE ticker_symbol = ?", (ticker_symbol,)
        )
        metadata = self.cursor.fetchone()

        if not metadata:
            raise ValueError(f"Metadata for ticker {ticker_symbol} not found.")

        metadata_dict = {
            "ticker_symbol": metadata[0],
            "csv_data_path": metadata[1],
            "start_date": metadata[2],
            "end_date": metadata[3],
            "min_value": metadata[4],
            "max_value": metadata[5],
            "min_volume": bytes_to_int(metadata[6]),
            "max_volume": bytes_to_int(metadata[7]),
            "value_count": metadata[8],
            "created_at": metadata[9],
            "modified_at": metadata[10],
        }

        if printable:
            metadata_str = " - "
            metadata_str += "\n - ".join(
                [f"{key}: {value}" for key, value in metadata_dict.items()]
            )
            metadata_str += "\n"
            return metadata_str
        else:
            return metadata_dict


if __name__ == "__main__":

    # Create table
    db_name = "stock_data.db"
    stock_database = StockDatabase(db_name)

    # Fetch and update stock data for a ticker (e.g., AAPL)
    ticker = "NVDA"
    start_date = "2019-01-01"
    data = stock_database.download_stock_data(ticker, start_date)

    # Save to CSV and update metadata
    csv_file = stock_database.save_to_csv(ticker, data)
    stock_database.update_metadata(ticker, csv_file, data)

    # Retrieve metadata for the ticker
    metadata = stock_database.get_metadata(ticker, printable=True)
    print(f"Metadata for ticker {ticker}:")
    print(metadata)
