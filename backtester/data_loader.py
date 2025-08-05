import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Load price data from CSV files or aggregate from a base timeframe.
    """
    def __init__(self, data_dir: str, symbol: str, timeframe: str, base_timeframe: str = None):
        self.data_dir = Path(data_dir)
        self.symbol = symbol
        self.timeframe = timeframe
        self.base_timeframe = base_timeframe

    def load(self) -> pd.DataFrame:
        # 1) Get the raw OHLC (or aggregated) data
        if self.base_timeframe:
            df = self._aggregate_from_lower()
        else:
            df = self._load_csv()

        # 2) Ensure downstream code can always reference df['price']
        if 'price' not in df.columns:
            # Prefer lowercase 'close'
            if 'close' in df.columns:
                df['price'] = df['close']
            # Fallback to uppercase 'Close'
            elif 'Close' in df.columns:
                df['price'] = df['Close']
            # Final fallback: take the first numeric column
            else:
                df['price'] = df.select_dtypes(include='number').iloc[:, 0]

        # 3) Return the augmented DataFrame
        return df


    def _read_csv(self, file: Path) -> pd.DataFrame:
        """
        Read CSV, normalize headers, parse whichever datetime column is present, and set as index.
        """
        df = pd.read_csv(file)
        # normalize column names: strip spaces and lowercase
        df.columns = df.columns.str.strip().str.lower()

        # detect datetime column
        candidates = [col for col in df.columns if col in ('date', 'timestamp', 'datetime', 'begin')]
        if not candidates:
            raise ValueError(f"No datetime column found in {file}")
        date_col = candidates[0]

        # parse dates and set index
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        return df

    def _load_csv(self) -> pd.DataFrame:
        """
        Load a single CSV with columns [datetime, open, high, low, close, volume]
        supporting multiple datetime column names.
        """
        file = self.data_dir / f"{self.symbol}_{self.timeframe}.csv"
        return self._read_csv(file)

    def _aggregate_from_lower(self) -> pd.DataFrame:
        """
        Aggregate from a lower timeframe, e.g., M1 -> H1
        """
        base_file = self.data_dir / f"{self.symbol}_{self.base_timeframe}.csv"
        df = self._read_csv(base_file)
        rule = self._timeframe_to_pandas_rule(self.timeframe)
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        df_agg = df.resample(rule).agg(agg).dropna()
        return df_agg

    @staticmethod
    def _timeframe_to_pandas_rule(tf: str) -> str:
        """
        Convert things like '1m', 'm1', '5m', '1h', '1d' into pandas resample rules.
        """
        tf_low = tf.lower()
        mapping = {
            '1m': '1T',  'm1': '1T',
            '5m': '5T',  'm5': '5T',
            '1h': '1H',  'h1': '1H',
            '1d': '1D',  'd1': '1D',
        }
        if tf_low not in mapping:
            raise ValueError(f"Unsupported timeframe: {tf}")
        return mapping[tf_low]
