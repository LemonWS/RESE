"""

Unified data loader for multi-system equilibrium experiments.

Expected data structure
-----------------------
Rows    : time/date observations
Columns : individual systems

Example
-------
Date, S1, S2, S3, ..., SN
t1,   y11, y21, y31, ..., yN1
t2,   y12, y22, y32, ..., yN2
...

The returned target matrix has shape

    Y.shape = (T, N)

where
    T = number of time observations
    N = number of systems.
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# ============================================================
# 1. Read different tabular file formats
# ============================================================

def read_file(file_path: Path) -> pd.DataFrame:
    """
    Read a supported tabular data file into a pandas DataFrame.

    Supported formats:
        .csv
        .xlsx / .xls
        .parquet
        .tsv
        .txt

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the input data file.

    Returns
    -------
    pd.DataFrame
        Loaded table.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)

    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)

    elif suffix in [".tsv", ".txt"]:
        df = pd.read_csv(file_path, sep="\t")

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}\n"
            "Supported formats are: "
            ".csv, .xlsx, .xls, .parquet, .tsv, .txt"
        )

    return df


# ============================================================
# 2. Convert non-CSV files to CSV
# ============================================================

def convert_to_csv(
    file_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Convert a supported tabular file to CSV.

    If the original file is already CSV, its path is returned
    without creating another file.

    Parameters
    ----------
    file_path : str
        Input file path.

    output_path : str, optional
        Path of the converted CSV file.
        If None, the CSV is created next to the original file.

    Returns
    -------
    str
        Path to the CSV file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {file_path}"
        )

    # Already CSV
    if path.suffix.lower() == ".csv":
        return str(path)

    # Read original file
    df = read_file(path)

    # Determine output location
    if output_path is None:
        csv_path = path.with_suffix(".csv")
    else:
        csv_path = Path(output_path)

    # Save standardized CSV
    df.to_csv(csv_path, index=False)

    return str(csv_path)


# ============================================================
# 3. Validate multi-system target data
# ============================================================

def _validate_target_data(
    df: pd.DataFrame,
    date_col: str,
    allow_missing: bool = False
) -> None:
    """
    Validate the standardized multi-system dataset.
    """

    # --------------------------------------------------------
    # Date column
    # --------------------------------------------------------

    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Need at least one system
    # --------------------------------------------------------

    target_cols = [
        col for col in df.columns
        if col != date_col
    ]

    if len(target_cols) == 0:
        raise ValueError(
            "No system target columns were found."
        )

    # --------------------------------------------------------
    # Numerical target columns
    # --------------------------------------------------------

    non_numeric = [
        col for col in target_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise ValueError(
            "All system target columns must be numerical.\n"
            f"Non-numeric columns found: {non_numeric}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    if not allow_missing:
        missing = df[target_cols].isna().sum()

        missing = missing[missing > 0]

        if len(missing) > 0:
            raise ValueError(
                "Missing values were found in the system data:\n"
                f"{missing.to_dict()}\n"
                "Set allow_missing=True if missing values "
                "should be retained."
            )


# ============================================================
# 4. Main data-loading function
# ============================================================

def load_system_data(
    file_path: str,
    date_col: Optional[str] = None,
    convert_csv: bool = True,
    csv_output_path: Optional[str] = None,
    sort_date: bool = True,
    allow_missing: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load multi-system time-series data.

    The expected structure is

        rows    -> time
        columns -> systems

    Parameters
    ----------
    file_path : str
        Input file path.

    date_col : str, optional
        Name of the date/time column.
        If None, the first column is treated as the date column.

    convert_csv : bool, default=True
        If True, non-CSV input files are converted to CSV.

    csv_output_path : str, optional
        Output location for converted CSV.

    sort_date : bool, default=True
        Sort observations chronologically.

    allow_missing : bool, default=False
        Whether NaN values are allowed in system target data.

    Returns
    -------
    dates : np.ndarray
        Time index, shape (T,).

    Y : np.ndarray
        Multi-system target matrix, shape (T, N).

    system_names : list[str]
        Names of the N systems.

    Notes
    -----
    Y[t, i] represents the observation of system i at time t.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {file_path}"
        )

    # --------------------------------------------------------
    # Convert to CSV when necessary
    # --------------------------------------------------------

    if convert_csv and path.suffix.lower() != ".csv":

        csv_path = convert_to_csv(
            file_path=file_path,
            output_path=csv_output_path
        )

        path = Path(csv_path)

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    df = read_file(path)

    # --------------------------------------------------------
    # Date column defaults to first column
    # --------------------------------------------------------

    if date_col is None:
        date_col = df.columns[0]

    # --------------------------------------------------------
    # Convert date column
    # --------------------------------------------------------

    try:
        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="raise"
        )
    except Exception as exc:
        raise ValueError(
            f"Column '{date_col}' cannot be converted "
            "to datetime."
        ) from exc

    # --------------------------------------------------------
    # Check duplicated dates
    # --------------------------------------------------------

    duplicated_dates = df[date_col].duplicated()

    if duplicated_dates.any():
        duplicates = (
            df.loc[duplicated_dates, date_col]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicated dates were found:\n"
            f"{duplicates[:10]}"
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    if sort_date:
        df = (
            df.sort_values(date_col)
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    _validate_target_data(
        df=df,
        date_col=date_col,
        allow_missing=allow_missing
    )

    # --------------------------------------------------------
    # Extract output
    # --------------------------------------------------------

    system_names = [
        col for col in df.columns
        if col != date_col
    ]

    dates = df[date_col].to_numpy()

    Y = df[system_names].to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    T, N = Y.shape

    print("=" * 60)
    print("Multi-System Dataset Loaded")
    print("=" * 60)
    print(f"File          : {path}")
    print(f"Date column   : {date_col}")
    print(f"Observations  : {T}")
    print(f"Systems       : {N}")
    print(f"Target shape  : {Y.shape}")
    print(f"Start date    : {dates[0]}")
    print(f"End date      : {dates[-1]}")
    print("=" * 60)

    return dates, Y, system_names


# ============================================================
# 5. Example
# ============================================================

if __name__ == "__main__":

    dates, Y, system_names = load_system_data(
        file_path="data/example.csv",
        date_col="Date"
    )

    print("\nSystem names:")
    print(system_names)

    print("\nTarget array:")
    print(Y)

    print("\nShape:")
    print(Y.shape)