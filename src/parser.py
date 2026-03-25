"""
src/parser.py
-------------
Parses raw HDFS log files into structured DataFrames.

HDFS log line example:
  081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049094627660 terminating
"""

import re
import pandas as pd
from pathlib import Path


# Regex pattern for HDFS log format
LOG_PATTERN = re.compile(
    r"(?P<Date>\d{6})\s+"
    r"(?P<Time>\d{6})\s+"
    r"(?P<Pid>\d+)\s+"
    r"(?P<Level>\w+)\s+"
    r"(?P<Component>[^:]+):\s+"
    r"(?P<Content>.+)"
)

# Block ID pattern — used to group logs into sequences
BLOCK_PATTERN = re.compile(r"blk_-?\d+")


def parse_line(line: str) -> dict | None:
    """Parse a single log line. Returns None if it doesn't match."""
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None
    entry = match.groupdict()
    # Extract block IDs from message content
    entry["BlockId"] = BLOCK_PATTERN.findall(entry["Content"])
    return entry


def parse_log_file(filepath: str) -> pd.DataFrame:
    """
    Parse an entire HDFS log file into a structured DataFrame.

    Args:
        filepath: Path to the raw .log file

    Returns:
        DataFrame with columns: Date, Time, Pid, Level, Component, Content, BlockId
    """
    records = []
    path = Path(filepath)

    print(f"[Parser] Reading: {path.name}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            entry = parse_line(line)
            if entry:
                records.append(entry)

    df = pd.DataFrame(records)
    print(f"[Parser] Parsed {len(df):,} log lines.")
    return df


def get_log_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the parsed log DataFrame."""
    print("\n--- Log Summary ---")
    print(f"Total entries : {len(df):,}")
    print(f"Columns       : {list(df.columns)}")
    print(f"\nLog Level distribution:")
    print(df["Level"].value_counts().to_string())
    print(f"\nTop 5 Components:")
    print(df["Component"].value_counts().head().to_string())
    print("-------------------\n")


if __name__ == "__main__":
    # Quick test with a sample file
    sample_logs = [
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049094627660 terminating",
        "081109 203616 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.7.224:61613",
        "081109 203618 148 ERROR dfs.DataNode: Exception in receiveBlock for block blk_-1608999687919862906",
    ]

    records = [parse_line(line) for line in sample_logs]
    df = pd.DataFrame([r for r in records if r])
    print(df[["Level", "Component", "BlockId"]].to_string())
