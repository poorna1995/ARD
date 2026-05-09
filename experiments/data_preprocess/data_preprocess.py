from pathlib import Path
import ast

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
GOLDEN_DIR = REPO_ROOT / "datasets" / "golden"


class DataPreprocessor:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def _read_processed(self) -> pd.DataFrame:
        parquet_path = PROCESSED_DIR / f"{self.dataset_name}.parquet"
        return pd.read_parquet(parquet_path)

    def _write_golden(self, df: pd.DataFrame) -> None:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        out_path = GOLDEN_DIR / f"{self.dataset_name}.parquet"
        df.to_parquet(out_path)

    def preprocess(self):
        if self.dataset_name == "gaia":
            self.preprocess_gaia()
        elif self.dataset_name == "mmlu_pro":
            self.preprocess_mmlu_pro()
        elif self.dataset_name == "math_hard":
            self.preprocess_math_hard()
        else:
            self.preprocess_swe_bench()

    def preprocess_gaia(self):
        df = self._read_processed()[["query", "file_name", "answer", "level"]].copy()
        file_name = df["file_name"].fillna("").astype(str).str.strip()
        has_file = file_name.ne("")
        df["query"] = df["query"].astype(str)
        df.loc[has_file, "query"] = (
            df.loc[has_file, "query"]
            + "\nAttached file name for reference: "
            + file_name[has_file]
        )
        self._write_golden(df)

    def preprocess_mmlu_pro(self):
        df = self._read_processed()[["query", "options", "answer", "category"]].copy()

        def format_options(value) -> str:
            if value is None:
                return ""

            if isinstance(value, (list, tuple)):
                parsed = value
            elif hasattr(value, "tolist") and not isinstance(value, str):
                parsed = value.tolist()
            else:
                try:
                    if pd.isna(value):
                        return ""
                except (TypeError, ValueError):
                    pass
                parsed = value

            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return ""
                if text.startswith("[") and text.endswith("]"):
                    try:
                        parsed = ast.literal_eval(text)
                    except (ValueError, SyntaxError):
                        parsed = text
                else:
                    parsed = text

            if isinstance(parsed, (list, tuple)):
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                lines = [
                    f"{labels[idx]}. {str(opt).strip()}"
                    for idx, opt in enumerate(parsed)
                    if idx < len(labels)
                ]
                return ", ".join(lines)

            return str(parsed)

        options_text = df["options"].apply(format_options)
        has_options = options_text.str.strip().ne("")
        df["query"] = (
            df["query"]
            .astype(str)
            .str.replace(r"\nOptions are:\s*.*$", "", regex=True)
        )
        df.loc[has_options, "query"] = (
            df.loc[has_options, "query"]
            + "\nOptions are: "
            + options_text[has_options]
        )
        self._write_golden(df)

    def preprocess_math_hard(self):
        df = self._read_processed()[["query", "answer", "level", "type"]].copy()
        self._write_golden(df)

    def preprocess_swe_bench(self):
        df = self._read_processed()[["query", "answer", "repo", "difficulty"]].copy()
        self._write_golden(df)


if __name__ == "__main__":
    data_preprocessor = DataPreprocessor(dataset_name="mmlu_pro")
    data_preprocessor.preprocess()