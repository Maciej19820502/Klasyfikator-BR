import pandas as pd
import re
import os
import logging
import platform
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# Enhanced Classification Heuristics
# -------------------------
class ContentClassifier:
    """Enhanced content classification with multiple detection strategies."""
    def __init__(self):
        self.patterns = {
            "TOC": [
                r"\.{3,}\s*\d+$",                                  # kropki + numer strony
                r"^\s*\d+\.?\s+.+\s+\d+\s*$",                      # sekcja + numer strony
                r"^\s*(Chapter|Rozdział|Część)\s+\d+",            # nagłówki rozdziałów
                r"^\s*\d+\.\d+\.?\s+",                            # 1.1 , 2.3.4
            ],
            "Header_Footer": [
                r"^\s*Strona\s+\d+",                              # "Strona 12"
                r"^\s*\d+\s*$",                                   # samotny numer strony
                r"©\s*\d{4}",                                     # copyright
                r"^(Rozdział|Chapter)\s+\d+\s*$",                 # goły nagłówek rozdziału
            ],
            "Bibliography": [
                r"^\s*\[\d+\]",                                   # [1], [23]
                r"^\s*\d+\.\s+[A-ZĄĆĘŁŃÓŚŹŻ].*?,.*?\d{4}",        # pozycje bibl.
                r"(?:doi:|DOI:)",                                 # DOI
                r"(https?://|www\.)",                             # URL
                r"(?:ISBN|ISSN):\s*[\d\-X]+",                     # ISBN/ISSN
            ],
            "Table_Figure": [
                r"^(Tabela|Table|Wykres|Chart|Rysunek|Figure|Diagram)\s+\d+",
                r"Studium przypadku",
                r"Case study",
                r"^\s*\|\s*.+\s*\|",                              # tabelki markdown
                r"^\s*[A-Za-z\s]+\s*\|\s*\d+",                    # wiersze tabel
            ],
            "Formula_Technical": [
                r"[A-Za-z]\s*=\s*[A-Za-z0-9\+\-\*/\(\)]+",        # proste wzory
                r"[\u2211\u222b\u2202\u221a\u00b1\u2264\u2265\u2260\u2248]",  # ∑ ∫ ∂ √ ± ≤ ≥ ≠ ≈
                r"\$.*?\$",                                       # LaTeX math mode
                r"Algorithm\s+\d+",
            ],
        }

    def detect_type(self, paragraph: str) -> str:
        """Priority-based content type detection."""
        text = str(paragraph).strip()
        if not text or len(text) < 3:
            return "Empty/Short"

        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    return category

        stats = self._calculate_text_stats(text)

        # Duża gęstość cyfr -> tabelaryczne/techniczne
        if stats["number_density"] > 0.15:
            return "Table_Figure"

        # Bardzo krótkie -> nagłówek/stopka
        if len(text.split()) < 5:
            return "Header_Footer"

        # Dużo interpunkcji + kropki/przecinki -> bibliografia/cytowania
        if stats["punct_density"] > 0.08 and ("," in text and "." in text):
            return "Bibliography"

        return "Content"

    def _calculate_text_stats(self, text: str) -> Dict[str, float]:
        words = text.split()
        total_chars = len(text)
        if total_chars == 0:
            return {"number_density": 0.0, "punct_density": 0.0, "avg_word_len": 0.0}

        numbers = len(re.findall(r"\d+", text))
        punctuation = len(re.findall(r"[.,;:!?\(\)\[\]{}\"]", text))

        return {
            "number_density": numbers / len(words) if words else 0.0,
            "punct_density": punctuation / total_chars,
            "avg_word_len": (sum(len(w) for w in words) / len(words)) if words else 0.0,
        }

# -------------------------
# Enhanced Analysis Engine - cross-platform
# -------------------------
class ClusterAnalyzer:
    """Main analysis engine with comprehensive reporting."""

    def __init__(self, project_name: str, base_dir: Optional[str] = None):
        self.project_name = project_name

        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            if platform.system() in ("Windows", "Darwin"):
                self.base_dir = Path.home() / "Documents" / "DocumentAnalysis" / "Projects" / project_name
            else:
                self.base_dir = Path.home() / "DocumentAnalysis" / "Projects" / project_name

        self.classifier = ContentClassifier()

        # File paths
        self.input_file = self.base_dir / "results_doc.csv"
        self.output_file = self.base_dir / "cluster_summary_v2.csv"
        self.output_folder = self.base_dir / "clusters_v2"
        self.report_file = self.base_dir / "analysis_report_v2.txt"

        logger.info(f"Initialized analyzer for project: {project_name}")
        logger.info(f"Base directory: {self.base_dir}")

    def load_and_validate_data(self) -> pd.DataFrame:
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        logger.info(f"Loading data from: {self.input_file}")
        df = pd.read_csv(self.input_file, encoding="utf-8")
        logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")

        # Validate required columns
        text_column = self._find_text_column(df)
        if not text_column:
            raise ValueError("No suitable text column found in the data")

        # Drop missing text
        missing_count = df[text_column].isna().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in text column; dropping")
            df = df.dropna(subset=[text_column])

        # Ensure Group column (fallback if needed)
        if "Group" not in df.columns:
            # jeśli nie ma, spróbuj alternatyw
            if "Cluster" in df.columns:
                df = df.rename(columns={"Cluster": "Group"})
            else:
                # Spróbuj wydedukować z drugiej kolumny
                if len(df.columns) > 1:
                    df = df.rename(columns={df.columns[1]: "Group"})
                else:
                    raise ValueError("No 'Group' column found in data")

        return df

    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        candidates = ["Paragraph", "Text", "Content", "text", "paragraph", "content"]
        for col in candidates:
            if col in df.columns:
                return col
        for col in df.columns:
            if df[col].dtype == "object":
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length and avg_length > 50:
                    return col
        return None

    def classify_content(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        logger.info("Classifying content types...")
        df["Type"] = df[text_column].apply(self.classifier.detect_type)
        type_counts = df["Type"].value_counts()
        logger.info("Classification results:")
        for content_type, count in type_counts.items():
            logger.info(f"  {content_type}: {count} paragraphs")
        return df

    def generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating cluster summary...")
        # Przyjmij pierwszą kolumnę tekstową jako sample
        sample_col = text_column = self._find_text_column(df)
        summary = (
            df.groupby(["Group", "Type"])
            .agg(
                Count=("Group", "size"),
                Sample_Text=(sample_col, lambda x: x.iloc[0] if len(x) > 0 else "")
            )
            .reset_index()
        )

        total_paragraphs = len(df)
        summary["Percentage"] = (summary["Count"] / total_paragraphs * 100).round(2)
        summary = summary.sort_values("Count", ascending=False)

        summary["Size_Category"] = pd.cut(
            summary["Count"],
            bins=[0, 1, 3, 10, 50, float("inf")],
            labels=["Unique", "Small", "Medium", "Large", "Very Large"]
        )
        return summary

    def export_clusters(self, df: pd.DataFrame, text_column: str) -> Dict[str, int]:
        logger.info("Exporting clusters to files...")
        self.output_folder.mkdir(exist_ok=True, parents=True)
        export_stats: Dict[str, int] = {}

        for content_type in df["Type"].unique():
            subset = df[df["Type"] == content_type].copy()
            if subset.empty:
                continue

            safe_type = str(content_type).replace("/", "_").replace(" ", "_").replace(".", "_")

            csv_path = self.output_folder / f"{safe_type}_clusters.csv"
            subset.to_csv(csv_path, index=False, encoding="utf-8")

            txt_path = self.output_folder / f"{safe_type}_clusters.txt"
            self._export_readable_text(subset, txt_path, text_column)

            stats_path = self.output_folder / f"{safe_type}_stats.txt"
            self._export_cluster_stats(subset, stats_path, content_type)

            export_stats[content_type] = len(subset)
            logger.info(f"Exported {len(subset)} paragraphs to {safe_type} files")

        return export_stats

    def _export_readable_text(self, subset: pd.DataFrame, file_path: Path, text_column: str):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Content Type: {subset['Type'].iloc[0]}\n")
            f.write(f"Total Paragraphs: {len(subset)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

            current_group = None
            for _, row in subset.iterrows():
                if row["Group"] != current_group:
                    current_group = row["Group"]
                    cluster_size = len(subset[subset["Group"] == current_group])
                    f.write(f"\nCLUSTER {current_group} ({cluster_size} paragraphs):\n")
                    f.write("-" * 40 + "\n")
                f.write(f"{row[text_column]}\n\n")

    def _export_cluster_stats(self, subset: pd.DataFrame, file_path: Path, content_type: str):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"CLUSTER STATISTICS: {content_type}\n")
            f.write("=" * 50 + "\n\n")

            cluster_sizes = subset.groupby("Group").size().sort_values(ascending=False)

            f.write(f"Total Paragraphs: {len(subset)}\n")
            f.write(f"Number of Clusters: {len(cluster_sizes)}\n")
            f.write(f"Average Cluster Size: {cluster_sizes.mean():.2f}\n")
            f.write(f"Largest Cluster: {cluster_sizes.max()} paragraphs\n")
            f.write(f"Smallest Cluster: {cluster_sizes.min()} paragraphs\n\n")

            f.write("Cluster Size Distribution:\n")
            f.write("-" * 25 + "\n")
            for group, size in cluster_sizes.items():
                f.write(f"Cluster {group}: {size} paragraphs\n")

    def generate_analysis_report(self, df: pd.DataFrame, summary: pd.DataFrame,
                                 export_stats: Dict[str, int]) -> str:
        report_content = []
        report_content.append("DOCUMENT SIMILARITY ANALYSIS REPORT")
        report_content.append("=" * 50)
        report_content.append(f"Project: {self.project_name}")
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"Total Paragraphs Analyzed: {len(df)}")
        report_content.append("")

        report_content.append("CONTENT TYPE BREAKDOWN")
        report_content.append("-" * 30)
        type_counts = df["Type"].value_counts()
        for content_type, count in type_counts.items():
            percentage = (count / len(df)) * 100
            report_content.append(f"{content_type}: {count} ({percentage:.1f}%)")
        report_content.append("")

        report_content.append("CLUSTER ANALYSIS")
        report_content.append("-" * 20)
        total_clusters = df["Group"].nunique()
        unique_content = len(df[df["Group"] == "UNIQUE"])
        similar_clusters = total_clusters - (1 if unique_content > 0 else 0)

        report_content.append(f"Total Unique Clusters: {total_clusters}")
        report_content.append(f"Similar Content Clusters: {similar_clusters}")
        report_content.append(f"Unique Content: {unique_content} paragraphs")
        report_content.append("")

        if similar_clusters > 0:
            clustered_content = len(df[df["Group"] != "UNIQUE"])
            similarity_rate = (clustered_content / len(df)) * 100
            report_content.append("SIMILARITY INSIGHTS")
            report_content.append("-" * 25)
            report_content.append(f"Content with Similarities: {similarity_rate:.1f}%")
            if similarity_rate > 30:
                report_content.append("High similarity detected - consider content optimization")
            elif similarity_rate > 15:
                report_content.append("Moderate similarity - review largest clusters")
            else:
                report_content.append("Low similarity - content appears well-optimized")
        report_content.append("")

        report_content.append("LARGEST SIMILARITY CLUSTERS")
        report_content.append("-" * 35)
        top_clusters = summary.head(10)
        for _, row in top_clusters.iterrows():
            if row["Group"] != "UNIQUE":
                report_content.append(f"Cluster {row['Group']} ({row['Type']}): {row['Count']} paragraphs")
        report_content.append("")

        report_content.append("EXPORTED FILES")
        report_content.append("-" * 18)
        for content_type, count in export_stats.items():
            report_content.append(f"{content_type}: {count} paragraphs")

        report_text = "\n".join(report_content)
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        return report_text

    def run_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            # Load and validate data
            df = self.load_and_validate_data()
            text_column = self._find_text_column(df)

            # Classify
            df = self.classify_content(df, text_column)

            # Summary
            summary = self.generate_summary(df)

            # Export
            summary.to_csv(self.output_file, index=False, encoding="utf-8")
            logger.info(f"Summary saved to: {self.output_file}")

            # Export clusters
            export_stats = self.export_clusters(df, text_column)

            # Report
            report = self.generate_analysis_report(df, summary, export_stats)
            logger.info(f"Analysis report saved to: {self.report_file}")

            # Display quick preview
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE - SUMMARY RESULTS")
            print("=" * 60)
            print(summary.head(15).to_string(index=False))

            print(f"\nFiles generated:")
            print(f"   • Summary: {self.output_file}")
            print(f"   • Clusters: {self.output_folder}/")
            print(f"   • Report: {self.report_file}")

            return df, summary

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

def main():
    """Main execution with argument parsing."""
    parser = argparse.ArgumentParser(description="Enhanced Document Cluster Analysis Tool")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--base-dir", "-d", help="Custom base directory (overrides default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        analyzer = ClusterAnalyzer(args.project, args.base_dir)
        analyzer.run_analysis()

        print("\nAnalysis completed successfully!")
        print("Next steps:")
        print("   1. Review the analysis report for insights")
        print("   2. Examine cluster files for detailed content")
        print("   3. Run analyze_results_v3.py for AI summarization")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Check that your input file exists and contains the expected columns")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
