import os
import re
import csv
import json
import logging
import platform
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI

# -------------------------
# Konfiguracja logowania
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Domyślne ustawienia
# -------------------------
DEFAULT_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-4o-mini"
MAX_PARAGRAPHS_FOR_SUMMARY = 10


# -------------------------
# Funkcje pomocnicze
# -------------------------
def load_data(project_name: str) -> pd.DataFrame:
    """Wczytaj dane CSV z wcześniejszej analizy (results_doc.csv)."""
    if platform.system() == "Windows":
        base_dir = Path.home() / "Documents" / "DocumentAnalysis" / "Projects" / project_name
    elif platform.system() == "Darwin":  # macOS
        base_dir = Path.home() / "Documents" / "DocumentAnalysis" / "Projects" / project_name
    else:
        base_dir = Path.home() / "DocumentAnalysis" / "Projects" / project_name

    input_file = base_dir / "results_doc.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego: {input_file}")

    df = pd.read_csv(input_file)
    logger.info(f"Wczytano {len(df)} wierszy z {input_file}")
    return df, base_dir


def detect_language(text: str) -> str:
    """Prosta detekcja języka (PL/EN)."""
    polish_chars = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    polish_count = sum(1 for c in text if c in polish_chars)
    return "pl" if polish_count > len(text) * 0.01 else "en"


def summarize_cluster(
    client: Optional[OpenAI],
    paragraphs: List[str],
    group_name: str,
    model: str,
    fallback: str,
    max_paragraphs: int
) -> str:
    """Wygeneruj AI-podsumowanie klastra podobnych akapitów."""
    if not client:
        sample = paragraphs[:3] if len(paragraphs) > 3 else paragraphs
        return (
            f"⚠️ Offline mode (brak API key). "
            f"Przykładowe akapity:\n" + "\n".join(sample)
        )

    sample_paragraphs = paragraphs[:max_paragraphs]
    combined_text = " ".join(sample_paragraphs)
    language = detect_language(combined_text)
    lang_instruction = "Odpowiadaj po polsku." if language == "pl" else "Respond in English."

    prompt = f"""
You are analyzing a document for duplicate or repetitive content.
{lang_instruction}

These paragraphs were identified as belonging to the same cluster: {group_name}
Total paragraphs in cluster: {len(paragraphs)}

Example paragraphs:
{chr(10).join(sample_paragraphs)}

Write a short 2–4 sentence summary that explains:
1. The main topic or theme
2. Why these paragraphs are considered similar
3. What type of information is being repeated

Summary:
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Błąd w modelu {model}: {e}, próba fallback {fallback}")
        try:
            response = client.chat.completions.create(
                model=fallback,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e2:
            logger.error(f"Podsumowanie nieudane: {e2}")
            return f"❌ Summary unavailable: {e2}"


# -------------------------
# Główna analiza
# -------------------------
def run_analysis(project: str, model: str, fallback: str, max_paragraphs: int):
    df, base_dir = load_data(project)

    # Połącz akapity wg grup
    grouped: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        group = str(row["Group"])
        text = str(row["Paragraph"])
        grouped.setdefault(group, []).append(text)

    logger.info(f"Znaleziono {len(grouped)} grup (klastrów).")

    # Inicjalizacja OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None

    summaries: Dict[str, str] = {}
    for g, paragraphs in grouped.items():
        logger.info(f"Generowanie podsumowania dla grupy {g} ({len(paragraphs)} akapitów)...")
        summaries[g] = summarize_cluster(client, paragraphs, g, model, fallback, max_paragraphs)

    # Eksport wyników
    output_file = base_dir / "cluster_summary_v3.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Group", "Summary"])
        for g, summary in summaries.items():
            writer.writerow([g, summary])

    json_file = base_dir / "cluster_summary_v3.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    report_file = base_dir / "analysis_report_v3.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("DOCUMENT SIMILARITY - AI SUMMARIES\n")
        f.write("=" * 50 + "\n")
        f.write(f"Project: {project}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for g, summary in summaries.items():
            f.write(f"--- {g} ---\n{summary}\n\n")

    print("\n============================================================")
    print("AI SUMMARIZATION COMPLETE")
    print("============================================================")
    print(f"CSV:   {output_file}")
    print(f"JSON:  {json_file}")
    print(f"Report:{report_file}")
    print("============================================================")

    return summaries


# -------------------------
# Uruchomienie z linii poleceń
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="AI-powered cluster summarization (v3)")
    parser.add_argument("--project", "-p", required=True, help="Project name (must exist in Projects folder)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model OpenAI (default: {DEFAULT_MODEL})")
    parser.add_argument("--fallback", default=FALLBACK_MODEL, help=f"Fallback model (default: {FALLBACK_MODEL})")
    parser.add_argument("--max-paragraphs", type=int, default=MAX_PARAGRAPHS_FOR_SUMMARY,
                        help=f"Maksymalna liczba akapitów do podsumowania (default: {MAX_PARAGRAPHS_FOR_SUMMARY})")
    args = parser.parse_args()

    run_analysis(args.project, args.model, args.fallback, args.max_paragraphs)


if __name__ == "__main__":
    main()
