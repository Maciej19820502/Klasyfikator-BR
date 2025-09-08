import pandas as pd
import os
import sys
import platform
from pathlib import Path

# ==============================
# CONFIGURATION - FIXED for cross-platform
# ==============================
PROJECT_NAME = "test_pl"  # Change to your project name

# FIXED: Cross-platform base directory
if platform.system() == "Windows":
    BASE_DIR = Path.home() / "Documents" / "DocumentAnalysis" / "Projects" / PROJECT_NAME
elif platform.system() == "Darwin":  # macOS
    BASE_DIR = Path.home() / "Documents" / "DocumentAnalysis" / "Projects" / PROJECT_NAME
else:  # Fallback for other systems
    BASE_DIR = Path.home() / "DocumentAnalysis" / "Projects" / PROJECT_NAME

# Input and output paths
INPUT_FILE = BASE_DIR / "results_doc.csv"
OUTPUT_FOLDER = BASE_DIR / "clusters"

# ==============================
# CLUSTER EXTRACTION FUNCTIONS
# ==============================

def validate_input_file(file_path):
    """Validate that the input CSV file exists and is readable."""
    if not file_path.exists():
        print(f"Error: Input file not found: {file_path}")
        print(f"Make sure you've run the main analysis first (run_document.py)")
        return False
    
    if file_path.stat().st_size == 0:
        print(f"Error: Input file is empty: {file_path}")
        return False
    
    return True

def load_and_validate_data(file_path):
    """Load CSV data and validate required columns."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        if df.empty:
            print("Error: CSV file is empty")
            return None
        
        # Identify text and group columns
        text_column = None
        group_column = None
        
        # Standard column names based on business case
        if 'Paragraph' in df.columns:
            text_column = 'Paragraph'
        elif 'Text' in df.columns:
            text_column = 'Text'
        else:
            # Use first column as text column
            text_column = df.columns[0]
            print(f"Warning: Using '{text_column}' as text column")
        
        if 'Group' in df.columns:
            group_column = 'Group'
        elif 'Cluster' in df.columns:
            group_column = 'Cluster'
        else:
            # Use second column as group column
            if len(df.columns) > 1:
                group_column = df.columns[1]
                print(f"Warning: Using '{group_column}' as group column")
            else:
                print("Error: No group column found")
                return None
        
        print(f"Loaded {len(df)} rows from {file_path}")
        print(f"Text column: '{text_column}'")
        print(f"Group column: '{group_column}'")
        
        return df, text_column, group_column
        
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty or corrupted")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return None
    except UnicodeDecodeError:
        print("Error: File encoding issue. Trying different encodings...")
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            print("Successfully loaded with latin-1 encoding")
        except:
            try:
                df = pd.read_csv(file_path, encoding='cp1250')
                print("Successfully loaded with cp1250 encoding")
            except Exception as e:
                print(f"Error: Could not read file with any encoding: {e}")
                return None
        return None
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return None

def extract_clusters(df, text_column, group_column, output_folder):
    """Extract clusters to separate files."""
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Group statistics
    group_stats = df[group_column].value_counts()
    unique_count = group_stats.get('UNIQUE', 0)
    cluster_count = len(group_stats) - (1 if 'UNIQUE' in group_stats else 0)
    
    print(f"\nCluster Statistics:")
    print(f"   • Total paragraphs: {len(df)}")
    print(f"   • Unique paragraphs: {unique_count}")
    print(f"   • Similar clusters: {cluster_count}")
    
    # Extract each cluster
    extracted_clusters = []
    
    for group_name, subset in df.groupby(group_column):
        if group_name == 'UNIQUE':
            print(f"Skipping {len(subset)} unique paragraphs")
            continue
        
        # File paths
        csv_path = output_folder / f"{group_name}.csv"
        txt_path = output_folder / f"{group_name}.txt"
        
        try:
            # Save CSV file
            subset.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Save TXT file with clean formatting
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"# Cluster: {group_name}\n")
                f.write(f"# Paragraphs: {len(subset)}\n")
                f.write("# " + "="*50 + "\n\n")
                
                for i, (_, row) in enumerate(subset.iterrows(), 1):
                    paragraph_text = str(row[text_column]).strip()
                    f.write(f"{i}. {paragraph_text}\n\n")
            
            extracted_clusters.append({
                'cluster': group_name,
                'paragraph_count': len(subset),
                'csv_file': csv_path.name,
                'txt_file': txt_path.name
            })
            
            print(f"Saved {len(subset)} paragraphs to {group_name}.txt")
            
        except Exception as e:
            print(f"Error saving cluster {group_name}: {e}")
            continue
    
    return extracted_clusters

def create_extraction_summary(extracted_clusters, output_folder):
    """Create a summary file of extracted clusters."""
    summary_path = output_folder / "extraction_summary.csv"
    
    if extracted_clusters:
        summary_df = pd.DataFrame(extracted_clusters)
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        print(f"Created extraction summary: {summary_path}")
        
        # Display summary
        print(f"\nExtraction Summary:")
        print(f"   • Clusters extracted: {len(extracted_clusters)}")
        print(f"   • Total similar paragraphs: {summary_df['paragraph_count'].sum()}")
        print(f"   • Average paragraphs per cluster: {summary_df['paragraph_count'].mean():.1f}")
        
        # Show largest clusters
        print(f"\nLargest Clusters:")
        top_clusters = summary_df.nlargest(5, 'paragraph_count')
        for _, row in top_clusters.iterrows():
            print(f"   • {row['cluster']}: {row['paragraph_count']} paragraphs")

def main():
    """Main execution function."""
    print("Document Similarity Analysis - Cluster Extractor")
    print("=" * 60)
    print(f"Project: {PROJECT_NAME}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    # Validate input file
    if not validate_input_file(INPUT_FILE):
        sys.exit(1)
    
    # Load and validate data
    result = load_and_validate_data(INPUT_FILE)
    if result is None:
        sys.exit(1)
    
    df, text_column, group_column = result
    
    # Extract clusters
    print(f"\nExtracting clusters...")
    extracted_clusters = extract_clusters(df, text_column, group_column, OUTPUT_FOLDER)
    
    # Create summary
    create_extraction_summary(extracted_clusters, OUTPUT_FOLDER)
    
    print(f"\nCluster extraction completed successfully!")
    print(f"All cluster files saved in: {OUTPUT_FOLDER}")
    print(f"\nNext steps:")
    print(f"   1. Review extracted clusters in the '{OUTPUT_FOLDER.name}' folder")
    print(f"   2. Run 'analyze_results_v2.py' to classify content types")
    print(f"   3. Run 'analyze_results_v3.py' to generate AI summaries")

if __name__ == "__main__":
    main()
