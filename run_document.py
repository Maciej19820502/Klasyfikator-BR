#!/usr/bin/env python3
"""
Main execution script for document similarity analysis
Processes documents and generates multiple output formats
COMPLETED VERSION for Windows and macOS
"""

import os
import sys
import argparse
import time
import platform
from pathlib import Path
from datetime import datetime
import json
import logging

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from similarity_tool import (
        process_document, save_to_docx, save_to_csv, save_to_json,
        get_project_directory, logger, config
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure similarity_tool.py is in the same directory")
    sys.exit(1)

# ==============================
# CONFIGURATION
# ==============================
class ProjectConfig:
    """Project-specific configuration for Windows and macOS"""
    
    def __init__(self):
        # Model selection
        # "gpt-4o"      → High quality, best for Polish language
        # "gpt-4o-mini" → Budget option for testing
        self.MODEL_NAME = "gpt-4o"
        
        # Project settings - THESE SHOULD BE CHANGED BY USER
        self.PROJECT_NAME = "my_document_analysis"   # ← Change to your project name
        self.PROJECT_FILE = "document.pdf"          # ← Change to your file name
        
        # Advanced settings
        self.EPS = 0.3              # Similarity threshold (0.1-1.0, lower = stricter)
        self.MIN_SAMPLES = 2        # Minimum cluster size
        self.CHUNK_SIZE = 20        # Paragraphs per processing batch
        
        # Output options
        self.GENERATE_DOCX = True   # Create Word document with analysis
        self.GENERATE_CSV = True    # Create CSV with detailed data
        self.GENERATE_JSON = True   # Create JSON for further processing
        
        # Platform detection
        self.SYSTEM = platform.system()
        self.IS_WINDOWS = self.SYSTEM == "Windows"
        self.IS_MACOS = self.SYSTEM == "Darwin"
        
        # Validate platform
        if self.SYSTEM not in ["Windows", "Darwin"]:
            raise RuntimeError(f"❌ Unsupported platform: {self.SYSTEM}. This tool supports Windows and macOS only.")
        
        # Setup project directory
        self.project_dir = get_project_directory(self.PROJECT_NAME)
        self.input_file = self.project_dir / self.PROJECT_FILE

# Create global config instance
project_config = ProjectConfig()

# ==============================
# UTILITY FUNCTIONS
# ==============================
def validate_environment() -> bool:
    """Validate that the environment is ready for analysis"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check if input file exists
    if not project_config.input_file.exists():
        # Try to find any document in project directory
        found_files = []
        for ext in ['.pdf', '.docx', '.txt']:
            found_files.extend(project_config.project_dir.glob(f"*{ext}"))
        
        if found_files:
            # Use first found file
            project_config.input_file = found_files[0]
            project_config.PROJECT_FILE = found_files[0].name
            print(f"📄 Using found document: {project_config.PROJECT_FILE}")
        else:
            issues.append(f"No document found in {project_config.project_dir}")
            issues.append("Place your document (.pdf, .docx, .txt) in the project directory")
    
    # Check write permissions
    try:
        test_file = project_config.project_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception:
        issues.append(f"Cannot write to project directory: {project_config.project_dir}")
    
    if issues:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    return True

def estimate_processing_time(file_path: Path) -> dict:
    """Estimate processing time and requirements"""
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Rough estimates based on file size
        estimated_pages = max(1, int(file_size_mb * 20))  # ~50KB per page
        estimated_paragraphs = estimated_pages * 10
        estimated_time_minutes = max(1, estimated_paragraphs // 100)  # ~100 paragraphs per minute
        
        return {
            "file_size_mb": round(file_size_mb, 2),
            "estimated_pages": estimated_pages,
            "estimated_paragraphs": estimated_paragraphs,
            "estimated_time_minutes": estimated_time_minutes
        }
    except Exception:
        return {"error": "Could not estimate processing time"}

def create_output_files(annotated_paragraphs: list, summaries: dict) -> dict:
    """Create all output files and return their paths"""
    output_files = {}
    base_name = project_config.PROJECT_FILE.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
    
    try:
        # CSV output (always generated for further analysis)
        if project_config.GENERATE_CSV:
            csv_file = project_config.project_dir / "results_doc.csv"
            save_to_csv(annotated_paragraphs, str(csv_file))
            output_files["csv"] = csv_file
            logger.info(f"📊 CSV saved: {csv_file}")
        
        # DOCX output (human-readable report)
        if project_config.GENERATE_DOCX:
            docx_file = project_config.project_dir / f"{base_name}_analysis.docx"
            save_to_docx(annotated_paragraphs, summaries, str(docx_file))
            output_files["docx"] = docx_file
            logger.info(f"📄 DOCX saved: {docx_file}")
        
        # JSON output (machine-readable data)
        if project_config.GENERATE_JSON:
            json_file = project_config.project_dir / f"{base_name}_analysis.json"
            save_to_json(annotated_paragraphs, summaries, str(json_file))
            output_files["json"] = json_file
            logger.info(f"📋 JSON saved: {json_file}")
        
    except Exception as e:
        logger.error(f"Error creating output files: {e}")
        print(f"⚠️ Some output files may be incomplete: {e}")
    
    return output_files

def generate_analysis_report(annotated_paragraphs: list, summaries: dict, 
                           processing_time: float, output_files: dict) -> str:
    """Generate comprehensive analysis report"""
    total_paragraphs = len(annotated_paragraphs)
    unique_paragraphs = sum(1 for p in annotated_paragraphs if not p.startswith("[SIMILAR-"))
    similar_clusters = len(summaries)
    similar_paragraphs = total_paragraphs - unique_paragraphs
    
    similarity_rate = (similar_paragraphs / total_paragraphs * 100) if total_paragraphs > 0 else 0
    
    # Create detailed report
    report_lines = [
        "="*60,
        "📊 DOCUMENT SIMILARITY ANALYSIS REPORT",
        "="*60,
        f"🖥️  System: {project_config.SYSTEM}",
        f"📁 Project: {project_config.PROJECT_NAME}",
        f"📄 Document: {project_config.PROJECT_FILE}",
        f"⏱️  Processing time: {processing_time:.1f} seconds",
        f"📅 Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "📈 STATISTICS:",
        f"   • Total paragraphs: {total_paragraphs}",
        f"   • Similar clusters: {similar_clusters}",
        f"   • Similar paragraphs: {similar_paragraphs}",
        f"   • Unique paragraphs: {unique_paragraphs}",
        f"   • Similarity rate: {similarity_rate:.1f}%",
        ""
    ]
    
    # Add interpretation
    report_lines.append("🎯 INTERPRETATION:")
    if similarity_rate > 30:
        report_lines.append("   ⚠️ High similarity detected - consider content optimization")
        report_lines.append("   📝 Review largest clusters for potential consolidation")
    elif similarity_rate > 15:
        report_lines.append("   ℹ️ Moderate similarity - review largest clusters")
        report_lines.append("   📝 Some content optimization opportunities exist")
    else:
        report_lines.append("   ✅ Low similarity - content appears well-optimized")
        report_lines.append("   📝 Document shows good content diversity")
    
    report_lines.extend([
        "",
        "📁 OUTPUT FILES:",
    ])
    
    for file_type, file_path in output_files.items():
        report_lines.append(f"   • {file_type.upper()}: {file_path.name}")
    
    report_lines.extend([
        "",
        f"📂 All files saved to: {project_config.project_dir}",
        "",
        "🔍 NEXT STEPS:",
        "   1. Review results_doc.csv for detailed paragraph analysis",
        "   2. Open the analysis report (DOCX) for human-readable summary",
        "   3. Run analyze_results_v2.py for content type classification",
        "   4. Run analyze_results_v3.py for AI-powered cluster summaries",
        "",
        "💡 For additional analysis:",
        "   • Use extract_clusters.py to separate similar content",
        "   • Use extract_examples.py to get representative samples",
        "="*60
    ])
    
    return "\n".join(report_lines)

def open_results_folder():
    """Open the results folder in the system file explorer"""
    try:
        if project_config.IS_WINDOWS:
            os.startfile(project_config.project_dir)
        elif project_config.IS_MACOS:
            os.system(f'open "{project_config.project_dir}"')
        
        print(f"📁 Opened results folder: {project_config.project_dir}")
    except Exception as e:
        logger.warning(f"Could not open folder: {e}")
        print(f"📁 Results saved to: {project_config.project_dir}")

def save_analysis_report(report_content: str):
    """Save the analysis report to a text file"""
    try:
        report_file = project_config.project_dir / "analysis_report.txt"
        report_file.write_text(report_content, encoding='utf-8')
        logger.info(f"📋 Analysis report saved: {report_file}")
    except Exception as e:
        logger.warning(f"Could not save analysis report: {e}")

# ==============================
# MAIN EXECUTION FUNCTION
# ==============================
def main():
    """Main execution function with comprehensive error handling"""
    parser = argparse.ArgumentParser(
        description="Document Similarity Analysis Tool - Main Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_document.py
  python run_document.py --project my_thesis --file thesis.pdf
  python run_document.py --eps 0.4 --min-samples 3
  python run_document.py --no-docx --no-json
        """
    )
    
    # Command line arguments
    parser.add_argument("--project", "-p", type=str,
                       help=f"Project name (default: {project_config.PROJECT_NAME})")
    
    parser.add_argument("--file", "-f", type=str,
                       help=f"Input document file (default: auto-detect)")
    
    parser.add_argument("--eps", type=float, 
                       help=f"Similarity threshold (default: {project_config.EPS})")
    
    parser.add_argument("--min-samples", type=int,
                       help=f"Minimum cluster size (default: {project_config.MIN_SAMPLES})")
    
    parser.add_argument("--model", type=str, choices=["gpt-4o", "gpt-4o-mini"],
                       help=f"OpenAI model (default: {project_config.MODEL_NAME})")
    
    parser.add_argument("--no-docx", action="store_true",
                       help="Skip DOCX report generation")
    
    parser.add_argument("--no-json", action="store_true", 
                       help="Skip JSON export")
    
    parser.add_argument("--open-folder", action="store_true",
                       help="Open results folder after analysis")
    
    parser.add_argument("--estimate-only", action="store_true",
                       help="Only show processing estimates, don't run analysis")
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.project:
        project_config.PROJECT_NAME = args.project
        project_config.project_dir = get_project_directory(args.project)
    
    if args.file:
        project_config.PROJECT_FILE = args.file
        project_config.input_file = project_config.project_dir / args.file
    
    if args.eps:
        project_config.EPS = args.eps
    
    if args.min_samples:
        project_config.MIN_SAMPLES = args.min_samples
    
    if args.model:
        project_config.MODEL_NAME = args.model
    
    if args.no_docx:
        project_config.GENERATE_DOCX = False
    
    if args.no_json:
        project_config.GENERATE_JSON = False
    
    try:
        print("🚀 Document Similarity Analysis Tool")
        print("="*50)
        print(f"🖥️  System: {project_config.SYSTEM}")
        print(f"📊 Project: {project_config.PROJECT_NAME}")
        print(f"📁 Directory: {project_config.project_dir}")
        print(f"📄 Document: {project_config.PROJECT_FILE}")
        
        # Validate environment
        print("\n🔍 Validating environment...")
        if not validate_environment():
            return 1
        
        print("✅ Environment validation passed")
        
        # Show processing estimates
        estimates = estimate_processing_time(project_config.input_file)
        if "error" not in estimates:
            print(f"\n📈 Processing estimates:")
            print(f"   • File size: {estimates['file_size_mb']} MB")
            print(f"   • Estimated paragraphs: {estimates['estimated_paragraphs']}")
            print(f"   • Estimated time: {estimates['estimated_time_minutes']} minutes")
            
            if args.estimate_only:
                return 0
            
            # Confirm for large files
            if estimates['estimated_time_minutes'] > 10:
                response = input(f"\n⚠️ Large file detected. Continue? (y/n): ").lower().strip()
                if response not in ['y', 'yes']:
                    print("Analysis cancelled")
                    return 0
        
        # Start analysis
        print(f"\n🔄 Starting analysis...")
        print(f"   • Similarity threshold: {project_config.EPS}")
        print(f"   • Minimum cluster size: {project_config.MIN_SAMPLES}")
        print(f"   • OpenAI model: {project_config.MODEL_NAME}")
        
        start_time = time.time()
        
        # Run document processing
        annotated_paragraphs, summaries = process_document(
            str(project_config.input_file),
            model_name=project_config.MODEL_NAME,
            eps=project_config.EPS,
            min_samples=project_config.MIN_SAMPLES,
            chunk_size=project_config.CHUNK_SIZE
        )
        
        processing_time = time.time() - start_time
        
        if not annotated_paragraphs:
            print("❌ No content found in document")
            return 1
        
        # Generate output files
        print("\n💾 Creating output files...")
        output_files = create_output_files(annotated_paragraphs, summaries)
        
        # Generate and display analysis report
        report_content = generate_analysis_report(
            annotated_paragraphs, summaries, processing_time, output_files
        )
        
        print(f"\n{report_content}")
        
        # Save report to file
        save_analysis_report(report_content)
        
        # Open results folder if requested
        if args.open_folder:
            open_results_folder()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results available in: {project_config.project_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")
        print(f"💡 Check the logs in: {project_config.project_dir / 'logs'}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
