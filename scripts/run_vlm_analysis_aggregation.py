#!/usr/bin/env python3
"""
VLM Analysis Aggregation Script
Aggregates and analyzes VLM analysis results for statistical insights.
"""

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, Any, List, Set
import statistics

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Common stop words and filler words to exclude from text analysis
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'undetectable', 'none', 'other', 'unknown', 'unclear', 'visible', 'not', 'no', 'yes'
}


class VLMAnalysisAggregator:
    """Aggregates VLM analysis results for statistical analysis."""
    
    def __init__(self, schema_path: str):
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load the VLM analysis schema."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema from {self.schema_path}: {e}")
            raise
            
    def find_analysis_files(self, vlm_analysis_path: Path, prompt_group: str = None) -> List[Path]:
        """Find all analysis JSON files, optionally filtered by prompt group."""
        analysis_files = []
        
        if prompt_group:
            # Look in specific prompt group directory
            prompt_dir = vlm_analysis_path / prompt_group
            if prompt_dir.exists():
                analysis_files.extend(prompt_dir.glob('*.json'))
        else:
            # Look in all prompt directories
            for prompt_dir in vlm_analysis_path.iterdir():
                if prompt_dir.is_dir() and prompt_dir.name.startswith('prompt_'):
                    analysis_files.extend(prompt_dir.glob('*.json'))
                    
        return sorted(analysis_files)
        
    def extract_meaningful_words(self, text: str) -> List[str]:
        """Extract meaningful words from text, excluding stop words."""
        if not isinstance(text, str):
            return []
            
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        meaningful_words = [
            word for word in words 
            if len(word) > 2 and word not in STOP_WORDS
        ]
        
        return meaningful_words
        
    def aggregate_field_data(self, field_name: str, field_def: Dict[str, Any], values: List[Any]) -> Dict[str, Any]:
        """Aggregate data for a specific field based on its type."""
        field_type = field_def.get("type", "open")
        
        # Filter out None/null values
        clean_values = [v for v in values if v is not None]
        
        if not clean_values:
            return {"type": field_type, "total_responses": 0, "data": {}}
            
        result = {
            "type": field_type,
            "total_responses": len(clean_values),
            "data": {}
        }
        
        if field_type == "options":
            # Count frequency of each option
            counter = Counter(clean_values)
            result["data"] = {
                "counts": dict(counter),
                "percentages": {option: (count / len(clean_values)) * 100 
                              for option, count in counter.items()}
            }
            
        elif field_type == "range":
            # Calculate statistics for numerical values
            numeric_values = []
            for v in clean_values:
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    continue
                    
            if numeric_values:
                result["data"] = {
                    "count": len(numeric_values),
                    "mean": statistics.mean(numeric_values),
                    "median": statistics.median(numeric_values),
                    "std_dev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
                    "min": min(numeric_values),
                    "max": max(numeric_values)
                }
            
        elif field_type == "open":
            # Extract and count meaningful words
            all_words = []
            
            for value in clean_values:
                if isinstance(value, str):
                    words = self.extract_meaningful_words(value)
                    all_words.extend(words)
                elif isinstance(value, list):
                    # Handle list of strings
                    for item in value:
                        if isinstance(item, str):
                            words = self.extract_meaningful_words(item)
                            all_words.extend(words)
                            
            word_counter = Counter(all_words)
            
            result["data"] = {
                "total_words": len(all_words),
                "unique_words": len(word_counter),
                "top_words": dict(word_counter.most_common(20)),
                "word_frequency": dict(word_counter)
            }
            
        return result
        
    def aggregate_object_data(self, obj_schema: Dict[str, Any], obj_data_list: List[Dict[str, Any]], context: str = "") -> Dict[str, Any]:
        """Aggregate data for a schema object (like composition, setting, etc.)."""
        results = {}
        
        for field_name, field_def in obj_schema.items():
            if not isinstance(field_def, dict):
                continue
                
            if "type" in field_def:
                # This is a field definition
                values = [obj.get(field_name) for obj in obj_data_list if isinstance(obj, dict)]
                field_context = f"{context}.{field_name}" if context else field_name
                results[field_name] = self.aggregate_field_data(field_name, field_def, values)
                
            else:
                # This is a nested object (like hair in appearance)
                nested_data_list = [obj.get(field_name, {}) for obj in obj_data_list if isinstance(obj, dict)]
                nested_context = f"{context}.{field_name}" if context else field_name
                results[field_name] = self.aggregate_object_data(field_def, nested_data_list, nested_context)
                
        return results
        
    def aggregate_people_data(self, people_data_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Aggregate data for the people array."""
        # Flatten all people from all files
        all_people = []
        for people_list in people_data_list:
            if isinstance(people_list, list):
                all_people.extend(people_list)
                
        if not all_people:
            return {"total_people": 0, "sections": {}}
            
        people_template = self.schema.get("people", [{}])[0] if self.schema.get("people") else {}
        
        results = {
            "total_people": len(all_people),
            "people_per_video": {
                "mean": len(all_people) / len(people_data_list) if people_data_list else 0,
                "distribution": Counter([len(people_list) if isinstance(people_list, list) else 0 
                                       for people_list in people_data_list])
            },
            "sections": {}
        }
        
        # Aggregate each section
        for section_name in ["demographics", "appearance", "role_and_agency"]:
            if section_name in people_template:
                section_data_list = [person.get(section_name, {}) for person in all_people]
                results["sections"][section_name] = self.aggregate_object_data(
                    people_template[section_name], 
                    section_data_list, 
                    f"people.{section_name}"
                )
                
        return results
        
    def aggregate_analysis_files(self, analysis_files: List[Path]) -> Dict[str, Any]:
        """Aggregate data from multiple analysis files."""
        logger.info(f"Aggregating {len(analysis_files)} analysis files...")
        
        # Load all analysis data
        all_analyses = []
        failed_files = []
        
        for file_path in analysis_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_analyses.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                failed_files.append(str(file_path))
                
        if not all_analyses:
            logger.error("No valid analysis files found")
            return {}
            
        logger.info(f"Successfully loaded {len(all_analyses)} files ({len(failed_files)} failed)")
        
        # Initialize results
        aggregated_results = {
            "metadata": {
                "total_files": len(analysis_files),
                "successfully_loaded": len(all_analyses),
                "failed_files": failed_files,
                "schema_version": self.schema.get("schema_version", "unknown")
            },
            "aggregated_data": {}
        }
        
        # Aggregate each main category
        main_categories = ["people", "composition", "setting", "cultural_flags", "overall_notes"]
        
        for category in main_categories:
            if category not in self.schema:
                continue
                
            logger.info(f"  Aggregating {category}...")
            
            if category == "people":
                # Special handling for people array
                people_data_list = [analysis.get("people", []) for analysis in all_analyses]
                aggregated_results["aggregated_data"][category] = self.aggregate_people_data(people_data_list)
            else:
                # Standard object aggregation
                category_data_list = [analysis.get(category, {}) for analysis in all_analyses]
                aggregated_results["aggregated_data"][category] = self.aggregate_object_data(
                    self.schema[category], 
                    category_data_list, 
                    category
                )
                
        return aggregated_results
        
    def save_aggregated_results(self, results: Dict[str, Any], output_path: Path):
        """Save aggregated results to JSON file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"✅ Aggregated results saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Aggregate VLM analysis results for statistical analysis")
    
    parser.add_argument(
        '--vlm-analysis-path',
        type=str,
        required=True,
        help='Path to vlm_analysis directory'
    )
    
    parser.add_argument(
        '--prompt-group',
        type=str,
        help='Specific prompt group to analyze (e.g., prompt_001). If not specified, analyzes all groups.'
    )
    
    parser.add_argument(
        '--schema-path',
        type=str,
        default='src/analysis/vlm_analysis/vlm_analysis_schema_new.json',
        help='Path to VLM analysis schema'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        help='Output path for aggregated results (defaults to vlm_analysis_path/aggregated_results.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Validate arguments
    vlm_analysis_path = Path(args.vlm_analysis_path)
    schema_path = Path(args.schema_path)
    
    if not vlm_analysis_path.exists():
        logger.error(f"VLM analysis directory not found: {vlm_analysis_path}")
        return 1
        
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return 1
        
    # Set output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        if args.prompt_group:
            output_path = vlm_analysis_path / f"aggregated_results_{args.prompt_group}.json"
        else:
            output_path = vlm_analysis_path / "aggregated_results_all.json"
    
    try:
        # Initialize aggregator
        logger.info(f"🔄 Initializing VLM Analysis Aggregator")
        logger.info(f"  Analysis directory: {vlm_analysis_path}")
        logger.info(f"  Schema: {schema_path}")
        logger.info(f"  Prompt group: {args.prompt_group or 'ALL'}")
        
        aggregator = VLMAnalysisAggregator(str(schema_path))
        
        # Find analysis files
        analysis_files = aggregator.find_analysis_files(vlm_analysis_path, args.prompt_group)
        
        if not analysis_files:
            logger.error("No analysis files found")
            return 1
            
        logger.info(f"📊 Found {len(analysis_files)} analysis files")
        
        # Aggregate results
        logger.info(f"🚀 Starting aggregation...")
        aggregated_results = aggregator.aggregate_analysis_files(analysis_files)
        
        # Save results
        aggregator.save_aggregated_results(aggregated_results, output_path)
        
        # Print summary
        logger.info(f"📈 Aggregation Summary:")
        metadata = aggregated_results.get("metadata", {})
        logger.info(f"  Files processed: {metadata.get('successfully_loaded', 0)}/{metadata.get('total_files', 0)}")
        
        aggregated_data = aggregated_results.get("aggregated_data", {})
        
        if "people" in aggregated_data:
            people_stats = aggregated_data["people"]
            logger.info(f"  Total people analyzed: {people_stats.get('total_people', 0)}")
            
        for category in ["composition", "setting", "cultural_flags"]:
            if category in aggregated_data:
                logger.info(f"  {category.capitalize()}: ✓")
                
        logger.info(f"✅ Aggregation completed successfully")
        logger.info(f"📄 Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
