"""
LLM Model Evaluation and Comparison

This module provides tools for evaluating and comparing the performance of different
LLM models used in the Mimir political context analyzer.
"""
import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define evaluation metrics
METRICS = {
    "latency": {
        "name": "Response Time",
        "unit": "seconds",
        "description": "Time taken to generate a response",
        "lower_is_better": True
    },
    "token_count": {
        "name": "Token Usage",
        "unit": "tokens",
        "description": "Number of tokens used for the response",
        "lower_is_better": True
    },
    "user_rating": {
        "name": "User Rating",
        "unit": "stars (1-5)",
        "description": "User satisfaction rating",
        "lower_is_better": False
    }
}

class ModelEvaluator:
    """Class for evaluating and comparing model performance"""
    
    def __init__(self, data_dir=None):
        """Initialize the evaluator with a data directory"""
        if data_dir is None:
            # Default to evaluations directory in app/data
            self.data_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "data", 
                "evaluations"
            )
        else:
            self.data_dir = data_dir
            
        os.makedirs(self.data_dir, exist_ok=True)
        self.reports_dir = os.path.join(self.data_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def load_evaluations(self) -> List[Dict]:
        """Load all evaluation data from JSON files"""
        all_evaluations = []
        
        try:
            for filename in os.listdir(self.data_dir):
                if not filename.endswith('.json') or filename.startswith('report_'):
                    continue
                    
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        all_evaluations.append(data)
                except Exception as e:
                    logger.error(f"Error loading evaluation file {filename}: {str(e)}")
                    
            logger.info(f"Loaded {len(all_evaluations)} evaluation records")
            return all_evaluations
            
        except Exception as e:
            logger.error(f"Error loading evaluations: {str(e)}")
            return []
            
    def get_evaluation_dataframe(self) -> pd.DataFrame:
        """Convert evaluation data to a pandas DataFrame for analysis"""
        evaluations = self.load_evaluations()
        
        if not evaluations:
            return pd.DataFrame()
            
        records = []
        
        for eval_data in evaluations:
            # Basic metadata
            record = {
                "query_id": eval_data.get("query_id", "unknown"),
                "timestamp": eval_data.get("timestamp", ""),
                "model": eval_data.get("model", "unknown"),
                "input_text": eval_data.get("input_text", "")[:100],  # Truncate for readability
                "total_process_time": eval_data.get("total_process_time", 0)
            }
            
            # Extract metrics from each processing step
            metrics = eval_data.get("metrics", {})
            
            # Reformatting metrics
            if "reformatting" in metrics:
                record["reformatting_latency"] = metrics["reformatting"].get("latency", 0)
                record["reformatting_tokens"] = metrics["reformatting"].get("token_count", 0)
                
            # News query extraction metrics
            if "news_query_extraction" in metrics:
                record["news_query_latency"] = metrics["news_query_extraction"].get("latency", 0)
                record["news_query_tokens"] = metrics["news_query_extraction"].get("token_count", 0)
                
            # Summarization metrics
            if "summarization" in metrics:
                record["summarization_latency"] = metrics["summarization"].get("latency", 0)
                record["summarization_tokens"] = metrics["summarization"].get("token_count", 0)
                
            # User feedback if available
            feedback = eval_data.get("user_feedback", {})
            if feedback:
                record["user_rating"] = feedback.get("rating", None)
                record["user_comments"] = feedback.get("comments", "")
                
            records.append(record)
            
        return pd.DataFrame(records)
        
    def generate_model_comparison_report(self, save_path=None) -> Dict:
        """
        Generate a detailed model comparison report
        
        Args:
            save_path: Path to save the report visualizations
            
        Returns:
            Dictionary with report data
        """
        df = self.get_evaluation_dataframe()
        
        if df.empty:
            logger.warning("No evaluation data available for report generation")
            return {"error": "No evaluation data available"}
            
        # Group by model
        model_groups = df.groupby("model")
        
        # Basic statistics per model
        model_stats = {}
        for model_name, group in model_groups:
            model_stats[model_name] = {
                "count": len(group),
                "total_process_time": {
                    "mean": group["total_process_time"].mean(),
                    "std": group["total_process_time"].std(),
                    "min": group["total_process_time"].min(),
                    "max": group["total_process_time"].max()
                }
            }
            
            # Add stats for each processing step if available
            for metric in ["reformatting_latency", "news_query_latency", "summarization_latency"]:
                if metric in group.columns:
                    model_stats[model_name][metric] = {
                        "mean": group[metric].mean(),
                        "std": group[metric].std()
                    }
                    
            # Add token usage stats if available
            for metric in ["reformatting_tokens", "news_query_tokens", "summarization_tokens"]:
                if metric in group.columns and not group[metric].isna().all():
                    model_stats[model_name][metric] = {
                        "mean": group[metric].mean(),
                        "std": group[metric].std()
                    }
                    
            # Add user rating stats if available
            if "user_rating" in group.columns and not group["user_rating"].isna().all():
                valid_ratings = group["user_rating"].dropna()
                if len(valid_ratings) > 0:
                    model_stats[model_name]["user_rating"] = {
                        "mean": valid_ratings.mean(),
                        "std": valid_ratings.std(),
                        "count": len(valid_ratings)
                    }
        
        # Generate visualizations
        if save_path:
            try:
                self._create_visualization(df, save_path)
                logger.info(f"Saved visualizations to {save_path}")
            except Exception as e:
                logger.error(f"Error creating visualizations: {str(e)}")
        
        # Create summary report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(df),
            "models_evaluated": list(model_stats.keys()),
            "model_statistics": model_stats,
            "visualization_path": save_path
        }
        
        # Save the report to a JSON file
        report_filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(self.reports_dir, report_filename)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved report to {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
        
        return report
    
    def _create_visualization(self, df: pd.DataFrame, save_path: str):
        """Create visualization charts for the model comparison"""
        models = df["model"].unique()
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Comparison Analysis", fontsize=16)
        
        # 1. Total processing time comparison
        self._plot_processing_time(df, axes[0, 0])
        
        # 2. Step-specific latency comparison
        self._plot_step_latency(df, axes[0, 1])
        
        # 3. Token usage comparison (if available)
        if "summarization_tokens" in df.columns and not df["summarization_tokens"].isna().all():
            self._plot_token_usage(df, axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, "Token usage data not available", 
                           ha='center', va='center', fontsize=12)
        
        # 4. User ratings comparison (if available)
        if "user_rating" in df.columns and not df["user_rating"].isna().all():
            self._plot_user_ratings(df, axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, "User rating data not available", 
                           ha='center', va='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_processing_time(self, df: pd.DataFrame, ax):
        """Plot total processing time comparison"""
        df.boxplot(column="total_process_time", by="model", ax=ax)
        ax.set_title("Total Processing Time by Model")
        ax.set_ylabel("Time (seconds)")
        ax.set_xlabel("")
        
        # Add mean values as text
        for model in df["model"].unique():
            mean_time = df[df["model"] == model]["total_process_time"].mean()
            y_pos = df[df["model"] == model]["total_process_time"].max() * 1.1
            ax.text(list(df["model"].unique()).index(model) + 1, y_pos, 
                   f"Mean: {mean_time:.2f}s", ha='center')
    
    def _plot_step_latency(self, df: pd.DataFrame, ax):
        """Plot latency breakdown by processing step"""
        latency_cols = ["reformatting_latency", "news_query_latency", "summarization_latency"]
        latency_data = []
        
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            row = {"Model": model}
            
            for col in latency_cols:
                if col in model_data.columns:
                    row[col.replace("_latency", "")] = model_data[col].mean()
            
            latency_data.append(row)
        
        latency_df = pd.DataFrame(latency_data)
        latency_df.set_index("Model", inplace=True)
        
        latency_df.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Average Latency by Processing Step")
        ax.set_ylabel("Time (seconds)")
        ax.legend(title="Processing Step")
        
        # Add total values as text
        for i, model in enumerate(latency_df.index):
            total = latency_df.loc[model].sum()
            ax.text(i, total + 0.1, f"Total: {total:.2f}s", ha='center')
    
    def _plot_token_usage(self, df: pd.DataFrame, ax):
        """Plot token usage comparison"""
        token_cols = [col for col in df.columns if "tokens" in col]
        
        if not token_cols:
            ax.text(0.5, 0.5, "Token usage data not available", 
                   ha='center', va='center', fontsize=12)
            return
        
        token_data = []
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            row = {"Model": model}
            
            for col in token_cols:
                if col in model_data.columns and not model_data[col].isna().all():
                    step_name = col.replace("_tokens", "")
                    row[step_name] = model_data[col].mean()
            
            token_data.append(row)
        
        token_df = pd.DataFrame(token_data)
        
        if len(token_df) > 0 and "Model" in token_df.columns:
            token_df.set_index("Model", inplace=True)
            token_df.plot(kind="bar", ax=ax)
            ax.set_title("Average Token Usage by Model")
            ax.set_ylabel("Token Count")
            ax.legend(title="Processing Step")
        else:
            ax.text(0.5, 0.5, "Insufficient token usage data", 
                   ha='center', va='center', fontsize=12)
    
    def _plot_user_ratings(self, df: pd.DataFrame, ax):
        """Plot user ratings comparison"""
        if "user_rating" not in df.columns or df["user_rating"].isna().all():
            ax.text(0.5, 0.5, "User rating data not available", 
                   ha='center', va='center', fontsize=12)
            return
        
        rating_data = []
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            ratings = model_data["user_rating"].dropna()
            
            if len(ratings) > 0:
                # Count occurrences of each rating
                rating_counts = ratings.value_counts().sort_index()
                row = {"Model": model}
                
                for rating in range(1, 6):
                    row[f"{rating} Stars"] = rating_counts.get(rating, 0)
                
                rating_data.append(row)
        
        if rating_data:
            rating_df = pd.DataFrame(rating_data)
            rating_df.set_index("Model", inplace=True)
            rating_df.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("User Ratings by Model")
            ax.set_ylabel("Count")
            ax.legend(title="Rating")
            
            # Add average rating text
            for i, model in enumerate(rating_df.index):
                avg_rating = df[df["model"] == model]["user_rating"].mean()
                count = df[df["model"] == model]["user_rating"].count()
                ax.text(i, rating_df.loc[model].sum() + 0.5, 
                       f"Avg: {avg_rating:.1f}/5 (n={count})", ha='center')
        else:
            ax.text(0.5, 0.5, "Insufficient user rating data", 
                   ha='center', va='center', fontsize=12)
    
    def get_latest_report(self) -> Dict:
        """Get the most recent model comparison report"""
        report_files = [f for f in os.listdir(self.reports_dir) if f.startswith("model_comparison_")]
        
        if not report_files:
            return {"error": "No reports available"}
            
        # Sort by creation time (newest first)
        report_files.sort(reverse=True)
        latest_file = os.path.join(self.reports_dir, report_files[0])
        
        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading latest report: {str(e)}")
            return {"error": f"Error loading report: {str(e)}"}

# Helper functions for the REST API
def store_evaluation_data(data: Dict):
    """Store evaluation data to a JSON file"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluations")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, f"{data['query_id']}.json")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Stored evaluation data for query {data['query_id']}")

def update_evaluation_feedback(query_id: str, rating: int, comments: str = ""):
    """Update evaluation data with user feedback"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluations")
    file_path = os.path.join(data_dir, f"{query_id}.json")
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        data['user_feedback'] = {
            'rating': rating,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Updated feedback for query {query_id} with rating {rating}")
    else:
        logger.warning(f"Evaluation data not found for query ID: {query_id}")
        raise FileNotFoundError(f"Evaluation data not found for query ID: {query_id}")

def generate_model_comparison_reports():
    """Generate and return evaluation reports"""
    evaluator = ModelEvaluator()
    visualization_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "data", 
        "evaluations", 
        "reports",
        f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    report = evaluator.generate_model_comparison_report(save_path=visualization_path)
    return report

if __name__ == "__main__":
    """When run directly, generate and display the latest report"""
    evaluator = ModelEvaluator()
    
    # Set up the report path
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "data", 
        "evaluations",
        "reports"
    )
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(
        reports_dir,
        f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    # Generate the report
    report = evaluator.generate_model_comparison_report(save_path=report_path)
    
    if "error" in report:
        print(f"Error generating report: {report['error']}")
    else:
        print(f"Report generated successfully with {report['total_evaluations']} evaluations")
        print(f"Models evaluated: {', '.join(report['models_evaluated'])}")
        print(f"Visualization saved to: {report_path}")
        
        # Print summary statistics
        print("\nModel Comparison Summary:")
        print("=" * 50)
        
        for model, stats in report["model_statistics"].items():
            print(f"\nModel: {model}")
            print(f"Sample size: {stats['count']}")
            print(f"Average processing time: {stats['total_process_time']['mean']:.2f}s ± {stats['total_process_time']['std']:.2f}s")
            
            if "user_rating" in stats:
                print(f"Average user rating: {stats['user_rating']['mean']:.1f}/5 (n={stats['user_rating']['count']})")
            else:
                print("No user ratings available") 