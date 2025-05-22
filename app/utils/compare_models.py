from Mimir.app.utils.evaluation import ModelEvaluator
import os
import matplotlib.pyplot as plt
import pandas as pd

def compare_specific_query_models(query_ids):
    """
    Compare specific evaluations by their query IDs and generate visualizations
    
    Args:
        query_ids: List of query IDs to compare
    """
    # Initialize the evaluator
    evaluator = ModelEvaluator()
    
    # Load all evaluations and filter to just the ones we want
    all_evals = evaluator.load_evaluations()
    specific_evals = [eval_data for eval_data in all_evals if eval_data.get("query_id") in query_ids]
    
    if not specific_evals:
        print("No matching evaluations found")
        return
        
    # Convert to DataFrame for easier analysis
    records = []
    for eval_data in specific_evals:
        # Basic metadata
        record = {
            "query_id": eval_data.get("query_id", "unknown"),
            "model": eval_data.get("model", "unknown"),
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
            
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Create visualizations
    create_comparison_visualizations(df)
    
def create_comparison_visualizations(df):
    """Create visualizations for comparing models on the same query"""
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison for Same Query", fontsize=16)
    
    # 1. Total processing time comparison
    plot_processing_time(df, axes[0, 0])
    
    # 2. Step-specific latency comparison
    plot_step_latency(df, axes[0, 1])
    
    # 3. Token usage comparison
    plot_token_usage(df, axes[1, 0])
    
    # 4. Output summary length comparison
    plot_output_summary(df, axes[1, 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
    
    # Save the visualization
    reports_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "app", 
        "data", 
        "evaluations", 
        "reports"
    )
    os.makedirs(reports_dir, exist_ok=True)
    
    save_path = os.path.join(
        reports_dir,
        f"same_query_comparison.png"
    )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")

def plot_processing_time(df, ax):
    """Plot total processing time comparison"""
    df.plot(kind='bar', x='model', y='total_process_time', ax=ax, color='skyblue')
    ax.set_title("Total Processing Time by Model")
    ax.set_ylabel("Time (seconds)")
    ax.set_xlabel("")
    
    # Add values as text
    for i, v in enumerate(df['total_process_time']):
        ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')

def plot_step_latency(df, ax):
    """Plot latency breakdown by processing step"""
    # Reshape data for stacked bar chart
    latency_data = []
    
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        row = {"Model": model}
        
        # Add latency for each step
        row["Reformatting"] = model_data["reformatting_latency"].iloc[0]
        row["Query Extraction"] = model_data["news_query_latency"].iloc[0]
        row["Summarization"] = model_data["summarization_latency"].iloc[0]
        
        latency_data.append(row)
    
    latency_df = pd.DataFrame(latency_data)
    latency_df.set_index("Model", inplace=True)
    
    latency_df.plot(kind="bar", stacked=True, ax=ax, colormap='viridis')
    ax.set_title("Latency by Processing Step")
    ax.set_ylabel("Time (seconds)")
    ax.legend(title="Processing Step")
    
    # Add total values as text
    for i, model in enumerate(latency_df.index):
        total = latency_df.loc[model].sum()
        ax.text(i, total + 0.1, f"Total: {total:.2f}s", ha='center')

def plot_token_usage(df, ax):
    """Plot token usage comparison"""
    # Reshape data for stacked bar chart
    token_data = []
    
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        row = {"Model": model}
        
        # Add token usage for each step
        row["Reformatting"] = model_data["reformatting_tokens"].iloc[0]
        row["Query Extraction"] = model_data["news_query_tokens"].iloc[0]
        row["Summarization"] = model_data["summarization_tokens"].iloc[0]
        
        token_data.append(row)
    
    token_df = pd.DataFrame(token_data)
    token_df.set_index("Model", inplace=True)
    
    token_df.plot(kind="bar", stacked=True, ax=ax, colormap='plasma')
    ax.set_title("Token Usage by Processing Step")
    ax.set_ylabel("Token Count")
    ax.legend(title="Processing Step")
    
    # Add total values as text
    for i, model in enumerate(token_df.index):
        total = token_df.loc[model].sum()
        ax.text(i, total + 100, f"Total: {int(total)}", ha='center')

def plot_output_summary(df, ax):
    """Plot a comparison of output summary lengths"""
    summary_lengths = []
    
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        query_id = model_data["query_id"].iloc[0]
        
        # Load the full evaluation to get the summary text
        evaluator = ModelEvaluator()
        all_evals = evaluator.load_evaluations()
        
        for eval_data in all_evals:
            if eval_data.get("query_id") == query_id and eval_data.get("model") == model:
                summary = eval_data.get("metrics", {}).get("summarization", {}).get("result", "")
                summary_lengths.append({"Model": model, "Length": len(summary)})
                break
    
    summary_df = pd.DataFrame(summary_lengths)
    
    summary_df.plot(kind='bar', x='Model', y='Length', ax=ax, color='lightgreen')
    ax.set_title("Output Summary Length (characters)")
    ax.set_ylabel("Character Count")
    ax.set_xlabel("")
    
    # Add values as text
    for i, row in enumerate(summary_df.itertuples()):
        ax.text(i, row.Length + 50, f"{row.Length}", ha='center')

if __name__ == "__main__":
    # The three query IDs we want to compare
    query_ids = [
        "852f982d-0abc-4e4e-ad11-1e664b2a8889",  # tinyllama-1.1b
        "69e33bc3-c436-482b-9001-cd64dd92977c",  # gpt-4
        "949345a8-484e-4d0c-8ee6-f39bfd7dbf47"   # gpt-3.5-turbo
    ]
    
    compare_specific_query_models(query_ids) 