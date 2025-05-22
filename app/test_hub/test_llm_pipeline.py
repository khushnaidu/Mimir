from llm_pipeline import query_reformatter, news_query_extractor, context_summarizer

if __name__ == "__main__":
    # Example text for reformatting and extraction
    text = "The US election and climate change policy in 2024. Candidates are debating aggressive action versus economic caution."
    print("\n--- Query Reformatter ---")
    print(query_reformatter(text))

    print("\n--- News Query Extractor ---")
    print(news_query_extractor(text))

    # Example context for summarization
    context = (
        "Reddit users are debating the impact of the 2024 US election on climate policy. "
        "Some believe strong action will be taken, while others are skeptical. "
        "News articles highlight both the promises made by candidates and the challenges ahead."
    )
    print("\n--- Context Summarizer ---")
    print(context_summarizer(context))
