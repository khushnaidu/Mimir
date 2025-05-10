from app.utils.data_acquisition import RedditDataFetcher
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Perform comprehensive data acquisition."""
    logger.info("Starting comprehensive data acquisition...")
    
    # Initialize fetcher
    fetcher = RedditDataFetcher()
    
    try:
        # 1. Fetch recent posts (using multi-timeframe approach)
        logger.info("Fetching recent posts with multi-timeframe approach...")
        recent_posts = fetcher.fetch_recent_posts()  # Using Reddit's native time filters and sorting
        logger.info(f"Fetched {len(recent_posts)} recent posts")
        
        # 2. Fetch topic-based archival posts
        logger.info("Fetching topic-based archival posts...")
        archival_posts = fetcher.fetch_topic_posts()
        logger.info(f"Fetched {len(archival_posts)} archival posts")
        
        # 3. Combine and deduplicate posts
        all_posts = recent_posts + archival_posts
        seen_ids = set()
        unique_posts = []
        
        for post in all_posts:
            if post['id'] not in seen_ids:
                unique_posts.append(post)
                seen_ids.add(post['id'])
        
        logger.info(f"Combined dataset contains {len(unique_posts)} unique posts")
        
        # 4. Save to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path('app/data') / f'reddit_political_posts_{timestamp}.json'
        fetcher.save_to_json(unique_posts, str(output_path))
        
        # 5. Print summary
        print("\n=== Data Acquisition Summary ===")
        print(f"\nTotal unique posts: {len(unique_posts)}")
        
        # Recent vs archival distribution
        recent_count = len([p for p in unique_posts if p['source'] == 'recent'])
        archive_count = len([p for p in unique_posts if p['source'] == 'archive'])
        print(f"\nPost sources:")
        print(f"Recent posts: {recent_count}")
        print(f"Archival posts: {archive_count}")
        
        # Time filter distribution for recent posts
        time_filter_counts = {}
        post_type_counts = {}
        for post in unique_posts:
            if post['source'] == 'recent':
                time_filter = post['time_filter']
                post_type = post['post_type']
                time_filter_counts[time_filter] = time_filter_counts.get(time_filter, 0) + 1
                post_type_counts[post_type] = post_type_counts.get(post_type, 0) + 1
        
        print("\nRecent posts by time filter:")
        for time_filter, count in sorted(time_filter_counts.items()):
            print(f"{time_filter}: {count} posts")
            
        print("\nRecent posts by type:")
        for post_type, count in sorted(post_type_counts.items()):
            print(f"{post_type}: {count} posts")
        
        # Subreddit distribution
        subreddit_counts = {}
        for post in unique_posts:
            subreddit = post['subreddit']
            subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
        
        print("\nPosts per subreddit:")
        for subreddit, count in sorted(subreddit_counts.items()):
            print(f"r/{subreddit}: {count} posts")
        
        # Topic distribution
        topic_counts = {}
        for post in unique_posts:
            if post['topic']:
                topic = post['topic']
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        print("\nPosts per topic:")
        for topic, count in sorted(topic_counts.items()):
            print(f"{topic}: {count} posts")
        
        # Date range
        dates = [datetime.fromtimestamp(post['created_utc']) for post in unique_posts]
        print(f"\nDate range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
        
        # Engagement metrics
        total_score = sum(post['score'] for post in unique_posts)
        total_comments = sum(post['num_comments'] for post in unique_posts)
        print(f"\nEngagement metrics:")
        print(f"Average score: {total_score/len(unique_posts):.2f}")
        print(f"Average comments: {total_comments/len(unique_posts):.2f}")
        
        print(f"\nData saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during data acquisition: {str(e)}")
        raise

if __name__ == '__main__':
    main() 