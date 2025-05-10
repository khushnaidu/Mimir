import os
import praw
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditDataFetcher:
    def __init__(self):
        """Initialize the Reddit data fetcher with PRAW."""
        load_dotenv()
        
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        self.subreddits = [
            'politics',
            'news',
            'conservative',
            'neoliberal',
            'moderatepolitics',
            'libertarian',
            'socialism'
        ]
        
        # Define topic mappings for archival search
        self.topics = {
            "abortion": ["abortion", "roe v wade", "planned parenthood"],
            "climate change": ["climate change", "global warming", "green energy"],
            "gun rights": ["guns", "gun control", "second amendment", "NRA"],
            "immigration": ["immigration", "border wall", "asylum", "ICE"],
            "elections": ["election fraud", "ballot", "voting machines"],
            "healthcare": ["medicare", "obamacare", "health insurance", "public option"],
            "taxes": ["income tax", "capital gains", "tax reform"],
            "foreign policy": ["ukraine", "nato", "china", "israel", "gaza", "taiwan"],
            "social justice": ["racism", "BLM", "DEI", "affirmative action", "police brutality"],
            "education": ["student loans", "critical race theory", "school choice"],
            "free speech": ["censorship", "twitter files", "content moderation"],
            "technology & AI": ["artificial intelligence", "surveillance", "facial recognition"]
        }
        
        # Define time windows (in days)
        self.time_windows = {
            'last_24h': 1,
            'last_week': 7,
            'last_2w': 14,
            'last_month': 30,
            'last_3m': 90,
            'last_6m': 180,
            'last_year': 365
        }
    
    def _get_time_window_dates(self, window_name: str) -> Tuple[datetime, datetime]:
        """
        Get start and end dates for a time window.
        
        Args:
            window_name (str): Name of the time window
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if window_name not in self.time_windows:
            raise ValueError(f"Invalid time window: {window_name}")
            
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.time_windows[window_name])
        return start_date, end_date
    
    def _filter_posts_by_time_window(self, posts: List, window_name: str, limit: int) -> List:
        """
        Filter posts to a specific time window.
        
        Args:
            posts (List): List of PRAW submission objects
            window_name (str): Name of the time window
            limit (int): Maximum number of posts to return
            
        Returns:
            List of posts within the time window
        """
        start_date, end_date = self._get_time_window_dates(window_name)
        filtered_posts = []
        
        for post in posts:
            post_date = datetime.fromtimestamp(post.created_utc)
            if start_date <= post_date <= end_date:
                filtered_posts.append(post)
                if len(filtered_posts) >= limit:
                    break
        
        return filtered_posts
    
    def fetch_topic_posts(self, topics: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
        """
        Fetch archival posts based on topic keywords.
        
        Args:
            topics (Optional[Dict[str, List[str]]]): Topic mapping to override default topics
            
        Returns:
            List of dictionaries containing post data
        """
        if topics is None:
            topics = self.topics
            
        seen_post_ids: Set[str] = set()
        all_posts = []
        
        for topic, keywords in topics.items():
            logger.info(f"Fetching posts for topic: {topic}")
            
            for subreddit_name in self.subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for keyword in keywords:
                    try:
                        for post in subreddit.search(keyword, sort='top', time_filter='all', limit=20):
                            if post.id not in seen_post_ids:
                                post_data = self._process_post(post)
                                post_data.update({
                                    'source': 'archive',
                                    'topic': topic,
                                    'search_keyword': keyword
                                })
                                all_posts.append(post_data)
                                seen_post_ids.add(post.id)
                    except Exception as e:
                        logger.error(f"Error fetching posts for topic '{topic}', keyword '{keyword}' in r/{subreddit_name}: {str(e)}")
        
        logger.info(f"Fetched total of {len(all_posts)} unique archival posts")
        return all_posts
    
    def fetch_recent_posts(self) -> List[Dict]:
        """
        Fetch diverse recent posts from configured subreddits.
        
        Strategy:
        1. For each subreddit:
           - Get top posts using Reddit's native time filters (day, week, month, year, all)
           - Get controversial posts (week: 100, year: 200)
           - Get hot, rising, and new posts (50 each)
        2. Remove duplicates
        3. Ensure temporal diversity
        
        Returns:
            List of dictionaries containing post data
        """
        seen_post_ids: Set[str] = set()
        all_posts = []
        
        # Define time filters and their post limits for top posts
        top_time_filters = {
            'day': 100,
            'week': 100,
            'month': 100,
            'year': 100,
            'all': 100
        }
        
        for subreddit_name in self.subreddits:
            logger.info(f"Fetching posts from r/{subreddit_name}")
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # 1. Fetch top posts using Reddit's native time filters
            for time_filter, limit in top_time_filters.items():
                try:
                    logger.info(f"Fetching top posts from r/{subreddit_name} for {time_filter}")
                    for post in subreddit.top(time_filter=time_filter, limit=limit):
                        if post.id not in seen_post_ids:
                            post_data = self._process_post(post)
                            post_data.update({
                                'source': 'recent',
                                'topic': None,
                                'post_type': 'top',
                                'time_filter': time_filter
                            })
                            all_posts.append(post_data)
                            seen_post_ids.add(post.id)
                except Exception as e:
                    logger.error(f"Error fetching top posts from r/{subreddit_name} ({time_filter}): {str(e)}")
            
            # 2. Fetch controversial posts
            try:
                # Get controversial posts from past week
                logger.info(f"Fetching controversial posts from r/{subreddit_name} (week)")
                for post in subreddit.controversial(time_filter='week', limit=100):
                    if post.id not in seen_post_ids:
                        post_data = self._process_post(post)
                        post_data.update({
                            'source': 'recent',
                            'topic': None,
                            'post_type': 'controversial',
                            'time_filter': 'week'
                        })
                        all_posts.append(post_data)
                        seen_post_ids.add(post.id)
                
                # Get controversial posts from past year
                logger.info(f"Fetching controversial posts from r/{subreddit_name} (year)")
                for post in subreddit.controversial(time_filter='year', limit=200):
                    if post.id not in seen_post_ids:
                        post_data = self._process_post(post)
                        post_data.update({
                            'source': 'recent',
                            'topic': None,
                            'post_type': 'controversial',
                            'time_filter': 'year'
                        })
                        all_posts.append(post_data)
                        seen_post_ids.add(post.id)
            except Exception as e:
                logger.error(f"Error fetching controversial posts from r/{subreddit_name}: {str(e)}")
            
            # 3. Fetch other post types (current only)
            post_types = {
                'hot': lambda sr: sr.hot(limit=50),
                'rising': lambda sr: sr.rising(limit=50),
                'new': lambda sr: sr.new(limit=50)
            }
            
            for post_type, fetch_func in post_types.items():
                try:
                    logger.info(f"Fetching {post_type} posts from r/{subreddit_name}")
                    for post in fetch_func(subreddit):
                        if post.id not in seen_post_ids:
                            post_data = self._process_post(post)
                            post_data.update({
                                'source': 'recent',
                                'topic': None,
                                'post_type': post_type,
                                'time_filter': 'current'  # These are current posts
                            })
                            all_posts.append(post_data)
                            seen_post_ids.add(post.id)
                except Exception as e:
                    logger.error(f"Error fetching {post_type} posts from r/{subreddit_name}: {str(e)}")
        
        # Sort posts by score to ensure quality
        all_posts.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Fetched total of {len(all_posts)} unique recent posts")
        return all_posts
    
    def _process_post(self, post) -> Dict:
        """
        Process a Reddit post into a standardized format.
        
        Args:
            post: PRAW submission object
            
        Returns:
            Dictionary containing processed post data
        """
        # Get top comments
        post.comments.replace_more(limit=0)
        top_comments = []
        for comment in post.comments.list()[:5]:  # Get top 5 comments
            top_comments.append({
                'author': str(comment.author),
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc
            })
        
        return {
            'id': post.id,
            'title': post.title,
            'text': post.selftext,
            'url': post.url,
            'author': str(post.author),
            'score': post.score,
            'upvote_ratio': post.upvote_ratio,
            'num_comments': post.num_comments,
            'created_utc': post.created_utc,
            'subreddit': post.subreddit.display_name,
            'top_comments': top_comments,
            'flair': post.link_flair_text,
            'is_original_content': post.is_original_content,
            'is_self': post.is_self,
            'permalink': post.permalink
        }
    
    def save_to_json(self, posts: List[Dict], output_path: str) -> None:
        """
        Save posts to a JSON file with metadata.
        
        Args:
            posts (List[Dict]): List of post dictionaries
            output_path (str): Path to save the JSON file
        """
        # Prepare metadata
        metadata = {
            'total_posts': len(posts),
            'subreddits': list(set(post['subreddit'] for post in posts)),
            'date_range': {
                'earliest': min(post['created_utc'] for post in posts),
                'latest': max(post['created_utc'] for post in posts)
            },
            'generated_at': datetime.now().isoformat(),
            'post_types': {
                'recent': len([p for p in posts if p['source'] == 'recent']),
                'archive': len([p for p in posts if p['source'] == 'archive'])
            },
            'topics': list(set(p['topic'] for p in posts if p['topic']))
        }
        
        # Calculate engagement metrics
        total_score = sum(post['score'] for post in posts)
        total_comments = sum(post['num_comments'] for post in posts)
        metadata['engagement'] = {
            'avg_score': total_score / len(posts),
            'avg_comments': total_comments / len(posts),
            'total_score': total_score,
            'total_comments': total_comments
        }
        
        data = {
            'posts': posts,
            'metadata': metadata
        }
        
        # Create directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(posts)} posts to {output_path}")
    
    def fetch_post_comments(self, post_id: str) -> List[Dict]:
        """
        Fetch all comments for a specific post.
        
        Args:
            post_id (str): The ID of the post
            
        Returns:
            List of dictionaries containing comment data
        """
        post = self.reddit.submission(id=post_id)
        post.comments.replace_more(limit=None)  # Get all comments
        
        comments = []
        for comment in post.comments.list():
            comments.append({
                'id': comment.id,
                'author': str(comment.author),
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc,
                'parent_id': comment.parent_id
            })
        
        return comments 