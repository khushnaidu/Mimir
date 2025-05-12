import aiohttp
import asyncio
from typing import List, Dict

class NewsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = "5f127c1cd18842fd8380c946fd50ad63"
        self.base_url = "https://newsapi.org/v2"

    async def search_news(self, queries: List[str]) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            results = []
            for query in queries:
                params = {
                    'q': query,
                    'apiKey': self.api_key,
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 5
                }
                print("[DEBUG] NewsAPI query:", params['q'])
                async with session.get(f"{self.base_url}/everything", params=params) as response:
                    results.append(await response.json())
            return results

if __name__ == "__main__":
    import asyncio

    api_key = "5f127c1cd18842fd8380c946fd50ad63"  # Or load from env
    client = NewsAPIClient(api_key)

    async def test():
        queries = [
            "Kashmir",
            "India Pakistan conflict",
            "quantum computing",
            "Pope Leo XIV"
        ]
        results = await client.search_news(queries)
        for q, res in zip(queries, results):
            print(f"\nQuery: {q}")
            print(f"Total results: {res.get('totalResults')}")
            if res.get('articles'):
                for art in res['articles']:
                    print(" -", art.get('title'))
            else:
                print("No articles found.")

    asyncio.run(test())
