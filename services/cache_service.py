import os
import httpx

class CacheService:
    """Service for interacting with the semantic cache API."""

    def __init__(self):
        self.cache_host = os.getenv("CACHE_API_HOST", "shomadhan.shafinhasnat.me")
        self.cache_port = os.getenv("CACHE_API_PORT", "")
        if self.cache_port:
            self.base_url = f"https://{self.cache_host}:{self.cache_port}"
        else:
            self.base_url = f"https://{self.cache_host}"

    async def get_cached_response(self, question: str, location: str = "") -> dict:
        """
        Check if cache is available for the question/location and return the full response payload if found, else None.
        """
        payload = {"question": question, "location": ""}
        print(f"Cache payload: {payload}")
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.base_url}/api/ask", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                print(f"*****Cache response: {data}")
                # If the response is in the new format (with 'results'), return as is
                if data.get("results"):
                    return data
        return None

    async def enrich(self, question: str, answer: str, location: str, product_ids: list, results: list) -> dict:
        """
        Enrich the cache by sending the full response payload to the enrich API.
        """
        url = f"{self.base_url}/api/enrich"
        payload = {
            "question": question,
            "answer": answer,
            "location": "",
            "product_ids": product_ids,
            "results": results
        }
        print(f"Enrich payload: {payload}")
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            return resp.json() if resp.status_code == 200 else {"success": False, "message": resp.text}

    async def ask(self, question: str, answer: str, location: str, product_ids: list, results: list) -> dict:
        """
        Ask the cache API by sending the full response payload to the enrich API (same as enrich, but named for clarity).
        """
        url = f"{self.base_url}/api/enrich"
        payload = {
            "question": question,
            "answer": answer,
            "location": location,
            "product_ids": product_ids,
            "results": results
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            return resp.json() if resp.status_code == 200 else {"success": False, "message": resp.text}

    @staticmethod
    def extract_product_ids_from_cache_response(cache_response: dict) -> list[str]:
        """
        Given a cache response dict, return a list of product IDs from the 'results' field.
        """
        return [item["id"] for item in cache_response.get("results", []) if "id" in item]
