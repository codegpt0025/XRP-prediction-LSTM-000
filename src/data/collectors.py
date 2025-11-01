import time
import requests
from typing import Dict, Any

class AdaptiveCollector:
    """
    An intelligent data collector that learns API behaviors, such as rate limits,
    and adapts its fetching strategy accordingly.
    """

    def __init__(self, api_name: str, base_url: str):
        """
        Initializes the AdaptiveCollector.

        Args:
            api_name (str): The name of the API.
            base_url (str): The base URL for the API.
        """
        self.api_name = api_name
        self.base_url = base_url
        self.rate_limits = self._learn_limits()
        self.quality_score: float = 0.5
        self.last_request_time: float = 0

    def _learn_limits(self) -> Dict[str, Any]:
        """
        Learns the rate limits of the API.
        """
        # This would involve making controlled requests to discover rate limits.
        # For this example, we'll use a default.
        return {'requests': 60, 'per_minute': 1}

    def fetch(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetches data from the API with adaptive rate limiting and retries.

        Args:
            endpoint (str): The API endpoint to fetch from.
            params (Dict[str, Any]): The parameters for the API request.

        Returns:
            Dict[str, Any]: The JSON response from the API.
        """
        # Adaptive rate limiting
        time_since_last_request = time.time() - self.last_request_time
        required_delay = 60 / self.rate_limits['requests']
        if time_since_last_request < required_delay:
            time.sleep(required_delay - time_since_last_request)

        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            self.last_request_time = time.time()
            self.quality_score = 0.9 * self.quality_score + 0.1  # Increase quality score on success
            return response.json()
        except requests.exceptions.RequestException as e:
            self.quality_score *= 0.9  # Decrease quality score on failure
            print(f"Error fetching from {self.api_name}: {e}")
            return {}
