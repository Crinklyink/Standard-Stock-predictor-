import requests

class FidelityBrokerage:
    """Minimal placeholder for Fidelity brokerage API integration."""
    BASE_URL = "https://api.fidelity.com"  # Placeholder URL

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def get_account_info(self):
        """Fetch account information from Fidelity (stub)."""
        raise NotImplementedError("Fidelity API integration not implemented yet.")
