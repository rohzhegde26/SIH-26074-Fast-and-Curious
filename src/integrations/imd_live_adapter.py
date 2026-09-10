"""
src/integrations/imd_live_adapter.py

Operational IMD FTP/SFTP Ingestion Adapter (Stage-2 Production Stub).
Connects to ftp-service.imd.gov.in over TLS with ministry-whitelisted credentials.
"""

from typing import Any, Dict


class IMDLiveAdapter:
    """
    Adapter client for automated ingestion of operational IMD 0.25° gridded daily binary/NetCDF feeds.
    Target endpoint: ftp-service.imd.gov.in:21/pub/data/gridded/daily_0.25deg.
    """

    def __init__(self, ftp_host: str = "ftp-service.imd.gov.in", port: int = 21):
        self.ftp_host = ftp_host
        self.port = port

    def fetch_grid(self, forecast_date: str) -> Dict[str, Any]:
        """
        Polls and downloads the 08:30 IST gridded daily rainfall binary payload from IMD gateway.
        Raises NotImplementedError in air-gapped development / stage-1 deployment.
        """
        raise NotImplementedError(
            "Live IMD FTP ingestion requires ministry IP-whitelisted credentials (Stage-2 deployment)."
        )
