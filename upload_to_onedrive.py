#!/usr/bin/env python3
"""
Upload MWR_Combined_ZipCode_Risk_v2.csv to OneDrive via Microsoft Graph API.
Used by GitHub Actions after mwr_rescore.py generates a fresh CSV.

SETUP REQUIRED (one-time):
1. Register an app in Azure AD (portal.azure.com)
2. Add Microsoft Graph permissions: Files.ReadWrite.All
3. Generate a client secret
4. Get a refresh token via OAuth flow
5. Add these as GitHub Secrets:
   - ONEDRIVE_CLIENT_ID
   - ONEDRIVE_CLIENT_SECRET
   - ONEDRIVE_TENANT_ID
   - ONEDRIVE_REFRESH_TOKEN
"""

import os
import sys
import requests

# Configuration from environment
CLIENT_ID = os.getenv("ONEDRIVE_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("ONEDRIVE_CLIENT_SECRET", "")
TENANT_ID = os.getenv("ONEDRIVE_TENANT_ID", "")
REFRESH_TOKEN = os.getenv("ONEDRIVE_REFRESH_TOKEN", "")

# File to upload
LOCAL_FILE = "MWR_Combined_ZipCode_Risk_v2.csv"
ONEDRIVE_PATH = "MWR_Automation_Data/MWR_Combined_ZipCode_Risk_v2.csv"

GRAPH_URL = "https://graph.microsoft.com/v1.0"


def get_access_token() -> str:
    """Get a fresh access token using refresh token."""
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token",
        "scope": "https://graph.microsoft.com/.default"
    }
    resp = requests.post(url, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]


def upload_file(token: str):
    """Upload file to OneDrive using Microsoft Graph API."""
    file_size = os.path.getsize(LOCAL_FILE)
    print(f"📤 Uploading {LOCAL_FILE} ({file_size / 1024 / 1024:.1f} MB)...")

    # For files < 4MB, use simple upload
    # For files > 4MB, use upload session (our CSV is ~18MB)
    if file_size < 4 * 1024 * 1024:
        # Simple upload
        url = f"{GRAPH_URL}/me/drive/root:/{ONEDRIVE_PATH}:/content"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream"
        }
        with open(LOCAL_FILE, "rb") as f:
            resp = requests.put(url, headers=headers, data=f)
        resp.raise_for_status()
    else:
        # Upload session for large files
        url = f"{GRAPH_URL}/me/drive/root:/{ONEDRIVE_PATH}:/createUploadSession"
        headers = {"Authorization": f"Bearer {token}"}
        session_resp = requests.post(url, headers=headers,
                                     json={"item": {"@microsoft.graph.conflictBehavior": "replace"}})
        session_resp.raise_for_status()
        upload_url = session_resp.json()["uploadUrl"]

        # Upload in 5MB chunks
        chunk_size = 5 * 1024 * 1024
        with open(LOCAL_FILE, "rb") as f:
            chunk_start = 0
            while chunk_start < file_size:
                chunk = f.read(chunk_size)
                chunk_end = min(chunk_start + len(chunk) - 1, file_size - 1)

                headers = {
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {chunk_start}-{chunk_end}/{file_size}"
                }
                resp = requests.put(upload_url, headers=headers, data=chunk)

                if resp.status_code not in [200, 201, 202]:
                    print(f"❌ Upload failed at byte {chunk_start}: {resp.status_code}")
                    resp.raise_for_status()

                chunk_start = chunk_end + 1
                pct = chunk_start / file_size * 100
                print(f"   {pct:.0f}% uploaded...")

    print("✅ Upload complete!")


def main():
    if not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID, REFRESH_TOKEN]):
        print("⚠️ OneDrive credentials not configured. Skipping upload.")
        print("   To enable: add ONEDRIVE_CLIENT_ID, ONEDRIVE_CLIENT_SECRET,")
        print("   ONEDRIVE_TENANT_ID, ONEDRIVE_REFRESH_TOKEN to GitHub Secrets.")
        sys.exit(0)  # Exit cleanly - don't fail the workflow

    if not os.path.exists(LOCAL_FILE):
        print(f"❌ {LOCAL_FILE} not found!")
        sys.exit(1)

    token = get_access_token()
    upload_file(token)


if __name__ == "__main__":
    main()
