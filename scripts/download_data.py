#!/usr/bin/env python3
"""
Download cybersecurity datasets for RAG indexing and CPT.
Downloads: ATT&CK STIX, CWE/CAPEC, NVD CVEs, Sigma rules
"""

import os
import json
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import subprocess

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filepath: Path, desc: str = None):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(filepath, 'wb') as f, tqdm(
        desc=desc or filepath.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

    return filepath


def download_attack_stix():
    """Download MITRE ATT&CK STIX data"""
    print("\n[1/4] Downloading MITRE ATT&CK STIX...")

    attack_dir = DATA_DIR / "attack"
    attack_dir.mkdir(exist_ok=True)

    # Enterprise ATT&CK
    urls = {
        "enterprise-attack.json": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
        "mobile-attack.json": "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json",
        "ics-attack.json": "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json"
    }

    for filename, url in urls.items():
        filepath = attack_dir / filename
        download_file(url, filepath, desc=f"ATT&CK {filename}")

    print(f"✓ ATT&CK STIX saved to {attack_dir}")


def download_cwe_capec():
    """Download CWE and CAPEC data"""
    print("\n[2/4] Downloading CWE/CAPEC...")

    cwe_dir = DATA_DIR / "cwe"
    capec_dir = DATA_DIR / "capec"
    cwe_dir.mkdir(exist_ok=True)
    capec_dir.mkdir(exist_ok=True)

    # CWE XML (latest version)
    cwe_url = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
    cwe_zip = cwe_dir / "cwec_latest.xml.zip"
    download_file(cwe_url, cwe_zip, desc="CWE XML")

    # Extract CWE
    with zipfile.ZipFile(cwe_zip, 'r') as zip_ref:
        zip_ref.extractall(cwe_dir)
    cwe_zip.unlink()  # Remove zip

    # CAPEC XML (latest version)
    capec_url = "https://capec.mitre.org/data/xml/capec_latest.xml"
    capec_file = capec_dir / "capec_latest.xml"
    download_file(capec_url, capec_file, desc="CAPEC XML")

    print(f"✓ CWE saved to {cwe_dir}")
    print(f"✓ CAPEC saved to {capec_dir}")


def download_nvd_cves():
    """Download NVD CVE data (recent years for MVP)"""
    print("\n[3/4] Downloading NVD CVEs...")

    nvd_dir = DATA_DIR / "nvd"
    nvd_dir.mkdir(exist_ok=True)

    # Download recent years (2020-2025 for MVP, can expand)
    years = [2020, 2021, 2022, 2023, 2024, 2025]

    for year in years:
        url = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
        filepath = nvd_dir / f"nvdcve-1.1-{year}.json.gz"

        try:
            download_file(url, filepath, desc=f"NVD CVE {year}")

            # Extract gz
            import gzip
            import shutil
            with gzip.open(filepath, 'rb') as f_in:
                with open(filepath.with_suffix(''), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            filepath.unlink()  # Remove gz
        except Exception as e:
            print(f"Warning: Could not download {year} CVEs: {e}")

    print(f"✓ NVD CVEs saved to {nvd_dir}")


def download_sigma_rules():
    """Clone Sigma rule repository"""
    print("\n[4/4] Downloading Sigma rules...")

    sigma_dir = DATA_DIR / "sigma"

    if sigma_dir.exists():
        print(f"Sigma already exists at {sigma_dir}, pulling latest...")
        subprocess.run(["git", "-C", str(sigma_dir), "pull"], check=True)
    else:
        print(f"Cloning Sigma repository...")
        subprocess.run([
            "git", "clone",
            "https://github.com/SigmaHQ/sigma.git",
            str(sigma_dir)
        ], check=True)

    print(f"✓ Sigma rules saved to {sigma_dir}")


def main():
    """Download all datasets"""
    print("=" * 60)
    print("Cybersecurity RAG Data Downloader")
    print("=" * 60)

    try:
        download_attack_stix()
        download_cwe_capec()
        download_nvd_cves()
        download_sigma_rules()

        print("\n" + "=" * 60)
        print("✓ All datasets downloaded successfully!")
        print(f"Data saved to: {DATA_DIR}")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        raise


if __name__ == "__main__":
    main()
