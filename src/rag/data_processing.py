"""
Data processing pipeline for cybersecurity knowledge sources.
Parses and chunks ATT&CK, CWE/CAPEC, NVD CVEs, and Sigma rules.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
import yaml
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Document:
    """Unified document format for RAG"""
    id: str
    content: str
    metadata: Dict[str, Any]
    source_type: str  # "attack", "cwe", "capec", "cve", "sigma"


class AttackProcessor:
    """Process MITRE ATT&CK STIX bundles"""

    def __init__(self, stix_file: Path):
        with open(stix_file) as f:
            self.data = json.load(f)

    def process(self) -> List[Document]:
        """Extract techniques, tactics, mitigations, groups"""
        documents = []

        for obj in self.data.get("objects", []):
            obj_type = obj.get("type")

            # Process attack-patterns (techniques)
            if obj_type == "attack-pattern":
                doc_id = obj.get("id")
                name = obj.get("name", "Unknown")
                description = obj.get("description", "")

                # Get tactics
                tactics = []
                for phase in obj.get("kill_chain_phases", []):
                    if phase.get("kill_chain_name") == "mitre-attack":
                        tactics.append(phase.get("phase_name"))

                # Get external ID (e.g., T1059)
                external_id = None
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        external_id = ref.get("external_id")
                        break

                content = f"ATT&CK Technique: {name} ({external_id})\n\n"
                content += f"Description: {description}\n\n"
                if tactics:
                    content += f"Tactics: {', '.join(tactics)}\n\n"

                # Add detection info if available
                x_mitre = obj.get("x_mitre_detection", "")
                if x_mitre:
                    content += f"Detection: {x_mitre}\n\n"

                documents.append(Document(
                    id=external_id or doc_id,
                    content=content,
                    metadata={
                        "name": name,
                        "tactics": tactics,
                        "type": "technique",
                        "external_id": external_id,
                    },
                    source_type="attack"
                ))

            # Process mitigations
            elif obj_type == "course-of-action":
                doc_id = obj.get("id")
                name = obj.get("name", "Unknown")
                description = obj.get("description", "")

                external_id = None
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        external_id = ref.get("external_id")
                        break

                content = f"ATT&CK Mitigation: {name} ({external_id})\n\n"
                content += f"Description: {description}\n\n"

                documents.append(Document(
                    id=external_id or doc_id,
                    content=content,
                    metadata={
                        "name": name,
                        "type": "mitigation",
                        "external_id": external_id,
                    },
                    source_type="attack"
                ))

            # Process threat groups
            elif obj_type == "intrusion-set":
                doc_id = obj.get("id")
                name = obj.get("name", "Unknown")
                description = obj.get("description", "")
                aliases = obj.get("aliases", [])

                external_id = None
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        external_id = ref.get("external_id")
                        break

                content = f"Threat Group: {name} ({external_id})\n\n"
                if aliases:
                    content += f"Aliases: {', '.join(aliases)}\n\n"
                content += f"Description: {description}\n\n"

                documents.append(Document(
                    id=external_id or doc_id,
                    content=content,
                    metadata={
                        "name": name,
                        "aliases": aliases,
                        "type": "group",
                        "external_id": external_id,
                    },
                    source_type="attack"
                ))

        return documents


class CWEProcessor:
    """Process CWE XML data"""

    def __init__(self, xml_file: Path):
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        # Handle XML namespace
        self.ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}

    def process(self) -> List[Document]:
        """Extract CWE weaknesses"""
        documents = []

        # Process Weaknesses
        for weakness in self.root.findall('.//cwe:Weakness', self.ns):
            cwe_id = weakness.get('ID')
            name = weakness.get('Name', 'Unknown')

            # Get description
            desc_elem = weakness.find('.//cwe:Description', self.ns)
            description = desc_elem.text if desc_elem is not None else ""

            # Get extended description
            ext_desc_elem = weakness.find('.//cwe:Extended_Description', self.ns)
            ext_description = ext_desc_elem.text if ext_desc_elem is not None else ""

            content = f"CWE-{cwe_id}: {name}\n\n"
            content += f"Description: {description}\n\n"
            if ext_description:
                content += f"Extended Description: {ext_description}\n\n"

            documents.append(Document(
                id=f"CWE-{cwe_id}",
                content=content,
                metadata={
                    "name": name,
                    "cwe_id": cwe_id,
                },
                source_type="cwe"
            ))

        return documents


class CAPECProcessor:
    """Process CAPEC XML data"""

    def __init__(self, xml_file: Path):
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        self.ns = {'capec': 'http://capec.mitre.org/capec-3'}

    def process(self) -> List[Document]:
        """Extract CAPEC attack patterns"""
        documents = []

        for pattern in self.root.findall('.//capec:Attack_Pattern', self.ns):
            capec_id = pattern.get('ID')
            name = pattern.get('Name', 'Unknown')

            # Get description
            desc_elem = pattern.find('.//capec:Description', self.ns)
            description = desc_elem.text if desc_elem is not None else ""

            content = f"CAPEC-{capec_id}: {name}\n\n"
            content += f"Description: {description}\n\n"

            documents.append(Document(
                id=f"CAPEC-{capec_id}",
                content=content,
                metadata={
                    "name": name,
                    "capec_id": capec_id,
                },
                source_type="capec"
            ))

        return documents


class NVDProcessor:
    """Process NVD CVE JSON feeds"""

    def __init__(self, json_file: Path):
        with open(json_file) as f:
            self.data = json.load(f)

    def process(self) -> List[Document]:
        """Extract CVE entries"""
        documents = []

        for item in self.data.get("CVE_Items", []):
            cve = item.get("cve", {})
            cve_id = cve.get("CVE_data_meta", {}).get("ID")

            # Get description
            descriptions = cve.get("description", {}).get("description_data", [])
            description = descriptions[0].get("value", "") if descriptions else ""

            # Get CVSS scores
            impact = item.get("impact", {})
            cvss_v3 = impact.get("baseMetricV3", {}).get("cvssV3", {})
            cvss_score = cvss_v3.get("baseScore", "N/A")
            severity = cvss_v3.get("baseSeverity", "N/A")

            content = f"CVE: {cve_id}\n\n"
            content += f"Description: {description}\n\n"
            content += f"CVSS Score: {cvss_score} ({severity})\n\n"

            # Get CWE references
            problem_type = cve.get("problemtype", {}).get("problemtype_data", [])
            cwes = []
            for pt in problem_type:
                for desc in pt.get("description", []):
                    cwe_id = desc.get("value", "")
                    if cwe_id.startswith("CWE-"):
                        cwes.append(cwe_id)

            if cwes:
                content += f"Related CWEs: {', '.join(cwes)}\n\n"

            documents.append(Document(
                id=cve_id,
                content=content,
                metadata={
                    "cve_id": cve_id,
                    "cvss_score": cvss_score,
                    "severity": severity,
                    "cwes": cwes,
                },
                source_type="cve"
            ))

        return documents


class SigmaProcessor:
    """Process Sigma detection rules"""

    def __init__(self, sigma_dir: Path):
        self.sigma_dir = sigma_dir

    def process(self) -> List[Document]:
        """Extract Sigma rules from YAML files"""
        documents = []

        # Find all .yml files in rules directory
        rules_dir = self.sigma_dir / "rules"
        if not rules_dir.exists():
            rules_dir = self.sigma_dir  # Fallback to root

        for yaml_file in rules_dir.rglob("*.yml"):
            try:
                with open(yaml_file) as f:
                    rule = yaml.safe_load(f)

                if not rule or not isinstance(rule, dict):
                    continue

                rule_id = rule.get("id", str(yaml_file))
                title = rule.get("title", "Unknown")
                description = rule.get("description", "")
                status = rule.get("status", "")
                level = rule.get("level", "")

                # Get MITRE ATT&CK tags
                tags = rule.get("tags", [])
                attack_tags = [t for t in tags if t.startswith("attack.")]

                content = f"Sigma Rule: {title}\n\n"
                content += f"Description: {description}\n\n"
                if level:
                    content += f"Level: {level}\n"
                if status:
                    content += f"Status: {status}\n"
                if attack_tags:
                    content += f"ATT&CK Tags: {', '.join(attack_tags)}\n"

                # Add detection logic
                detection = rule.get("detection", {})
                content += f"\nDetection Logic:\n{yaml.dump(detection, default_flow_style=False)}\n"

                documents.append(Document(
                    id=rule_id,
                    content=content,
                    metadata={
                        "title": title,
                        "level": level,
                        "status": status,
                        "attack_tags": attack_tags,
                        "file": str(yaml_file.relative_to(self.sigma_dir)),
                    },
                    source_type="sigma"
                ))

            except Exception as e:
                # Skip invalid YAML files
                continue

        return documents


def process_all_data(data_dir: Path) -> List[Document]:
    """Process all data sources and return unified documents"""
    all_documents = []

    print("Processing cybersecurity knowledge sources...")

    # Process ATT&CK
    attack_dir = data_dir / "attack"
    if attack_dir.exists():
        for stix_file in attack_dir.glob("*.json"):
            print(f"Processing {stix_file.name}...")
            processor = AttackProcessor(stix_file)
            docs = processor.process()
            all_documents.extend(docs)
            print(f"  Extracted {len(docs)} documents")

    # Process CWE
    cwe_dir = data_dir / "cwe"
    if cwe_dir.exists():
        for xml_file in cwe_dir.glob("*.xml"):
            if "cwec" in xml_file.name.lower():
                print(f"Processing {xml_file.name}...")
                processor = CWEProcessor(xml_file)
                docs = processor.process()
                all_documents.extend(docs)
                print(f"  Extracted {len(docs)} documents")

    # Process CAPEC
    capec_dir = data_dir / "capec"
    if capec_dir.exists():
        for xml_file in capec_dir.glob("*.xml"):
            print(f"Processing {xml_file.name}...")
            processor = CAPECProcessor(xml_file)
            docs = processor.process()
            all_documents.extend(docs)
            print(f"  Extracted {len(docs)} documents")

    # Process NVD CVEs
    nvd_dir = data_dir / "nvd"
    if nvd_dir.exists():
        for json_file in nvd_dir.glob("*.json"):
            print(f"Processing {json_file.name}...")
            processor = NVDProcessor(json_file)
            docs = processor.process()
            all_documents.extend(docs)
            print(f"  Extracted {len(docs)} documents")

    # Process Sigma
    sigma_dir = data_dir / "sigma"
    if sigma_dir.exists():
        print(f"Processing Sigma rules...")
        processor = SigmaProcessor(sigma_dir)
        docs = processor.process()
        all_documents.extend(docs)
        print(f"  Extracted {len(docs)} documents")

    print(f"\n✓ Total documents processed: {len(all_documents)}")
    return all_documents
