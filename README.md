# ARTV: Advanced Red-Teaming for Vulnerabilities Toolkit

An agent-based framework developed for the OpenAI `gpt-oss-20b` Safety Hackathon. This toolkit provides a structured, extensible, and automated system to discover, verify, and document high-impact vulnerabilities in Large Language Models.

## Features

- **Agent-Based Testing:** Includes 10 specialized agents for vulnerability classes like Sabotage, Multi-Turn Deception, and Covert Data Exfiltration.
- **Reproducible Findings:** Generates structured JSON reports compatible with competition standards, including detailed reproducibility metadata.
- **Automated Verification Harness:** A built-in system to programmatically verify findings with deterministic settings.
- **Advanced Behavioral Analysis:** Deeply analyzes model responses for subtle signs of deception, strategic planning, and misalignment.

## Installation

This toolkit is designed to be self-contained and can be installed directly from the source directory using pip:

```bash
pip install .
