#!/usr/bin/env python3
"""Query Azure OpenAI for data analysis tasks.

This script provides BAA-compliant data analysis via Azure OpenAI.
Use this for direct data exploration while keeping within DUA compliance.

Usage:
    python scripts/azure_query.py "Your question here"
    python scripts/azure_query.py --file path/to/data.json "Analyze this data"

Environment variables (set in .env):
    AZURE_OPENAI_ENDPOINT - Azure OpenAI resource URL
    AZURE_OPENAI_API_KEY - API key
    AZURE_OPENAI_DEPLOYMENT - Model deployment name (e.g., "gpt-5.2")
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try default locations


def get_client() -> AzureOpenAI:
    """Create Azure OpenAI client from environment."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key:
        print("Error: Missing required environment variables.")
        print("Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
        sys.exit(1)

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def query(prompt: str, file_content: str | None = None, model: str | None = None, max_file_chars: int = 50000) -> str:
    """Send query to Azure OpenAI.

    Args:
        prompt: The question or instruction.
        file_content: Optional file content to include.
        model: Model deployment name (uses env var if not specified).
        max_file_chars: Maximum characters of file content to include.

    Returns:
        Model response text.
    """
    client = get_client()
    deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")

    # Build the full prompt
    full_prompt = prompt
    if file_content:
        # Truncate large files to avoid token limits
        if len(file_content) > max_file_chars:
            truncated = file_content[:max_file_chars]
            full_prompt = f"""Data (truncated to first {max_file_chars} characters):
```
{truncated}
```

{prompt}"""
        else:
            full_prompt = f"""Data:
```
{file_content}
```

{prompt}"""

    # Use developer role and max_completion_tokens for reasoning models
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "developer",
                "content": "You are a medical data analyst helping with research on the AI-READI diabetes dataset. Provide clear, accurate analysis."
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        max_completion_tokens=4096,
    )

    return response.choices[0].message.content or ""


def main():
    parser = argparse.ArgumentParser(
        description="Query Azure OpenAI for data analysis"
    )
    parser.add_argument("prompt", help="Question or instruction")
    parser.add_argument(
        "--file", "-f",
        help="Path to file to include in query"
    )
    parser.add_argument(
        "--model", "-m",
        help="Model deployment name (default: from env)"
    )

    args = parser.parse_args()

    load_env()

    # Load file content if specified
    file_content = None
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        file_content = file_path.read_text()

    # Query and print result
    result = query(args.prompt, file_content, args.model)
    print(result)


if __name__ == "__main__":
    main()
