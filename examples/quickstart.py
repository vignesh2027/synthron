"""
Synthron Quickstart — 1-line usage example.

Setup:
    pip install synthron
    cp .env.example .env
    # Add your GEMINI_API_KEY and GROQ_API_KEY to .env

Run:
    python examples/quickstart.py
"""

import asyncio
import sys
import os

# Add parent dir to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from synthron import run

    print("=" * 60)
    print("  SYNTHRON QUICKSTART")
    print("=" * 60)
    print()

    # 1-line usage — it's really this simple
    result = await run("What are the top 3 benefits of Python for AI development?")

    print("RESULT:")
    print("-" * 40)
    print(result)
    print("-" * 40)
    print("\n✅ Done! Synthron is working.")


if __name__ == "__main__":
    asyncio.run(main())
