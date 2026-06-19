from __future__ import annotations

import argparse
import json

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ai_core.api_health import check_all_apis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether configured LLM API keys can call their models")
    parser.add_argument(
        "--provider",
        choices=["all", "gemini", "google", "openai"],
        default="gemini",
        help="Which provider to check. Default: gemini",
    )
    parser.add_argument("--gemini-model", default=None, help="Gemini model to test. Defaults to GEMINI_MODEL or gemini-2.5-flash")
    parser.add_argument("--openai-model", default=None, help="OpenAI model to test. Defaults to OPENAI_MODEL or gpt-4o-mini")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main() -> None:
    if load_dotenv:
        load_dotenv()

    args = build_parser().parse_args()
    results = check_all_apis(
        provider=args.provider,
        gemini_model=args.gemini_model,
        openai_model=args.openai_model,
        timeout=args.timeout,
    )

    if args.json:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
        return

    print("=== API Health Check ===")
    for result in results:
        latency = f", latency={result.latency_ms}ms" if result.latency_ms is not None else ""
        print(f"[{result.status.upper()}] {result.provider} model={result.model}, key={result.key_env}{latency}")
        print(f"  {result.message}")
        if result.provider == "Gemini" and result.status != "ok":
            print("  建議：確認 .env 有 GOOGLE_API_KEY 或 GEMINI_API_KEY，並可先用 GEMINI_MODEL=gemini-2.5-flash 測試。")


if __name__ == "__main__":
    main()
