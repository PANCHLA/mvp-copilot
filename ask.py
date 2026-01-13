# ask.py
import requests
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Your question for the copilot")
    parser.add_argument("-k", "--top", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("-m", "--mode", default="local", help="LLM mode (stub, openrouter, local)")
    args = parser.parse_args()

    payload = {
        "q": args.question,
        "k": args.top,
        "mode": args.mode
    }

    response = requests.post("http://127.0.0.1:8000/query", json=payload)
    response.raise_for_status()
    
    # Get the response data
    # New format: {"answer": <str>, "sources": [...], "prompt": <str>}
    response_data = response.json()

    # print answer
    ans = response_data.get("answer", "")
    print(ans.replace('\\n', '\n'))

    # print sources (if any)
    sources = response_data.get("sources", [])
    if sources:
        print("\nSOURCES:")
        for s in sources:
            # show file, start-end and summary/score if present
            file = s.get("file") or s.get("id") or "<unknown>"
            start = s.get("start", "?")
            end = s.get("end", "?")
            summary = s.get("summary") or s.get("text") or ""
            score = s.get("score")
            line = f"- {file}:{start}-{end}"
            if summary:
                line += f"  — {summary[:200]}{'...' if len(summary)>200 else ''}"
            if score is not None:
                line += f"  (score={score:.4f})"
            print(line)

    # print a trimmed prompt for debugging (optional)
    prompt = response_data.get("prompt", "")
    if prompt:
        print("\nPROMPT (trimmed):")
        print(prompt[:1000] + ("\n...[truncated]" if len(prompt) > 1000 else ""))

if __name__ == "__main__":
    main()