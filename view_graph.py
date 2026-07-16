"""Visualize the compiled LangGraph graph (Layer 3: Self-RAG).

Usage:
    python view_graph.py            # print Mermaid source + save graph.png

Outputs:
  1. Mermaid source printed to the terminal — paste it into
     https://mermaid.live if the PNG step fails, or drop it straight into the
     README inside a ```mermaid code fence (GitHub renders it natively).
  2. graph.png in the repo root — rendered via the mermaid.ink web service
     (needs internet; skipped gracefully if unreachable).

Note: importing src.graph loads Chroma/BM25 lazily, so this works even before
ingestion — we only need the graph *structure*, not the index.
"""
from src.graph import graph


def main() -> None:
    g = graph.get_graph()

    # 1) Mermaid source — always works, no extra dependencies.
    mermaid = g.draw_mermaid()
    print("=== Mermaid source (paste into https://mermaid.live or the README) ===\n")
    print(mermaid)

    # 2) PNG — uses the mermaid.ink web API under the hood.
    try:
        png_bytes = g.draw_mermaid_png()
        out = "graph.png"
        with open(out, "wb") as f:
            f.write(png_bytes)
        print(f"\nSaved rendered graph to {out}")
    except Exception as e:  # offline / service hiccup — Mermaid source above suffices
        print(f"\n(PNG render skipped: {e})")
        print("Use the Mermaid source above instead.")


if __name__ == "__main__":
    main()
