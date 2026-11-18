# EvalForge MCP server

Exposes the EvalForge engine as MCP tools so an AI coding agent (Claude Code, Cursor, etc.) can create a dataset, run an eval against a RAG endpoint, and check a run for regressions.

Tools: `engine_health`, `list_datasets`, `add_dataset`, `run_eval`, `get_report`, `check_regression`.

```bash
pip install -r requirements.txt
EVALFORGE_URL=http://localhost:8000 python server.py   # stdio transport
python -m pytest -q                                      # 13 tests, no engine or API key needed
```

Claude Code: `claude mcp add evalforge -- python /path/to/evalforge/mcp_server/server.py`

`check_regression` fails any metric that drops more than `tolerance` (default 0.02) between two runs, so it can gate a prompt or retriever change.
