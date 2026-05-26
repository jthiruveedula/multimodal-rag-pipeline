## 2025-05-14 - [Initial Analysis]
**Learning:** Found multiple low-hanging fruits for performance optimization:
1. `ingest_text` calls embedding API sequentially for each chunk, which is O(N) network calls.
2. `generate_answer` re-instantiates `GenerativeModel` on every call.
**Action:** Refactor `ingest_text` to use batched embedding calls and move model initialization to `__init__`.
