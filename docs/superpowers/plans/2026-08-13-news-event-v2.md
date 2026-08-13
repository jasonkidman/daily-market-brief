# News Event V2 Implementation Plan

**Goal:** Upgrade RSS news processing to event-level deduplication and market-driven Top 8 selection without changing RSS sources, page layout, market logic, or deployment.

**Architecture:** Keep deterministic URL/title deduplication as stage zero. Add a separate DeepSeek event clustering stage that produces validated event groups and program-selected representative candidates; pass only those representatives, historical event summaries, and existing market context to the existing selection transport.

**Constraints:** Do not add Yahoo Finance news, URLs to model payloads, new DeepSeek clients, or commits/pushes. Stage A failures must fall back to one candidate per event. Stage B remains candidate-only and is the only stage that can degrade the visible news module.

### Task 1: Event schema and representatives

- [ ] Add `src/news_event_prompt.py` with the clustering-only system prompt and fixed topic groups.
- [ ] Add failing validation and representative tests in `tests/test_news_events.py`.
- [ ] Implement parsing, exact coverage validation, event-map construction, and deterministic representative choice in `src/news_events.py`.

### Task 2: Clustering transport and fallback

- [ ] Add failing tests for payload URL exclusion, three-attempt behavior, candidate cap, singleton bypass, and fallback.
- [ ] Implement `cluster_news_events()` using `call_deepseek` as injected transport and candidate-per-event fallback.

### Task 3: Stage B and history

- [ ] Add failing tests for event-enriched selection output and topic-concentration prompt language.
- [ ] Update `src/news_prompt.py` and `src/deepseek_client.py` so Stage B accepts event representatives and resolves protected event metadata locally.
- [ ] Add a backward-compatible `_recent_news_events()` helper for legacy and V2 reports.

### Task 4: Pipeline integration

- [ ] Add integration tests for the V2 offline fixture and Stage A fallback isolation.
- [ ] Integrate deterministic dedupe, clustering, representatives, history, selection, and concise pipeline logs in `src/main.py`.

### Task 5: Verification

- [ ] Run `python -m pytest -v`.
- [ ] Run offline generation into `work/news-v2` and `python -m src.smoke --base-dir work/news-v2`.
- [ ] If the local DeepSeek credential is present, run a real generation and inspect selected events; otherwise report that limitation.
