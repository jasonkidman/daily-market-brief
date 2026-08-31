# Phase 1 news-source addition: NASA/NASASpaceflight priority A/B (2026-08-31)

Controlled comparison on the SAME 101 deduplicated real candidates (single fetch),
toggling only NASA/NASASpaceflight priority between P0 and P1, to isolate the effect
of the promotion from RSS feed content drift between separate live fetches.

## Scenario A: NASA/NASASpaceflight = P0 (current config)
Top-50 by source: CNBC Top News=8, Bloomberg Technology=2, Bloomberg Markets=8,
Bloomberg Economics=8, TechCrunch=6, NASA=5, BBC Business=3, BBC News=8,
NASASpaceflight=1, The Verge=1

## Scenario B: NASA/NASASpaceflight forced back to P1
Top-50 by source: CNBC Top News=10, Bloomberg Technology=2, Bloomberg Markets=11,
Bloomberg Economics=8, TechCrunch=6, BBC Business=3, BBC News=8, NASA=1, The Verge=1

## Diff: 5 candidates swapped in / 5 swapped out when NASA promoted to P0

Now included (NASA/NASASpaceflight):
- NASA: Panorama Showcasing the 34-Meter Antennas of the DSN's Goldstone Complex
- NASA: APOD: 2026 August 31 - Launch of the Roman Space Telescope
- NASA: Ribbon-Cutting Event for NASA Deep Space Network's Deep Space Station 23
- NASA: NASA Deep Space Network's New Goldstone Antenna Goes Online
- NASASpaceflight: SLS picks up the pace, four vehicles in production

Now excluded (displaced from the Stage A top-50 cap):
- Bloomberg Markets: Sungrow Shares Slump After Profit Tumbles by Almost a Third
- CNBC Top News: BYD shares slide as fierce China competition dents first-half earnings
- Bloomberg Markets: Radiant World's Senior China Trader Quits as Pressure Builds
- CNBC Top News: China's Xi builds diplomatic clout with multiple state visits ahead of Trump summit
- Bloomberg Markets: India Share Sale Pipeline Swells as Firms Rush to Raise Funds

## Conclusion

All 5 displaced candidates are non-US noise (Chinese/foreign equities, routine
diplomacy) that Stage B's selection criteria would exclude regardless. None of the
core financial_markets/macro_policy candidates (Fed/rates, US Treasury statements,
US tariff/trade policy, US bond/FX moves) were displaced. Promoting NASA and
NASASpaceflight to P0 is safe: it lets the SpaceX/space-specialist sources compete
in the Stage A 50-candidate cap without measurably degrading US financial/macro
coverage.

See also:
- 2026-08-31-phase1-after-p0.json: full real-network pipeline run (RSS fetch through
  Stage B) with NASA/NASASpaceflight at P0 -- 4 selected.
- An earlier real-network run with NASA/NASASpaceflight still at P1 (not saved to
  disk, see the conversation record) selected 9, versus 1 in the pre-phase-1
  production baseline (data/news_snapshots/2026-08-31.json, run 33351478765) --
  the swing between the two live P1/P0 runs (9 -> 4) is attributable to RSS feed
  content drift between fetches (a ~15 minute gap; e.g. the highest-scoring Warsh
  Fed-speech bond-market story had rolled out of the CNBC/Bloomberg feed windows by
  the second fetch), not the priority change -- see the controlled same-pool
  comparison above for the isolated effect.
