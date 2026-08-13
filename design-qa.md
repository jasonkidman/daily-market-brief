# Design QA — v5 visual migration

## Source of truth

- Reference: `/Users/mayansen/Downloads/daily_market_brief_screenshot_matched_v5.html`
- Dynamic implementation: `http://127.0.0.1:8765/`
- Render fixture: `work/v5-final-preview`, report date `2026-08-12`

## Compared state

- Desktop viewport: 1440 × 1000
- Mobile viewport: 375 × 812
- Reference and implementation were inspected at the same desktop viewport.
- Full-page evidence:
  - `work/design-qa/v5-reference-1440.png`
  - `work/design-qa/v5-implementation-1440.png`
  - `work/design-qa/v5-side-by-side-1440.png`

## Geometry verification

At 1440px, the reference and implementation matched these measured values:

- page width: 1406px at x=17px
- header: 1406px wide, 86px high, 18px 0 14px padding
- title: 31px, 32.5px rendered height
- market cards: 460.7px wide, 84.8px high, 13px 14px padding
- context cards: 344px wide, 94.3px high, 15px 14px padding
- breadth grid: 449.9px / 956.1px columns with 12px gap
- drawdown grid: two 697px columns with 12px gap
- all module borders, radii, colors, typography, and desktop grid ratios use the v5 CSS baseline

Dynamic-content height differences are expected and limited to the market summary text, market-health summary, retained drawdown status labels, and actual news count.

## Responsive verification

At 375px:

- viewport width: 375px
- document scroll width: 375px
- page width: 341px with 17px side gutters
- header, market, context, breadth, drawdown, and news grids become single-column
- breadth summary remains a compact two-column internal grid
- sector groups become a single column
- drawdown cards keep all fields readable without overflow

## Interaction verification

- Market Context help opens on click, reports `aria-expanded=true`, and closes with Escape.
- Market Breadth help opens on click, reports `aria-expanded=true`, and closes with Escape.
- History select retains relative targets and invokes the archive loading overlay before navigation.
- Sector rows retain hover/focus tooltips.
- Browser console log: empty.

## Findings and patches

- Replaced the legacy presentation DOM with the v5 module DOM instead of adapting the old layout.
- Migrated the v5 stylesheet as the baseline and added only functional state, tooltip, archive loading, dynamic-count, and mobile compatibility rules.
- Replaced every static example value with Jinja fields from the report schema.
- Preserved required interactions inside the v5 DOM.
- No P0, P1, or P2 visual defects remain in the inspected desktop and mobile states.

## Final result

passed
