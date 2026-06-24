# Confirmed starting-XI drop-in

When the official teamsheet is released (~1 hour before kickoff), the model can
re-value both teams from the *actual* eleven and update that match's
prediction. Lineups arrive two ways:

1. **Automatic** — `scripts/update_live.py` tries Sofascore. It is best-effort
   (Cloudflare often returns 403), so it may silently find nothing.
2. **Manual (always reliable)** — drop a JSON file here and it is picked up on
   the next run of `update_live.py`.

## File name

`<YYYY-MM-DD>_<Home>_<Away>.json` — spaces become underscores, exactly matching
the fixture's date and team names as they appear in the dataset, e.g.:

```
2026-06-27_Croatia_Ghana.json
2026-06-17_England_Croatia.json
```

## Format

`home` / `away` are the eleven **starters**, as plain names (or objects with a
`position` to override the database). Names are matched accent/case-insensitively
and by surname, so "Saka" resolves to "Bukayo Saka".

```json
{
  "confirmed": true,
  "source": "manual",
  "home": ["David Raya", "Pedri", "Rodri", "Lamine Yamal", "..."],
  "away": ["Dominik Livaković", "Modrić", "Gvardiol", "..."]
}
```

See `EXAMPLE.json` for a full eleven-a-side template.

## Apply it

```bash
python scripts/update_live.py --no-scrape        # manual files only
python scripts/update_live.py                     # also try Sofascore
```

This writes `predictions/live_updates.json` (shown in the dashboard's Fixtures
tab). The 15-min GitHub Action does the same automatically during the
tournament.

> Note: with the formula-only valuation, the override reliably reflects coarse
> squad quality (league tier / age) and big lineup changes, but cannot tell
> apart two elite XIs that differ by one rested star — that needs per-player
> market values added to the database.
