# SYS-002 — Audio workbench

Part of **SRD (Systems Research & Development)**. Designation **SYS-002**, site **SRD-01**.

Generates five reference test signals as 48 kHz, 16-bit mono PCM WAV files (a 1 kHz sine, a logarithmic sweep, a click train, silence and a clipped sine), reads them back, and checks each one against closed-form expectations computed independently of the generator (32 checks, including a cross-check of the `wave`-based reader against an independent `struct`-based one in `wavparse.py`). Python standard library only; nothing is played back.

> **Listening safety:** if you play these files, start at a **low volume**. `sine_1k_clipped_2x.wav` is clipped at full scale (0 dBFS); the sine, sweep and clicks peak at about half of full scale (about −6 dBFS). Raise the level slowly.

## Use

```bash
python -I generate.py --out ~/.local/state/lab/audio-workbench/
python -I check.py ~/.local/state/lab/audio-workbench/
python -I -m unittest discover -s tests        # simulated; temporary directories only
```

`--out` must be outside this repository; the generator refuses otherwise.

## Documentation

Decision numbers (`D-nnn`) in commit messages and this README refer to SRD's private lab log; they are kept for traceability.

The specification, decisions, acceptance criteria and experiment results (EXP-002, the five-signal round-trip experiment) live in the lab's knowledge vault (CEREBRUM), record `SYS-002 — Audio workbench`. Evidence cites this repository as `SYS-002@<commit>`.

History: this code lived in the vault at `projects/audio-workbench/` until M05.5 moved it here with `git subtree split`. Messages and dates are preserved; the author was rewritten to SRD BODY (D-068), and the commits were rewritten, so their hashes differ from the vault's. Before publication all commit dates were rewritten to UTC (same moments, offset `+0000`), which changed the hashes once more; the table shows the current ones. Evidence written before the move cites vault commits; this mapping links them:

| Vault commit | This repository |
| --- | --- |
| `39bd364` | `5c6c34d` |

## Licence

Code: MIT License, see `LICENSE`. Copyright (c) 2026 SRD (Systems Research & Development). Documentation in this repository: CC BY 4.0 (D-074).
