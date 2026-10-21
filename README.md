# SYS-002 — Audio workbench

Part of **SRD (Systems Research & Development)**. Designation **SYS-002**, site **SRD-01**.

Generates five reference test signals as 48 kHz, 16-bit mono PCM WAV files (a 1 kHz sine, a logarithmic sweep, a click train, silence and a clipped sine), reads them back, and checks each one against closed-form expectations computed independently of the generator (27 checks). Python standard library only; nothing is played back.

## Use

```bash
python -I generate.py --out ~/.local/state/lab/audio-workbench/
python -I check.py ~/.local/state/lab/audio-workbench/
python -I -m unittest discover -s tests        # simulated; temporary directories only
```

`--out` must be outside this repository; the generator refuses otherwise.

## Documentation

The specification, decisions, acceptance criteria and experiment results (EXP-002, the five-signal round-trip experiment) live in the lab's knowledge vault (CEREBRUM), record `SYS-002 — Audio workbench`. Evidence cites this repository as `SYS-002@<commit>`.

History: this code lived in the vault at `projects/audio-workbench/` until M05.5 moved it here with `git subtree split`. Messages, authors and dates are preserved, but the commits were rewritten, so their hashes differ from the vault's. Evidence written before the move cites vault commits; this mapping links them:

| Vault commit | This repository |
| --- | --- |
| `39bd364` | `d91e6a3` |
