# Notes for AI assistants

Read **[README.md](README.md)** first — it explains what this repository is for,
how to install and run the tool, and the license. Then read
**[CONTRIBUTING.md](CONTRIBUTING.md)** for the ground rules: offline by contract,
English-only source, pinned dependencies, and the cross-platform parity gate with
MycoSpec (Android).

## Responsible & honest use
@.claude/RESPONSIBLE_AI_USE.md

## Things that are easy to get wrong here

- **Do not add network access.** FSSA requests no network permission and ships no
  telemetry. Suggestions that call out to an API, a model service or analytics
  will be rejected.
- **Do not "fix" the tkinter stubs** in `tests/`. They are stubbed on purpose so
  the suite is identical headless and on a display.
- **Do not edit a number without touching both implementations.** The Python side
  and the Android side must agree; see CONTRIBUTING.md §5.
