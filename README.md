# Null's Royale Reverse Engineering

Python host adapter with a read-only Rust memory sidecar.

## Python adapter

| Responsibility | Modules |
| --- | --- |
| CLI, session setup, and match decisions | `live_ladder.py`, `policy_match.py` |
| FAIR observations, battlefield history, and placement masks | `live_observation.py`, `battlefield.py`, `placement.py` |
| Touch coordinates, play validation, and execution receipts | `touch_layout.py`, `live_actions.py`, `play_ledger.py` |
| Build identity, authorization, and payload validation | `game_profile.py`, `live_settings.py`, `live_contracts.py` |
| Device lifecycle and reader interface | `external_source.py`, `external_memory.py`, `memory_transport.py` |
| Battle discovery and snapshot extraction | `memory_types.py`, `memory_discovery.py`, `memory_snapshot.py` |
| Public card events, tower identity, and snapshot serialization | `public_snapshot.py` |

The CLI and model integration tests use the shared gameplay and training modules in the original checkout. The component tests run independently with `python -m unittest native_runner.tests.test_adapter_components`. On Windows, Python needs IANA time zone data for `Europe/Paris`.

## Rust sidecar

`memory_reader/src/main.rs` contains the device-side read-only reader. Its build script verifies the static executable before copying it into `out/`.

## Disclaimer

This project is provided as-is for research and educational use. I am not responsible for any misuse of the code or for consequences arising from its use. Only use it with software and devices you own or are authorized to test.
