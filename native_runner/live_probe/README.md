# Null's Royale Reverse Engineering

Python host adapter with a read-only Rust memory sidecar.

## Python modules

- `live_ladder.py` — CLI and policy-session orchestration.
- `live_actions.py` — validated, human-paced Android touch dispatch.
- `live_observation.py` — FAIR observation construction.
- `live_contracts.py` and `live_settings.py` — authorization and pinned build settings.
- `external_source.py` — ADB lifecycle, process attestation, and live client.
- `external_memory.py` — compatibility entry point for the memory reader.
- `memory_transport.py` — bounded sidecar protocol and transport interface.
- `memory_types.py` — layouts, address types, and shared limits.
- `memory_discovery.py` — active battle discovery.
- `memory_snapshot.py` — snapshot extraction and card-event tracking.

Shared gameplay and model modules are intentionally omitted; run the adapter from the original repository checkout.

## Rust sidecar

`memory_reader/src/main.rs` contains the device-side read-only reader. Its build script emits the static executable into `out/`.

## Disclaimer

This project is provided as-is for research and educational use. I am not responsible for any misuse of the code or for consequences arising from its use. Only use it with software and devices you own or are authorized to test.
