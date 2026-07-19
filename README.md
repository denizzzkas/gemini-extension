# Gemini AI — Imperal Cloud Extension

Generate images and videos with Google's Gemini models, right from Imperal
Cloud's chat or its own panel — **Nano Banana Pro** for studio-quality image
generation/editing, and **Gemini Omni Flash** for fast text-to-video.

Built on the [Imperal SDK](https://panel.imperal.io) (`imperal-sdk`), talking
directly to the Gemini [Interactions API](https://ai.google.dev/) over REST
(no `google-genai` dependency).

## Features

- **`generate_image`** — turn a text prompt into an image (Nano Banana Pro).
  Supports **reference images for character/scene consistency**: pass
  `reference_generation_ids` (up to 6 IDs from your own
  `list_generation_history` or a prior `generate_image` call's
  `generation_id`) to reuse the exact character/setting from earlier
  generations — e.g. "same antagonist, new pose". Only this extension's own
  saved generations work as references (arbitrary external images pasted
  into chat aren't accepted yet — re-generate or re-save them here first).
- **`generate_video`** — turn a text prompt into a short video (Gemini Omni Flash).
- **`check_gemini_connection`** — verify the configured API key is valid and reachable.
- **`list_generation_history`** — list your past generations (each item includes
  its `id`, reusable as a `reference_generation_ids` entry).
- **Skeleton refresh** (`skeleton_refresh_gemini_stats`) — feeds Webbee a
  lightweight snapshot (key configured?, image/video counts, last prompt)
  on a 5-minute TTL, with no extra network call.
- **Gemini Studio panel** — a center-slot Panel UI with prompt forms for
  image/video generation and a history list with inline previews (saved
  media is uploaded to `ctx.storage` and normalized into an absolute,
  clickable URL — not the bare storage path the raw API can return).
- **App-level health check** — a bounded reachability probe of the
  Gemini API itself (per-user key status lives in `check_gemini_connection`
  and the skeleton snapshot, not in the app-level probe).

## Bring your own key (per-user)

This extension declares a single secret, `gemini_api_key`
(`scope="user"`, `write_mode="user"`): **each user connects their own key**
privately via **Panel → Secrets** — nobody shares a key, nobody's usage
counts against someone else's Google Cloud quota or billing. Nobody else can
read or overwrite it from chat; only the Panel Secrets UI can set/rotate it,
and it's never visible to other users of this extension.

Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
Note: the Gemini API requires **Google Cloud Billing** enabled on the
project behind the key — the free tier's quota for these models is `0`.

## Project layout

```
app.py                 Extension setup, secret declaration, health check
gemini_config.py        Model ids, store collection, limits/timeouts
clients/gemini_client.py   REST client for the Gemini Interactions API
return_models.py        Pydantic response models
handlers/generate.py    Chat functions: generate_image / generate_video / check_gemini_connection / list_generation_history
handlers/skeleton.py    Skeleton refresh (gemini_stats)
handlers/panel.py       Gemini Studio panel UI
main.py                 Entry point
tests/                  pytest suite (generate, skeleton, panel)
scripts/smoke_test.py   Standalone script to hit the real Gemini API directly
```

## Development

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m imperal_sdk.cli.main build .       # generate imperal.json
./venv/bin/python -m imperal_sdk.cli.main validate .    # validate the manifest
./venv/bin/python -m pytest -q                          # run the test suite
```

### Smoke-testing against the real API

`scripts/smoke_test.py` is a standalone script (stdlib only, no SDK/venv
needed) for testing directly against Google's API with your own key:

```bash
export GEMINI_API_KEY=your-key-here
python3 scripts/smoke_test.py image "a cat astronaut on the moon"
python3 scripts/smoke_test.py video "a paper airplane flying through a city"
```

## License

[LGPL-3.0](LICENSE)
