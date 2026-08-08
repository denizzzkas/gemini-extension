# Gemini AI — Imperal Cloud Extension

Generate images and videos with Google's Gemini models, right from Imperal
Cloud's chat or its own panel — the full **Nano Banana** image-model family
(Pro / 2 / 2 Lite / legacy), and **Gemini Omni Flash** for fast text-to-video.

Built on the [Imperal SDK](https://panel.imperal.io) (`imperal-sdk`), talking
directly to the Gemini [Interactions API](https://ai.google.dev/) over REST
(no `google-genai` dependency).

## Features

- **Dedicated image-generation tools** — each Nano Banana model has its own
  explicit tool, so model choice, pricing, and capability never depend on a
  hidden parameter:
  | model id | label | notes |
  |---|---|---|
  | `gemini-3-pro-image` | Nano Banana Pro (default) | premium, 4K, up to 5 reference images |
  | `gemini-3.1-flash-image` | Nano Banana 2 | balanced, up to 4 reference images |
  | `gemini-3.1-flash-lite-image` | Nano Banana 2 Lite | fastest/cheapest, no multi-reference support |
  | `gemini-2.5-flash-image` | Nano Banana (legacy) | kept for compatibility; Google recommends 2 Lite instead |

  Supports **reference images for character/scene consistency**: use a
  generation ID returned by one of the dedicated image tools, or select a
  saved image in Gemini Studio, to reuse the exact character/setting from an
  earlier generation — e.g. "same antagonist, new pose". You can also upload
  a PNG or JPEG reference in chat; it is stored in your own reference library
  and can be passed to the next generation.
- **`generate_video`** — turn a text prompt into a short video (Gemini Omni Flash).
- **`check_gemini_connection`** — verify the configured API key is valid and reachable.
- **Skeleton refresh** (`skeleton_refresh_gemini_stats`) — feeds Webbee a
  lightweight snapshot (key configured?, image/video counts, last prompt)
  on a 5-minute TTL, with no extra network call.
- **Gemini Studio panel** — a stable left generator/history panel plus a
  center detail view. It shows panel-safe previews, the full prompt, model,
  copy-prompt control, reference details, and a real original-file download
  whenever the panel payload can safely carry it.
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

## Video models: Gemini Omni Flash only, by design

`generate_video` supports Gemini Omni Flash (`gemini-omni-flash-preview`)
and deliberately does NOT offer Google's other video model, **Veo** — this
is a final decision, not a pending task.

Veo uses a separate, asynchronous API contract (`predictLongRunning` +
polling `operations.get`) that hands back the finished video as an
external `video.uri`, not inline bytes like Omni Flash/the image models.
The only way for a browser to fetch that URI directly is appending the
user's own Gemini API key to the URL in plain text (`...&key=...`) —
Google's SDKs have no other client-side download path for it. There is
also no safe way to proxy the raw video bytes through this extension
instead: the Imperal SDK's `ctx.http` decodes any non-JSON response body
as UTF-8 text, which corrupts binary data, and `ctx.storage` only serves
this extension's own internal storage, not arbitrary external URLs.

Exposing a user's personal API key in a rendered link is a structural
security regression for a platform whose whole secrets model exists to
keep that key from ever reaching plaintext outside the vault — so Veo
stays unintegrated until Google or the Imperal SDK offers a safe binary
download path. Gemini Omni Flash remains the one supported video model.

## Project layout

```
app.py                 Extension setup, secret declaration, health check
gemini_config.py        Model ids, store collection, limits/timeouts
clients/gemini_client.py   REST client for the Gemini Interactions API
return_models.py        Pydantic response models
handlers/generate.py    Chat function: generate_video
handlers/image_tools.py Dedicated per-model image-generation chat functions
handlers/status.py      Connection status handler
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
