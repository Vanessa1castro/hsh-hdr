# HSH Studio – HDR editor

Private web app for Home Sweet Home Productions. Paste the Dropbox link of a shoot's RAW
folder (or upload RAWs), and it:

1. sorts the RAWs into bracket sets (auto-detected from the camera's exposure data),
2. aligns and merges each set,
3. finishes it in the HSH look – bright clean whites, natural wood tones, soft contrast,
   crisp window views, lens distortion removed and verticals levelled,
4. shows a before/after gallery, and saves full-size JPEGs into an `edited` folder next
   to the RAWs in Dropbox (plus "Download all" as a zip).

## Setup (click-only)

The step-by-step guide is in the HSH Studio setup doc. Short version:

1. **GitHub** – create a private repository and upload these files.
2. **Railway** – New Project → Deploy from GitHub repo → this repo.
   - Variables: `APP_PASSWORD` (team password), `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET`
   - Add a Volume mounted at `/data`
   - Settings → Networking → Generate Domain
3. **Dropbox** – App Console → Create app → Scoped access → Full Dropbox.
   - Permissions: `account_info.read`, `files.metadata.read`, `files.content.read`,
     `files.content.write`, `sharing.read`
   - Redirect URI: `https://YOUR-RAILWAY-DOMAIN/dropbox/callback`
4. Open the site, sign in, click **Connect Dropbox**, paste a RAW folder link.

## Settings (Railway variables)

| Variable | What it does |
|---|---|
| `APP_PASSWORD` | Password your team uses to sign in (required) |
| `DROPBOX_APP_KEY` / `DROPBOX_APP_SECRET` | From the Dropbox App Console |
| `KEEP_DAYS` | How long finished edits stay on the site (default 14) |
| `DATA_DIR` | Where jobs are stored (default `/data`, the Railway volume) |

## Notes

- One photo is edited at a time; each takes ~3–4 GB of memory while it runs.
- From Dropbox, RAWs are fetched one photo at a time and deleted right after, so even
  large shoots fit in a small volume. Browser uploads are stored until the job finishes.
- Panasonic Lumix S1 RAWs use a built-in decoder with a measured lens profile. Other
  cameras (Canon, Nikon, Sony, Fujifilm, DNG…) are read with LibRaw; they get the same
  look but no lens-distortion profile yet.

## Code map

- `app.py` – web pages, sign-in, API
- `jobs.py` – job queue and the background worker
- `dropbox_client.py` – Dropbox sign-in, shared-link download, upload
- `engine/pipeline.py` – bracket grouping, merge and the finishing look
- `engine/raw_io.py`, `engine/rw2.py` – RAW reading
