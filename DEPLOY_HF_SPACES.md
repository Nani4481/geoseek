# DEPLOY_HF_SPACES.md — putting the real, working prototype online for free

GeoSeek's own backend (torch + faiss-cpu + rasterio + opencv, RemoteCLIP +
FC-Siam-diff + optionally YOLO26s-OBB all loaded at once) needs more RAM than
most "free" web-host tiers actually give you (typically 512 MB–1 GB). **Hugging
Face Spaces' free Docker "CPU basic" tier gives 2 vCPU / 16 GB RAM**, which is
the realistic free option that can run this without code changes — the same
`Dockerfile` already in this repo, unchanged.

The one thing a hosting platform can't supply for you: your **staged data**
(the FAISS index, the SQLite catalog, the model weights, the ingested
imagery — everything under `data/`). That lives only on the machine you ran
staging on (see RUN.md), was never committed to this GitHub repo (`data/` is
gitignored — it's gigabytes), and has to travel with you to wherever you
deploy. The steps below assume you're doing them **on that machine**, not
inside a fresh clone.

## Why a separate Space repo, not this GitHub repo directly

A Hugging Face Space is its own git repository (`huggingface.co/spaces/<you>/<name>`),
independent of GitHub. You push to *that* repo; Spaces builds whatever
`Dockerfile` it finds at the root and serves the running container. This
keeps your real data out of the public GitHub repo (where it's gitignored on
purpose) while still letting it live in the Space's own repo via Git LFS,
which Spaces supports natively and generously on the free tier.

## Steps

### 1. Create the Space

1. Sign in at [huggingface.co](https://huggingface.co) (free account).
2. **New Space** → SDK: **Docker** → Hardware: **CPU basic** (free) → any
   visibility (public is simplest for judges to open the link with no login).
3. Note the git URL it gives you:
   `https://huggingface.co/spaces/<your-username>/<space-name>`

### 2. Clone the new Space next to your existing GeoSeek checkout

Run this on the machine where `data/` is actually staged (per RUN.md):

```powershell
git clone https://huggingface.co/spaces/<your-username>/<space-name> geoseek-space
cd geoseek-space
git lfs install
```

### 3. Copy the project in, data included

```powershell
# from geoseek-space/, with your real GeoSeek checkout at ..\geoseek
robocopy ..\geoseek . /E /XD .git
del .dockerignore
```

Deleting `.dockerignore` here is deliberate: the copy this repo ships
excludes `data/` on purpose (for the bind-mount workflow in
`docker-compose.yml`), but a Space has no bind mounts — the Dockerfile's
`COPY . .` needs to actually see your data this time.

### 4. Tell Git LFS which files are large before adding anything

```powershell
git lfs track "data/**/*.faiss" "data/**/*.sqlite" "data/**/*.pt" "data/**/*.pth" `
              "data/**/*.safetensors" "data/**/*.png" "data/**/*.jpg" "data/**/*.jpeg" `
              "data/**/*.tif" "data/**/*.tiff"
git add .gitattributes
```

### 5. Add the Space metadata block to the top of README.md

Hugging Face reads a small YAML header at the very top of `README.md` to
know how to build/serve the Space. `app_port: 8000` tells it to route to the
container's existing port — the `Dockerfile`'s own `EXPOSE 8000` / `--port
8000` doesn't need to change.

```yaml
---
title: GeoSeek
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---
```

Paste that above the existing first line of `README.md` in `geoseek-space/`
(leave the rest of the file — the real project README — exactly as is; HF
shows it below the header as the Space's description).

### 6. Commit and push

```powershell
git add -A
git commit -m "Deploy GeoSeek with staged data"
git push
```

This upload is large (your `data/` folder, however much you've staged
locally) — it will take a while on a normal connection. Hugging Face then
builds the image (same `Dockerfile`, so the same few minutes it takes
locally) and starts the container.

### 7. Get the link

Once the Space's build finishes (the Space page shows live build logs), the
running app is at:

```
https://<your-username>-<space-name>.hf.space/app/
```

That's the link to give the judges.

## If you don't want to upload everything

`data/` breaks down as (see SRS §13 / docs/EVALUATION_REPORT.md §2.3 for the
exact figures at the corpus size they were measured at):

- `data/index/` — the FAISS + SQLite catalog (small, a few hundred MB)
- `data/models/` — RemoteCLIP + OpenCLIP checkpoints (~1.2 GB)
- `data/datasets/` — raw staged imagery (this is the large one)
- `data/detections/` — object-detection observation data, only needed for the Object Detect tab

All of it is needed for full functionality exactly as it runs locally; if
your staged corpus is already just your working AOIs (e.g. Ayodhya) rather
than the full multi-region benchmark corpus, uploading all of `data/` as-is
is simplest and safest — trimming risks silently breaking a tab whose imagery
turns out to be read live from `data/datasets/` rather than a smaller cache.

## Free-tier limits worth knowing

- **Sleep on inactivity.** A public CPU-basic Space that gets no traffic for
  a while goes to sleep and takes a short moment to wake on the next visit —
  normal, not a bug.
- **No persistent volume on the free tier.** Anything the app *writes* at
  runtime (new watch areas, new analyst decisions, newly ingested scenes)
  lives only in that running container and is lost on a restart/redeploy.
  Fine for a judged demo of the existing catalog; not a substitute for the
  real deployment's audit trail if you want decisions to persist.
- **Storage.** Free public Spaces comfortably hold multi-GB repos via LFS;
  a full raw-imagery corpus in the tens of GB may eventually need a paid
  storage add-on — cross that bridge only if you actually hit it.
