# Copy-paste prompt to continue the live GeoSeek deployment

Continue the existing deployment in `C:\Users\Prash\Downloads\SIH 2026\geoseek`.
Do not recreate resources, delete bucket data, change architecture, or expose
credentials. Read `deploy/gcp/README.md` and inspect current state first.

The serving deployment is complete and publicly verified:

- URL: `https://geoseek.8.234.117.216.sslip.io/app/`
- The user explicitly requested no username/password. The site and APIs are
  anonymous over HTTPS; startup metadata and the live Caddy config reflect this.
- Project `geoseek-510206`; region/zone `asia-south1` / `asia-south1-a`.
- VM `geoseek-demo`: `e2-standard-2`, CPU-only, one FastAPI/Uvicorn worker.
- Reserved IP `8.234.117.216`; Caddy has a valid Let's Encrypt certificate.
- Persistent disk `geoseek-data`: 200 GiB pd-balanced; `.seed-complete` exists.
- Pinned image:
  `asia-south1-docker.pkg.dev/geoseek-510206/geoseek/app@sha256:142a6b9dd0b0966d504cb9f933cded397938c022f8145504a2588b18554e0de9`.
- VPC/firewalls/IAM/runtime and build service accounts, private buckets,
  Artifact Registry, Ops Agent and ₹1,000 budget alerts exist. The former demo
  password secret and stale proxy hash file were deleted.
- Ops Agent is active. FastAPI is bound only to `127.0.0.1:8000`; public port
  8000 was tested blocked. Caddy alone listens publicly on 80/443.
- Current observed container memory: app ~1.415 GiB of its 6 GiB limit; proxy
  ~15 MiB. Data disk 87 GiB used/110 GiB free; boot disk 5.7/30 GiB used.
- Public smoke passed with valid TLS and no credentials: anonymous `/app/`,
  health, RemoteCLIP+FAISS search, thumbnail, candidate overlay,
  discovery map and detection observations returned 200.
- A labeled decision `dec_a1bb04185bd743449027b78c9b961c87` and watch area
  `watch_ab3f2ee5faff4bf7bd33dcbbf7587424` survived an application-container
  restart, proving SQLite persistence.
- Caddy provides TLS/reverse proxy only. Startup recreates the lightweight proxy
  on boot so the checked-in public-access policy remains effective.
- Bootstrap now derives the 81 required imagery directories from deployed
  SQLite and syncs only the verified 34.09 GiB serving subset. It does not seed
  DOTA/xView/OSCD training archives onto a fresh serving disk.
- Local regression after the bootstrap change: 21 selected config/catalog tests
  passed; deployment Python files compiled; Git Bash syntax checks passed.
- Previous broader validation: 92 passed, 3 skipped, with four pre-existing
  frontend vendor-provenance failures documented in the runbook.

The independent full source-archive upload is also complete. The full local
`data/datasets` archive is 145,898,432,683 bytes and 76,138 files. GCS reports
145,898,432,685 bytes and 76,139 objects: an exact match after adding the 2-byte
`.serving-upload-complete` marker. The final idempotent rsync exited 0 with no
remaining transfers. `verify_serving_upload.py` then reconfirmed all 350
SQLite-referenced imagery files and 36,602,517,424 bytes with
`SERVING_UPLOAD_VERIFIED`.

For a later archive refresh, keep composite uploads disabled on this network and
run the same idempotent command until a final no-op pass exits 0:

```powershell
gcloud config set storage/parallel_composite_upload_enabled False
gcloud storage rsync --recursive data/datasets gs://geoseek-510206-archive/datasets --no-user-output-enabled
```

If future work is requested:

1. Preserve the single-service, CPU-only architecture and persistent-disk data.
2. Snapshot the data disk before application/database updates.
3. Re-run the anonymous public smoke and persistence checks after changes.
4. Stop the VM outside demo hours to conserve trial credits; the disk, archive,
   reserved IP and certificate state persist.

Keep the architecture unchanged: one same-origin FastAPI service, no CORS, no
GPU, no GCS FUSE for SQLite/COG reads, no index rebuild, no runtime Hugging Face
download, and no Oriented R-CNN, transformers or GeoPandas.
