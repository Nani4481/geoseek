#!/usr/bin/env bash
# GCE startup script. Invoked with the attached runtime service account (ADC).
set -euo pipefail
exec > >(tee -a /var/log/geoseek-bootstrap.log) 2>&1
meta() { curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
BUCKET=$(meta geoseek-bucket)
PREFIX=$(meta satellite-prefix)
MODELS=$(meta model-bucket)
IMAGE=$(meta geoseek-image)
PROJECT=$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/project-id)
HOST=$(meta geoseek-host)
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y docker.io curl ca-certificates gnupg python3
systemctl enable --now docker
if ! command -v gcloud >/dev/null; then
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
  echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update
  apt-get install -y google-cloud-cli
fi
DEVICE=/dev/disk/by-id/google-geoseek-data
test -b "$DEVICE"
FSTYPE=$(blkid -s TYPE -o value "$DEVICE" || true)
if [ -z "$FSTYPE" ]; then
  # Only the explicitly named, newly-created deployment disk may be formatted.
  test -z "$(wipefs --no-act --noheadings --output TYPE "$DEVICE")"
  mkfs.ext4 -m 0 "$DEVICE"
elif [ "$FSTYPE" != ext4 ]; then
  echo "Refusing to modify unexpected filesystem: $FSTYPE" >&2
  exit 1
fi
mkdir -p /srv/geoseek
UUID=$(blkid -s UUID -o value "$DEVICE")
if ! grep -q "UUID=$UUID " /etc/fstab; then
  echo "UUID=$UUID /srv/geoseek ext4 defaults,nofail 0 2" >> /etc/fstab
fi
mountpoint -q /srv/geoseek || mount /srv/geoseek
ROOT=/srv/geoseek/app-data
mkdir -p "$ROOT" /srv/geoseek/caddy/data /srv/geoseek/caddy/config
if [ ! -f /srv/geoseek/.seed-complete ]; then
  echo "Waiting for the SQLite-referenced serving imagery completion marker..."
  until gcloud storage objects describe "gs://$BUCKET/$PREFIX/.serving-upload-complete" >/dev/null 2>&1; do
    sleep 30
  done
  # Copy the small serving snapshot first so its catalog can select exactly the
  # imagery directories referenced by SQLite. The bucket can also contain raw
  # training/staging archives that the web service never reads.
  for name in index change_model detections discovery; do
    gcloud storage rsync --recursive "gs://$BUCKET/$name" "$ROOT/$name"
  done
  gcloud storage rsync --recursive "gs://$MODELS/models" "$ROOT/models"
  gcloud storage cp "gs://$BUCKET/provenance_manifest.json" "$ROOT/provenance_manifest.json"
  gcloud storage cp "gs://$BUCKET/release-manifest.json" "$ROOT/release-manifest.json"
  python3 - "$ROOT/index/tiles.sqlite" > /tmp/geoseek-serving-dirs.txt <<'PY'
import pathlib, sqlite3, sys

database = pathlib.Path(sys.argv[1])
with sqlite3.connect(database) as connection:
    values = [row[0] for row in connection.execute(
        "SELECT DISTINCT dataset_dir FROM observations ORDER BY dataset_dir"
    )]
for value in values:
    normalized = str(value).replace('\\', '/').rstrip('/')
    marker = '/datasets/'
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    elif normalized.startswith('datasets/'):
        normalized = normalized[len('datasets/'):]
    path = pathlib.PurePosixPath(normalized)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise SystemExit(f"Unsafe dataset_dir in catalog: {value!r}")
    print(path.as_posix())
PY
  while IFS= read -r directory; do
    mkdir -p "$ROOT/datasets/$directory"
    gcloud storage rsync --recursive \
      "gs://$BUCKET/$PREFIX/$directory" "$ROOT/datasets/$directory"
  done < /tmp/geoseek-serving-dirs.txt
  python3 - "$ROOT" <<'PY'
import hashlib, json, pathlib, sqlite3, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / 'release-manifest.json').read_text())
for item in manifest['files']:
    path = root / item['path']
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    assert h.hexdigest() == item['sha256'], path
with sqlite3.connect((root / 'index/tiles.sqlite').as_uri()+'?mode=ro', uri=True) as c:
    assert c.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
PY
  touch /srv/geoseek/.seed-complete
fi
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin "${IMAGE%%/*}"
docker pull "$IMAGE"
if ! docker container inspect geoseek >/dev/null 2>&1; then
  docker run -d --name geoseek --restart unless-stopped \
    --log-driver=gcplogs --log-opt gcp-project="$PROJECT" \
    --memory=6g --cpus=2 -p 127.0.0.1:8000:8000 \
    -e GEOSEEK_DEVICE=cpu -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
    -e GDAL_CACHEMAX=256 -e HF_HUB_OFFLINE=1 \
    -e FAISS_INDEX_PATH=/app/data/index/tiles.faiss \
    -e DATABASE_URL=sqlite:////app/data/index/tiles.sqlite \
    -e MODEL_PATH=/app/data/models/RemoteCLIP-ViT-B-32.pt \
    -v "$ROOT:/app/data" "$IMAGE"
fi
cat > /srv/geoseek/caddy/Caddyfile <<EOF
$HOST {
    reverse_proxy 127.0.0.1:8000
    log {
        output stdout
        format json
    }
}
EOF
# Recreate the lightweight proxy on boot so the checked-in proxy policy is
# applied without changing the application container or persistent data.
docker rm -f geoseek-https >/dev/null 2>&1 || true
docker run -d --name geoseek-https --restart unless-stopped --network host \
  --log-driver=gcplogs --log-opt gcp-project="$PROJECT" \
  -v /srv/geoseek/caddy/Caddyfile:/etc/caddy/Caddyfile:ro \
  -v /srv/geoseek/caddy/data:/data -v /srv/geoseek/caddy/config:/config caddy:2
if ! systemctl is-active --quiet google-cloud-ops-agent; then
  curl -fsSLo /tmp/install-ops-agent.sh https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash /tmp/install-ops-agent.sh --also-install
fi
echo "GeoSeek startup dispatched: https://$HOST/app/ (check /health and /stats before demonstrating)"
