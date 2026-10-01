#!/usr/bin/env bash
# Run from repo root in Google Cloud Shell, AFTER upload and build in README.
set -euo pipefail
: "${PROJECT:=geoseek-510206}"
: "${REGION:=asia-south1}"
: "${ZONE:=asia-south1-a}"
: "${GCS_BUCKET:=$PROJECT-archive}"
: "${MODEL_BUCKET:=$GCS_BUCKET}"
: "${SATELLITE_DATA_PREFIX:=datasets}"
: "${IMAGE:?Set IMAGE to the built Artifact Registry digest reference}"
SA="geoseek-runtime@$PROJECT.iam.gserviceaccount.com"
gcloud compute networks create geoseek-net --subnet-mode=custom --project="$PROJECT"
gcloud compute networks subnets create geoseek-subnet --network=geoseek-net \
  --range=10.40.0.0/24 --region="$REGION" --project="$PROJECT"
gcloud compute firewall-rules create geoseek-https --network=geoseek-net \
  --allow=tcp:80,tcp:443 --source-ranges=0.0.0.0/0 --target-tags=geoseek-web --project="$PROJECT"
gcloud compute firewall-rules create geoseek-iap-ssh --network=geoseek-net \
  --allow=tcp:22 --source-ranges=35.235.240.0/20 --target-tags=geoseek-web --project="$PROJECT"
gcloud compute addresses create geoseek-ip --region="$REGION" --project="$PROJECT"
IP=$(gcloud compute addresses describe geoseek-ip --region="$REGION" --project="$PROJECT" --format='value(address)')
# sslip.io supplies public DNS for a temporary demo without purchasing a domain.
# Set DEMO_HOST to a domain you own (A record to $IP) for long-term use.
HOST="${DEMO_HOST:-geoseek.$IP.sslip.io}"
gcloud compute disks create geoseek-data --type=pd-balanced --size=200GB --zone="$ZONE" --project="$PROJECT"
gcloud compute instances create geoseek-demo --project="$PROJECT" --zone="$ZONE" \
  --machine-type=e2-standard-2 --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-balanced \
  --disk=name=geoseek-data,device-name=geoseek-data,mode=rw,boot=no,auto-delete=no \
  --subnet=geoseek-subnet --address="$IP" --tags=geoseek-web \
  --service-account="$SA" --scopes=cloud-platform \
  --shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring \
  --metadata="enable-oslogin=TRUE,geoseek-bucket=$GCS_BUCKET,model-bucket=$MODEL_BUCKET,satellite-prefix=$SATELLITE_DATA_PREFIX,geoseek-image=$IMAGE,geoseek-host=$HOST" \
  --metadata-from-file=startup-script=deploy/gcp/bootstrap-vm.sh
printf '\nProvisioning started. URL after startup/certificate checks: https://%s/app/\n' "$HOST"
