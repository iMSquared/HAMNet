#!/usr/bin/env bash
set -ex

# Parse command line arguments
DOWNLOAD_DGN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --dgn)
      DOWNLOAD_DGN=true
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Download DGN (optional)
if [[ "${DOWNLOAD_DGN}" == "true" ]]; then
  echo "Downloading DGN dataset..."

  hf download "imm-unicorn/corn-public" "DGN.tar.gz" \
  --repo-type model \
  --local-dir "/input/DGN"
    
  cd /input/DGN
  tar -xzf "DGN.tar.gz"
  rm -f "DGN.tar.gz"
  cd -
else
  DEST_DIR="/tmp/docker"
  REPO_ID="HAMNet/public" # dataset repo id

  mkdir -p "${DEST_DIR}"

  # Download episode for training
  FILE="eps-fr3-near-and-rand-1024x384.pth"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "${DEST_DIR}"

  # Download episode for testing
  FILE="eps-demo-16x8.pth"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "${DEST_DIR}"

  # Download sysid
  FILE="new_sysid.pkl"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "${DEST_DIR}"

  DEST_DIR="/input/robot"
  mkdir -p "${DEST_DIR}"

  # Download robot cloud (Optional)
  FILE="custom_v3_cloud.pkl"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "${DEST_DIR}"
fi
