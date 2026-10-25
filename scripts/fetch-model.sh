#!/usr/bin/env bash
# Downloads the EverParakeet model files into Model/, the folder the app bundles, and verifies every file against the
# CHECKSUMS.sha256 published in the same Hugging Face revision.
#
#   scripts/fetch-model.sh             Standard 1.1.1 (default)
#   scripts/fetch-model.sh --compact   Compact 1.1
#
# Needs only bash, curl and shasum. Safe to run again: files that already match their checksum are kept, and an
# interrupted download resumes.
#
# Both revisions are pinned by full commit hash, so what you get does not change if the repository is updated.
set -euo pipefail

STANDARD_REPO="EverPaper/everparakeet-tdt-0.6b-v3-coreai"
STANDARD_REV="3273ae066c0d7f46940de4917fb731fdb89558b7"
COMPACT_REPO="EverPaper/everparakeet-tdt-0.6b-v3-compact"
COMPACT_REV="aeb5fb8656cb57d3298e645e11cb20ee5614eaf0"

repo="$STANDARD_REPO"; rev="$STANDARD_REV"; label="Standard 1.1.1"
case "${1:-}" in
  "") ;;
  --compact) repo="$COMPACT_REPO"; rev="$COMPACT_REV"; label="Compact 1.1" ;;
  -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "usage: $0 [--compact]" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/Model"
base="https://huggingface.co/$repo/resolve/$rev"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "EverParakeet $label ($repo @ ${rev:0:8})"

# The checksum list of this exact revision. Keep only the files the app needs.
curl -fsSL --retry 3 "$base/CHECKSUMS.sha256" -o "$work/all.sha256"
grep -E '^[0-9a-f]{64}  \./(models|tokenizer)/' "$work/all.sha256" > "$work/needed.sha256"
# Every path must stay inside Model/ (the grep above already limits them to models/ and tokenizer/).
if grep -qE '(^|/)\.\.(/|$)' "$work/needed.sha256"; then echo "Unexpected path in CHECKSUMS.sha256" >&2; exit 1; fi
total="$(wc -l < "$work/needed.sha256" | tr -d ' ')"
[ "$total" -gt 0 ] || { echo "No model files listed in CHECKSUMS.sha256" >&2; exit 1; }

# A different variant in Model/ would leave stale files next to the new ones.
if [ -f "$dest/.variant" ] && [ "$(cat "$dest/.variant")" != "$repo@$rev" ]; then
  echo "Model/ holds a different variant; replacing it."
  rm -rf "$dest/models" "$dest/tokenizer"
fi
mkdir -p "$dest"

matches() { # file expected-hash
  [ -f "$1" ] && [ "$(shasum -a 256 "$1" | cut -d' ' -f1)" = "$2" ]
}

n=0
while read -r hash path; do
  n=$((n + 1))
  rel="${path#./}"
  out="$dest/$rel"
  if matches "$out" "$hash"; then
    echo "[$n/$total] have  $rel"
    continue
  fi
  echo "[$n/$total] fetch $rel"
  mkdir -p "$(dirname "$out")"
  # -C - resumes a partial .part file. Downloads go to .part first so a broken run never leaves a half file in place.
  curl -fL --retry 3 --retry-delay 2 -C - --progress-bar -o "$out.part" "$base/$rel"
  mv "$out.part" "$out"
done < "$work/needed.sha256"

echo "Verifying $total files against CHECKSUMS.sha256"
( cd "$dest" && shasum -a 256 -c "$work/needed.sha256" --quiet )
echo "$repo@$rev" > "$dest/.variant"
echo "OK. Model/ holds EverParakeet $label. Build and run the app (see README)."
