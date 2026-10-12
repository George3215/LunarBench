#!/usr/bin/env bash
# Fetch the workspace-level third-party code/model clones at their pinned commits.
#
# These repositories live in the outer workspace (default <workspace>/third_party),
# NOT inside the MoonSim Git repository. The manifest docs/third_party_models.json
# is the single source of truth; this script only executes it.
#
# Usage:
#   bash tools/fetch_third_party.sh                 # clone/checkout everything
#   bash tools/fetch_third_party.sh --list          # show dirs, origins, commits
#   bash tools/fetch_third_party.sh --weights       # print weight download commands
#   bash tools/fetch_third_party.sh --only NaVILA   # limit to one directory
#
# Override the destination with LUNARBENCH_THIRD_PARTY.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${HERE}/../docs/third_party_models.json"
ROOT="${LUNARBENCH_THIRD_PARTY:-$(cd "${HERE}/.." && pwd)/../third_party}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "manifest not found: ${MANIFEST}" >&2
  exit 1
fi

mode="clone"
only=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)    mode="list"; shift ;;
    --weights) mode="weights"; shift ;;
    --only)    only="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

repos_tsv() {
  python3 - "${MANIFEST}" "${only}" <<'PY'
import json, sys
manifest, only = sys.argv[1], sys.argv[2]
data = json.load(open(manifest, encoding="utf-8"))
for r in data.get("repositories", []):
    if only and r["dir"] != only:
        continue
    print("\t".join([r["dir"], r["origin"], r["commit"], r.get("state", "")]))
PY
}

weights_lines() {
  python3 - "${MANIFEST}" "${only}" <<'PY'
import json, sys
manifest, only = sys.argv[1], sys.argv[2]
data = json.load(open(manifest, encoding="utf-8"))
for r in data.get("repositories", []):
    if only and r["dir"] != only:
        continue
    for w in r.get("weights", []):
        if not str(w.get("type", "")).startswith("huggingface"):
            continue
        repo = w["repo"]
        local = w.get("local_path") or repo.split("/")[-1]
        extra = " --repo-type dataset" if w.get("repo_type") == "dataset" else ""
        print(f"# {r['dir']}: {w.get('source', '')}")
        print(f"huggingface-cli download {repo}{extra} --local-dir {r['dir']}/{local}")
PY
}

case "${mode}" in
  list)
    printf '%-20s %-56s %s\n' "DIR" "ORIGIN" "COMMIT"
    while IFS=$'\t' read -r dir origin commit state; do
      printf '%-20s %-56s %s  (%s)\n' "${dir}" "${origin}" "${commit}" "${state}"
    done < <(repos_tsv)
    exit 0
    ;;
  weights)
    echo "# 权重不随仓库发布；下面命令会把权重下到 third_party/<dir>/ 下。"
    echo "# Hugging Face 在本机可能不可达，需要时先在目标机器验证网络。"
    weights_lines
    exit 0
    ;;
esac

mkdir -p "${ROOT}"
echo "third_party 根目录：${ROOT}"
rc=0

while IFS=$'\t' read -r dir origin commit state; do
  target="${ROOT}/${dir}"
  echo
  echo "== ${dir} (${state}) =="
  if [[ -d "${target}/.git" ]]; then
    echo "   已存在，抓取并切换到 ${commit}"
    if ! git -C "${target}" fetch --tags --quiet origin; then
      echo "   !! fetch 失败，保留现有内容" >&2; rc=1; continue
    fi
  else
    echo "   克隆 ${origin}"
    if ! git clone --quiet "${origin}" "${target}"; then
      echo "   !! clone 失败，跳过" >&2; rc=1; continue
    fi
  fi
  if git -C "${target}" checkout --quiet "${commit}" 2>/dev/null; then
    echo "   ok: $(git -C "${target}" rev-parse --short HEAD)"
  else
    echo "   !! 无法切换到 ${commit}（该提交可能未被抓取），当前：$(git -C "${target}" rev-parse --short HEAD)" >&2
    rc=1
  fi
done < <(repos_tsv)

echo
echo "完成。可用的权重下载命令：bash tools/fetch_third_party.sh --weights"
echo "不可运行的模型（无公开权重/代码）见 docs/THIRD_PARTY.md，不要把缺席当成对照结果。"
exit "${rc}"
