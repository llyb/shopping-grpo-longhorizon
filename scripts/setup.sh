#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOPSIM_ROOT="${ROOT}/environments/ShopSimulator"
SHOP_ENV_ROOT="${SHOPSIM_ROOT}/shop_env"
ENV_DIR="${SHOPSIM_ROOT}/.venv-shopsim"
COMPRESSED_PRODUCTS="${SHOP_ENV_ROOT}/data/fine_items_eval_train_all.json.gz"
PRODUCTS="${SHOP_ENV_ROOT}/data/items_eval_train.json"
EXPECTED_PRODUCT_SHA256="57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"
MAIN_PYTHON="${MAIN_PYTHON:-3.12}"
SHOPSIM_PYTHON="${SHOPSIM_PYTHON:-3.10}"

if [[ ! -f "${COMPRESSED_PRODUCTS}" ]]; then
  echo "Missing embedded product archive: ${COMPRESSED_PRODUCTS}" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the isolated ShopSimulator environment" >&2
  exit 1
fi

# PyPI 源：默认走清华 TUNA 镜像加速 uv 的下载；海外网络或需要直连官方源时
# 设置 SHOPPING_PYPI_MIRROR=https://pypi.org/simple 覆盖即可。只影响本脚本的子进程。
PYPI_MIRROR="${SHOPPING_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_DEFAULT_INDEX="${PYPI_MIRROR}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

# 受限或低速网络下可设置 SHOPPING_GIT_MIRROR 让 git 走镜像拉取固定版本的 transformers
# （例如 SHOPPING_GIT_MIRROR=https://gh-proxy.com）；只影响本脚本的子进程，不写入全局配置。
GIT_MIRROR="${SHOPPING_GIT_MIRROR:-}"
if [[ -n "${GIT_MIRROR}" ]]; then
  GIT_MIRROR="${GIT_MIRROR%/}"
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.${GIT_MIRROR}/https://github.com/.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/"
fi

cd "${ROOT}"
uv sync --python "${MAIN_PYTHON}" --extra sft --extra grpo

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  uv venv --python "${SHOPSIM_PYTHON}" "${ENV_DIR}"
fi
uv pip install \
  --python "${ENV_DIR}/bin/python" \
  -r "${SHOP_ENV_ROOT}/requirements.txt"

if [[ ! -f "${PRODUCTS}" ]]; then
  temporary_products="${PRODUCTS}.preparing"
  trap 'rm -f "${temporary_products}"' EXIT
  gzip -cd "${COMPRESSED_PRODUCTS}" > "${temporary_products}"
  actual_sha256="$(sha256sum "${temporary_products}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${EXPECTED_PRODUCT_SHA256}" ]]; then
    echo "Product data SHA-256 mismatch: ${actual_sha256}" >&2
    exit 1
  fi
  mv "${temporary_products}" "${PRODUCTS}"
  trap - EXIT
fi

actual_sha256="$(sha256sum "${PRODUCTS}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_PRODUCT_SHA256}" ]]; then
  echo "Existing product data SHA-256 mismatch: ${actual_sha256}" >&2
  exit 1
fi

cd "${SHOP_ENV_ROOT}"
PYTHONPATH=. "${ENV_DIR}/bin/python" scripts/build_index.py

cd "${ROOT}"
"${ROOT}/.venv/bin/python" scripts/apply_verl_dynamic_sampling_patch.py

echo "Shopping Agent training environment is ready."
echo "Product SHA-256: ${actual_sha256}"
echo "Index: ${SHOP_ENV_ROOT}/search_engine/products.sqlite3"
