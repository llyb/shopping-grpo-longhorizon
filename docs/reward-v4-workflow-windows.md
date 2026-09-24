# Reward v4 工作流 · Windows 本地采集手册（§1–§4）

本手册是 `docs/reward-v4-workflow.md` 的原生 Windows 版，只覆盖该手册的 §1 环境变量、
§2 冻结互斥任务池、§3 启动 ShopSimulator、§4 采集并构建 SFT 数据。**不使用 WSL**，
全部命令在 Windows 本机的 Git Bash 中执行。

`$RUN_ROOT` 的命名、目录布局与产物文件名与 Linux 版逐字一致（`splits/`、
`sft-collection/`），因此 Windows 侧采完数据后，把同一个 `$RUN_ROOT` 交给 Linux 环境，
即可直接从 Linux 手册 §5 续跑。这里只改"怎么启动"，不改"产出什么"。

Windows 只承担 CPU 部分。§5 起的 SFT 训练、GRPO、vLLM 评测依赖 torch / veRL / vLLM，
原生 Windows 装不起来，必须在 Linux 或 WSL 执行。

## 0. 与 Linux 手册的差异对照

| 环节 | Linux 手册写法 | 本手册（Windows 原生） |
| --- | --- | --- |
| 主环境解释器 | `.venv/bin/python` | `.venv/Scripts/python.exe` |
| ShopSimulator 解释器 | `.venv-shopsim/bin/python` | `.venv-shopsim/Scripts/python.exe` |
| 下载前的证书环境 | 无需处理 | 必须先 `unset SSL_CERT_FILE SSL_CERT_DIR`（§1.1） |
| 启动环境的编码 | 无需处理（Linux locale 即 UTF-8） | 必须 `export PYTHONUTF8=1`（§4.1，原因见 §6.6） |
| `RUN_ROOT` | 一次性 `date` 生成，单终端从头跑到尾 | 记进 `outputs/reward-v4-current.txt` 跨终端复用（§2、§6.7） |
| `PYTHONPATH` 分隔符 | `:` | `;`（Windows 语义，Git Bash 下也必须用 `;`） |
| 环境安装 | `bash scripts/setup.sh` | 手工三步（§1.2–§1.4）。`setup.sh` 硬编码 `bin/python`，且默认带 `--extra sft --extra grpo`，本机都不适用 |
| 启动环境 | `bash scripts/start_environment.sh` | 直接 `python pack_api.py`（§4.1） |
| 契约预检 | 必跑 | **本机必然失败，本手册不执行**，原因见 §6.1 |

**终端分工**：终端 A 常驻跑数据准备与采集；终端 B 常驻跑 ShopSimulator。Linux 手册里
"另开一个终端"的约定不变，只是终端 B 的启动方式不同。

## 1. 一次性环境准备

### 1.1 前置检查

```bash
cd /d/shopping-grpo-longhorizon
export ROOT="$(pwd -W)"     # 关键：Git Bash 里取 D:/... 形式的 Windows 路径
export UV="$HOME/.local/bin/uv.exe"
"$UV" --version

# 本机 SSL_CERT_FILE 指向一个已被删掉的 conda 证书路径，会让 uv 连不上 PyPI，必须先清掉
unset SSL_CERT_FILE SSL_CERT_DIR

git config core.autocrlf   # 预期 true（本机默认值），只影响 §6.1 的哈希诊断
```

之后所有 `$ROOT/...` 都展开成 `D:/...`。不要混用 `/d/...` 形式，MSYS 会对形似 POSIX
路径的参数做改写，容易在下游 Python 里变成别名不一致的路径。

**`SSL_CERT_FILE` 是必清项。** 当前 Windows 用户级环境变量把它设成了
`D:\miniconda/ssl/cacert.pem`，而该文件并不存在（miniconda 的真实证书包在
`D:\miniconda\Library\ssl\cacert.pem`）。uv 启动时会警告
`Invalid SSL_CERT_FILE ... No default certificates will be trusted`，随后任何下载都以
`invalid peer certificate: UnknownIssuer` 失败。清掉这两个变量后 uv 回退到自带信任根，
可以正常下载。永久修法见 §6.4。

### 1.2 主环境（只装 CPU 采集依赖）

```bash
cd "$ROOT"
"$UV" sync --python 3.12
export PYTHON="$ROOT/.venv/Scripts/python.exe"
"$PYTHON" -V
```

预期看到 `Installed 3 packages`（`colorama==0.4.6`、`shopping-grpo==0.1.0` 可编辑安装、
`tqdm==4.70.0`），随后 `$PYTHON -V` 打印 `Python 3.12.13`。

- **不要加 `--extra sft --extra grpo`**：那两组依赖（torch / transformers / veRL / vLLM）
  只在 Linux + CUDA 上有意义，原生 Windows 装不上。
- 采集链路本身零第三方依赖：`collect_sft_data.py` 只用到标准库和 `shopping_grpo` 自身，
  所以这个 `.venv` 除了 tqdm 什么都没有，也不影响采集。
- 因此若 `uv` 不可用，用任意 ≥3.10 的 CPython 直接跑 §3–§5 也可行；`.venv` 只是为了让
  解释器版本与仓库约定（3.12）一致。
- 出现 `Failed to hardlink files; falling back to full copy` 只是 UV 缓存在 C:、仓库在 D:
  导致的跨盘回退，不影响结果；嫌吵可以 `export UV_LINK_MODE=copy`。

### 1.3 ShopSimulator 独立环境（Python 3.10）

```bash
export SHOPSIM_ROOT="$ROOT/environments/ShopSimulator"
export SHOPSIM_PY="$SHOPSIM_ROOT/.venv-shopsim/Scripts/python.exe"

"$UV" venv --python 3.10 "$SHOPSIM_ROOT/.venv-shopsim"
"$UV" pip install --python "$SHOPSIM_PY" -r "$SHOPSIM_ROOT/shop_env/requirements.txt"
"$SHOPSIM_PY" -c "import flask, numpy, gym; print(flask.__version__, numpy.__version__, gym.__version__)"
```

版本必须是 3.10（与 `scripts/setup.sh` 的 `SHOPSIM_PYTHON=3.10` 一致）：
`requirements.txt` pin 了 `gym==0.24.0` 与 `numpy==1.26.4`，在更高版本 Python 上装不出来。
`.venv-shopsim` 命中 `.gitignore` 的 `.venv-*/`，不会污染工作区。

实测（本机 2026-09-20）：uv 会先下载 `cpython-3.10.21`，然后 `Built gym==0.24.0` 成功，
共装 17 个包，约 1 分钟。

**已知偏差：numpy 实际装成 2.2.6，不是 `requirements.txt` 里的 1.26.4。**
原因是仓库根 `pyproject.toml` 有 `[tool.uv] override-dependencies = ["numpy==2.2.6"]`
（为 veRL 准备的），而 `uv pip install` 在项目目录下执行时会套用这个 override。Linux 的
`scripts/setup.sh` 也在 `cd "$ROOT"` 之后执行同一条命令，所以两边行为一致，这是仓库既有口径，
不是 Windows 特有问题。

不用管它：ShopSimulator 全仓只在
`web_agent_site/envs/web_agent_text_env.py` 两处用到 numpy（`np.cumsum(weights).tolist()`，
1.x/2.x 结果相同），gym 只作为 `gym.Env` 基类和 `register` 使用。实测在 numpy 2.2.6 下
索引构建、服务启动、检索回放全部正常（见 §1.4、§1.5、§4）。gym 启动时会打印一条
"does not support NumPy 2.0" 的警告，可忽略。

若你确实要按 `requirements.txt` 还原，必须在**项目目录外**执行（否则 override 又会生效）：

```bash
cd "$HOME"   # 换到任意中立目录
"$UV" pip install \
  --python "D:/shopping-grpo-longhorizon/environments/ShopSimulator/.venv-shopsim/Scripts/python.exe" \
  -r "D:/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/requirements.txt"
```

注意这会让 Windows 侧的环境与 Linux `setup.sh` 产出的环境不一致，除非两边都这么做。

### 1.4 展开商品语料（`setup.sh` 的 gzip + sha256 步骤）

```bash
"$PYTHON" - <<'PY'
import gzip, hashlib, os, shutil
from pathlib import Path

root = Path(os.environ["ROOT"])
data = root / "environments/ShopSimulator/shop_env/data"
archive = data / "fine_items_eval_train_all.json.gz"
products = data / "items_eval_train.json"
expected = "57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not products.is_file():
    staging = products.with_suffix(".preparing")
    with gzip.open(archive, "rb") as source, staging.open("wb") as sink:
        shutil.copyfileobj(source, sink, 1024 * 1024)
    actual = sha256(staging)
    if actual != expected:
        staging.unlink()
        raise SystemExit(f"产品语料 SHA-256 不匹配: {actual}")
    staging.replace(products)

actual = sha256(products)
if actual != expected:
    raise SystemExit(f"已有产品语料 SHA-256 不匹配: {actual}")
print("product corpus ok:", actual)
PY
```

`data/*.json` 已被 `shop_env/.gitignore` 忽略（只放行 `*.json.gz`），解压产物不会进版本库。

实测：解压得到 133.8 MB 的 `items_eval_train.json`，sha256 与上面常量精确一致，打印
`product corpus ok: 57b10950...`。这一步幂等，重跑会跳过解压只做校验。

### 1.5 构建检索索引

```bash
cd "$SHOPSIM_ROOT/shop_env"
"$SHOPSIM_PY" scripts/build_index.py
```

产出 `search_engine/products.sqlite3`（约 54 MB）与 `search_engine/products.manifest.json`。
`build_index.py` 自己把 `shop_env` 根插进 `sys.path`，不需要设 `PYTHONPATH`。
索引缺失时 ShopSimulator 会直接拒绝启动（`start.sh` 里就有这道检查）。

实测输出：`product_count: 23421`、`search_version: shopsimulator-multifield-bm25-v2`、
`product_data_sha256` 与 §1.4 的常量一致、`index_sha256: 46c182b21029ff6a8d91d9683ba63d2ddd8ff149382114e29904c27efa84a7f2`。
`product_count` 与 §3 里 23421 的 task_id 总数同源，对不上说明语料被换过。

## 2. 一次性环境变量（对应 Linux 手册 §1）

```bash
export ROOT="$(cd /d/shopping-grpo-longhorizon && pwd -W)"
export PYTHON="$ROOT/.venv/Scripts/python.exe"
export SHOPSIM_ROOT="$ROOT/environments/ShopSimulator"
export SHOPSIM_PY="$SHOPSIM_ROOT/.venv-shopsim/Scripts/python.exe"

# RUN_ROOT 必须跨终端稳定：一旦定下来就记到文件里，后续终端复用同一条路径。
# 否则每开一个新终端都会 date 出一个新目录，§3 写的 splits 在 §5 里就找不到了（见 §6.7）。
export RUN_ROOT_FILE="$ROOT/outputs/reward-v4-current.txt"
if [[ -f "$RUN_ROOT_FILE" ]]; then
  export RUN_ROOT="$(cat "$RUN_ROOT_FILE")"
else
  export RUN_ROOT="$ROOT/outputs/reward-v4-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$RUN_ROOT"
  printf '%s\n' "$RUN_ROOT" > "$RUN_ROOT_FILE"
fi
echo "RUN_ROOT=$RUN_ROOT"

export PYTHONPATH="$ROOT/src;$ROOT"
export SHOPPING_ENVIRONMENT_VERSION="shopsimulator-environment-v2.1"
export SHOPPING_ENV_MANIFEST="$ROOT/data/environment.json"
export SHOPPING_TOOL_CONFIG="$ROOT/configs/tools.json"

# Windows 中文环境默认编码是 GBK，ShopSimulator 读取 UTF-8 语料会失败；见 §6.6
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

export OPENAI_BASE_URL="http://10.128.202.100:3010/v1"
export OPENAI_MODEL="qwen3.7-max"
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY && echo
export OPENAI_API_KEY

if [[ "$OPENAI_BASE_URL" == */chat/completions ]]; then
  export OPENAI_CHAT_URL="$OPENAI_BASE_URL"
else
  export OPENAI_CHAT_URL="${OPENAI_BASE_URL%/}/chat/completions"
fi
```

- **每个新终端都要重跑这一整块**（终端 B 至少要有 `ROOT` / `SHOPSIM_PY` / `PYTHONUTF8`）。
  因为 `RUN_ROOT` 会被记进 `outputs/reward-v4-current.txt`，重跑不会换目录，`$RUN_ROOT` 在各终端
  始终指向同一处。
- 想开新一轮（新目录）时，先 `rm "$ROOT/outputs/reward-v4-current.txt"` 再重跑这一块。
- `$RUN_ROOT` 与 Linux 版同名同结构，是为了让 Linux 侧续跑 §5–§8 时不用改任何路径。
- API key 只存在于当前 shell：metadata 会记录 endpoint 与模型名，但代码显式拒绝序列化密钥。
  收工后执行 `unset OPENAI_API_KEY`。
- `PYTHONPATH` 必须用 `;`。写成 `"$ROOT/src:$ROOT"` 会被当成一个不存在的目录名，
  之后 `import shopping_grpo` 直接失败。

### 2.1 教师接口预检 + thinking 探针（**采集前必须做**）

```bash
curl --fail-with-body --request POST \
  --url "$OPENAI_CHAT_URL" \
  --header "Authorization: Bearer $OPENAI_API_KEY" \
  --header 'Content-Type: application/json' \
  --data "{\"model\":\"$OPENAI_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你是什么模型？\"}],\"temperature\":0.0,\"max_tokens\":64,\"stream\":false}"
```

再用显式关闭思考的版本跑一次，比较两条返回的 `content`：

```bash
curl --fail-with-body --request POST \
  --url "$OPENAI_CHAT_URL" \
  --header "Authorization: Bearer $OPENAI_API_KEY" \
  --header 'Content-Type: application/json' \
  --data "{\"model\":\"$OPENAI_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你是什么模型？\"}],\"temperature\":0.0,\"max_tokens\":64,\"stream\":false,\"thinking\":{\"type\":\"disabled\"}}"
```

判读规则：

- 两条都返回非空 `content` → 可以继续。
- 只有第二条非空（第一条 `content` 是空串、token 被思考吃满）→ **不要开始正式采集**。
  `collect_sft_data.py` 用的客户端只在 `model` 以 `deepseek-v4` 开头时才注入
  `thinking={"type":"disabled"}`（`src/shopping_grpo/evaluation/rollout.py:200`），
  `qwen3.7-max` 走的是"不带该字段"的分支。若网关对 qwen 也默认开思考，所有 rollout 都会因
  `content` 为空而作废，跑几小时拿到 0 条有效数据。
  仓库里另一份客户端（`src/shopping_grpo/evaluation/model_client.py:28`）的族列表是
  `("deepseek-v4", "glm-")`，同样不含 qwen，所以两处都不能替你兜住。
  真要修，就是把 qwen 前缀加进那段判断，属于代码改动，先决策再动手。
- 不要用 `--thinking`：它走 `thinking={"type":"enabled"}` + `reasoning_effort`，
  并要求后续消息带 `reasoning_content`；Linux 手册的采集命令也没开。

## 3. 冻结互斥任务池（对应 Linux 手册 §2）

```bash
"$PYTHON" scripts/build_data_splits.py \
  --output-dir "$RUN_ROOT/splits" \
  --sft-candidates 3000 \
  --grpo-train 1000 \
  --grpo-validation 100 \
  --development 200 \
  --seed 20260917
```

- 必须在仓库根目录执行：脚本用相对路径定位语料与 Final-200。
- 生成的 `$RUN_ROOT/splits/metadata.json` 里必须 `isolation.valid: true`，否则不要往下走。
- 跑完立刻核一次落盘位置，避免后面在别的终端找不到（原因见 §6.7）：

```bash
echo "$RUN_ROOT"; ls -l "$RUN_ROOT/splits"
# 期望看到 sft_candidates.jsonl(3000 行) / grpo_train.jsonl(1000) /
#          grpo_validation.jsonl(100) / development.jsonl(200) / metadata.json
```

- 容量已实测（本机、当天）：语料共 23421 个 task_id，其中 `contract_admissible` 23102 个；
  排除 Final-200 的 200 个 task_id 和同需求文本后，可用唯一 task 22895 个，远大于请求的
  4300 个。不会因容量不足退出。

## 4. 启动 ShopSimulator（对应 Linux 手册 §3）

### 4.1 终端 B：常驻服务

```bash
export ROOT="D:/shopping-grpo-longhorizon"        # 换成你本机实际路径
export SHOPSIM_PY="$ROOT/environments/ShopSimulator/.venv-shopsim/Scripts/python.exe"

export SHOP_ENVIRONMENT_VERSION="shopsimulator-environment-v2.1"
export SHOP_ENV_CONFIG="$ROOT/environments/ShopSimulator/shop_env/configs/environment.json"
export SHOP_SEARCH_INDEX="$ROOT/environments/ShopSimulator/shop_env/search_engine/products.sqlite3"
export SHOP_MAX_STEPS=35
export SHOPSIM_ENV_SLOTS=8
export SHOPSIM_PORT=5700

# 必须：Windows 中文环境默认编码是 GBK，环境初始化会读 UTF-8 语料直接崩，见 §6.6
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

cd "$ROOT/environments/ShopSimulator/shop_env/shop_env"
"$SHOPSIM_PY" pack_api.py
```

这是 `scripts/start_environment.sh` + `shop_env/start.sh` 的等价 Windows 写法，两者都没法
直接用：前者检查 `.venv-shopsim/bin/python`，后者用 `exec python pack_api.py`。

几个不能省的点：

- **必须 `cd` 到 `shop_env/shop_env`**：`pack_api.py` 用 `sys.path.append("../")` 定位
  `web_agent_site`，并把日志写进当前目录的 `shop_agent.log`。在别的目录启动会 `ImportError`。
- 终端 B 是新 shell，`$ROOT` 之类的变量不会继承，要么像上面重新 export，要么把 §2 的块复制过来。
- `SHOPSIM_ENV_SLOTS` 必须 ≥ 采集的 `--workers`（每个 worker 独占一个 slot；slot 用尽时
  服务端会重试 5 次后返回错误）。
- 启动时要逐个初始化环境，等日志出现 `Running on http://0.0.0.0:5700` 才算就绪。
  实测 `SHOPSIM_ENV_SLOTS=1` 时约 12 秒就绪；`=8` 会明显更久，先探路可以降到 1–2。
- Windows 防火墙首次会弹窗，允许本地访问即可（服务只在本机内使用）。
- Flask 开发服务器默认 `threaded=True`，所以 `--workers 8` 能真正并发，不需要额外配置。

就绪的正常日志形态（`SHOPSIM_ENV_SLOTS=8`，实测）：

```text
[INFO] Environment 0 is being initialized
Products loaded.                      <- §6.6 的编码坑就死在这一步之前
Keys cleaned.
100%|...| 23421/23421 [00:00<00:00, ...]
Loaded 23421 goals.
[INFO] Environment 1..7 is being initialized
 * Serving Flask app 'pack_api' (lazy loading)
[INFO]  * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5700
```

`Loaded 23421 goals.` 与 §1.5 的 `product_count` 同源，是"语料装对了"的旁证。

**注意：用浏览器打开 `http://127.0.0.1:5700/` 会返回 `404 Not Found`，这是正常的，不是故障。**
`pack_api.py` 只注册了一个路由——`POST /api/shop_agent`，而且是纯 JSON API，没有任何 HTML 页面：

```python
# environments/ShopSimulator/shop_env/shop_env/pack_api.py:42
@app.route('/api/shop_agent', methods=['POST'])
```

`web_agent_site/app.py` 里那套带 `/` 路由和 `templates/*.html` 的 WebShop 网页界面**没有被
`pack_api.py` 挂载**（它只 import 了 `web_agent_site.utils` 和 `WebAgentTextEnv`），
所以浏览器访问根路径、`/favicon.ico` 一定 404。采集链路走的是
`shopping_grpo.environment.client.ShopAgentEnv`，只打这个 POST 端点，与网页 UI 无关。

### 4.2 终端 A：服务可用性烟测

想一条命令确认服务活着，用 `release_all` 探针（幂等、不占 slot）：

```bash
curl -s -X POST http://127.0.0.1:5700/api/shop_agent \
  -H 'Content-Type: application/json' \
  -d '{"action":"release_all"}'
# 期望：{"result":{"message":"All environments have been initialized"}}   HTTP 200
```

真正的功能烟测还是跑客户端回放：

```bash
cd "$ROOT"
export PYTHONPATH="$ROOT/src;$ROOT"
"$PYTHON" scripts/smoke_shop_env.py --base-url http://127.0.0.1:5700
```

成功时打印 `outputs/smoke/task_0000_<时间戳>.json` 的路径，内容是 `search[乳胶枕]` 的
原始动作回放。端口占用排查：`netstat -ano | grep 5700`。

实测这份回放里 `reset` 会返回 `environment_version=shopsimulator-environment-v2.1`、
`observation_version=shopping-observation-v2`、`goal_options`、`instruction` 和
`user_persona`；`search[乳胶枕]` 这一步的 `result` 里 `done`/`over` 均为 `false`，
`observation_state.page_type=search_home`，正文是 `Page 1 of 8 (Total results: 150)`
开头的 150 条商品列表。用这个判断环境真在干活，而不是只看进程有没有起。

脚本的 `--output-dir` 默认写到 `outputs/smoke/`；想换位置显式传 `--output-dir` 即可。

## 5. 采集并构建 SFT 数据（对应 Linux 手册 §4）

### 5.1 先小样跑通（Windows 上强烈建议）

正式跑 1200 条之前，先花几分钟确认"环境 + 教师 + 验收链路"三件事都通：

```bash
cd "$ROOT"
export PYTHONPATH="$ROOT/src;$ROOT"

"$PYTHON" scripts/collect_sft_data.py \
  --tasks "$RUN_ROOT/splits/sft_candidates.jsonl" \
  --held-out-tasks "$ROOT/data/evaluation/tasks.jsonl" \
  --output-dir "$RUN_ROOT/sft-collection-smoke" \
  --limit 12 \
  --attempts-per-task 1 \
  --workers 1 \
  --base-url http://127.0.0.1:5700 \
  --llm-base-url "$OPENAI_BASE_URL" \
  --model "$OPENAI_MODEL" \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 512 \
  --max-steps 35 \
  --timeout 180 \
  --context-window 24576 \
  --observation-token-budget 1536 \
  --observation-detail-token-budget 4096 \
  --observation-generic-token-budget 768
```

`--workers 1` 时不能同时给 `--target-accepted`（脚本会直接拒绝）。跑完先看
`$RUN_ROOT/sft-collection-smoke/reject_stats.json`。

**这个命令执行期间屏幕上不会有任何输出**，直到全部跑完才打印一行 `collected_raw=` 和汇总 JSON。
按 `--limit` 模式（带 `--target-accepted` 也一样）采集器不逐条打印进度，看起来像卡住了，其实在跑。
判断"是否真在推进"看这两处：

```bash
# 1) 已落盘轨迹数（每完成一条追加一行；这是最直接的进度）
wc -l "$RUN_ROOT/sft-collection-smoke/raw.jsonl"

# 2) 环境端日志（能看到 agent 正在做什么动作）
tail -f "$ROOT/environments/ShopSimulator/shop_env/shop_env/shop_agent.log"
```

实测节奏：12 条 / 约 6 分钟 ≈ **30 秒一条**（每条最多 35 步，每步一次教师调用）。

**实测结果**（本机 2026-09-20，12 条）：
`total=12 accepted=9 rejected=3`，`train=8 validation=1`，即 75% 通过率。

**3 条拒绝全是 `ContextBudgetError`**，这是预期内的结构性失败，不是配置错：

```text
prompt uses 23738 / 24542 / 24834 tokens, above input budget 23552
```

`input_budget = context-window - max-tokens - context-safety-margin
             = 24576 - 512 - 512 = 23552`。轨迹步数一多（实测在第 10–13 步）上下文就撑破预算，
  而 `--context-compaction` 没开时（Linux 手册也没开）采集器选择直接报错，该条记为
  `has_error` → 拒绝，**不会中止整轮采集**。

要不要处理它？**默认不用**。`sft_candidates` 有 3000 条、`--attempts-per-task 2` 共 6000 次机会，
按 75% 通过率仍远够 1200。只想减少这类浪费再考虑 `--context-compaction`（保留固定 prompt，
把较旧的完整工具调用组整体丢掉换空间），代价是它偏离 Linux 手册的口径、可比性下降，属于要单独决策的事。

**教师返回内容是正常的**：9 条成功轨迹里都是真实的工具调用，没有出现 §2.1 担心的
"content 空串"现象。qwen3.7-max 在这个网关上不会因为思考吃满 max_tokens 而废掉 —— 
§2.1 的双 curl 探针仍建议跑一次留档，但已不再是阻塞项。

其它拒绝原因的分层读法：

- `reward_v4_not_gold_purchase` / `strict_success_required` / `incomplete_key_evidence` ——
  "教师没买对"或证据不全 → 完全正常，是这类任务的实际难度，直接放大。
- `has_error` + `status_not_done` + `trajectory_not_done` 同时出现 —— 看 `error.type`：
  `ContextBudgetError` 属上面那种结构性拒绝；若是网络/超时类则先查服务与网关。

**小样结果自检**（决定要不要放大前跑一次，全部期望 ✅）：

```bash
export ROOT="$(pwd -W)"; export PYTHONPATH="$ROOT/src;$ROOT"
"$PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
from shopping_grpo.collection.sft import ALLOWED_MESSAGE_KEYS
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

D = Path(os.environ["RUN_ROOT"]) / "sft-collection-smoke"
rows = lambda n: [json.loads(l) for l in (D / n).read_text(encoding="utf-8").splitlines() if l.strip()]
acc, sft, train, val = rows("accepted.jsonl"), rows("sft.jsonl"), rows("train.jsonl"), rows("validation.jsonl")
m = json.loads((D / "metadata.json").read_text(encoding="utf-8"))


def peak(r):
    return max((e.get("input_tokens", 0) for e in (r.get("context_turn_tokens") or [])), default=0)


print("1 计数一致      :", len(train) + len(val) == len(sft) == m["accepted"])
print("2 训练行键      :", {tuple(sorted(r)) for r in sft} == {("messages", "task_id", "tools", "trajectory_id")})
print("3 tools 同 schema:", all(json.dumps(r["tools"], sort_keys=True) == json.dumps(SHOP_TOOL_SCHEMAS, sort_keys=True) for r in sft))
print("4 消息键白名单  :", not any(set(msg) - ALLOWED_MESSAGE_KEYS for r in sft for msg in r["messages"]))
print("5 无多工具调用  :", not any(len(msg.get("tool_calls") or []) > 1 for r in sft for msg in r["messages"] if msg["role"] == "assistant"))
print("6 全部 buy_now  :", all(any(s.get("tool_name") == "buy_now" or s.get("env_action") == "click[Buy Now]" for s in (r.get("steps") or [])) for r in acc))
print("7 长度上限余量  : max 峰值 =", max(peak(r) for r in acc), "< 24576 =", max(peak(r) for r in acc) < 24576)
print("8 train/val 隔离:", not ({r["task_id"] for r in train} & {r["task_id"] for r in val}))
print("9 与 Final-200  :", not ({r["task_id"] for r in acc} & {json.loads(l)["task_id"] for l in Path("data/evaluation/tasks.jsonl").read_text(encoding="utf-8").splitlines()}))
print("10 奖励口径     :", all((r["terminal_result"]["reward_detail"] or {}).get("strict_success") is True for r in acc))
print("11 sha256 对账  :", all(hashlib.sha256((D / f"{n}.jsonl").read_bytes()).hexdigest() == i["sha256"] for n, i in m["files"].items()))
print("12 不含 API key :", not any("key" in k.lower() for k in m["collection_config"]))
PY
```

第 7 条最关键：`build_supervised_example` 对超过 `--max-length` 的行是**直接丢弃**
（`shopping_grpo/training/sft/dataset.py:86` 返回 `None` → 计入 `dropped`），不是截断。
所以小样里若出现峰值逼近 24576 的行，正式采集放大后会有相当比例的样本在训练阶段被静默丢掉。

### 5.2 正式采集

```bash
"$PYTHON" scripts/collect_sft_data.py \
  --tasks "$RUN_ROOT/splits/sft_candidates.jsonl" \
  --held-out-tasks "$ROOT/data/evaluation/tasks.jsonl" \
  --output-dir "$RUN_ROOT/sft-collection" \
  --target-accepted 1200 \
  --attempts-per-task 2 \
  --workers 8 \
  --validation-ratio 0.10 \
  --seed 20260917 \
  --base-url http://127.0.0.1:5700 \
  --llm-base-url "$OPENAI_BASE_URL" \
  --model "$OPENAI_MODEL" \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 512 \
  --max-steps 35 \
  --timeout 180 \
  --context-window 24576 \
  --observation-token-budget 1536 \
  --observation-detail-token-budget 4096 \
  --observation-generic-token-budget 768
```

与 Linux 手册 §4 逐参数一致，只有 `--model` 由 `$OPENAI_MODEL`（`qwen3.7-max`）驱动。

- 可断点续跑：重跑同一条命令会跳过已完成的 `(task_id, attempt_index)`。
- 中断（Ctrl-C）后进程以退出码 2 结束，已写入的 `raw.jsonl` 保留，直接重跑即可。
- 基础设施故障（环境不可达）会停在当前任务之后退出，不会把坏样本混进结果。

**⚠ `--target-accepted 1200` 数的不是"1200 条唯一合格轨迹"。** 它数的是
**通过验收门的轨迹条数**（`_accepted_count()` 逐条累加 `acceptance_reasons()`，**不按 task 去重**）；
而最终产物 `accepted.jsonl` / `train+validation` 由 `build_collection_artifacts` **按 task 去重**生成。
两者在 `--attempts-per-task 2` 下会差很多，因为同一 task 的两次尝试相邻入队、`--temperature 0.0`
下结果高度相关——2026-09-21 实测：**1200 条通过轨迹只对应 665 个唯一 task，535 条是同题重复**
（535 个 task 两次都成功，130 个成功一次）。详见 §6.8。

因此这个参数要按"唯一 task 数"来定。`--attempts-per-task 1` 时关系是**确定的**
（每个 task 只会贡献 0 或 1 条通过轨迹，不存在重复）：

```text
最终 metadata.accepted = 已有唯一数 + (target − 已有计数)
```

本轮实测 `已有唯一数 = 665`、`已有计数 = 1200`，所以想拿到 ≥1200：

```text
target ≥ 1200 + (1200 − 665) = 1735     实战取 1800 → 落在约 1265
```

```bash
# 补采到 ~1265 条唯一合格轨迹：attempts-per-task 降到 1，去掉重复浪费
--target-accepted 1800 --attempts-per-task 1
```

用默认的 `--attempts-per-task 2` 想达到 1200 唯一 task，实测比例是
`counted : unique ≈ 1200 : 665 ≈ 1.8 : 1`，目标得设到 **约 2200**（但会多花一倍的模型调用）。

主 SFT 集只接收同时满足以下条件的轨迹：`reward_version=shopsimulator-reward-v4`、
`reward_type=gold_purchase`、`reward_valid=true`、`sampling_invalid=false`、
`purchase_success=true`、`termination_reason=gold_purchase`、`strict_success=true`、
`evidence_coverage=1.0`，且没有 `error` / `release_error`、没有一步发出多个 tool_call。

### 5.3 验收门禁

```bash
"$PYTHON" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["RUN_ROOT"]) / "sft-collection" / "metadata.json"
m = json.loads(p.read_text(encoding="utf-8"))
assert m.get("accepted", 0) >= 1200, m
assert m.get("train", 0) > 0 and m.get("validation", 0) > 0, m
print("SFT acceptance gate passed:", m["accepted"], "accepted")
PY

"$PYTHON" scripts/verify_dataset_isolation.py \
  --dataset sft_train="$RUN_ROOT/sft-collection/train.jsonl" \
  --dataset sft_validation="$RUN_ROOT/sft-collection/validation.jsonl" \
  --dataset grpo_train="$RUN_ROOT/splits/grpo_train.jsonl" \
  --dataset grpo_validation="$RUN_ROOT/splits/grpo_validation.jsonl" \
  --dataset development="$RUN_ROOT/splits/development.jsonl" \
  --dataset evaluation="$ROOT/data/evaluation/tasks.jsonl" \
  --report "$RUN_ROOT/dataset-isolation.json"
```

隔离检查以非零状态退出就停下，不要手工删几行继续。

第一条断言（`accepted >= 1200`）**只有在 §5.2 把 `--target-accepted` 按唯一 task 数设对时才会通过**。
按默认的 `--attempts-per-task 2` + `--target-accepted 1200` 跑，实测会停在 665，这条断言必然失败 ——
不是数据坏了，而是目标值定错了口径（§6.8）。补采的办法见 §5.5。

### 5.4 产物清单与落盘位置

正式采集（§5.2）写进 `--output-dir`，即 `$RUN_ROOT/sft-collection/`：

```text
$RUN_ROOT/sft-collection/
├── raw.jsonl            所有轨迹（含被拒的）的原始记录 —— 唯一真相，续跑靠它，不要删
├── accepted.jsonl       通过验收的轨迹全文（带完整审计字段）
├── rejected.jsonl       被拒轨迹
├── reject_stats.json    逐条拒绝原因计数
├── sft.jsonl            通过验收轨迹转成的 SFT 行（action-only，剥掉审计字段）
├── train.jsonl          ★ 训练用训练集（validation_ratio 之外的 90%）
├── validation.jsonl     ★ 训练用验证集（默认 10%）
└── metadata.json        计数 + 每个文件的 rows/sha256 + collection_config（不含 API key）
```

按默认 `$RUN_ROOT`，完整路径就是：

```text
D:/shopping-grpo-longhorizon/outputs/reward-v4-20260920-210802/sft-collection/
```

`train.jsonl` / `validation.jsonl` 就是 Linux 手册 §5 训练要吃的两份文件；`metadata.json` 里
`files.<name>.sha256` 可用于跨平台拷贝后核对完整性。

进度与耗时：实测 `--workers 8` 下 2260 条轨迹用了 **约 2 小时 10 分**（30 秒/条的单 worker 值在
8 worker 下变为约 28 秒/条的有效吞吐）。但**"到 1200 就停"并不等于 1200 条唯一轨迹**，见 §6.8。
采集中途可以用 §5.1 的两条命令盯进度；调度器一次只在途 `--workers` 条、达到目标就停，
所以它不会跑满全部候选。

### 5.5 补采到足够的唯一轨迹

`--target-accepted` 数的是"通过验收门的轨迹条数"，而 `metadata.accepted` 是"按 task 去重后的数量"。
所以**当唯一数不够时，重跑原命令没有用**：`_accepted_count()` 读现有 `raw.jsonl` 已经得到 1200，
`remaining = 1200 - 1200 = 0`，它会立刻停止、只重新派生一遍产物。

正确的补采方式是**抬高目标 + 把重复浪费去掉**：

```bash
cd "$ROOT"
export PYTHONPATH="$ROOT/src;$ROOT"

"$PYTHON" scripts/collect_sft_data.py \
  --tasks "$RUN_ROOT/splits/sft_candidates.jsonl" \
  --held-out-tasks "$ROOT/data/evaluation/tasks.jsonl" \
  --output-dir "$RUN_ROOT/sft-collection" \
  --target-accepted 1800 \
  --attempts-per-task 1 \
  --workers 8 \
  --validation-ratio 0.10 \
  --seed 20260917 \
  --base-url http://127.0.0.1:5700 \
  --llm-base-url "$OPENAI_BASE_URL" \
  --model "$OPENAI_MODEL" \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 512 \
  --max-steps 35 \
  --timeout 180 \
  --context-window 24576 \
  --observation-token-budget 1536 \
  --observation-detail-token-budget 4096 \
  --observation-generic-token-budget 768
```

注意每个参数行末尾都要有 `\`，漏一个就会把下一行当成新命令（`bash -n 文件` 可以先自查）。

为什么这样设：

- `--attempts-per-task 1`：候选表变成"每个 task 只跑一次"，**结构上不可能产生同题重复**，
  于是后续新增的每条通过轨迹都等于一个新增唯一 task。已经跑过的 `(task, 0)` 会被
  `completed_task_attempts()` 跳过，所以它只吃没碰过的 task。
- `--target-accepted 1800`：本轮起点是 `已有计数 1200 / 已有唯一 665`。`attempts-per-task 1`
  下关系是确定的 —— `最终 accepted = 665 + (target − 1200)`，所以 `target ≥ 1735` 才够 1200；
  取 1800 落在约 **1265**，留了 5% 余量。
- 容器里还有 **1870 个没跑过的 task**（`sft_candidates` 3000 条，已用 1130 个），
  按单次尝试通过率实测约 0.53 估算，补 600 个唯一约需新跑 1130 个，够用。
- `--workers 8` 下预计 **1–1.5 小时**（2260 条轨迹实测 2h10m，这次只需约 1130 条）。

跑完重跑 §5.3 的两条命令，`accepted` 应达到 1200 以上。

**另一种做法**（不改采集方式、接受更多模型调用）：保持 `--attempts-per-task 2`，
把目标设为 `1200 × 1.8 ≈ 2200`，让去重后落在 1200 附近。

**最彻底的做法**是修计数口径：把 `scripts/collect_sft_data.py:_accepted_count()` 与
`collect_until_target()` 的增量计数改成"按 task 去重后计数"，让 `--target-accepted` 的语义
与最终 `metadata.accepted` 一致。这是对核心脚本的行为改动，需要单独决策；改完 Linux 手册 §4
的同一条命令也会跟着变正确。

**另一种做法**（不改采集方式、接受更多模型调用）：保持 `--attempts-per-task 2`，
把目标设为 `1200 × 1.8 ≈ 2200`，让去重后落在 1200 附近。

**最彻底的做法**是修计数口径：把 `scripts/collect_sft_data.py:_accepted_count()` 与
`collect_until_target()` 的增量计数改成"按 task 去重后计数"，让 `--target-accepted` 的语义
与最终 `metadata.accepted` 一致。这是对核心脚本的行为改动，需要单独决策；改完 Linux 手册 §4
的同一条命令也会跟着变正确。

通过后这批数据就冻结了。下一步：把整个 `$RUN_ROOT` 拷到 Linux 环境，从 Linux 手册 §5
（SFT 训练与合并）接着跑。拷贝时用 `scp` / `robocopy` 这类二进制方式，不要让任何工具改写
JSONL 的换行。

## 6. Windows 侧已知偏差与硬门禁

### 6.1 环境契约预检在本机必然失败（当天实测）

```bash
cd "$ROOT"
export PYTHONPATH="$ROOT/src;$ROOT"          # 注意：靠包路径导入，$ROOT 必须在 PYTHONPATH 里
export SHOPPING_ENVIRONMENT_VERSION="shopsimulator-environment-v2.1"
export SHOPPING_ENV_MANIFEST="$ROOT/data/environment.json"
export SHOPPING_TOOL_CONFIG="$ROOT/configs/tools.json"
"$PYTHON" -c 'from scripts.check_grpo_runtime import validate_environment_contract; validate_environment_contract()'
```

它校验 `data/environment.json` 的 `runtime_files_sha256`，14 个文件里 **10 个对不上**，
分两类：

- **真实内容漂移 8 个**：`comparators.py`、`config.py`、`goal.py`、`reward.py`、
  `reward_features.py`、`termination.py`、`variant_price.py`、`web_agent_text_env.py`。
  按 LF 归一化后仍然对不上。
- **仅换行符导致的 2 个**：`evidence.py`、`reward_v4.py`。本机 `core.autocrlf=true`
  且仓库没有 `.gitattributes`，checkout 时把 LF 写成了 CRLF；按 LF 归一化后哈希与 manifest 一致。

影响面：这条门禁挡住 Linux 手册 §1 的探针和 §6 的 GRPO 启动，**不影响本手册的 §3–§5**——
划分与采集链路不读 `data/environment.json`。所以 Windows 侧直接不执行这条探针。

不要为了让预检通过而手改 `data/environment.json` 的哈希：那是签入的冻结资产，改了等于伪造
证据链，后续所有轨迹的 reward 口径都失去可验证性。真要修，正确做法是重新生成 manifest，
或者先把 8 个漂移文件的真实来源确认清楚，属于单独决策。同理，加 `.gitattributes`
声明 `* -text` 能消掉那 2 个 CRLF 假失配，但会改变所有人的 checkout 行为，也需要单独评估。

### 6.2 路径、导入与 shell 细节

- 统一用 `pwd -W` 得到的 `D:/...` 形式传给 Python，不要传 `/d/...`。
- `PYTHONPATH` 分隔符必须是 `;`。
- `scripts/*.py` 之间用的是平铺导入（`from dataset_isolation import ...`），所以只能以
  `python scripts/xxx.py` 的形式运行——此时 `scripts/` 会自动进入 `sys.path`。
  用 `-m scripts.xxx` 或从别处 import 都会 `ModuleNotFoundError`。
- Windows 下直接调用 `python.exe` 即可，不需要 `winpty`。

### 6.3 与本次采集无关但容易踩的陈旧描述

`docs/evaluation.md` 里仍有"183 分母"和 r3 提示词版本的说法，实际是 200 题 / r4；
属于文档笔误，不影响本手册的划分与采集。知道即可，别照着改数据。

### 6.4 `SSL_CERT_FILE` 失效（uv/pip 下载全部失败）

报错形态：uv 启动时先警告
`warning: Invalid SSL_CERT_FILE. Path does not exist: D:\miniconda/ssl/cacert.pem.
No default certificates will be trusted.`，随后

```text
× Failed to download `tqdm==4.70.0`
├─▶ client error (Connect)
╰─▶ invalid peer certificate: UnknownIssuer
```

根因不是网络，而是 Windows 用户级环境变量 `SSL_CERT_FILE` 指向了不存在的
`D:\miniconda\ssl\cacert.pem`（miniconda 的证书实际在 `D:\miniconda\Library\ssl\cacert.pem`）。
变量存在但文件不存在时，uv 不会回退到自带信任根，而是"没有可信证书"。

**本次会话的修法**（已实测，`uv sync` 随即成功）：§1.1 里的
`unset SSL_CERT_FILE SSL_CERT_DIR`。

**永久修法**（二选一，改的是用户级环境变量，需你自己确认后再执行）：

```powershell
# 方案 A：直接删除这个坏变量（uv / pip 都用自带或 certifi 的信任根，不需要它）
[Environment]::SetEnvironmentVariable('SSL_CERT_FILE', $null, 'User')

# 方案 B：改指向 miniconda 真实存在的证书包，保留"自定义信任根"的语义
[Environment]::SetEnvironmentVariable('SSL_CERT_FILE', 'D:\miniconda\Library\ssl\cacert.pem', 'User')
```

改完要重开终端才生效（Git Bash 从 Windows 进程环境继承，不会热更新）。

若清掉之后仍然报 `UnknownIssuer`，说明链路上有企业级 TLS 拦截：这时保留坏变量的清理，
改用 `"$UV" sync --python 3.12 --native-tls` 让 uv 走 Windows 证书存储（schannel），
或者把拦截代理的根证书导入 Windows 证书存储后再用 `--native-tls`。

### 6.5 一个只在 WorkBuddy 托管终端里出现的干扰

如果在 WorkBuddy 自己的终端里跑 `uv sync`，它会在构建阶段报
`OSError: [safe-delete][SAFE_DELETE_FAIL_CLOSED] ... windows-sandbox-recycle-bin-unavailable`。
这是因为该终端把 `PYTHONPATH` 指向了自带的 shim 目录，uv 的构建子进程继承了它。
在自己的 MINGW64 终端里不会有这个问题；万一遇到，`unset PYTHONPATH` 即可。

### 6.6 GBK 默认编码导致环境起不来（**启动 ShopSimulator 必踩**）

报错形态：`pack_api.py` 打过 gym 的警告、打印
`[INFO] Environment 0 is being initialized`，然后立刻崩在

```text
File "...\web_agent_site\engine\engine.py", line 241, in load_products
    products = json.load(f)
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa3 in position 81: illegal multibyte sequence
```

原因不在数据：`engine.py` 有两处 `open()` 没显式指定编码

```python
# environments/ShopSimulator/shop_env/web_agent_site/engine/engine.py
line 110:  with open(path) as f:        # human goals
line 240:  with open(filepath) as f:    # products
```

`open()` 不带 `encoding` 时用 `locale.getpreferredencoding(False)`。Linux 上是 UTF-8，所以
Linux 手册没这个问题；Windows 中文环境返回 `cp936`（GBK），于是比 `gym==0.24.0` 更早炸掉——
`gym` 的 warning 只是噪音，真正的错误是这一行 `UnicodeDecodeError`。

**修法（推荐，零改动）**：启动前 `export PYTHONUTF8=1`。
它开启 Python UTF-8 Mode，把 `open()` 的默认编码、stdio 和 `logging.FileHandler` 全部切到
UTF-8，等于让 Windows 与 Linux 参考环境行为一致。实测：

```bash
# 不带 PYTHONUTF8：preferred encoding = cp936, utf8_mode = 0 → 必崩
# 带 PYTHONUTF8=1：utf8_mode = 1 → 单 slot 约 12 秒就绪，
#                  search[乳胶枕] 返回 "Page 1 of 8 (Total results: 150)"
```

`PYTHONIOENCODING=utf-8` 在 UTF-8 Mode 下已是冗余，但一起写上更保险（stdout 重定向到文件时有用），
它也让 `shop_agent.log` 里中文正常落盘。

自检一行：

```bash
"$SHOPSIM_PY" -c "import locale, sys; print(locale.getpreferredencoding(False), sys.flags.utf8_mode)"
# 期望输出：utf-8 1     若是 cp936 0，说明该 shell 还没 export PYTHONUTF8=1
```

**替代修法（不推荐）**：给 `engine.py` 那两处加 `encoding="utf-8"`。`engine.py` 不在
`data/environment.json` 的 14 个受哈希约束文件里，改它不会触发 §6.1 的门禁；但它属于内置环境源码，
改动会让本机环境与其它平台不完全一致，能用环境变量解决就别动源码。

**采集侧不受影响**：`collect_sft_data.py`、`evaluate_shop_benchmark.py`、
`shopping_grpo.collection.*`、`shopping_grpo.evaluation.*` 的读写要么显式 `encoding="utf-8"`，
要么是二进制模式（`open("rb")`），实测在 GBK 默认编码下也能正常跑完划分与采集。
但终端 A 一起 `export PYTHONUTF8=1` 没有坏处，建议两个终端都设。

### 6.7 采集时找不到 `splits/sft_candidates.jsonl`（`$RUN_ROOT` 在换终端后变了）

报错形态：

```text
File "...\scripts\collect_sft_data.py", line 267, in main
    for task in load_tasks(args.tasks)
FileNotFoundError: [Errno 2] No such file or directory:
  'D:\\shopping-grpo-longhorizon\\outputs\\reward-v4-20260920-213142\\splits\\sft_candidates.jsonl'
```

**不是文件被删了，是 `$RUN_ROOT` 换了一个新目录。** 报错里的目录名带时间戳
（`213142` = 21:31:42），如果它和你跑 §3 时看到的时间戳不同，就说明：§3 写进了 A 目录，
之后你在新终端里又执行了一遍 §2 的 export 块，`date +%Y%m%d-%H%M%S` 生成了 B 目录，
§5 于是去 B 目录里找 splits —— 那里是空的。

诊断（看磁盘上到底有哪些轮次、哪一轮有 splits）：

```bash
cd "$ROOT"
ls -d outputs/reward-v4-*/ 
for d in outputs/reward-v4-*/; do echo "$d -> $(ls "$d" 2>/dev/null | tr '\n' ' ')"; done
```

**修法 A（推荐，不重跑）**：把已有 splits 的那一轮固定为当前轮，并写进标记文件：

```bash
export RUN_ROOT="$ROOT/outputs/reward-v4-20260920-210802"   # 换成上面查到的、含 splits 的目录
printf '%s\n' "$RUN_ROOT" > "$ROOT/outputs/reward-v4-current.txt"
echo "$RUN_ROOT"; ls -l "$RUN_ROOT/splits"
```

之后任何新终端只要照 §2 重新 export 一遍，`$RUN_ROOT` 都会解析到同一条路径。
顺手删掉那个空目录：`rmdir "$ROOT/outputs/reward-v4-213142"` 之类。

**修法 B**：在当前（空的）`$RUN_ROOT` 里重跑一次 §3。划分是确定性的（`--seed 20260917`），
内容与上一轮逐字节相同，只是白等一次。

**根因与预防**：Linux 手册里 `RUN_ROOT="$ROOT/outputs/reward-v4-$(date +%Y%m%d-%H%M%S)"`
在"一块终端从头跑到尾"的假设下没问题；Windows 上要开两个终端（终端 A 采集、终端 B 服务），
每次重跑 export 块都会生成新时间戳，所以 §2 改成了"存在标记文件就复用"的写法。
判断当前 `$RUN_ROOT` 是否指对了目录，只需：

```bash
echo "$RUN_ROOT"; ls -l "$RUN_ROOT/splits" 2>&1 | head
```

### 6.8 `--target-accepted` 数的是轨迹数，不是唯一合格轨迹数（**实测踩到**）

**现象**：按 §5.2 默认参数（`--target-accepted 1200 --attempts-per-task 2`）跑完，
§5.3 的门禁 `assert accepted >= 1200` 失败，`metadata.json` 里 `accepted` 只有 665。

**机制**：两个计数口径不一致。

| 环节 | 计数方式 | 位置 |
| --- | --- | --- |
| 采集器停止条件 | 逐条累加 `acceptance_reasons()`，**不按 task 去重** | `scripts/collect_sft_data.py:_accepted_count()` / `collect_until_target()` |
| 最终产物 | **按 task 去重**，同题第二条记 `duplicate_task` 丢弃 | `src/shopping_grpo/collection/sft.py:144-147` |

`--attempts-per-task 2` 让同一 task 的两次尝试在候选队列里**相邻**
（`[(t1,0),(t1,1),(t2,0),(t2,1),...]`，`test_rollout.py:491` 有断言），
在 `--temperature 0.0` 下两次结果高度相关，于是"两次都成功"是常态。

**2026-09-21 实测**（`--target-accepted 1200 --attempts-per-task 2 --workers 8`，耗时 2h10m）：

```text
total                    2260
通过验收门的轨迹条数      1200   ← 采集器数到 1200 就停了
  其中唯一 task           665
  每 task 通过次数        535 个 task 两次都成功，130 个成功一次
metadata.accepted          665   ← build 阶段按 task 去重后
duplicate_tasks_excluded   535
train / validation     599 / 66
```

`665 + 535 = 1200` 精确闭合。也就是说 **1200 里 44.6% 是同题重复**，被去重丢掉了。

**顺带确认的两件事**：

- 主失败模式是 `ContextBudgetError` **908 条**（占 2260 的 40%），另 1 条
  `ObservationProjectionError`、3 条 `invalid_action_limit`；质量拒绝 1060 条。
  这与 §5.1 小样里观察到的结构性拒绝一致，只是比例更高。
- 长度是安全的：665 条通过轨迹的峰值 token 为 **8857 ~ 23511，无一超过 24576**，
  所以训练阶段不会有样本被 `build_supervised_example` 静默丢弃。

**预防**：把 `--target-accepted` 按"唯一 task 数"来设（§5.2 的口径说明、§5.5 的补采做法）。
另外 `tests/test_collect_sft_data_cli.py` 里的 target 用例全部是 `attempts_per_task=1`，
这个组合没有被测试覆盖；如果你要动这块逻辑，记得先补一个 `attempts_per_task=2` 的用例。

## 7. 恢复与清理原则（对应 Linux 手册 §9）

- `raw.jsonl` 是唯一的原始真相，续跑和重算派生文件都靠它，不要删。
- 采集中断后重跑同一条命令即可，已完成的 `(task_id, attempt_index)` 会被跳过。
- 隔离检查失败时重新跑 §3 生成 split，不要手删几行继续。
- ShopSimulator 侧报错先看 `environments/ShopSimulator/shop_env/shop_env/shop_agent.log`。
- 收工：终端 B `Ctrl-C`，终端 A `unset OPENAI_API_KEY`。
- 环境 manifest、Reward v4 合同或环境配置文件一旦变化，这批 split 与已采集数据全部失效，
  必须重跑。
