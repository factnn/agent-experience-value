# Stage1 forward15/32：取得已有产物并复现分析

本说明对应用户事前收窄的15条在线RL训练与32个前向评测面板。15条训练已经完成整批审阅（E321）；评测是否齐全，以原控制段、完整证书和带时间的NEXT快照为准。本说明不宣称32个面板已经完成，也不新增训练、评测或准入检查。

## 环境与代码

在独立仓库副本中记录所检出提交。实际RL环境为`.venv-rl`，依赖锁为[`pipeline/requirements-rl-lock.txt`](../pipeline/requirements-rl-lock.txt)，入口依赖见[`pipeline/requirements-rl.txt`](../pipeline/requirements-rl.txt)。本次读取的核心包元数据及文件引用见[`rl/bfcl_stage1_forward15_reproduction_metadata_20261016.json`](../rl/bfcl_stage1_forward15_reproduction_metadata_20261016.json)。该文件是当前环境快照，不是补写的历史执行凭据。

本批使用Python 3.10.12、torch 2.5.1+cu121、transformers 5.2.0、TRL 0.29.0。`pipeline/runtime_versions_train.json`标注的是旧`.venv-train`环境；`pipeline/runtime_versions.json`也不是本批RL依赖锁。复现时按RL锁选择环境，保留原模型和配置；基础模型是已登记的Qwen3-4B，其取得与指纹校验沿用原master及组件谱系。

原科学代码按冻结协议中的`input_sha256`逐文件绑定，不能把后来Git HEAD的变化当成原运行代码被替换。协议为[`BFCL_SOURCE_STAGE1_FORWARD15_PROTOCOL_20261015.json`](../rl/BFCL_SOURCE_STAGE1_FORWARD15_PROTOCOL_20261015.json)，15条训练完成证书为[`audit_status_forward15_20261015.json`](../rl/bfcl_source_stage1_train_20261015/audit_status_forward15_20261015.json)。

## 检查点取得

每个训练终点的`evidence_manifest.json`已经记录服务器路径、文件大小和SHA256，并区分Git文本与服务器二进制。Git clone不包含二进制。需要取得15个终点的adapter权重和完整`components/learning_state.pt`，保留全部原训练重复；相同政策权重不能成为丢弃某个完整学习状态的理由。

共同晚阶段父状态固定为`rl/bfcl_source_stage0_train_20261012/s0_uniform_r0`的完整7步终点，见[原父状态绑定](../rl/bfcl_late_stage_parent_binding_20261012.json)。接续需要其完整优化器、历史与RNG状态，政策权重单独复制不足以恢复原学习干预。

以下命令只复制已有文件。访问者须已有服务器与项目目录的授权，使用自己的SSH配置；仓库不提供凭据或公网检查点下载服务。在独立工作目录中运行：

```bash
git rev-parse HEAD
RL_SSH=your_authorized_ssh_alias
RL_REMOTE=/share/project/zpy/RL/agent_experience_project_20260928
mkdir -p restored-stage1/rl/bfcl_source_stage0_train_20261012
mkdir -p restored-stage1/rl/bfcl_intervention_20261008
scp -P 2222 -r "$RL_SSH:$RL_REMOTE/rl/bfcl_source_stage1_train_20261015" restored-stage1/rl/
scp -P 2222 -r "$RL_SSH:$RL_REMOTE/rl/bfcl_source_stage0_train_20261012/s0_uniform_r0" restored-stage1/rl/bfcl_source_stage0_train_20261012/
scp -P 2222 -r "$RL_SSH:$RL_REMOTE/rl/bfcl_intervention_20261008/common" restored-stage1/rl/bfcl_intervention_20261008/
```

最后一项是父状态原谱系中的共同起点，另按[Stage0取得说明](../68_stage0_artifact_retrieval.md)校验。基础模型文件需另行按原注册引用取得，优先复用既有缓存。上面的复制不重新计费父状态训练成本，也不启动科学子进程。

使用Git版本中的manifest作为校验依据，不用复制来的manifest替换它：

```python
from pathlib import Path
import hashlib, json

archive = Path("restored-stage1")
protocol = json.loads(Path("rl/BFCL_SOURCE_STAGE1_FORWARD15_PROTOCOL_20261015.json").read_text())
assert len(protocol["jobs"]) == 15
new_files, new_bytes = 0, 0
for entry in protocol["jobs"]:
    job = json.loads(Path(entry["path"]).read_text())
    directory = Path(job["output"])
    manifest = json.loads((directory / "evidence_manifest.json").read_text())
    assert manifest["original_job"] == entry
    for leaf, expected in manifest["files"].items():
        if expected["publication"] != "server_artifact":
            continue
        path = archive / directory / leaf
        assert path.stat().st_size == expected["bytes"], path
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        assert h.hexdigest() == expected["sha256"], path
        new_files += 1
        new_bytes += expected["bytes"]
assert new_files == 30
print({"stage1_endpoint_binary_files": new_files, "bytes": new_bytes})
```

固定父状态的`evidence_manifest.json`同样以Git原引用校验其文件。将核验后的文件放入独立仓库副本的原相对路径；保留原始结果，分析输出选择新的目录。

## 完整32面板之后的CPU重放

全部面板原始证据在`rl/bfcl_source_stage1_forward_eval_20261015`，实际封存预测在`rl/bfcl_source_stage1_forward15_prediction_seal_20261015`。必须先取得完整32面板及原完成证书；部分面板不能代替完整G/V或预测误差报告。

```bash
CUDA_VISIBLE_DEVICES= .venv-rl/bin/python pipeline/report_bfcl_stage1_forward15_science.py \
  --manifest rl/bfcl_source_stage1_forward_jobs_20261015/manifest.json \
  --manifest-sha256 01bd201c1ccfab44582e7bfc61148ea0216eccb917197496a1f144548a9e1191 \
  --output rl/reproduced_stage1_forward15
```

该入口重放已有原始证据，核对15条训练、32个面板及原预测，再计算G/V和误差。它不生成新轨迹、不更新参数。输出目录须尚不存在。gpt21建议的整体偏差/来源间误差分解另由`pipeline/describe_bfcl_stage1_forward_errors.py`读取完整报告及其SHA256；主评分继续使用原封存预测。

取得文件、核对版本和CPU重放不构成迁移收益结论。原五模型与辅助H1、完整学习状态与新梯度、共享祖先/重复依赖、真实成本及失败账本仍须在整批科学报告中分别解释。21道保留目标题继续关闭，晚阶段完整51/104及阶段内重新拟合仍按原决定延后。
