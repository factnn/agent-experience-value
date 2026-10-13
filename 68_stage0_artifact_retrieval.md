# Stage0 已完成证据与检查点取得

本说明对应原51训练及其迁移评测主证据链。Git保存冻结协议、完成的原始文本证据、审计、adapter配置及各分支evidence_manifest；102个终点二进制（51个adapter权重和51个完整学习状态）保留在授权服务器。克隆仓库后仍需取得这些二进制才能加载原终点；本项目尚未建立公网检查点下载地址。

服务器工作根为 /share/project/zpy/RL/agent_experience_project_20260928。访问者须已有该服务器和本项目目录的授权SSH访问；使用自己的SSH配置，向项目维护者取得访问权限，不在仓库保存凭据。下面假设SSH配置别名为 ssh.platform-multi.baai.ac.cn，端口2222；按自己的获授权配置替换RL_SSH。

在独立的仓库副本中，记录所检出Git提交，并将服务器产物复制到新的restored-stage0目录：

```bash
git rev-parse HEAD
RL_SSH=ssh.platform-multi.baai.ac.cn
RL_REMOTE=/share/project/zpy/RL/agent_experience_project_20260928
mkdir -p restored-stage0/rl/bfcl_intervention_20261008
scp -P 2222 -r "$RL_SSH:$RL_REMOTE/rl/bfcl_intervention_20261008/common" restored-stage0/rl/bfcl_intervention_20261008/
scp -P 2222 -r "$RL_SSH:$RL_REMOTE/rl/bfcl_source_stage0_train_20261012" restored-stage0/rl/
```

这两次复制取得共同起点和51终点的原目录、配置及原始证据，不启动GPU作业。终点102个二进制目前合计2,426,196,086字节；该数不包含共同起点、基础模型和文本证据。按本地Git中的冻结引用核对复制结果，不能用复制来的manifest替换作为校验依据的Git版本：

```python
from pathlib import Path
import hashlib, json

archive = Path("restored-stage0")
def verify(rel, expected, size=None):
    p = archive / rel
    if size is not None:
        assert p.stat().st_size == size, rel
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == expected, rel

protocol = json.loads(Path("rl/BFCL_SOURCE_STAGE0_EXECUTION_PROTOCOL_20261012.json").read_text())
for rel, digest in protocol["input_sha256"].items():
    if rel.startswith("rl/bfcl_intervention_20261008/common/"):
        verify(rel, digest)
manifests = sorted(Path("rl/bfcl_source_stage0_train_20261012").glob("*/evidence_manifest.json"))
assert len(manifests) == 51
for mf in manifests:
    for leaf, entry in json.loads(mf.read_text())["files"].items():
        verify(str(mf.parent / leaf), entry["sha256"], entry["bytes"])
print("Common frozen files and all51 branch artifact bytes verified")
```

共同起点的完整学习状态为 rl/bfcl_intervention_20261008/common/components/learning_state.pt，冻结payload SHA为67da25b41bd57f3b9a5fe7f67f1fa8b0431b9962b9a666e314ec758416ab4c59。每个终点的adapter/adapter_model.safetensors用于政策权重；components/learning_state.pt还包含接续需要的学习状态。components/manifest.json保存payload、基础模型revision及配置谱系。基础模型与环境另按原master.input_sha256、pipeline/requirements-rl.txt、pipeline/runtime_versions.json和pipeline/fetch_assets.py取得并核对；新下载的8B模型不替代本阶段冻结基础模型。

完整51审计与执行段合并入口为 rl/bfcl_source_stage0_train_20261012/audit_status.json（E145），真实训练成本见同目录training_cost_and_exposure.json。完成的fit面板在 rl/bfcl_source_stage0_fit_eval_20261013，每个evidence_manifest区分Git证据与服务器产物；实时queue和PID以服务器为准，Git中的进度文字是带日期快照。

原训练规则见49，后续评测与预测封存顺序见52；当前决策见experiments/NEXT.md，科学范围与未覆盖目标见experiments/SCOPE_REVIEW.md。历史smoke命令只重现其旧工程条件。主项目只接续尚未执行的已登记job，既有终点和历史失败保留。资产取得或SHA校验不等于迁移收益、预测验证或整个研究完成。
