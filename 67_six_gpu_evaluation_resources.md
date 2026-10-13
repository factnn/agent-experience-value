# 六卡评测行政适配

2026-10-13T09:47:32.793871+00:00

用户最新明确重申六卡上限，当前绑定experiments/SSH_FUTURE_GPU_CAP_V2_20261013.json；前一条四卡登记保留为历史。未来fit80与heldout24可使用实际空闲的0/1/4/5/6/7。仅修改尚未进入任何live V4冻结清单的三个评测文件，不修改原训练协议、job或旧guard；修改前后完整冻结字节核查通过。

显式resource_authorization path/sha必须同时出现在manifest与当前phase authority；authority允许列表和并行6完全一致，远端Git必须包含该cap文件。默认旧四卡路径保留。六卡路径执行同一dose审计/审阅/旧scheduler退出、<=64MiB且无compute PID检查，不伪造四卡plan绕守卫。线程池和启动记录采用验证过的资源列表；无自动重试。

主代理已读取实际diff及旧preflight，对照保留所有科学门槛：完整51终点及审阅、原80/24面板/seed/任务、硬77400秒/面板和27,262,976新token总界、当前停止要求、预测封存发布及21题关闭。25项CPU检查（既有20加新增5）通过，含缺失/错误资源授权、dose/占卡拒绝、cap远端发布及四/六卡80面板内容除输出路径相同。它们使用合成或mock条件，不能称真实评测或完整科学证据。

尚无fit/heldout manifest、phase authority或GPU面板。三个原V4训练仍需自然完成、逐条raw audit及整批四根合并/scope审阅；随后登记当前owner、原budget和对应phase/global停止方式，发布核验后才能启动。六卡仅改变并行，不增加科学任务、token预算或取消封存/审计；不能为填满卡而重跑训练。
