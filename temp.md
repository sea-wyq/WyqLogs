
10.94 后台管理网站和账号密码

<https://10.0.200.100>
<systemtest@vsphere.local>
System@323

纪律要求/价值观践行：

严守公司制度，始终遵循开发流程规范。

业绩经历：

主导类脑云平台可观测性方案设计：
    1. 完成监控指标完善；
    2. 完成日志监控方案升级；
    3. 完成监控告警;
    4. 完成秒级监控；
    5. 完成多云监控；
输出方案设计和技术文档《秒级监控方案设计》，《多云监控方案设计》，《多云日志方案设计》，《监控指标汇总详解》，《告警方案流程验证》，《Prometheus版本升级方案》。

参与类脑云平台功能开发：
    1. 完成训练任务适配DeepSpeed，ColossalAI训练框架；
    2. 完成训练任务集成Tesorboard面板；
    3. 完成训练环境健康检测功能开发；
    4. 完成模型仓库功能开发。
输出方案设计和技术文档《Colossal-AI 类脑云集成方案》，《DeepSpeed调研和落地设计》，《基于llama2模型的类脑云全功能全流程使用教程》，《分布式训练容错与诊断-环境检测》。

支撑项目功能开发：
    1. 芜湖大数据项目：对接第三方云平台服务功能。
    2. 算力异构项目：对接跨集群协同训练功能和提供演示示例。
    3. 宿州项目：NPU环境DeepSeek服务部署。
    4. 38所项目：MindSpore/DeepSpeed/ColossalAI等框架基于npu环境构建演示示例。
    5. 新疆项目：服务器性能测试（基于大模型训练进行评估）。

项目经验要求
作为公司一般项目负责人完成至少3个公司一般项目，并取得预期成果，产出相应技术文档。

岗位专业知识要求
1、基础知识：熟悉主要计算机相关网络、存储、内存等原理，能够通过合理设计集成不同模块的优点，解决一些系统问题。
2、系统知识：熟悉不同系统架构主要业务流及其特点，熟悉容器化技术及相关原理，能够根据实际业务场景，通过调整/增加少量模块完成对业务场景的支持，能够借鉴常见系统特点，灵活应用，解决系统问题。

基础知识：了解网络，存储，内存等原理。

系统知识：

针对不同计算卡服务器，通过容器化部署不同的监控exporter来增加，或者定制化exporter对应计算卡的监控能力。
针对不同的device-plugin 完成对不同资源设备的纳管
针对不同sci组件，完成对不同存储服务的支持。

岗位技能要求

1. 技术调研：能够围绕产品/项目需求对模块的方案有调研能力，并形成系统性的调研文档，调研结论对技术规划和方案设计具有指导意义并取得成果，至少2次。
2.设计能力：能够根据业务需求和场景特点，并考虑当前技术成熟度，主导技术模块中的技术方案设计（存储/网络/效率等），在业务实际场景中证明取得至少2次成效。
3.代码能力：能够在遵循代码规范要求，开展有效代码评审基础上，主导基本模块的按时、高质量的代码实现；能够及时发现并解决开发过程中的代码问题，代码评审记录10次。
4.实践能力：能够在开展有效实验方案和代码评审基础上，针对技术模块的一般问题，独立进行实验设计，对实验结果科学分析提出并改进思路，在规定的资源内完成预设的目标，至少3次。

DeepSpeed，ColossalAI 大模型框架调研和落地方案设计
指标告警方案调研和落地方案设计
主导监控方案升级，减少存储压力和提升查询性能。
主导秒级监控方案设计，减少存储压力，提供秒级监控能力
主导多云监控方案设计，减少存储压力，提升查询性能，提供统一查询能力。
模块开发能力Tesorboard，环境健康检测，模型仓库

组织贡献：

与同事间友好沟通，构建舒适的工作氛围。

努力输出技术方案，与部门同事共同进步。

 labels:
    job-type.system.hero.ai: TrainingJob
    resourcepool.system.hero.ai: a14078179380686848491473
    system.hero.ai/job-name: a15787703687180288704327
    system.hero.ai/job-namespace: hero-user
    tenant.id: "12027394361880000"
    user.id: bx57d89f0a785e427d88a6a2d8573d86a4

 labels:
    fuse.serverful.fluid.io/inject: "true"
    job-type.system.hero.ai: TrainingJob
    resourcepool.system.hero.ai: a14078179380686848491473
    system.hero.ai/job-name: a15787703687180288704327
    system.hero.ai/job-namespace: hero-user
    tenant.id: "12027394361880000"
    user.id: bx57d89f0a785e427d88a6a2d8573d86a4
    volcano.sh/job-name: tj-a15787703687180288704327
    volcano.sh/job-namespace: hero-user
    volcano.sh/queue-name: default
    volcano.sh/task-spec: worker

经过验证trainingjob得label会直接传递到podlabel上面。

监控指标存储 : 提供多方面指标（cpu、内存、网络、gpu等）的数据采集，并按照时间维度进行指标存储，支持用户监控指标查询功能。
弹性秒级监控 : 提供监控指标按照秒级频率存储，支持用户针对网络指标开启秒级监控功能。
异常诊断与容错： 提供集群环境常见问题监控诊断，并在检测到后即时恢复，支持用户使用集群计算卡健康检测恢复，ssh连通信检测恢复，通信连通性检测恢复等功能
日志集中管理： 提供日志数据集中存储，支持用户使用任务日志导出到本地，日志导出到存储桶，关键字搜索等功能。
告警与通知： 提供监控指标设置和告警功能配置，支持用户使用监控告警策略设置，接收告警通知设置。
事件采集与管理：提供集群事件存储和管理，支持用户使用事件查看功能。

npu_chip_info_utilization *on ( vdie_id) group_left npu_container_info{pod_name="nb-a15802149899595776962049-a15802149899595776962049-0"}

roce 网络容器化

roce 网卡在安装到节点的时候，每个网卡都会配置一个ip。

rdma-shared-dev-plugin + multus-cni

所以不通过下发pod挂载整个计算节点的资源进行测试，而是通常ssh接口来进行访问。

但仍存在问题，mpirun 命令要求节点ssh免密登录

不管使用不使用pod，进行通信库测试都需要使用gpu显存，这会影响用户的使用体验。

所以在针对通信库测试这项，需要目标节点没有任务参与才能开始进行测试。

如果通信库和带宽时延不需要使用很大的数据量进行验证，对用户的影响就会比较

服务设计

通过CR 直接调用接口 在每个节点执行对应命令。

设计一个daemenset服务，这对节点资源做健康检查，那如何进行多节点的通信验证?

异常检测可以通过告警进行设置的可以通过告警进行触发，无法通过告警进行配置触发的可以通过定时任务进行触发。

dcgm-exporter 镜像本身就存在

nvlink 检测

nvidia-smi nvlink -e -i 0

ecc 错误检查

nvidia-smi -q -d ECC

GPU 掉卡检查

通过nvidia-smi命令或NVML定期查询系统中 GPU 设备的数量和状态。若检测到 GPU 设备数量减少或设备状态异常（如 “Not Present”），则判定发生 GPU 掉卡故障。同时，记录掉卡的时间、GPU 设备 ID 等信息，以便后续分析。

存储挂载检查

定期使用mount命令检查存储挂载点的状态，确保挂载点已正确挂载且可读写。同时，通过尝试读取和写入测试文件来验证存储的可用性。例如，在挂载点下创建一个临时文件并写入数据，然后读取该文件，检查数据是否一致。若挂载点未挂载或读写操作失败，则判定存储挂载存在问题。

npu_chip_info_temperature{pod_name=""} *on (pod) group_left(node) kube_pod_info{node="$node"}

(npu_chip_info_hbm_used_memory{pod_name=""} /npu_chip_info_hbm_total_memory{pod_name=""} *100)*on (pod) group_left(node) kube_pod_info{node="$node"}

kubectl get pvc -n monitoring | grep prometheus | awk '{print $1}' | xargs kubectl delete pvc -n monitoring

mpirun -np 2 -H a100-44,a100-43 \
--allow-run-as-root  \
--output-filename log_output \
--merge-stderr-to-stdout \
--mca mpi_debug 1 \
-x NCCL_IB_GID_INDEX=3 \
-x NCCL_DEBUG=INFO \
-x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
./all_reduce_perf -b 512M -e 16G  -f 2 -g 8

mpirun -np 2 -H a100-44,a100-43  --allow-run-as-root -bind-to none -map-by slot -mca coll_hcoll_enable 0 -mca pml ob1 -mca btl_tcp_if_include  bond0 -mca btl ^openib -x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=bond0  -x NCCL_IB_HCA=^mlx5_8 -x NCCL_IB_TC=128 -x NCCL_IB_QPS_PER_CONNECTION=8 -x NCCL_DEBUG=INFO -x NCCL_ALGO=Ring ./all_reduce_perf -b 32M -e 8G  -f 2 -g 8

mpirun -np 2 -H 10.1.30.2,10.1.30.1  --allow-run-as-root -bind-to none -map-by slot -mca coll_hcoll_enable 0 -mca pml ob1 -mca btl_tcp_if_include  bond0 -mca btl ^openib -x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=bond0  -x NCCL_IB_HCA=^mlx5_8 -x NCCL_IB_TC=128 -x NCCL_IB_QPS_PER_CONNECTION=8 -x NCCL_DEBUG=INFO -x NCCL_ALGO=Ring ./all_reduce_perf -b 32M -e 8G  -f 2 -g 8

mpirun -np 2 -H a100-44,a100-43  --allow-run-as-root -bind-to none -map-by slot -mca coll_hcoll_enable 0 -mca pml ob1 -mca btl_tcp_if_include  bond0 -mca btl ^openib -x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=bond0  -x NCCL_IB_HCA=^mlx5_8 -x NCCL_IB_TC=128 -x NCCL_IB_QPS_PER_CONNECTION=8 -x NCCL_DEBUG=INFO -x NCCL_ALGO=Ring ./all_reduce_perf -b 32M -e 8G  -f 2 -g 8

针对方案总涉及的很多验证想和需求点，可用通过表格的形式来呈现。

deep research

rag

1. 文档解析阶段

文件类型不同， 文本划分。划分切块策略有很多，效果不同。

milvus 向量数据块。

2. embdding 阶段

3. rerank 阶段
topk

4. 问答回复

问题+promt 输入LLM模型

创新药
油气
贵金属
稀土

先考虑设计，讨论方案，再考虑执行，结果不达预期要给他反馈，继续讨论。

- 把他当作一个人，用对人的口吻和他说话。
- 执行具体的行动前，都应该先讨论，反复论证
- 在他给出方案后，也可以分享自己的想法，让他比较论证。
- 确定方案后，再执行。
- 对执行的结果，任何看不明白的地方都要继续讨论，请教他。
- 如果他写错了，也需要以谦卑的语气询问是不是错了，而不是命令他怎么改，除非非常确定。
- 直到完全确认达到预期，结束讨论。
- 一次最好只讨论一个话题。
- 对一些规范性问题，要记录到一个文档里。
- 下次需要遵循某个规范时，要主动引用那个文件，以防他忘记规范。
- 不要主动改代码，让他去改，因为小改动都可能触发深层次的影响，他考虑更周全。
- 要一直站在产品经理的角度、架构师的角度思考问题，细节交给他。
- 必须要 code review，不能当甩手掌柜、全盘接收，否则容易积重难返，也可能让他错误理解需求。
