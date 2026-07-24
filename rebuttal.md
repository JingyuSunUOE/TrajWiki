以下是三位审稿人分别对论文提出的问题和局限。但考虑到时间和成本的局限，我最多只能补充一到两个小规模试验。

## 当前补充实验协议更新

下面原始分析中的“40--60 query 小规模实验”已经完成为 60-query pilot。由于该
pilot 的结果已经被查看，后续扩展不能再被称为独立复现或 preregistered
experiment。当前正式补充实验采用固定的 `rebuttal_200_v1`
**post-hoc nested extension**：它包含 pilot 的全部 60 题，并从同一个 282-query
LoCoMo multi-hop population 中扩展到 200 题，seed 为 7，分层配额为
9 个 strict deep-history、132 个 update-sensitive 和 59 个 ordinary queries。
它继续复用同一份不可变 memory build，不重建 snapshots、trajectories 或 Wiki。

正式 answer-level 比较包含 11 个生成配置：

1. Full TrajWiki；
2. No Wiki Routing；
3. Latest Snapshot Only；
4. Flat Raw Memory (32K)；
5. Wiki Summaries Only；
6. Lifecycle State Hidden；
7. No Source-Support Constraint；
8. Full Context (32K)；
9. Naive Dense RAG；
10. Full Context (TrajWiki-Matched)；
11. Flat Raw Memory (TrajWiki-Matched)。

后两个 matched baselines 按 query 使用 regenerated Full TrajWiki 的完整 answer
prompt token 数作为上限，可以直接回应“更简单 baseline 在同 token budget
下是否相近”的质疑。32K 与 matched Flat Raw 使用完全相同的检索排序，只改变
context budget；Full Context 只按完整 message 边界截断。所有新答案由
GPT-4o-mini 生成，并由不知道 method identity 的 Claude Sonnet 4.6 独立判断。
Observed Full Pipeline 单列，不能和统一 prompt 的因果消融混为一谈。Mem0 不再
发生 API 调用，只对已经保存并完成 282-query 对齐验证的答案重新进行独立判断，
并明确标记为 historical external baseline。

60-query parent 的 540 个 GPT generations 和 660 个 Claude judgments 必须逐
job 验证 source hash、协议配置和 prompt SHA 后才能复用。预计新调用为 1,660
个 GPT generations 和 1,940 个 Claude judgments，共 3,600 个新 provider
calls；含父实验后共有 4,800 个 successful logical provider jobs。5,300 的
application-level metered attempt cap 为临时失败和 resume 保留余量；provider
SDK 内部不可见的重试不单独计数。该计数是实验计划，不是结果。

同一 sampling manifest 还会用于离线 retrieval/cost/auditability/failure
analysis、retrieval-weight 与 cutoff robustness，以及 24-case、两名标注者的
counterbalanced provenance audit。统计同时报告普通 paired bootstrap、按总体
stratum prevalence 加权的 bootstrap 和 dialogue-cluster bootstrap。置信区间
仅表示 query sampling uncertainty，不表示多次 generation variance。

证据边界保持不变：本轮不能声称完成了 no-ADD/REVISE/DEPRECATE memory rebuild、
真实 no-repair/no-retry ablation、GraphRAG/RAPTOR 对比、Qwen3-8B regression
重跑、token break-even、医疗安全有效性或合规删除。下面较早的 40--60 query
建议保留为 pilot 设计过程记录；若与本节冲突，以本节和 README 中的当前协议为准。

首先是第一位审稿人：
Questions:
1.Could the authors provide end-to-end answer-level ablations for removing the Memory Wiki, removing trajectory expansion, using only latest snapshots, and using flat raw-memory retrieval? The current counterfactual ablation is useful, but answer-level results would more directly show how each component affects final QA performance.

2.The main table shows that TrajWiki does not consistently outperform all baselines across all backbones, subsets, and metrics. Could the authors discuss failure cases or settings where Full Context or Mem0 remains better? My score would increase if the paper provided a more balanced analysis of these cases.

3.How sensitive is TrajWiki to errors in claim extraction, trajectory matching, and wiki page construction? Since many stages rely on structured LLM outputs, an error propagation analysis or manual audit of memory update accuracy would help clarify the reliability of the framework.

4.The reported token cost is substantial. Could the authors provide a more concrete deployment analysis, such as cost per user/session, memory update frequency, amortized query cost, and comparison with simpler memory baselines under the same budget?

5.For MedMT-Bench, the paper evaluates memory-relevant medical subsets, but medical dialogue also raises safety and reliability concerns. Could the authors discuss whether incorrect memory updates or outdated medical facts may create downstream risks, and whether additional safeguards are needed?

Limitations:
The limitations are partially addressed. The authors acknowledge that TrajWiki is not provenance-perfect, depends on structured LLM outputs, may suffer from summary-mediated information loss, and provides proxy diagnostics rather than human-verified audit accuracy. However, the paper would benefit from a deeper discussion of practical deployment limitations, especially computational cost, memory growth, privacy concerns in long-term user memory, and risks in medical dialogue settings.

然后是第二位审稿人：

Questions:
In Table 2, "Latest Snapshot Only" gets nearly the same retrieval coverage as full TrajWiki (Gold Ref Cov 0.613 vs 0.610) without using any historical snapshot depth. Could you provide answer-level scores under this configuration? The answer-level scores would help to show how much trajectory depth matters for answering quality in addition to audit purposes.

In table 1, there are a few cells where TrajWiki shows regression: 1. Qwen3-8B LoCoMo Multi-Hop, TrajWiki gets F1 30.25 while Full Context gets 39.05; 2. Qwen3-8B LoCoMo Open Domain, TrajWiki gets F1 16.89 while Full Context gets 20.67; 3. Qwen3-8B LoCoMo Open Domain, TrajWiki gets F1 43.46 while Naive RAG gets 46.32. Could you add a brief discussions of what might have caused the regressions?

For the scoring weights in Appendix A.2/A.5, do you have results of other choices of constants/hyper-parameters you tried? This is a lot of hyper-parameters/constants, it would be very useful to know how robust of TrajWiki is to hyper-parameter/constants choices.

最后是第三位审稿人：
Questions:
Provenance quality -> The diagnostic metrics are all automated proxies. Have you considered even a small-scale human evaluation to validate that the provenance chains actually help human auditors identify and correct errors faster?
In the agent literature, "trajectory" overwhelmingly refers to a sequence of observation-action pairs. In this paper, it means a version history of a knowledge item, which is a fundamentally different concept. Consider using a less overloaded term (e.g., revision history).
The paper uses the same model as both the answer generator and the LLM judge. This introduces potential self-preference bias, which could inflate absolute Acc scores and may not uniformly affect all systems. Consider using a stronger, independent model as judge.
The "counterfactual ablation" replays saved intermediate diagnostics without re-running answer generation for ablated variants. This means it can only report retrieval-level proxy metrics but not actual answer quality differences. Why not run proper end-to-end ablations? If the barrier is computational cost, this itself highlights a practical limitation that the system is too expensive to ablate properly.
Limitations:
The paper evaluates on medical domain data but never specifies how the memory store is persisted/protected/governed. The storage mechanism (database? filesystem? in-memory?) is undescribed. The system by design permanently retains immutable episodic snapshots (nothing is deleted, only deprecated), but medical data is subject to regulations such as HIPAA and GDPR.

## 总体判断

三位审稿人的意见看似很多，实际上可以压缩为 **六个共同问题簇**。其中最重要的不是方法动机或新颖性，而是：

1. **当前证据能否证明各组件确实改善最终回答，而不只是改善 retrieval proxy；**
2. **主结果是否受到 judge 偏差、弱模型和特定指标的影响；**
3. **多阶段 LLM 管线是否可靠，错误会在哪里传播；**
4. **高成本、持续增长的记忆和高风险数据能否实际部署。**

最需要重视的是第一点。三位审稿人都独立指出：当前 Table 2 是 offline counterfactual analysis，没有重新生成答案，因此不能证明组件对最终 QA 的贡献；而且 `Latest Snapshot Only` 的 Gold Ref Coverage 甚至略高于 Full TrajWiki，0.613 对 0.610。论文自己也明确说明这些变体没有 rerun answer generation。

所以，有限实验预算应该首先用于一个真正的、但规模受控的 answer-level ablation。

---

## 一、可以归并在一起回答的问题

| 统一问题簇                                  | 涉及的审稿意见                            | 最适合的处理方式                           | 优先级 |
| -------------------------------------- | ---------------------------------- | ---------------------------------- | --- |
| **A. 各组件对最终回答是否必要**                    | R1-Q1；R2-Q1；R3-Q4                  | 小规模 end-to-end ablation            | 最高  |
| **B. Judge 偏差与评价可信度**                  | R3-Q3；也影响 R1-Q2、R2-Q2              | 用独立 judge 重评已有输出，可和 A 合并           | 最高  |
| **C. 非一致性结果与失败场景**                     | R1-Q2；R2-Q2                        | 平衡性澄清 + 基于已有日志的错误分析                | 高   |
| **D. 结构化 LLM 错误、错误传播与 provenance 有效性** | R1-Q3；R3-Q1；部分 R2-Q3               | 现有机制澄清 + 小型人工审计或 stage-wise funnel | 中高  |
| **E. 成本、扩展性和超参数鲁棒性**                   | R1-Q4；R2-Q3；R3-Q4 的成本批评            | 主要依靠已有日志和离线分析，不必大规模实验              | 中高  |
| **F. 医疗安全、隐私、存储和治理**                   | R1-Q5、R1 limitations；R3 limitation | 明确承认适用边界，补充具体治理设计                  | 高   |
| **G. “trajectory”术语过载**                | R3-Q2                              | 直接接受并澄清术语                          | 低   |

实际写 rebuttal 时，不必逐条重复回答。比较好的组织方式是：

1. **End-to-end ablations and independent evaluation**：一起回复 R1-Q1、R2-Q1、R3-Q3/Q4。
2. **Balanced performance and failure modes**：一起回复 R1-Q2、R2-Q2。
3. **Reliability, error propagation, and hyperparameter robustness**：一起回复 R1-Q3、R2-Q3、R3-Q1。
4. **Deployment cost, privacy, and medical safety**：一起回复 R1-Q4/Q5/limitations 和 R3 limitation。
5. 最后一句单独回复 terminology。

---

# 二、最推荐补充的第一个实验

## 小规模 answer-level ablation，并使用独立 judge

这是最值得做、也最能同时解决多个审稿意见的实验。

### 1. 实验范围

建议只做：

* **LoCoMo multi-hop**
* **GPT-4o-mini backbone**
* 大约 **40–60 个问题**
* 复用已经构建好的 snapshots、trajectories、wiki 和原始对话，不重新进行完整 memory construction

只选择 GPT-4o-mini 和 multi-hop 是合理的，因为：

* 当前 counterfactual ablation 本来就是在这个设置上进行的；
* multi-hop 是论文最核心的优势场景；
* 可以直接和 Table 2 对齐；
* 不需要为三个 backbone 全部重跑。

### 2. 不要只做纯随机采样

建议进行预先定义的分层采样：

* 一半是 **shallow cases**：gold evidence 可由最新 snapshot 覆盖；
* 一半是 **deep-history cases**：至少一个 gold source 只位于较早 snapshot，或者涉及 revision、deprecated claim、conflict。

这样才能真正回答 R2 的问题：为什么 aggregate Gold Ref Coverage 几乎一样，但 trajectory depth 仍可能有用。

仅随机抽样可能被大量浅层问题淹没，最后得到 Full 和 Latest Snapshot 几乎无差异，却无法判断历史深度在目标场景中的作用。Figure 4 已经说明多数问题只需要浅层历史，但存在需要更深 snapshot 的长尾。

### 3. 建议比较的配置

至少包括：

1. **Full TrajWiki**
2. **Direct Trajectory Retrieval / No Memory Wiki**
3. **Latest Snapshot Only**
4. **Flat Raw-Memory Retrieval**

预算允许时再加入：

5. **Wiki Summaries Only / No Trajectory Expansion**

所有配置使用相同的：

* context budget；
* answer generator；
* answer prompt；
* 输出格式；
* query subset。

最好仍沿用 Table 2 的 32K 上限，避免审稿人认为 no-Wiki variant 因预算太小被人为削弱。也可以报告实际 context tokens，因为 Full TrajWiki 的平均上下文远小于上限。

### 4. `Wiki Summaries Only` 需要特别定义

这是一个容易产生“机械性结论”的变体。

当前系统要求答案必须引用可见 source refs；wiki summaries 本身没有源消息引用，因此严格保留该验证机制时，这个变体会大量直接 abstain。那样虽然忠实于系统，但 answer-level 结果几乎是由规则预先决定的。

更有信息量的处理方式是：

* 让 summary-only variant 从 wiki summaries 生成答案；
* 同时报告：

  * **raw answer accuracy**；
  * **经过 source-support validation 后的 grounded accuracy / abstention rate**。

这样可以区分：

* wiki summaries 是否包含足以猜对答案的语义；
* trajectory expansion 是否是实现 source grounding 所必需的。

不必为这两个值进行两次生成，只需在同一输出上分别计算验证前后结果。

### 5. 指标

建议报告：

* canonicalized F1；
* canonicalized BLEU-1；
* **独立 judge accuracy**；
* abstention rate；
* valid-source-support rate；
* context tokens；
* overall subset 和 deep-history subset 分开报告。

小样本下最好增加：

* paired bootstrap 95% CI，或者
* pairwise win/tie/loss。

这里的置信区间只表示 query sampling uncertainty，不应表述成多次模型运行方差。

### 6. 用同一个实验同时解决 judge 问题

新 ablation 不要再使用 GPT-4o-mini 自评，而应使用一个：

* 不参与答案生成；
* 更强；
* 不知道 method identity；
* 答案顺序随机化

的独立 judge。

此外，可以把同一 subset 上已经保存的 **Full Context、Mem0 和完整 TrajWiki 输出** 一并交给这个独立 judge，无需重新生成。这样一个实验就能同时回答：

* R1/R2/R3 的 answer-level ablation；
* R3 的 self-preference concern；
* R1/R2 的 Full Context、Mem0 regression concern。

论文当前确实对 GPT-4o-mini 和 Qwen3-8B 使用了同模型作为 generator 和 judge；Qwen3-32B 则由 Qwen3-8B 判断，虽然不是完全相同的模型，但仍是同一模型家族。

### 7. 对不同实验结果要预先准备不同结论

不要把实验设计成只能支持某个预期结果。

* 如果 Full 明显优于 Latest，尤其在 deep-history subset 上：可以有力证明历史 snapshot 对 QA 有贡献。
* 如果 overall 基本持平，但 deep-history subset 上 Full 更好：将贡献准确表述为 **long-tail/update-sensitive benefit**。
* 如果两者在 deep-history 上也持平：应主动缩小论断，承认当前 QA benchmark 并未证明 trajectory depth 的普遍 answer-level 增益，其主要已验证价值是 provenance、conflict history 和 auditability。
* 如果 No-Wiki 在答案准确率上接近 Full，但成本更高：将 Wiki 的核心贡献表述为 **candidate-space reduction、context efficiency 和 diagnosability**，而不是所有场景中的准确率提升。
* 如果 flat raw memory 竞争力较强：强调结构化记忆在长尾、多跳、更新敏感和审计场景中的价值，而非简单声称平坦检索始终更差。

这种预先限定结论的方式会让 rebuttal 更可信。

---

# 三、第二个小实验应该选什么

## 首选：小规模人工 provenance / memory-update audit

因为独立 judge 已经可以并入第一个实验，所以第二个实验最值得用于 R1-Q3 和 R3-Q1。

建议做一个非常克制的人工审计，而不是大型 user study：

* 20–30 个案例；
* 最好两名 annotator；
* 混合包含：

  * 正确答案；
  * page-routing miss；
  * trajectory-selection miss；
  * answer-synthesis error；
  * update/conflict cases。

给 annotator 的任务可以是：

1. 找到支持或反驳答案的原始 source message；
2. 判断最早出现错误的阶段：

   * claim extraction；
   * trajectory assignment；
   * wiki organization；
   * retrieval；
   * answer synthesis；
3. 记录完成时间。

对一半案例提供 full dialogue，对另一半提供 TrajWiki audit packet，第二位 annotator 使用反向分配，避免条件和案例本身混淆。

报告：

* source localization accuracy；
* failure-stage identification accuracy；
* median audit time；
* annotator agreement；
* 被审计链条中 claim/trajectory/wiki 错误的比例。

这会同时回答：

* structured LLM update 的实际正确性；
* provenance chain 是否真的帮助人找错；
* 自动 failure-localization proxy 是否大致可信。

论文当前只能证明 audit packet 更紧凑，例如 source count 从 594.7 降至 195.1、估计 token 数从 20.33K 降至 11.34K；论文也明确承认这不能替代真实的人类审计时间和错误率。

因此，即便这个人工审计很小，它对论文核心的“auditability”主张也有直接价值。

### 人力不足时的替代方案

若短时间内无法安排两名 annotator，第二选择是一个**纯离线的超参数扰动分析**：

* 对 A.5 retrieval weights 做 ±20% 或 ±50% 扰动；
* 报告 Gold Trajectory R@15、Gold Ref Coverage 和排名稳定性；
* 对 A.2 只计算 top-3 trajectory candidate shortlist 在扰动后的保留率，不重跑 LLM trajectory decisions。

不过从审稿影响来看：

> 小型人工 audit 的价值高于完整的权重 sweep。

因为 hyperparameter concern 只有一位审稿人提出，而 provenance/reliability concern 横跨第一和第三位审稿人，并且更贴近论文的中心贡献。

---

# 四、哪些问题主要通过澄清和已有日志即可回答

## 1. Full Context、Mem0 和 Naive RAG 的回归场景

这一点不需要新生成实验，但需要更平衡的表述和一些已有输出的 case analysis。

按 Table 1 的 15 个指标逐格统计，TrajWiki 是：

* GPT-4o-mini：15 项中最好 14 项；
* Qwen3-32B：15 项中最好 12 项；
* Qwen3-8B：15 项中最好 8 项。

因此，回归主要集中在最小的 Qwen3-8B backbone，而不是均匀分布于所有设置。

这可以支持一个合理但需要日志验证的解释：

* TrajWiki 给生成器提供了更复杂的结构化上下文、历史状态和 provenance 信息；
* 较弱 backbone 更容易发生 evidence selection error、overgeneration 或 instruction-following failure；
* Full Context 或简洁 memory 对简单、局部问题有时反而更容易使用。

但这只能作为**待案例验证的解释**，不能直接作为事实写入 rebuttal。

### 第二位审稿人的第三个例子标错了列

审稿人写道：

> Qwen3-8B Open Domain: TrajWiki 43.46 vs Naive RAG 46.32

这两个数字实际对应的是 **Single-Hop F1**，不是 Open-Domain。Open-Domain F1 是 16.89 对 12.18；43.46 和 46.32 位于 Single-Hop F1 列。

回复时可以温和地说：

> “We believe the third cited comparison corresponds to Single-Hop F1 rather than Open-Domain F1; we nevertheless agree that this is a genuine regression and analyze it below.”

不要把重点放在“审稿人看错了”，而是继续正面解释真实回归。

### 两个前述 regression 其实是 metric-specific

Qwen3-8B multi-hop：

* Full Context F1：39.05；
* TrajWiki F1：30.25；
* 但 judge Acc 是 39.36 对 42.14，TrajWiki 更高；
* TrajWiki BLEU-1 也略高，32.14 对 31.16。

Qwen3-8B open-domain：

* Full Context F1/BLEU 更高；
* 但 TrajWiki Acc 是 39.78，Full Context 是 31.54。

因此，不能简单写成“TrajWiki 在这些任务上整体更差”。更准确的是：

> TrajWiki 在部分 Qwen3-8B lexical/slot-level metrics 上回归，但在其中两个设置的 judge accuracy 上仍更高。

这也进一步说明独立 judge 和 case analysis 很重要。需要检查：

* TrajWiki 是否生成了较长或带额外 item 的答案，从而降低 F1 precision；
* 是否属于正确但表达方式不同；
* 是否是真正漏掉必需事实；
* 是否出现 unsupported extras。

### 建议修正论文主张

把：

> consistently outperforms all/most baselines

收紧为：

> achieves the strongest overall performance, with the clearest gains for stronger backbones, multi-hop reasoning, and medical conflict-sensitive settings, while showing regressions on several Qwen3-8B and simple-query metrics.

第一位审稿人已经明确表示，平衡分析会提高评分。这是成本非常低、回报很高的一项修改。

---

## 2. Structured LLM 错误和错误传播

论文已经有不少保护机制，可以在回复中集中说明：

* claim 必须具有有效 source IDs 和 supporting quote；
* unsupported source IDs、空 claims 和 ungrounded exact terms 会被丢弃；
* deterministic fallback 保留 names、places、counts 等精确信息；
* trajectory matching 先确定性 shortlist，再由 structured model 判断；
* claim transition 的模型结果只是 hint，最终状态由 validated deterministic procedure 持久化；
* wiki compiler 有 deterministic fallback；
* final answer 的 source refs 必须在上下文中可见，无有效支持时转为 abstention。 

但这些只是**保护机制**，不是系统准确率的证明。不能只罗列 safeguards 来回应 reviewer。

即使不做人类 audit，也可以基于已有日志增加一个几乎零模型成本的 stage-wise funnel：

[
\text{gold source stored}
\rightarrow
\text{linked to a trajectory}
\rightarrow
\text{reachable from selected pages}
\rightarrow
\text{trajectory selected}
\rightarrow
\text{source included in context}
\rightarrow
\text{used in final support}
\rightarrow
\text{answer correct}.
]

报告每阶段保留率和条件正确率，例如：

* (P(\text{correct}\mid\text{gold source retrieved}))；
* (P(\text{correct}\mid\text{valid support ref}))；
* page miss、trajectory miss、answer-stage miss 各占多少。

论文目前已经显示 retrieved gold-reference coverage 为 0.615，但最终 support gold-reference coverage 只有 0.332，说明大量错误发生在证据已经检索到之后。

这个 funnel 会比现有自动 failure-stage 分类更直观，并且不需要新 LLM 调用。

---

## 3. Hyperparameter robustness

该问题适合“已有结果澄清 + 可选的廉价离线扰动”，不适合投入一次完整实验预算。

首先需要提醒审稿人，论文已经分析了主要容量参数：

* wiki page cutoff (t)；
* trajectory top-(k)；
* trajectory depth (m)。

Figures 3–4 展示了 coverage 与成本的变化，并解释了 (t=k=m=15) 的选择。

其次，需要解释 reviewer 所问的 A.2/A.5 内部常数和这三个主参数不完全相同：

* A.2 的权重主要用于给 trajectory candidates 排序和构造 top-3 shortlist；
* remote-model 实验最终还会进行 structured `CONTINUE/NEW` 判断；
* (\delta=0.72) 仅用于 mock deterministic fallback，不是实际实验中的唯一硬阈值。
* A.5 使用 dense/sparse ranking 后再做 RRF，很多绝对分数最终只通过相对 rank 进入融合，因此对统一尺度变化会相对不敏感；但 entity/facet/family bonuses 仍可能影响排序。

回复前需要诚实确认一件事：

* 这些常数究竟是 hand-set；
* 在小型 development set 上选择；
* 还是经过若干试验后固定。

论文当前并没有清楚交代选择过程。若没有系统 sweep，不要声称“extensively tuned”或“demonstrated robust”。更好的说法是：

> The constants were fixed once and reused across backbones/settings rather than tuned per test condition; we agree that systematic sensitivity analysis remains incomplete.

---

## 4. 成本和部署分析

这部分不需要重跑 baseline。论文已有足够日志，可以给出更具体、也更诚实的分析。

现有结果是：

* memory construction：23.71M tokens；
* query-time retrieval：10.84M；
* answer generation：6.92M；
* mixed repair/validation：1.81M；
* memory construction 占 provider latency 的 83.8%。

对报告的 282 个问题，粗略换算：

* retrieval：约 38.4K tokens/query；
* answer generation：约 24.5K tokens/query；
* 两者合计约 63.0K tokens/query；
* 若把所有 mixed repair 都分摊到查询上，则约 69.4K tokens/query。

这里应明确这是 benchmark-run average，不是现实产品中精确的“每用户成本”。

部署时可以给出一个简单的 amortization 表达式：

[
\bar C(Q)
=========

\frac{C_{\mathrm{memory\ build}}}{Q}
+
C_{\mathrm{query}},
]

其中 (Q) 是同一长期记忆被查询的次数。它解释了为什么 upfront memory construction 只有在后续反复查询时才可能摊薄。

同时必须承认：

* 论文没有给出同预算下 Full Context、Mem0 等 baseline 的完整 provider token 成本；
* 因而不能声称已经达到 token break-even；
* 当前系统的优势是 provenance、search-space reduction 和 reuse，而不是总 token 最少。

论文已经显示 Memory Wiki 将平均 candidate universe 从 130.4 降到 55.6，即 2.35 倍缩减，但论文也明确称当前实现是 computation-heavy，而非 token-minimizing。

### Memory update frequency 必须准确说明

当前实验实现并不是每个 turn 后在线修改 wiki page，而是在一个 sample 的 dialogue replay 完成后，从 trajectory store 刷新 wiki。

因此回复中应写：

* 当前测得的是 sample-level replay/build cost；
* 并没有测量逐 turn 在线更新成本；
* production 可采用 batch refresh、event-triggered refresh 和 caching，但这些属于未来工程设计，而非已验证结果。

不要把当前实验表述成已经证明了低成本的 per-turn incremental deployment。

---

## 5. 医疗安全、隐私、存储和治理

这些问题不适合临时补一个小型 benchmark。最有效的方式是明确划定研究范围，并给出具体的生产要求。

### 首先承认范围

MedMT-Bench 在论文中只是 memory stress test，不应被描述为：

* 临床可部署系统；
* 合规医疗记录系统；
* 能够可靠提供医疗建议的系统。

Source links、timestamps 和 deprecation 提高了可追踪性，但不能保证：

* 医疗事实正确；
* 用户病史最新；
* outdated facts 不会影响回答；
* 临床决策安全。

### “immutable”需要做一个重要澄清

可以把 append-only/immutable 解释为：

> 在一个仍处于有效 retention policy 内的 memory record 中，不通过覆盖旧版本来更新事实。

它不应意味着生产系统永远不能物理删除数据。

应明确指出，实际部署需要支持：

[
\text{hard-delete source message}
\Rightarrow
\text{invalidate/delete dependent snapshots, claims, embeddings, links, and wiki content}
\Rightarrow
\text{rebuild affected indices/pages}.
]

也就是说：

* deprecation 是 epistemic update；
* hard deletion 是 governance operation；
* 两者必须分开。

### 应补充的 safeguards

至少应讨论：

* explicit user consent；
* retention期限和用户可控删除；
* encryption at rest/in transit；
* role-based access control 和 tenant isolation；
* 对 source、derived claims、embeddings 和 page caches 的级联删除；
* medical fact freshness/validity intervals；
* conflicting medical memories 的 quarantine 或人工确认；
* 高风险回答的 external verification 和 human oversight；
* 不将记忆内容作为权威医疗记录或自主临床决策依据。

论文自己的 NeurIPS checklist 已承认当前版本没有充分讨论 privacy、consent、retention、deletion、access control 和 outdated/incorrect memories，因此这两位审稿人的批评和论文当前自述是完全一致的。

### 存储后端必须从代码中确认

这是不能仅凭论文补出的事实。回复前需要检查实现，明确写出当前原型实际使用的是：

* 内存对象；
* JSON/文件系统；
* SQLite；
* 向量数据库；
* 或其他持久化层。

还需要说明：

* 是否默认落盘；
* 是否加密；
* 运行结束后是否清理；
* 是否实现访问控制。

不要使用模糊的“stored securely in a database”，除非代码确实如此。即使当前实验只使用本地 benchmark artifacts，也应承认它不是 HIPAA/GDPR-compliant deployment implementation。

---

## 6. “trajectory”术语

这个问题没有必要争辩，直接接受即可。

不需要更改方法名 TrajWiki，但可在定义处改为：

> **memory revision trajectory**, i.e., a source-grounded revision history of one memory thread, rather than an observation–action trajectory used in agent planning.

后文可以交替使用：

* memory revision trajectory；
* revision history；
* memory evolution history。

这是一项几乎无成本、能体现善意的修改。

---

# 五、不建议把有限实验预算花在哪里

不建议在 rebuttal 期间进行：

1. 三个 backbone × 所有 ablation 的完整重跑；
2. 大规模内部权重网格搜索；
3. 同预算下所有 baseline 的完整成本重跑；
4. 新的医疗安全 benchmark；
5. 为证明 provenance 而做设计不充分、只有一名作者看几个案例的“human evaluation”。

前三项成本太高，后两项容易产生新的方法学问题。

最具性价比的安排是：

### 只能做一个实验

做：

> **分层的 answer-level ablation，并用独立 judge 统一评价 full、baseline 和 ablated outputs。**

它同时覆盖三位审稿人最重叠、最关键的问题。

### 可以做两个实验

第二个做：

> **20–30 案例的小型人工 provenance/update audit。**

若实在没有人力，再以 offline weight perturbation 替代。

---

# 六、建议的最终优先顺序

**必须处理：**

1. answer-level ablation；
2. 独立 judge；
3. 平衡地讨论 Qwen3-8B 和 simple/open-domain regressions；
4. 给出具体成本分解和 amortization，而不是笼统说可摊销；
5. 补充 storage、hard deletion、privacy 和 medical safety 边界。

**最好处理：**

6. stage-wise error-propagation funnel；
7. 小型人工 provenance/update audit；
8. 说明内部权重的选择过程和有限鲁棒性。

**一句话即可解决：**

9. “trajectory”术语过载。

整体上，这些 reviews 并没有否定 TrajWiki 的核心构想。三位审稿人共同要求的是把论文从“有吸引力的结构与 retrieval diagnostics”，进一步推到“有可信 answer-level 证据、评价更独立、部署边界更诚实”。有限资源下，一个设计良好的联合 ablation 表，再加一段严谨的局限性与失败分析，价值远高于分散地补很多零碎实验。
