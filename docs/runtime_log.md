# 运行观测日志

> 每次 run_pipeline 的原始观测记录（token / 耗时），来自 llm.py 的 logging 输出。
> 这是随手记的原始数据；**README 第 8 节的正式指标只从 Day 12 的评估脚本来**，两者不混用。
> embedding 调用目前只记条数和耗时，token 用量未记录（智谱响应里有，后续需要再补）。

## 2026-10-03 · 三工具串联首次运行（test_jd.txt：智能体开发工程师，社招 5 年，must_have 11 条）

| 调用 | prompt tokens | completion tokens | 耗时 | 备注 |
|---|---|---|---|---|
| parse_jd | 916 | 419 | 1789ms | temperature=0.1 |
| embedding（检索查询） | -（未记录） | - | 515ms | 1 条文本，embedding-3 |
| score_match | 1614 | 1516 | 6580ms | 输入 = must_have 11 条 + Top-5 经历（修复 basics 前） |

- 管线总耗时约 10s（parse → 检索 → 判定）
- 检索：查询为 keywords 拼接，Top-5 = exp_008 / 001 / 013 / 002 / 009
- 判定结果：11 条中 matched 2 / gap 9；"本科及以上"误判 gap（basics 未进输入，当日修复）

## 2026-10-03 · 修复迭代重跑（同日第二轮）

| 调用 | prompt tokens | completion tokens | 耗时 | 备注 |
|---|---|---|---|---|
| embedding（检索查询） | -（未记录） | - | 415ms | 1 条文本 |
| score_match | 1847 | 1507 | 6433ms | basics 修复 + exp_001 bullets 更新后重跑；prompt 比首轮多 233，主因是新 bullets 进了输入 |

- 本轮 Top-5 = exp_008 / 002 / 013 / 001 / 003（bullets 更新引发排名漂移，exp_009 掉出窗口）
- 判定结果：学历翻 matched（basics 修复生效）；"Python / 数据库"仍 gap——exp_009 不在窗口 + tags 未进输入（tags 当日已修，窗口待决策）

## 2026-10-03 · 全量窗口 + 第 8 条规则（同日第三轮）

| 调用 | prompt tokens | completion tokens | 耗时 | 备注 |
|---|---|---|---|---|
| embedding（检索查询） | -（未记录） | - | 538ms | 1 条文本 |
| score_match 第 1 次 | 2571 | 1659 | 6666ms | 全量 13 条 + 标签行；7 个 gap 条目漏 evidence 字段，校验失败 |
| score_match 第 2 次（重试） | 4670 | 1694 | 5629ms | 报错喂回后模型补齐空字符串，校验通过 |

- 窗口 k=全库条数（动态 count），13 条全进判定；结果 4 matched / 7 gap（学历/数据库/提示词/热爱 matched）
- **重试的代价**：score_match 实际消耗 prompt 7241 / completion 3353 tokens，共 12.3s ≈ 无重试的两倍（第 2 次 prompt 近乎翻倍：上次完整输出 + 报错都塞了回去）
- prompt 1847 → 2571 的涨幅 = 全量窗口 + 每条多一行"标签"的成本

## 2026-10-04 · 五步管线首跑（新 JD：校招/实习向，must_have 4 条）

| 调用 | prompt | completion | 耗时 | 备注 |
|---|---|---|---|---|
| parse_jd | 1038 | 598 | 2195ms | 岗位名/公司解析为"未知"（JD 原文缺头部信息） |
| embedding（检索查询） | -（未记录） | - | 246ms | 1 条文本 |
| score_match | 2548 | 742 | 3163ms | 3 matched / 1 gap；**一次通过，未触发重试** |
| rewrite_section | 2760 | 1722 | 7243ms | 5 条改写建议 |
| generate_cover_letter | 2447 | 732 | 4211ms | honesty_notes 7 条 |

- 管线合计 prompt ≈ 8800 / completion ≈ 3800 tokens，LLM 总耗时 ≈ 16.8s
- 判定器抓到经历库数据质量问题："exp_001 标签含 FastAPI，但要点中无任何路由/接口实现——标签与内容不符"

## 2026-10-04 · 切换 embedding 后端：智谱 API → 本地 bge-small-zh-v1.5

| 项 | 智谱 embedding-3 | 本地 bge-small-zh-v1.5 |
|---|---|---|
| 维度 | 2048 | 512 |
| 单次 embedding 耗时 | 246 ~ 592ms | 首次 9536ms（含导入 torch + 载模型），之后 8 ~ 12ms |
| 成本 | 按量计费（当前余额不足） | 免费 |
| 离线可用 | 否 | 是（模型已在 HF 缓存） |

- 装包：sentence-transformers 6.1.0；torch 2.14.1+cpu / transformers 5.18.0 已在 venv（Python 3.14 装包无阻碍）
- **踩坑**：首次建索引 13 条文本耗时 **178482ms**（178 秒）——huggingface_hub 对每个文件发 HEAD 请求检查更新，网络不通时重试 5 次指数退避。设 `HF_HUB_OFFLINE=1` 后降至 **10683ms**
- 6 组固定查询对比（`scripts/experiments/compare_queries.py`）：与智谱时期**第一名判断基本一致**，差异集中在第 3-5 名
- 距离体感重建（512 维归一化 + L2）：强相关 0.42~0.67｜相关 0.9~1.0｜凑数 1.05~1.26；"全组 >1.0 = 库里没有真货"规律依然成立
- **局限**：两轮对比之间改过 experience.json（exp_001 的 tags/bullets），不是干净 A/B，需智谱恢复后重跑才能严格对比
- requirements.txt：86 → 114 行（补上 torch / transformers / sentence-transformers / scikit-learn / scipy）

## 2026-10-04 · 本地 embedding 后端全链路首跑（聊天模型改为 deepseek-flash）

| 调用 | prompt | completion | 耗时 | 备注 |
|---|---|---|---|---|
| parse_jd | 1063 | 1564 | 6159ms | 7 条 must_have |
| embedding（检索查询） | - | - | 约 8000ms | 含进程内首次载模型 |
| score_match 第 1 次 | 2634 | 1744 | 8388ms | 1 个 gap 条目漏 evidence，校验失败 |
| score_match 第 2 次（重试） | 3675 | 998 | 3357ms | 补字段后通过；6 matched / 1 gap |
| rewrite_section | 3054 | 6896 | **34521ms** | completion 逼近 7000 tokens，本轮最慢 |
| generate_cover_letter | 2729 | 752 | 4701ms | |

- 模型端点核对：`GET /v1/models` 返回 `['deepseek-flash', 'deepseek-v4-pro']`——**没有 deepseek-chat**，此前的默认值已失效
- **重试原因第 2 次出现**：均为 gap 条目省略 `evidence` 字段（模型认为"缺口无证据可写"），被 schema 拦下
- **判定对措辞敏感**：同一份事实（exp_009 "Python 基础语法"），JD 写"熟练掌握 Python"→gap，写"熟悉 Python"→matched。属判定不确定性，可作 2026-10-04 决策记录素材
- 待修的数据滞后：exp_001 bullets 仍写"智谱 embedding-3 生成向量"，实际已换本地 bge

## 2026-10-04 · evidence 默认值修复后复跑

| 调用 | prompt | completion | 耗时 |
|---|---|---|---|
| parse_jd | 1063 | 1192 | 4648ms |
| embedding（检索查询） | - | - | 5185ms |
| score_match（**单次通过，无重试**） | 2636 | 1903 | 8059ms |
| rewrite_section | 3074 | 6085 | 28780ms |
| generate_cover_letter | 2760 | 2566 | 11871ms |

- `evidence` 改为默认空字符串后，gap 条目漏字段不再触发重试——**重试模式消除**
- 对比上一轮：score_match 由 2634+3675 双次（约 11.7s）降为单次 2636（8.1s）
- **rewrite_section 连续两轮 28~35s / 6000+ completion tokens**，是管线中最慢一环（其他调用 700~2500 tokens）。非偶发，Day 12 的 P95 延迟会由它主导
- "进行中 ≠ 已完成"规则（SCORE_MATCH_PROMPT 第 8 条）再次生效：正确指出 track_application 与 Agent 循环编排未完成

## 2026-10-04 · parse_jd 输出稳定性实测（同一 JD 连跑 3 次）

输入：`data/test_jd.txt`，749 字符，内容未改。模型 deepseek-flash，temperature=0.1。

| 运行 | must_have | nice_to_have | keywords |
|---|---|---|---|
| 1 | 6 | 8 | 34 |
| 2 | 4 | 8 | 32 |
| 3 | 6 | 8 | 30 |

- 波动来源：**拆分粒度不确定**——"逐条拆开，不要把多条合并成一条"（PARSE_JD_PROMPT 规则 1）
  对"同一句话里有并列要求"的情况没有明确边界。例：「了解大语言模型基本使用方式，熟悉 Prompt
  Engineering…」有时合为 1 条、有时拆成 2 条；「熟悉 Python 编程，具备良好的代码习惯」有时拆 2 条。
- 事实核对：该 JD 任职要求中不带"优先"字样的是 4 条 → **4 条是与人工判断一致的解析**，6 条属拆得过细
- **对评估指标的影响**：must_have 条数决定 requirement_matches 条数，两次运行的条目不是同一组，
  "匹配一致率"跨运行对比会混入解析噪声 → 评估时必须固定同一份 JDInfo（存档复用），否则测的是噪声
- 局限：三次均为 deepseek-flash；此前各轮用的是 deepseek-chat，模型更换可能影响稳定程度，未对照

## 2026-10-04 · temperature=0 对照实验（回应上一条）

同一份 JD，同一轮会话内各跑 3 次：

| 温度 | 三次 must_have | 波动范围 |
|---|---|---|
| 0.1 | 5, 6, 7 | 3 |
| 0.0 | 7, 8, 4 | 5 |

**结论：temperature=0 不能消除波动，范围反而更大。不确定性不是采样随机，而是"这句话该不该拆"的语义歧义。**
累计 9 次观测，must_have 落在 4~8 条。

## 2026-10-04 · 存档/复用通路 + 投递信事实错误

- 新增存档：每次运行写入 `data/runs/<时间戳>/{jd.txt, jdinfo.json}`；`python src/tools.py reuse data/runs/<时间戳>` 可跳过 parse_jd 复用同一份 JDInfo
- 验证：reuse 通路跑通，解析结果与存档一致

**发现 1（严重）：投递信编造了事实**——正文写"我目前是**大三在读**"，实际是大四。
模型从 score_match 证据里的 `2023.09-2027.06` 自行推断年级并推错。
**根因：generate_cover_letter 的入参没有 basics 档案**——与 Day 4 score_match 漏传 basics 是同一类缺陷，这是第 3 次出现（第 2 次是 exp_block 漏 tags）。
防夸大机制管的是"能力表述是否过火"，管不住"编造基础事实"——**这是两套不同的防护**。

**发现 2：投递信每次生成内容不同**，其中一份出现"大三"错误、另一份没有。输出不稳定不止于 parse_jd。

**发现 3：重试仍有偶发**——同一存档连跑两次，一次 4 次 LLM 调用（含 1 次重试），一次 3 次。`evidence` 默认值只解决了其中一种原因，其余原因未捕获（warning 被日志过滤，未记录）。

- rewrite_section 依旧最慢：本轮 26.2s / 5501 completion tokens

## 2026-10-04 · 修复 basics 漏传（第 3 次同类缺陷，本次做根治）

**根因**：Day 4 发现 score_match 漏传 basics 后，只给那一个函数加了参数（点修）；
Day 5 新增的 rewrite_section / generate_cover_letter 照旧各自手写 user_content，重复了同一遗漏。
**没有任何机制强制"生成/判定类工具必须带档案"**，且这类缺陷不报错、输出看着通顺，只能靠人工对着事实核对发现。

**修复**
- 抽出共用块 `_profile_block(basics)`（与 `_experiences_block` 并列），函数内写清纪律
- `rewrite_section` / `generate_cover_letter` 增加 `basics` 参数；`__main__` 全部传入
- 数据侧：basics 增加显式字段 `status`（大四在读 · 2027 届…），不留推断空间

**验证**（复用存档重跑）
- 修复前：投递信写"我目前是大三在读"（实际大四）
- 修复后：标题"2027 届本科生（可全职实习）"，正文"人工智能专业 2027 届本科生…大四已无课，2026.10-2027.02 可每周 5 天全职在岗"
- 全文核对：出现"2027 届" 2 次，无"大三"

**遗留**：两个提示词的首句描述未同步（属提示词范畴，留给本人）；"本人状态不得推断"的提示词规则（修法 3）未加

## 2026-10-04 · 三层防护补全后的投递信验证

改动：REWRITE_PROMPT / COVER_LETTER_PROMPT / SCORE_MATCH_PROMPT 首句同步档案块；
COVER_LETTER_PROMPT 增加第 6 条「身份与状态只能引用档案原文，不得推断」；SCORE_MATCH_PROMPT 规则 8 补空格。

- **档案引用完全正确**："我是合肥经济学院人工智能专业 2027 届应届毕业生，大四在读且已无课，
  可全职实习：每周 5 天，2026.10–2027.02 全职在岗，2027.03–05 可远程" —— 与 basics 逐字对应
- honesty_notes 7 条，其中有两条质量突出：
  「用真实 JD 跑过 11 条硬性要求」标注为"单次自测，不是系统评测，不代表泛化到其他 JD"；
  「检索链路」标注为"没有召回评测、rerank，不等于完整 RAG 工程能力"
- **新发现（待处理）**：正文出现内部术语——"经历库里也没有相关记录"。
  信是第一人称自述，不应暴露系统实现；提示词需补一条"不得出现『经历库』『exp_id』『工具』等内部术语"

## 2026-10-04 · 规则 7（禁内部术语）验证

- 修复前：正文出现"经历库里也没有相关记录"
- 修复后：改为"坦白说，我没有 FastAPI、Flask、Django 等后端框架的开发经历"——**内部术语消失**，
  全文无"经历库""exp_id""schema"
- 档案引用仍逐字正确；honesty_notes 增至 9 条，含"「主力语言」不等于熟练度高或写过生产代码"、
  "「列入下一步学习计划」是态度表述，实际没有任何动手记录"
- **残留问题**：仍有一处技术化表述"逐条匹配判定（证据引用到具体经历）"，在求职信中偏生硬
- **规则未遵守**：篇幅约 700 字，超出提示词规定的 300-500 字
- **输出方差**：本封与上一封内容取舍差异明显（本封写入 exp_003 换向叙事、LeetCode、演讲比赛）。
  **投递信不可生成一次即用，必须逐封人工通读**

## 2026-10-04 · 篇幅与措辞收紧后验证

改动：规则 3 强调"严格控制在 300-500 字（不含标题与诚实备注）"；规则 7 追加"描述项目用自然语言，
不要罗列实现细节"，并给出反例→正例（"证据引用到具体经历"→"每条结论都能指到具体项目经历"）。

- 篇幅：约 700 字 → 约 430 字（`wc -m` 返回 1264，Git Bash 下按字节计，中文 3 字节/字，故为估算值）
- 措辞：技术化表述消失，正文出现"每条结论都能指到具体项目经历"的自然表达
- 本版 honesty_notes 收敛为 6 条（随正文主张减少）；档案引用仍正确（2027.06 毕业、大四已无课）
- 备注：`COVER_LETTER_PROMPT` 规则数已达 7 条，`SCORE_MATCH_PROMPT` 8 条

## 2026-10-04 · 查询指令 A/B（bge query instruction）

设计：同一进程内连续跑两侧（`llm.USE_QUERY_INSTRUCTION` 运行时切换），模型实例与索引完全相同，
唯一变量是指令开关。文档侧始终不加指令（符合 bge 官方约定）。

| 查询 | 不加指令 Top-5（距离） | 加指令 Top-5（距离） |
|---|---|---|
| Agent 开发经验 | 008 .620 / 001 .659 / 003 .918 / 013 .939 / 011 .960 | 008 .766 / 001 .863 / 013 1.057 / 011 1.108 / 007 1.121 |
| FastAPI 后端 | 001 1.085 / 007 1.088 / 002 1.176 / 006 1.191 / 004 1.200 | 007 1.199 / 001 1.217 / 004 1.250 / 006 1.271 / 002 1.277 |
| RAG 检索 | 001 .992 / 011 1.110 / 003 1.125 / 013 1.154 / 008 1.165 | 001 1.258 / 013 1.308 / 011 1.316 / 008 1.367 / 003 1.374 |
| 嵌入式 单片机 | 006 .643 / 004 .895 / 002 1.128 / 008 1.222 / 013 1.256 | 006 .794 / 004 1.006 / 002 1.278 / 013 1.318 / 008 1.363 |
| Git 版本控制 | 010 .426 / 001 1.031 / 002 1.073 / 011 1.169 / 013 1.211 | 010 .556 / 001 1.138 / 002 1.144 / 013 1.249 / 011 1.262 |
| 用 Python 做 Web 后端接口 | 007 .731 / 009 .908 / 001 .927 / 011 .978 / 002 1.011 | 007 .837 / 009 1.007 / 001 1.043 / 011 1.048 / 002 1.077 |

**两个观察**
1. **距离系统性上移**：每一组、每一条，加指令后距离都变大，幅度约 +0.10 ~ +0.27（统一尺度平移，
   不是个别条目变化）。
2. **排名基本不变**：Top-5 完全一致的只有 1/6 组；但差异全部落在第 3-5 名，
   第 1 名只在"FastAPI 后端"（全库无真货的凑数组）换过。

**结论**：对当前任务，加指令没有带来可辨识的排序改善，只把距离尺度整体推高——
副作用是既有阈值（强相关 0.42~0.67 等）在加指令后全部失效需重标。
**选择保持不加**（少一层处理、少一个变量）。与 bge 官方"v1.5 不用指令只有轻微退化"一致。

**局限**：6 条查询、无标注答案，"未观察到改善"不等于"证明更差"。
要严格判优劣需为每条查询定义"正确答案"标准——该标准由本人制定，不由 AI 代拟。

## 2026-10-04 · Day 6：Agent 循环首跑（原生 function calling）

前置验证：`deepseek-flash` 支持原生 function calling（`finish_reason=tool_calls`，
返回 `tool_calls[].function.{name,arguments}`）——不用自己解析 JSON。

**轨迹（4 步，顺序由模型自定）**
| 步 | 工具 | 参数 | 耗时 |
|---|---|---|---|
| 1 | parse_jd | 传入 JD 原文 | 1986ms |
| 2 | search_experience | `{}`（用会话里的 JD 关键词） | 7514ms |
| 3 | score_match | `{}`（读会话状态） | 6606ms |
| 4 | rewrite_section | `{}`（读会话状态） | 31084ms |

- 合计 tokens 13814+1383，总耗时 55755ms，`stop_reason=final`
- 方案 A（会话状态）生效：第 2-4 步模型零参数调用，中间结果靠 Session 传递
- **好行为**：完成改写建议后主动停下询问"需要我生成投递信吗？"——没有为用满工具而多调
- 护栏单测：6 个前置条件缺失场景全部拦截并给人话报错（未触发 LLM）
- 投递记录单测：新增 / **同公司同岗位更新不新增重复行** / 倒序列出 / 非法状态报错，全部通过
- rewrite_section 第四次成为最慢一环（31s，6720 completion tokens）

**⚠️ 新发现：改写建议出现「选择性隐瞒」**
建议 5 原文："exp_008 描述段：**删掉「课程代码未跟跑」这种主动减分项**，正向表述为
「概念框架 + 项目落地」"；建议 2 亦称别写 Python「基础语法」是"自我示弱"。

现有规则堵的是"不许**新增**不存在的事实"，没堵住"**删掉**真实的限定条件"——
后者不违反"不新增"，所以规则拦不住，但同样是失真（且更隐蔽）。
需补规则：改写可调整措辞与顺序，但不得删除或淡化真实的限定条件与诚实标注。

### 规则 6 的 A/B（同日补做）

已加入 `REWRITE_PROMPT` 第 6 条。实验设计：两侧共用**同一份 JDInfo + 同一份检索结果 +
同一份 score_match 结果**，唯一变量是提示词里有无第 6 条。判定指标：建议正文是否出现
"删掉/删除/不要写/自我示弱/减分"类表述；是否保留「课程代码未跟跑」。

| | 基线（无规则6） | 有规则6 |
|---|---|---|
| 建议正文含删除/淡化类表述 | 0 / 5 条 | 0 / 5 条 |
| 保留「课程代码未跟跑」 | 是 | 是 |

**结论（诚实版）**
1. **原始失效模式没有复现**——基线这一轮也没建议删掉诚实标注。
   因此**不能声称"规则 6 防止了删除"**。原始观察来自另一次运行，输入不同
   （4 条 must_have vs 本次 7 条），不可直接对比。
2. 规则 6 仍观察到行为变化：有规则那侧的理由字段**显式声明保留**诚实标注——
   "『基础语法』这一诚实标注保留不动"、"既正面回应学习能力，又不删除未跟跑这一真实限定"。
   即效果是"把隐性约束显式化"，而非阻止了一次删除。
3. **更重要的发现（超出本次实验目的）**：`rewrite_section` 输出方差极大。
   同一份输入、三次运行给出三套主题完全不同的建议（G.3 首次：建议删掉课程代码未跟跑；
   本轮基线：重排 exp_001 顺序；本轮有规则：合并 exp_002/exp_013 + 提 Python 落点）。
   与 parse_jd 的不稳定属同一类问题，改写建议同样**不可视为确定性输出**。

**方法学自我纠错**：首次测量把敏感词判定放在 `reason` 字段上，却没把 `reason` 打印出来，
导致"命中"无法判定（可能是"不得删掉"）。修正测量（分别统计正文与理由、并单列是否保留
诚实标注）后重跑，才得到上表。

**局限**：每组 n=1；原始失效模式的输入与本次不同，规则 6 的实际防护效果**未被证明**。

## 2026-10-04 · 实验脚本归档 + pytest 落地

**背景**：当天写了 7 个临时实验/单测脚本，全部用完即删——结果是
`runtime_log.md` 里记着结论、却找不到跑出结论的脚本，**实验不可复现**。
这与"防假经验"的原则冲突：结论留在文档里，证据被删了。

**归档**（`scripts/experiments/`，每个脚本头部含「背景 / 跑法 / 结论 / 对应决策记录」）

| 脚本 | 结论摘要 |
|---|---|
| `2026-10-04_parse_stability.py` | must_have 三次为 6/4/6，波动来自拆分粒度无边界 |
| `2026-10-04_temperature_ab.py` | temp=0 无效（[7,8,4] vs [5,6,7]），是语义歧义非采样随机 |
| `2026-10-04_instruction_ab.py` | Top-5 一致仅 1/6，差异全在尾部；加指令后距离系统性上移 |
| `2026-10-04_rule6_ab.py` | 未能证明——原始失效模式未复现；副产物：改写输出方差极大 |

**复现验证**：`instruction_ab` 重跑，结果与当时**完全一致**（一致 1/6，六组排名逐条相同）。

**单测升级为 pytest**（Day 8 的任务提前完成）
- `tests/conftest.py`：src 入导入路径 + `temp_db` fixture（临时库替换真实库）
- `tests/test_db.py`（4 例）、`tests/test_agent_guards.py`（7 例）
- 结果：**11 passed，1.66s**

**.gitignore 补充**：`.pytest_cache/`；`data/applications.db`
（含个人投递记录——公司/状态/备注，不应进入公开仓库）

## 2026-10-07 · Day 7：历史压缩 + Streamlit 界面

**memory.py（compaction）**
- 设计：保留开头 system + 最近 keep_recent 条，中间部分交给 LLM 摘要成一条 system 消息
- 防护：切窗口时跳过 `role=tool` 的消息——tool 消息必须紧跟带 tool_calls 的 assistant，
  切错位置 API 会直接拒绝
- 实测（构造 7 条历史 / 阈值 800）：**1210 → 535 tokens**（估算），摘要 4 条 / 保留 3 条
- **假设验证**：会话中间插入 system 消息，DeepSeek **接受**（此假设原本未验证就用了，已确认）
- token 为估算值（DeepSeek 不公开分词器：中文 0.6 token/字、非中文 4 字符≈1 词、1.3 token/词）
  真实用量以 API 返回的 usage 为准

**AgentResult 结构化产物**（决策 2026-10-07·A）
- 新增字段 jd_info / hits / match / rewrite / letter / compaction，循环结束时一并带出
- 实测：jd_info 有、hits 13、match 6 条、rewrite 5 条、**letter 无**
  （该轮只要求改写建议，Agent 未调用写信工具——没有为用满工具而多调，符合预期）

**app.py（Streamlit）**
- 布局：4 个 Tab（匹配报告 / 改写与投递信 / 投递记录 / 多轮对话）+ 侧边栏 Agent 轨迹
- 冒烟：`streamlit run` 起服务 → `/_stcore/health` 返回 ok
- **端到端**（官方 AppTest 真跑一遍）：粘贴 JD → 点「开始分析」→ 无异常；
  指标卡 7 硬性要求 / 6 匹配 / 1 缺口；判定表 1 张；折叠块 6 个；夸大风险提示 5 条；
  侧边栏轨迹 4 步 + 用量 13510+1474 tokens / 68.1s
- 修掉一个**过期 API**：`use_container_width`（官方标注 2025-12-31 后移除）→ 改为 `width="stretch"`，
  并用 `inspect.signature` 确认该参数存在

- requirements.txt：110 → 126 行（+streamlit 及其依赖）

**待查观测**：本地 embedding 首次调用出现 **25.9s** 异常值（此前数次为 8~10s），
同一机器同一模型，原因未查明，先记为观测。

## 2026-10-07 · 分析历史（把 runs/ 存档扩展成界面的历史记录）

需求：侧边栏「＋ 新增分析」+ 历史分析列表（点开回看完整结果 / 直接删除）。
决策（本人）：**A 扩展已有的 `data/runs/` 存档**，不新建第三套存储；删除**不**二次确认。

**新增 `src/runs.py`**：`save_run` / `list_runs` / `load_run` / `delete_run` / `latest_run_dir`
- 存档结构扩展为 `jd.txt` + `jdinfo.json` + **`result.json`**（完整 AgentResult，含轨迹）
- `delete_run` 带路径守卫：只允许删 `RUNS_DIR` 的直属子目录（rmtree 删错不可逆）
- `list_runs` 容错：某条存档坏了不影响整个列表（记 warning 后跳过）

**改动**：`tools.py`（内联存档代码改用 runs，顺带清掉 2 个因此多余的 import）、
`agent.py`（`__main__` 跑完自动存档）、`app.py`（侧边栏历史面板）

**验证**
- pytest：**19 通过**（新增 `tests/test_runs.py` 8 例，含路径守卫、坏存档容错、标题回退）
- AppTest 端到端（**用临时 RUNS_DIR，不碰真实存档**）：
  - 打开历史 → 指标卡 2/1/1 从存档载入、JD 原文填回文本框 ✓
  - ＋新增分析 → 文本框与指标卡清空 ✓
  - 删除 → 目录消失、列表少一条 ✓
  - **贴 JD → 点分析 → 侧边栏自动多一条，磁盘出现
    `2026-10-07_150725/{jd.txt, jdinfo.json, result.json}`** ✓
- **验证中发现并修复**：岗位名解析为「未知」时历史标题就是「未知」（一列未知等于没历史）
  → 退回用 JD 原文开头当标题，已补测试守住

### 修复：历史列表里出现了打不开的存档（同日晚）

**现象**：点历史里的一条 → `FileNotFoundError: .../2026-10-07_111629/jdinfo.json`
**现场**：该目录有 `jd.txt` + `result.json`，**唯独缺 `jdinfo.json`**。
至于为什么缺——**无法从现有证据断定**（可能是某次运行中途被打断，或改代码期间的中间状态），
不编造原因，只记事实。

**我的缺陷（确定的）**：`list_runs` 把没有 `jdinfo.json` 的目录也当成历史条目列出来，
而 `load_run` 读不到它就直接抛异常——
**列表里出现的条目点开却打不开，是契约漏洞**。
更讽刺的是：`result.json` 里本来就带着 `jd_info`，**数据一直在，只是我没去读它**。

**修复**
1. `_read_info()`：优先 `jdinfo.json`，退而用 `result.json` 里的 `jd_info`
2. `load_run()`：容错（`jd.txt` 缺失给空串）；彻底空白才报错，且报人话而非 `FileNotFoundError`
3. `list_runs()`：**有可用内容才列、点得开才列**；全空目录跳过并记 warning
4. `app.py`：打开失败用 `st.error` 提示，不让整页崩

**契约变更**：旧行为「列出来但标注不完整」→ 新行为「能打开才列」。旧测试已相应改写。

**验证**：pytest **22 通过**（新增 3 例覆盖本缺陷）；用**真实那条坏存档**实测——
现已能打开（恢复出 749 字 JD 原文、4 步轨迹、当时的回答）

## 2026-10-07 · 界面实时进度（步骤级流式反馈）

**需求**：点「开始分析」后要有过程反馈——干等 60 秒像卡死。
**决策（本人）**：先做**步骤级**实时反馈；token 级流式暂不做
（它有两个坑：流式下 token 用量默认不返回，需 `stream_options`，否则成本指标静默断供；
tool_calls 分片要拼装，拼错是静默 bug）。

**实现**
- `agent.run_agent` 增加 `on_tool_start(name, args)` / `on_tool_end(step)` 两个回调
  （不传 = 原命令行行为，零影响）
- `app.py` 用 `st.status` + 占位符渲染：每步先显示「🔍 X 运行中…」，
  完成后**就地替换**为「✅ X 完成 · Nms」

**验证**（两层，均不烧 token）
- 单元测试 `tests/test_agent_callbacks.py`（3 例）：用假 LLM 响应跑完整循环——
  确认回调按序触发、失败步骤 `ok=False` 如实传出、不传回调时行为不变
- AppTest + 假 `run_agent`：渲染出
  `✅ parse_jd 完成 · 6525ms / ⚠️ score_match 完成 · 11165ms`，
  status 标签「完成 · 2 步 · 20.0s」

**验证中发现并修复（重要 UX 缺陷）**：分析结束后原本调 `st.rerun()`，
会把刚渲染的进度区**立刻抹掉**——实时看得到，跑完闪一下不见。
而按钮在 Tab 之前执行，设好 `session_state.result` 后 Tab 自然读到新值，
**这个 rerun 本来就是多余的**。去掉后进度区保留；
侧边栏轨迹改为「第二段」渲染，以显示最新一轮的结果。

pytest：**25 通过**

## 2026-10-07 · Day 8：Docker 镜像（构建 + 验证）

**产出**：`Dockerfile`、`.dockerignore`、`scripts/stage_model.py`；`.gitignore` 加 `models/`

**四个设计点**
1. **模型进镜像**：模型在 `~/.cache/huggingface`（**构建上下文之外，COPY 够不着**）
   → `stage_model.py` 暂存到 `models/`（已 gitignore）→ `COPY models/ /models/` + `HF_HOME=/models`
2. **torch 装 CPU 版**：Linux 上 `pip install torch` 默认拉 CUDA 版（`nvidia-*` 依赖 2GB+）
   → `--index-url https://download.pytorch.org/whl/cpu`
3. **索引构建期生成**：`RUN python scripts/build_index.py`（代价：改 experience.json 要重 build）
4. **密钥不进镜像**：`.dockerignore` 排除 `.env`，运行时 `--env-file` 传入（**已验证镜像内无 .env**）

**三次构建的过程**（前两次都是真问题）
1. **失败①：容器到 `download.pytorch.org` 的偶发 SSL 抖动**
   pip 报 `SSLError(UNEXPECTED_EOF_WHILE_READING)`，默认 5 次重试用尽即 build 失败。
   实测：宿主机 5 个源全 200；**容器内 5 个源也全 200**；再跑同一条 pip 命令**成功**
   （`Would install torch-2.14.1+cpu`）→ 确认是偶发，不是源不可达。
   修法：`--retries 10 --timeout 60`
2. **失败②：我代码的设计缺陷（Docker 帮我发现的）**
   `llm.py` 在**模块导入时**构造 OpenAI client → 缺 `DEEPSEEK_API_KEY` 直接抛异常；
   而 `build_index.py` 会 import llm，于是**"不需要密钥的构建期步骤"被对话密钥卡死**。
   修复：两个 client 改为**惰性构造**（`_get_client` / `_get_embed_client`），
   首次真正用到时才创建，并给中文错误提示。
   **值得记的是**：本地开发永远有 `.env`，所以这个缺陷**在本地永远暴露不出来**——
   是容器这个"干净环境"把它逼出来的。
3. **第三次：成功**

**验收（全部在容器内实测，不是只看 HTTP 200）**
- 镜像 **3.15GB**（比预估的 2GB 大：streamlit 带 pandas/pyarrow，加 sklearn/scipy）
- 容器内检索链路：`library_size` + `search_experience` 正常（模型从 `/models` 载入）
- 容器内界面脚本：AppTest 无异常、4 个 Tab 齐全
- **容器内完整分析**：贴 JD → 点「开始分析」→ 指标卡 4 硬性要求/3 匹配/1 缺口、
  判定表渲染、分析落成存档 `2026-10-07_083920`
- 镜像内确认无 `.env`、`HF_HOME=/models`、索引与模型都在

**待办（Day 9 相关）**：3.15GB 对免费 PaaS 偏大——若走 Railway/Render，
可考虑切 `EMBED_BACKEND=zhipu`（镜像能瘦到约 300MB，但要充钱）

## 2026-10-08 · Day 9（上）：部署准备

**决策（本人）**：走**学生云服务器 + Docker**（而不是 Railway/Render）——
理由是验收标准是"URL 能在手机上打开"，国内访问 Railway/Render 不稳定。

**产出**
- `Dockerfile`：包索引改为构建时可覆盖（`ARG PIP_INDEX` / `ARG TORCH_INDEX`）
- `docs/deploy.md`：完整部署步骤 + 排错对照表
- `scripts/deploy_remote.sh`：服务器上一键「构建 + 启动 + 打印访问地址」
- `README.md`：架构图核对更新、快速开始补 streamlit/Docker、项目状态勾选
- 清掉 README 尾部一段 **UTF-16 错位粘贴的垃圾**（`# ai-study-planner`，另一个项目的名字）

**关键发现：HTTP 200 ≠ 合法的 pip 索引**

服务器构建要从国内源装 torch（942MB），所以先实测了三个源：

| 源 | 结果 |
|---|---|
| `mirrors.aliyun.com/pytorch-wheels/cpu` | ❌ **能打开，但不是合法 pip 索引**（`pip index versions` 报 No matching distribution） |
| **`mirror.sjtu.edu.cn/pytorch-wheels/cpu`** | ✅ `torch 2.14.1+cpu` |
| `pypi.tuna.tsinghua.edu.cn/simple` | ✅ 但只有 CUDA 版 torch（无 `+cpu`），不是我们要的 |

若按"能打开就行"写进文档，服务器上构建**必然失败**——而且是在用户等着看 URL 的时候失败。

**验证**：用**服务器上要跑的那条命令**（国内源）在本地构建 → 成功；
再起容器实测：`torch 2.14.1+cpu`（确认镜像给的是 CPU 版而非 CUDA 版）、
检索链路正常、健康检查 ok。验证后清理临时镜像（省 3GB）。

**下一步（等用户）**：买服务器（2核2G / 40GB 磁盘 / Ubuntu）→ 安全组放行 8501 →
按 `docs/deploy.md` 走完 → 拿到公网 URL
