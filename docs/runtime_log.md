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
