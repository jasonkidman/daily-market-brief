# Daily Market Brief

一个每天生成的个人投资日报静态站点：展示美国三大指数、最多 8 条全球重要新闻，以及 S&P 500 / Nasdaq-100 的独立回撤加仓状态。系统只管理 ¥200,000 回撤备用金，不包含固定月度定投，也不提供投资建议或自动交易。

## 工作方式

- 行情：yfinance 日线 `Close`，由本地程序计算单日涨跌、YTD、历史最高收盘价和回撤。
- 新闻：仅从 `config/news_sources.yaml` 中的 RSS 获取候选，本地先去重，再交给 DeepSeek 筛选、分类、翻译与摘要。模型只返回候选 ID，原文 URL 由程序映射回来。
- 状态：`state/drawdown_state.json` 保存当前两个独立回撤周期，`state/drawdown_history.json` 保存已结束周期。行情校验失败时不会产生或修改任何新回撤信号。
- 输出：`data/reports/YYYY-MM-DD.json` 保存日报，`site/` 是唯一 Pages 发布目录。仅保留最近 7 个自然日。

## 本地运行

推荐 Python 3.11：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -v
```

真实数据完整生成（需要联网；未配置 Secret 时新闻区会安全降级）：

```bash
export DEEPSEEK_API_KEY="你的密钥"
python -m src.main
python -m src.smoke
```

不联网、不调用 DeepSeek 的确定性完整流程验证：

```bash
python -m src.main --offline-fixture --base-dir work/smoke --report-date 2026-08-12
python -m src.smoke --base-dir work/smoke
```

离线模式生成的页面会明确标注测试夹具，不应作为真实日报发布。

## 首次部署

1. 在 GitHub 仓库进入 `Settings → Secrets and variables → Actions → New repository secret`。
2. 创建名称为 `DEEPSEEK_API_KEY` 的 Secret，值为 DeepSeek API Key。代码只通过 `os.environ` 读取它；不要把 Key 写入任何文件。
3. 进入 `Settings → Pages → Build and deployment → Source`，选择 `GitHub Actions`。
4. 进入 `Actions → Daily Market Report → Run workflow`，手动运行一次。
5. 流程成功后，在该次运行的 `deploy` job 或 Pages 设置中查看站点 URL。

每日 workflow 使用 `Asia/Shanghai` 时区，在周一至周日每天 10:00 运行，也支持手动触发。它会安装依赖、获取并校验数据、更新状态、生成 JSON、保留 7 日、渲染、运行测试与冒烟检查、提交 `data/reports/`、`state/`、`site/` 的真实变化，再通过 GitHub 官方 Pages Actions 发布 `site/`。

## 实际加仓后的操作

完成实际买入后：

1. 进入 `Actions → 确认回撤加仓 → Run workflow`。
2. 选择指数 `sp500` 或 `nasdaq100`。
3. 选择档位 `tier_1` 至 `tier_4`。
4. 点击 `Run workflow`。

只有 `pending` 状态可以确认。`not_triggered` 会以“该档位尚未触发，禁止标记为已执行。”失败；`executed` 会保持不变。确认流程记录 `executed_at`、金额、指数、档位和周期 ID，只重渲染已有日报并重新发布，不会获取行情、RSS 或调用 DeepSeek。

每日任务与确认任务共用 `investment-report-state` 并发组，且 `cancel-in-progress: false`，避免同时修改状态。

## 配置

- `config/market.yaml`：固定三个指数名称与 yfinance ticker。
- `config/drawdown_rules.yaml`：总备用金、70/30 资金池与各档阈值/比例。
- `config/news_sources.yaml`：RSS URL 与 P0/P1/P2 优先级。单源失败只产生 warning，不会改用网页爬虫。

DeepSeek 使用 OpenAI Python SDK 兼容接口：`https://api.deepseek.com`、模型 `deepseek-v4-flash`、JSON 输出，并显式发送 `thinking: disabled`。连续三次校验或调用失败后，新闻区降级，行情与回撤仍继续生成。

## 安全与限制

- Yahoo Finance 与 RSS 都是外部数据源，可能临时不可用；页面会显示健康状态与具体 warning。
- RSS 源是否持续稳定由媒体决定，尤其 Reuters/AP 可能调整地址；失败不会阻断日报。
- GitHub Pages 实际发布状态只能在仓库启用 Pages、提交这些文件并运行 workflow 后验证。
- 不包含实时行情、分钟行情、登录、数据库、自动交易、券商 API、提醒、图表、用户体系、Vercel 或 React。
