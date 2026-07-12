const fs = require("fs");
const path = require("path");

const comparisonFile = process.argv[2];
const reportFile = process.argv[3];
const sourceRoot = process.argv[4];
const outputFile = process.argv[5];

if (!comparisonFile || !reportFile || !sourceRoot || !outputFile) {
  throw new Error("usage: node tools/build_release_triage.js <comparison.json> <release-report.json> <source-root> <output.md>");
}

const comparison = JSON.parse(fs.readFileSync(comparisonFile, "utf8"));
const reports = new Map(JSON.parse(fs.readFileSync(reportFile, "utf8")).cases.map((item) => [item.case_id, item]));
const rows = comparison.case_rows.filter((row) => row.detector_correct === false).map((row) => {
  const casePath = path.join(sourceRoot, ...row.case_id.split("/"));
  const sourceExists = fs.existsSync(casePath);
  const logsExist = fs.existsSync(path.join(casePath, "logs", "logs.json"));
  const errors = Array.isArray(row.errors) ? row.errors : [];
  const failure = errors.some((error) => /429|throttl/i.test(error))
    ? "vlm_429"
    : errors.some((error) => /timeout/i.test(error))
      ? "vlm_timeout"
      : errors.length
        ? "vlm_other"
        : "none";
  const category = !sourceExists
    ? "data_missing_case_dir"
    : !logsExist
      ? "data_missing_logs"
      : failure === "none"
        ? "logic_review"
        : "rerun_after_transport_fix";
  const report = reports.get(row.case_id);
  const vision = report?.vision || {};
  const correlator = report?.event_correlator || {};
  const reason = category !== "logic_review"
    ? category
    : Number(vision.vlm_frames || 0) === 0
      ? "no_vlm_frames"
      : Number(vision.vlm_events || 0) === 0
        ? "vlm_no_events"
        : Number(correlator.datalog_facts?.length || 0) === 0
          ? "correlation_no_match"
          : Number(report?.leak_reasoner?.leak_paths?.length || 0) > 0 && row.expected_conclusion !== "data_leak_risk_detected"
            ? "false_confirmed_leak"
            : Number(report?.leak_reasoner?.suspicious_behaviors?.length || 0) > 0 && row.expected_conclusion === "data_leak_risk_detected"
              ? "missing_leak_path"
              : "evidence_semantics_review";
  return { ...row, sourceExists, logsExist, failure, category, reason };
});

const categories = [
  ["data_missing_case_dir", "数据待确认：案例目录不存在"],
  ["data_missing_logs", "数据待确认：缺 logs/logs.json"],
  ["rerun_after_transport_fix", "重跑队列：VLM 429/超时"],
  ["logic_review", "语义复核队列：数据完整且请求成功"],
];

const lines = [
  "# Grid2 错例清单",
  "",
  `来源：\`${comparisonFile}\``,
  `生成时间：${new Date().toISOString()}`,
  "",
  "| 分类 | 数量 | 处理 |",
  "| --- | ---: | --- |",
];

for (const [category, label] of categories) {
  const count = rows.filter((row) => row.category === category).length;
  const action = category.startsWith("data_") ? "确认后从评测集移除或补齐原始数据" : category === "rerun_after_transport_fix" ? "使用低并发与重试机制重跑" : "保留原始数据，按证据链做针对性修复";
  lines.push(`| ${label} | ${count} | ${action} |`);
}

for (const [category, label] of categories) {
  const group = rows.filter((row) => row.category === category);
  if (!group.length) continue;
  lines.push("", `## ${label}`, "", "| 用例 | 期望 | 当前检测 | 证据断点 | VLM 问题 |", "| --- | --- | --- | --- | --- |");
  for (const row of group) {
    const failure = row.failure === "none" ? "无" : row.failure;
    lines.push(`| ${row.case_id} | ${row.expected_conclusion} | ${row.detector_conclusion} | ${row.reason} | ${failure} |`);
  }
}

fs.mkdirSync(path.dirname(outputFile), { recursive: true });
fs.writeFileSync(outputFile, `${lines.join("\n")}\n`, "utf8");
const rerunCaseIds = rows
  .filter((row) => row.category === "rerun_after_transport_fix" || row.category === "logic_review")
  .map((row) => row.case_id)
  .sort();
const rerunListFile = path.join(path.dirname(outputFile), "grid2_rerun_case_ids.txt");
fs.writeFileSync(rerunListFile, `${rerunCaseIds.join("\n")}\n`, "utf8");
console.log(outputFile);
console.log(rerunListFile);
