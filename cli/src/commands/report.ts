import { Command } from "commander";
import chalk from "chalk";
import { table } from "table";
import { getReport } from "../client/api";
import * as fs from "fs";
import * as path from "path";

export function reportCommand(): Command {
  return new Command("report")
    .description("Fetch and display a report for a completed eval run")
    .requiredOption("-r, --run-id <id>", "Eval run ID")
    .option("-o, --output <file>", "Save report to JSON file")
    .action(async (opts) => {
      try {
        const report = await getReport(opts.runId);

        console.log(chalk.bold.cyan(`\nEvalForge Report — Run ${report.run_id}`));
        console.log(chalk.gray("─".repeat(50)));

        const rows = [
          [chalk.bold("Metric"), chalk.bold("Score")],
          ["Precision@K",       formatScore(report.metrics.precision_at_k)],
          ["Recall@K",          formatScore(report.metrics.recall_at_k)],
          ["Faithfulness",      formatScore(report.metrics.faithfulness)],
          ["Relevance",         formatScore(report.metrics.relevance)],
          ["Coherence",         formatScore(report.metrics.coherence)],
          [chalk.bold("Overall Score"), chalk.bold(formatScore(report.metrics.overall_score))],
        ];

        console.log(table(rows));
        console.log(chalk.gray(`Total queries: ${report.total_queries} | Duration: ${report.duration_seconds.toFixed(2)}s`));

        if (opts.output) {
          const outPath = path.resolve(opts.output);
          fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
          console.log(chalk.green(`\nReport saved to ${outPath}`));
        }
      } catch (err: any) {
        console.error(chalk.red("Failed to fetch report: " + (err.response?.data?.detail || err.message)));
        process.exit(1);
      }
    });
}

function formatScore(score: number): string {
  const pct = (score * 100).toFixed(1) + "%";
  if (score >= 0.8) return chalk.green(pct);
  if (score >= 0.6) return chalk.yellow(pct);
  return chalk.red(pct);
}
