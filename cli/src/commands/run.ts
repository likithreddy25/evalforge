import { Command } from "commander";
import ora from "ora";
import chalk from "chalk";
import { table } from "table";
import { startEvalRun, healthCheck } from "../client/api";

export function runCommand(): Command {
  return new Command("run")
    .description("Run an evaluation suite against a RAG endpoint")
    .requiredOption("-d, --dataset <id>", "Dataset ID to evaluate against")
    .requiredOption("-e, --endpoint <url>", "RAG endpoint URL to evaluate")
    .option("-m, --model <model>", "LLM judge model", "gpt-4o-mini")
    .option("-k, --top-k <k>", "Top-K for retrieval evaluation", "3")
    .action(async (opts) => {
      // Health check
      const spinner = ora("Connecting to EvalForge engine...").start();
      const healthy = await healthCheck();
      if (!healthy) {
        spinner.fail(chalk.red("Cannot reach EvalForge engine at " + (process.env.EVALFORGE_API_URL || "http://localhost:8000")));
        console.log(chalk.yellow("Start the engine with: docker compose up"));
        process.exit(1);
      }
      spinner.succeed("Engine connected");

      // Run eval
      const runSpinner = ora(`Running eval on dataset ${chalk.cyan(opts.dataset)}...`).start();
      try {
        const result = await startEvalRun({
          dataset_id: opts.dataset,
          rag_endpoint: opts.endpoint,
          model: opts.model,
          top_k: parseInt(opts.topK),
        });

        runSpinner.succeed(`Eval complete — run ID: ${chalk.green(result.run_id)}`);

        // Print metrics table
        const rows = [
          [chalk.bold("Metric"), chalk.bold("Score")],
          ["Precision@K",       formatScore(result.metrics.precision_at_k)],
          ["Recall@K",          formatScore(result.metrics.recall_at_k)],
          ["Faithfulness",      formatScore(result.metrics.faithfulness)],
          ["Relevance",         formatScore(result.metrics.relevance)],
          ["Coherence",         formatScore(result.metrics.coherence)],
          [chalk.bold("Overall Score"), chalk.bold(formatScore(result.metrics.overall_score))],
        ];

        console.log("\n" + table(rows));
        console.log(chalk.gray(`Queries: ${result.total_queries} | Duration: ${result.duration_seconds.toFixed(2)}s`));
        console.log(chalk.gray(`Full report: evalforge report --run-id ${result.run_id}`));
      } catch (err: any) {
        runSpinner.fail(chalk.red("Eval failed: " + (err.response?.data?.detail || err.message)));
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
