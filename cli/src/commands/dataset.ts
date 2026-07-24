import { Command } from "commander";
import chalk from "chalk";
import ora from "ora";
import { addDataset, listDatasets } from "../client/api";
import * as fs from "fs";
import * as path from "path";

export function datasetCommand(): Command {
  const cmd = new Command("dataset").description("Manage eval datasets");

  cmd
    .command("add")
    .description("Add a Q&A dataset for evaluation")
    .requiredOption("-f, --file <path>", "Path to JSON file with Q&A pairs")
    .requiredOption("-d, --dataset-id <id>", "Dataset ID to store under")
    .action(async (opts) => {
      const filePath = path.resolve(opts.file);
      if (!fs.existsSync(filePath)) {
        console.error(chalk.red(`File not found: ${filePath}`));
        process.exit(1);
      }

      const raw = fs.readFileSync(filePath, "utf-8");
      const qaPairs = JSON.parse(raw);

      const spinner = ora(`Adding dataset ${chalk.cyan(opts.datasetId)}...`).start();
      try {
        const result = await addDataset({ dataset_id: opts.datasetId, qa_pairs: qaPairs });
        spinner.succeed(`Added ${chalk.green(result.count)} Q&A pairs to dataset ${chalk.cyan(opts.datasetId)}`);
      } catch (err: any) {
        spinner.fail(chalk.red("Failed: " + (err.response?.data?.detail || err.message)));
        process.exit(1);
      }
    });

  cmd
    .command("list")
    .description("List all available datasets")
    .action(async () => {
      try {
        const { datasets } = await listDatasets();
        if (datasets.length === 0) {
          console.log(chalk.yellow("No datasets found. Add one with: evalforge dataset add"));
        } else {
          console.log(chalk.bold("\nAvailable datasets:"));
          datasets.forEach((d) => console.log(`  • ${chalk.cyan(d)}`));
        }
      } catch (err: any) {
        console.error(chalk.red("Failed: " + (err.response?.data?.detail || err.message)));
        process.exit(1);
      }
    });

  return cmd;
}
