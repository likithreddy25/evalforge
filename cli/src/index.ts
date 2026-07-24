#!/usr/bin/env node
import { Command } from "commander";
import { runCommand } from "./commands/run";
import { reportCommand } from "./commands/report";
import { datasetCommand } from "./commands/dataset";

const program = new Command();

program
  .name("evalforge")
  .description("EvalForge — LLM & RAG pipeline evaluation framework")
  .version("0.1.0");

program.addCommand(runCommand());
program.addCommand(reportCommand());
program.addCommand(datasetCommand());

program.parse(process.argv);
