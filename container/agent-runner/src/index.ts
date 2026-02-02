/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 * Supports both Claude (via claude-agent-sdk) and OpenAI providers
 */

import fs from 'fs';
import path from 'path';
import { createIpcMcp } from './ipc-mcp.js';
import { streamLLM, LLMConfig } from './llm-client.js';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  llm?: LLMConfig;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

async function main(): Promise<void> {
  let input: ContainerInput;

  try {
    const stdinData = await readStdin();
    input = JSON.parse(stdinData);
    log(`Received input for group: ${input.groupFolder}`);
    log(`LLM provider: ${input.llm?.provider || 'claude (default)'}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`
    });
    process.exit(1);
  }

  const ipcMcp = createIpcMcp({
    chatJid: input.chatJid,
    groupFolder: input.groupFolder,
    isMain: input.isMain
  });

  let result: string | null = null;
  let newSessionId: string | undefined;

  // Add context for scheduled tasks
  let prompt = input.prompt;
  if (input.isScheduledTask) {
    prompt = `[SCHEDULED TASK - You are running automatically, not in response to a user message. Use mcp__nanoclaw__send_message if needed to communicate with the user.]\n\n${input.prompt}`;
  }

  // Configure LLM
  const llmConfig: LLMConfig = input.llm || { provider: 'claude' };
  const groupDir = '/workspace/group';
  const allowedTools = [
    'Bash',
    'Read', 'Write', 'Edit', 'Glob', 'Grep',
    'WebSearch', 'WebFetch',
    'mcp__nanoclaw__*'
  ];

  try {
    log('Starting agent...');

    for await (const event of streamLLM(
      llmConfig,
      prompt,
      input.sessionId,
      groupDir,
      allowedTools,
      { nanoclaw: ipcMcp },
      // Hooks are only supported in Claude mode
      llmConfig.provider === 'claude' ? {
        PreCompact: [] // Pre-compact hooks handled by claude-agent-sdk
      } : undefined
    )) {
      switch (event.type) {
        case 'session_init':
          newSessionId = event.sessionId;
          log(`Session initialized: ${newSessionId}`);
          break;
        case 'content':
          if (event.content) {
            result = event.content;
          }
          break;
        case 'done':
          log('Agent completed successfully');
          break;
        case 'error':
          throw new Error(event.error || 'Unknown LLM error');
      }
    }

    writeOutput({
      status: 'success',
      result,
      newSessionId
    });

  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId,
      error: errorMessage
    });
    process.exit(1);
  }
}

main();
