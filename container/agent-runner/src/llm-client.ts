/**
 * LLM Client abstraction layer
 * Supports both Claude (via claude-agent-sdk) and OpenAI providers
 */

import { query, HookCallback } from '@anthropic-ai/claude-agent-sdk';
import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';

export type LLMProvider = 'claude' | 'openai';

export interface LLMConfig {
  provider: LLMProvider;
  openaiModel?: string;
  openaiBaseUrl?: string;
  claudeModel?: string;
}

export interface LLMMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface LLMResponse {
  content: string;
  sessionId?: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export interface LLMStreamEvent {
  type: 'content' | 'done' | 'error' | 'session_init';
  content?: string;
  sessionId?: string;
  error?: string;
}

interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

interface ToolResult {
  tool_call_id: string;
  role: 'tool';
  content: string;
}

const log = (message: string): void => {
  console.error(`[llm-client] ${message}`);
};

/**
 * Read CLAUDE.md from group directory for system prompt
 */
function readSystemPrompt(groupDir: string): string | undefined {
  const claudePath = path.join(groupDir, 'CLAUDE.md');
  try {
    if (fs.existsSync(claudePath)) {
      return fs.readFileSync(claudePath, 'utf-8');
    }
  } catch (err) {
    log(`Failed to read CLAUDE.md: ${err instanceof Error ? err.message : String(err)}`);
  }
  return undefined;
}

/**
 * Archive conversation to markdown file (for OpenAI mode)
 */
function archiveConversation(groupDir: string, messages: LLMMessage[], title?: string): void {
  try {
    const conversationsDir = path.join(groupDir, 'conversations');
    fs.mkdirSync(conversationsDir, { recursive: true });

    const now = new Date();
    const date = now.toISOString().split('T')[0];
    const time = `${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}`;

    const name = title
      ? title.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 50)
      : `conversation-${time}`;

    const filename = `${date}-${name}.md`;
    const filePath = path.join(conversationsDir, filename);

    const formatDateTime = (d: Date) => d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });

    const lines: string[] = [
      `# ${title || 'Conversation'}`,
      '',
      `Archived: ${formatDateTime(now)}`,
      '',
      '---',
      ''
    ];

    for (const msg of messages) {
      const sender = msg.role === 'user' ? 'User' : 'Assistant';
      const content = msg.content.length > 2000
        ? msg.content.slice(0, 2000) + '...'
        : msg.content;
      lines.push(`**${sender}**: ${content}`);
      lines.push('');
    }

    fs.writeFileSync(filePath, lines.join('\n'));
    log(`Archived conversation to ${filePath}`);
  } catch (err) {
    log(`Failed to archive conversation: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/**
 * Parse XML-style messages from the prompt
 */
function parseMessages(prompt: string): LLMMessage[] {
  const messages: LLMMessage[] = [];

  // Check if this is our XML format
  if (prompt.includes('<messages>')) {
    const messageRegex = /<message sender="([^"]+)" time="[^"]+">([^<]+)<\/message>/g;
    let match;
    while ((match = messageRegex.exec(prompt)) !== null) {
      const sender = match[1];
      const content = match[2]
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"');

      // Determine role based on sender (simple heuristic)
      const role = sender.toLowerCase().includes('assistant') ||
        sender.toLowerCase().includes('claude') ||
        sender.toLowerCase().includes('andy')
        ? 'assistant'
        : 'user';

      messages.push({ role, content });
    }
  } else {
    // Plain text prompt - treat as single user message
    messages.push({ role: 'user', content: prompt });
  }

  return messages;
}

/**
 * Execute tools and return results (for OpenAI function calling)
 */
async function executeTools(
  toolCalls: ToolCall[],
  allowedTools: string[],
  mcpServers?: Record<string, any>
): Promise<ToolResult[]> {
  const results: ToolResult[] = [];

  for (const toolCall of toolCalls) {
    const toolName = toolCall.function.name;

    // Check if tool is allowed
    const isAllowed = allowedTools.some(pattern => {
      if (pattern.includes('*')) {
        const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '$');
        return regex.test(toolName);
      }
      return pattern === toolName;
    });

    if (!isAllowed) {
      results.push({
        tool_call_id: toolCall.id,
        role: 'tool',
        content: `Error: Tool "${toolName}" is not allowed. Allowed tools: ${allowedTools.join(', ')}`
      });
      continue;
    }

    try {
      // Handle MCP tools (nanoclaw IPC)
      if (toolName.startsWith('mcp__nanoclaw__')) {
        const mcpTool = toolName.replace('mcp__nanoclaw__', '');
        const args = JSON.parse(toolCall.function.arguments);
        const result = await executeMcpTool(mcpTool, args, mcpServers);
        results.push({
          tool_call_id: toolCall.id,
          role: 'tool',
          content: JSON.stringify(result)
        });
      } else if (toolName === 'Bash') {
        // Bash tool - execute in container
        const args = JSON.parse(toolCall.function.arguments);
        const result = await executeBash(args);
        results.push({
          tool_call_id: toolCall.id,
          role: 'tool',
          content: JSON.stringify(result)
        });
      } else if (toolName === 'Read' || toolName === 'Write' || toolName === 'Edit') {
        // File tools
        const args = JSON.parse(toolCall.function.arguments);
        const result = await executeFileTool(toolName, args);
        results.push({
          tool_call_id: toolCall.id,
          role: 'tool',
          content: JSON.stringify(result)
        });
      } else {
        results.push({
          tool_call_id: toolCall.id,
          role: 'tool',
          content: `Error: Tool "${toolName}" is not implemented in OpenAI mode`
        });
      }
    } catch (err) {
      results.push({
        tool_call_id: toolCall.id,
        role: 'tool',
        content: `Error executing ${toolName}: ${err instanceof Error ? err.message : String(err)}`
      });
    }
  }

  return results;
}

/**
 * Execute MCP tool via IPC
 */
async function executeMcpTool(
  toolName: string,
  args: Record<string, unknown>,
  mcpServers?: Record<string, any>
): Promise<unknown> {
  // The MCP server is passed via the mcpServers config
  // For now, return a placeholder indicating the tool was called
  // Full MCP implementation would require more complex IPC handling
  log(`MCP tool called: ${toolName}`);
  return { status: 'called', tool: toolName, args };
}

/**
 * Execute Bash command
 */
async function executeBash(args: { command: string; description?: string }): Promise<unknown> {
  const { exec } = await import('child_process');

  return new Promise((resolve) => {
    const timeout = 60000; // 1 minute timeout
    const child = exec(args.command, { timeout }, (error, stdout, stderr) => {
      if (error) {
        resolve({
          exitCode: error.code || 1,
          stdout: stdout || '',
          stderr: stderr || error.message
        });
      } else {
        resolve({
          exitCode: 0,
          stdout: stdout || '',
          stderr: stderr || ''
        });
      }
    });
  });
}

/**
 * Execute file tool (Read, Write, Edit)
 */
async function executeFileTool(
  toolName: string,
  args: Record<string, unknown>
): Promise<unknown> {
  const filePath = args.file_path as string;

  try {
    switch (toolName) {
      case 'Read': {
        if (!fs.existsSync(filePath)) {
          return { error: `File not found: ${filePath}` };
        }
        const content = fs.readFileSync(filePath, 'utf-8');
        return { content };
      }
      case 'Write': {
        const content = args.content as string;
        fs.mkdirSync(path.dirname(filePath), { recursive: true });
        fs.writeFileSync(filePath, content);
        return { success: true, bytesWritten: content.length };
      }
      case 'Edit': {
        if (!fs.existsSync(filePath)) {
          return { error: `File not found: ${filePath}` };
        }
        const oldString = args.old_string as string;
        const newString = args.new_string as string;
        let content = fs.readFileSync(filePath, 'utf-8');

        if (!content.includes(oldString)) {
          return { error: `Old string not found in file` };
        }

        content = content.replace(oldString, newString);
        fs.writeFileSync(filePath, content);
        return { success: true };
      }
      default:
        return { error: `Unknown file tool: ${toolName}` };
    }
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * Stream responses from Claude Agent SDK
 */
export async function* streamClaude(
  config: LLMConfig,
  prompt: string,
  sessionId: string | undefined,
  groupDir: string,
  allowedTools: string[],
  mcpServers?: Record<string, any>,
  hooks?: { PreCompact?: Array<{ hooks: HookCallback[] }> }
): AsyncGenerator<LLMStreamEvent> {
  const systemPrompt = readSystemPrompt(groupDir);

  try {
    let newSessionId: string | undefined;
    let result: string | null = null;

    for await (const message of query({
      prompt,
      options: {
        cwd: groupDir,
        resume: sessionId,
        allowedTools,
        permissionMode: 'bypassPermissions',
        allowDangerouslySkipPermissions: true,
        settingSources: ['project'],
        mcpServers,
        hooks: hooks || {}
      }
    })) {
      if (message.type === 'system' && message.subtype === 'init') {
        newSessionId = message.session_id;
        yield { type: 'session_init', sessionId: newSessionId };
      }

      if ('result' in message && message.result) {
        result = message.result as string;
        yield { type: 'content', content: result };
      }
    }

    yield { type: 'done' };
  } catch (err) {
    yield {
      type: 'error',
      error: err instanceof Error ? err.message : String(err)
    };
  }
}

/**
 * Stream responses from OpenAI API
 */
export async function* streamOpenAI(
  config: LLMConfig,
  prompt: string,
  sessionId: string | undefined,
  groupDir: string,
  allowedTools: string[],
  mcpServers?: Record<string, any>
): AsyncGenerator<LLMStreamEvent> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    yield { type: 'error', error: 'OPENAI_API_KEY environment variable is not set' };
    return;
  }

  const openai = new OpenAI({
    apiKey,
    baseURL: config.openaiBaseUrl
  });

  const model = config.openaiModel || 'gpt-4o';
  const systemPrompt = readSystemPrompt(groupDir);

  // Parse messages from prompt
  const messages = parseMessages(prompt);

  // Build OpenAI messages array
  const openaiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

  if (systemPrompt) {
    openaiMessages.push({ role: 'system', content: systemPrompt });
  }

  for (const msg of messages) {
    openaiMessages.push({ role: msg.role, content: msg.content });
  }

  // Define available tools
  const tools: OpenAI.Chat.ChatCompletionTool[] = [];

  if (allowedTools.includes('Bash') || allowedTools.includes('*')) {
    tools.push({
      type: 'function',
      function: {
        name: 'Bash',
        description: 'Execute a bash command in the container',
        parameters: {
          type: 'object',
          properties: {
            command: { type: 'string', description: 'The bash command to execute' },
            description: { type: 'string', description: 'Description of what the command does' }
          },
          required: ['command']
        }
      }
    });
  }

  if (allowedTools.includes('Read') || allowedTools.includes('*')) {
    tools.push({
      type: 'function',
      function: {
        name: 'Read',
        description: 'Read a file from the filesystem',
        parameters: {
          type: 'object',
          properties: {
            file_path: { type: 'string', description: 'Absolute path to the file' }
          },
          required: ['file_path']
        }
      }
    });
  }

  if (allowedTools.includes('Write') || allowedTools.includes('*')) {
    tools.push({
      type: 'function',
      function: {
        name: 'Write',
        description: 'Write content to a file',
        parameters: {
          type: 'object',
          properties: {
            file_path: { type: 'string', description: 'Absolute path to the file' },
            content: { type: 'string', description: 'Content to write' }
          },
          required: ['file_path', 'content']
        }
      }
    });
  }

  if (allowedTools.includes('Edit') || allowedTools.includes('*')) {
    tools.push({
      type: 'function',
      function: {
        name: 'Edit',
        description: 'Edit a file by replacing text',
        parameters: {
          type: 'object',
          properties: {
            file_path: { type: 'string', description: 'Absolute path to the file' },
            old_string: { type: 'string', description: 'Text to replace' },
            new_string: { type: 'string', description: 'Replacement text' }
          },
          required: ['file_path', 'old_string', 'new_string']
        }
      }
    });
  }

  // MCP tools
  const mcpToolPatterns = allowedTools.filter(t => t.startsWith('mcp__'));
  for (const pattern of mcpToolPatterns) {
    const toolName = pattern.replace(/\*$/, '').replace(/\*\*/, '');
    tools.push({
      type: 'function',
      function: {
        name: toolName,
        description: `MCP tool: ${toolName}`,
        parameters: {
          type: 'object',
          properties: {
            args: { type: 'object', description: 'Tool arguments' }
          },
          required: ['args']
        }
      }
    });
  }

  try {
    let fullContent = '';
    const maxIterations = 10;
    let iteration = 0;

    while (iteration < maxIterations) {
      iteration++;

      const response = await openai.chat.completions.create({
        model,
        messages: openaiMessages,
        tools: tools.length > 0 ? tools : undefined,
        tool_choice: tools.length > 0 ? 'auto' : undefined,
        stream: false
      });

      const choice = response.choices[0];
      const message = choice.message;

      // Handle tool calls
      if (message.tool_calls && message.tool_calls.length > 0) {
        // Add assistant message with tool calls
        openaiMessages.push({
          role: 'assistant',
          content: message.content || null,
          tool_calls: message.tool_calls
        });

        // Execute tools
        const toolResults = await executeTools(message.tool_calls, allowedTools, mcpServers);

        // Add tool results to messages
        for (const result of toolResults) {
          openaiMessages.push(result);
        }

        continue; // Get next completion with tool results
      }

      // Regular content response
      const content = message.content || '';
      fullContent += content;
      yield { type: 'content', content };

      // Archive conversation
      const allMessages = parseMessages(prompt);
      allMessages.push({ role: 'assistant', content: fullContent });
      archiveConversation(groupDir, allMessages);

      break;
    }

    // Generate a pseudo session ID for compatibility
    const newSessionId = sessionId || `openai-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    yield { type: 'session_init', sessionId: newSessionId };
    yield { type: 'done' };

  } catch (err) {
    yield {
      type: 'error',
      error: err instanceof Error ? err.message : String(err)
    };
  }
}

/**
 * Main LLM streaming function - routes to appropriate provider
 */
export async function* streamLLM(
  config: LLMConfig,
  prompt: string,
  sessionId: string | undefined,
  groupDir: string,
  allowedTools: string[],
  mcpServers?: Record<string, any>,
  hooks?: { PreCompact?: Array<{ hooks: HookCallback[] }> }
): AsyncGenerator<LLMStreamEvent> {
  if (config.provider === 'openai') {
    yield* streamOpenAI(config, prompt, sessionId, groupDir, allowedTools, mcpServers);
  } else {
    yield* streamClaude(config, prompt, sessionId, groupDir, allowedTools, mcpServers, hooks);
  }
}
