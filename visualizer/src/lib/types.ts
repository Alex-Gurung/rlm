// Types matching the RLM log format

export interface ModelUsageSummary {
  total_calls: number;
  total_input_tokens: number;
  total_output_tokens: number;
}

export interface UsageSummary {
  model_usage_summaries: Record<string, ModelUsageSummary>;
}

export interface RLMChatCompletion {
  root_model?: string;
  prompt: string | Record<string, unknown>;
  response: string;
  execution_time: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  usage_summary?: UsageSummary;
}

export interface StoreEvent {
  op: string;
  id: string;
  type: string;
  description: string;
  parents?: string[];
  tags?: string[];
  backrefs_count?: number;
  ts: number;
}

export interface BatchCall {
  prompts_count: number;
  model: string | null;
  execution_time: number;
  ts: number;
}

export interface CommitEvent {
  op: string;
  commit_id: string;
  creates_count?: number;
  links_count?: number;
  proposals_count?: number;
  status?: string;
  ts: number;
}

export interface REPLResult {
  stdout: string;
  stderr: string;
  locals: Record<string, unknown>;
  execution_time: number;
  rlm_calls: RLMChatCompletion[];
  store_events?: StoreEvent[];
  batch_calls?: BatchCall[];
  commit_events?: CommitEvent[];
}

export interface CodeBlock {
  code: string;
  result: REPLResult;
}

export interface RLMIteration {
  type?: string;
  iteration: number;
  timestamp: string;
  prompt: Array<{ role: string; content: string }>;
  response: string;
  code_blocks: CodeBlock[];
  final_answer: string | [string, string] | null;
  iteration_time: number | null;
  // Hierarchical run tracking
  run_id?: string;
  parent_run_id?: string | null;
  depth?: number;
}

// Worker lifecycle events for hierarchical tracking
export interface WorkerSpawnEvent {
  type: 'worker_spawn';
  timestamp: string;
  run_id: string;
  child_run_id: string;
  depth: number;
  worker_prompt_preview?: string;
}

export interface WorkerCompleteEvent {
  type: 'worker_complete';
  timestamp: string;
  run_id: string;
  child_run_id: string;
  depth: number;
  result_summary?: string;
}

// A hierarchical run (root or worker)
export interface HierarchicalRun {
  run_id: string;
  parent_run_id: string | null;
  depth: number;
  iterations: RLMIteration[];
  children: HierarchicalRun[];
  // Metadata
  config?: RLMConfigMetadata;
  workerPromptPreview?: string;
  resultSummary?: string;
}

// Metadata saved at the start of a log file about RLM configuration
export interface RLMConfigMetadata {
  root_model: string | null;
  task_name?: string | null;
  store_mode?: string | null;
  max_depth: number | null;
  max_iterations: number | null;
  backend: string | null;
  backend_kwargs: Record<string, unknown> | null;
  environment_type: string | null;
  environment_kwargs: Record<string, unknown> | null;
  other_backends: string[] | null;
  // Hierarchical tracking
  run_id?: string;
  parent_run_id?: string | null;
  depth?: number;
}

export interface RLMLogFile {
  fileName: string;
  filePath: string;
  iterations: RLMIteration[];
  metadata: LogMetadata;
  config: RLMConfigMetadata;
  // Hierarchical structure (built from flat logs)
  hierarchicalRuns?: HierarchicalRun;
}

export interface LogMetadata {
  totalIterations: number;
  totalCodeBlocks: number;
  totalSubLMCalls: number;
  totalStoreEvents: number;
  totalBatchCalls: number;
  totalBatchPrompts: number;
  totalCommitEvents: number;
  contextQuestion: string;
  finalAnswer: string | null;
  totalExecutionTime: number;
  hasErrors: boolean;
  // Hierarchical stats
  totalRuns: number;
  maxDepth: number;
}

export function extractFinalAnswer(answer: string | [string, string] | null): string | null {
  if (!answer) return null;
  if (Array.isArray(answer)) {
    return answer[1];
  }
  return answer;
}
