import { RLMIteration, RLMLogFile, LogMetadata, RLMConfigMetadata, HierarchicalRun, WorkerSpawnEvent, WorkerCompleteEvent, extractFinalAnswer } from './types';

// Extract the context variable from code block locals
export function extractContextVariable(iterations: RLMIteration[]): string | null {
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string') {
          return ctx;
        }
        try {
          return JSON.stringify(ctx, null, 2);
        } catch (e) {
          return String(ctx);
        }
      }
    }
  }
  return null;
}

// Default config when metadata is not present (backwards compatibility)
function getDefaultConfig(): RLMConfigMetadata {
  return {
    root_model: null,
    task_name: null,
    store_mode: null,
    max_depth: null,
    max_iterations: null,
    backend: null,
    backend_kwargs: null,
    environment_type: null,
    environment_kwargs: null,
    other_backends: null,
  };
}

export interface ParsedJSONL {
  iterations: RLMIteration[];
  config: RLMConfigMetadata;
  workerSpawnEvents: WorkerSpawnEvent[];
  workerCompleteEvents: WorkerCompleteEvent[];
  // Map of run_id -> config for each run
  runConfigs: Map<string, RLMConfigMetadata>;
}

export function parseJSONL(content: string): ParsedJSONL {
  const lines = content.trim().split('\n').filter(line => line.trim());
  const iterations: RLMIteration[] = [];
  const workerSpawnEvents: WorkerSpawnEvent[] = [];
  const workerCompleteEvents: WorkerCompleteEvent[] = [];
  const runConfigs = new Map<string, RLMConfigMetadata>();
  let config: RLMConfigMetadata = getDefaultConfig();

  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);

      // Check if this is a metadata entry
      if (parsed.type === 'metadata') {
        const parsedConfig: RLMConfigMetadata = {
          root_model: parsed.root_model ?? null,
          task_name: parsed.task_name ?? null,
          store_mode: parsed.store_mode ?? null,
          max_depth: parsed.max_depth ?? null,
          max_iterations: parsed.max_iterations ?? null,
          backend: parsed.backend ?? null,
          backend_kwargs: parsed.backend_kwargs ?? null,
          environment_type: parsed.environment_type ?? null,
          environment_kwargs: parsed.environment_kwargs ?? null,
          other_backends: parsed.other_backends ?? null,
          run_id: parsed.run_id,
          parent_run_id: parsed.parent_run_id ?? null,
          depth: parsed.depth ?? 0,
        };

        // Store config by run_id if available
        if (parsed.run_id) {
          runConfigs.set(parsed.run_id, parsedConfig);
        }

        // Use root config (depth 0 or no parent) as main config
        if (!parsed.parent_run_id || parsed.depth === 0) {
          config = parsedConfig;
        }
      } else if (parsed.type === 'iteration') {
        // This is an iteration entry
        iterations.push(parsed as RLMIteration);
      } else if (parsed.type === 'worker_spawn') {
        workerSpawnEvents.push(parsed as WorkerSpawnEvent);
      } else if (parsed.type === 'worker_complete') {
        workerCompleteEvents.push(parsed as WorkerCompleteEvent);
      }
    } catch (e) {
      console.error('Failed to parse line:', line, e);
    }
  }

  return { iterations, config, workerSpawnEvents, workerCompleteEvents, runConfigs };
}

export function extractContextQuestion(iterations: RLMIteration[]): string {
  if (iterations.length === 0) return 'No context found';
  
  const firstIteration = iterations[0];
  const prompt = firstIteration.prompt;
  
  // Look for user message that contains the actual question
  for (const msg of prompt) {
    if (msg.role === 'user' && msg.content) {
      // Try to extract quoted query
      const queryMatch = msg.content.match(/original query: "([^"]+)"/);
      if (queryMatch) {
        return queryMatch[1];
      }
      
      // Check if it contains the actual query pattern
      if (msg.content.includes('answer the prompt')) {
        continue;
      }
      
      // Take first substantial user message
      if (msg.content.length > 50 && msg.content.length < 500) {
        return msg.content.slice(0, 200) + (msg.content.length > 200 ? '...' : '');
      }
    }
  }
  
  // Fallback: look in system prompt for context info
  const systemMsg = prompt.find(m => m.role === 'system');
  if (systemMsg?.content) {
    const contextMatch = systemMsg.content.match(/context variable.*?:(.*?)(?:\n|$)/i);
    if (contextMatch) {
      return contextMatch[1].trim().slice(0, 200);
    }
  }
  
  // Check code block output for actual context
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string' && ctx.length < 500) {
          return ctx;
        }
      }
    }
  }
  
  return 'Context available in REPL environment';
}

/**
 * Build hierarchical run tree from flat iterations.
 * Groups iterations by run_id and builds parent-child relationships.
 */
export function buildHierarchicalRuns(
  iterations: RLMIteration[],
  workerSpawnEvents: WorkerSpawnEvent[],
  workerCompleteEvents: WorkerCompleteEvent[],
  runConfigs: Map<string, RLMConfigMetadata>
): HierarchicalRun | undefined {
  // Group iterations by run_id
  const runIterations = new Map<string, RLMIteration[]>();
  const runDepths = new Map<string, number>();
  const runParents = new Map<string, string | null>();

  for (const iter of iterations) {
    const runId = iter.run_id || 'root';
    if (!runIterations.has(runId)) {
      runIterations.set(runId, []);
    }
    runIterations.get(runId)!.push(iter);

    // Track depth and parent
    if (iter.depth !== undefined) {
      runDepths.set(runId, iter.depth);
    }
    if (iter.parent_run_id !== undefined) {
      runParents.set(runId, iter.parent_run_id);
    }
  }

  // Also get parent info from spawn events
  for (const spawn of workerSpawnEvents) {
    runParents.set(spawn.child_run_id, spawn.run_id);
    runDepths.set(spawn.child_run_id, spawn.depth + 1);
  }

  // Build worker prompt/result summaries
  const workerPrompts = new Map<string, string>();
  const workerResults = new Map<string, string>();

  for (const spawn of workerSpawnEvents) {
    if (spawn.worker_prompt_preview) {
      workerPrompts.set(spawn.child_run_id, spawn.worker_prompt_preview);
    }
  }
  for (const complete of workerCompleteEvents) {
    if (complete.result_summary) {
      workerResults.set(complete.child_run_id, complete.result_summary);
    }
  }

  // Build HierarchicalRun objects
  const runs = new Map<string, HierarchicalRun>();

  for (const [runId, iters] of runIterations) {
    runs.set(runId, {
      run_id: runId,
      parent_run_id: runParents.get(runId) ?? null,
      depth: runDepths.get(runId) ?? 0,
      iterations: iters.sort((a, b) => a.iteration - b.iteration),
      children: [],
      config: runConfigs.get(runId),
      workerPromptPreview: workerPrompts.get(runId),
      resultSummary: workerResults.get(runId),
    });
  }

  // Build tree structure
  let rootRun: HierarchicalRun | undefined;

  for (const run of runs.values()) {
    if (run.parent_run_id && runs.has(run.parent_run_id)) {
      runs.get(run.parent_run_id)!.children.push(run);
    } else if (run.depth === 0 || !run.parent_run_id) {
      // This is a root run (or the only run for backward compatibility)
      rootRun = run;
    }
  }

  // Sort children by first iteration timestamp
  for (const run of runs.values()) {
    run.children.sort((a, b) => {
      const aTime = a.iterations[0]?.timestamp || '';
      const bTime = b.iterations[0]?.timestamp || '';
      return aTime.localeCompare(bTime);
    });
  }

  // Fallback: if no hierarchical data, create a single "root" run
  if (!rootRun && iterations.length > 0) {
    rootRun = {
      run_id: 'root',
      parent_run_id: null,
      depth: 0,
      iterations: iterations,
      children: [],
    };
  }

  return rootRun;
}

export function computeMetadata(iterations: RLMIteration[]): LogMetadata {
  let totalCodeBlocks = 0;
  let totalSubLMCalls = 0;
  let totalStoreEvents = 0;
  let totalBatchCalls = 0;
  let totalBatchPrompts = 0;
  let totalCommitEvents = 0;
  let totalExecutionTime = 0;
  let hasErrors = false;
  let finalAnswer: string | null = null;
  
  for (const iter of iterations) {
    totalCodeBlocks += iter.code_blocks.length;
    
    // Use iteration_time if available, otherwise sum code block times
    if (iter.iteration_time != null) {
      totalExecutionTime += iter.iteration_time;
    } else {
      for (const block of iter.code_blocks) {
        if (block.result) {
          totalExecutionTime += block.result.execution_time || 0;
        }
      }
    }
    
    for (const block of iter.code_blocks) {
      if (block.result) {
        if (block.result.stderr) {
          hasErrors = true;
        }
        if (block.result.rlm_calls) {
          totalSubLMCalls += block.result.rlm_calls.length;
        }
        if (block.result.store_events) {
          totalStoreEvents += block.result.store_events.length;
        }
        if (block.result.batch_calls) {
          totalBatchCalls += block.result.batch_calls.length;
          for (const batch of block.result.batch_calls) {
            totalBatchPrompts += batch.prompts_count || 0;
          }
        }
        if (block.result.commit_events) {
          totalCommitEvents += block.result.commit_events.length;
        }
      }
    }
    
    if (iter.final_answer) {
      finalAnswer = extractFinalAnswer(iter.final_answer);
    }
  }
  
  // Compute hierarchical stats
  const runIds = new Set(iterations.map(i => i.run_id || 'root'));
  const depths = iterations.map(i => i.depth ?? 0);
  const maxDepth = depths.length > 0 ? Math.max(...depths) : 0;

  return {
    totalIterations: iterations.length,
    totalCodeBlocks,
    totalSubLMCalls,
    totalStoreEvents,
    totalBatchCalls,
    totalBatchPrompts,
    totalCommitEvents,
    contextQuestion: extractContextQuestion(iterations),
    finalAnswer,
    totalExecutionTime,
    hasErrors,
    totalRuns: runIds.size,
    maxDepth,
  };
}

export function parseLogFile(fileName: string, content: string): RLMLogFile {
  const { iterations, config, workerSpawnEvents, workerCompleteEvents, runConfigs } = parseJSONL(content);
  const metadata = computeMetadata(iterations);
  const hierarchicalRuns = buildHierarchicalRuns(iterations, workerSpawnEvents, workerCompleteEvents, runConfigs);

  return {
    fileName,
    filePath: fileName,
    iterations,
    metadata,
    config,
    hierarchicalRuns,
  };
}
