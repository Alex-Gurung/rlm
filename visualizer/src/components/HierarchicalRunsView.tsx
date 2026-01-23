'use client';

import { useState } from 'react';
import { ChevronRight, ChevronDown, Play, CheckCircle, AlertCircle, Layers } from 'lucide-react';
import { HierarchicalRun, RLMIteration } from '@/lib/types';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface HierarchicalRunsViewProps {
  rootRun: HierarchicalRun;
  onSelectIteration: (iteration: RLMIteration, run: HierarchicalRun) => void;
  selectedIteration?: RLMIteration | null;
}

interface RunNodeProps {
  run: HierarchicalRun;
  onSelectIteration: (iteration: RLMIteration, run: HierarchicalRun) => void;
  selectedIteration?: RLMIteration | null;
  level: number;
}

function RunNode({ run, onSelectIteration, selectedIteration, level }: RunNodeProps) {
  const [expanded, setExpanded] = useState(level < 2); // Auto-expand first 2 levels
  const hasChildren = run.children.length > 0;
  const hasIterations = run.iterations.length > 0;

  // Determine run status
  const lastIteration = run.iterations[run.iterations.length - 1];
  const hasFinalAnswer = lastIteration?.final_answer != null;
  const hasError = run.iterations.some(iter =>
    iter.code_blocks.some(block => block.result?.stderr)
  );

  // Count total iterations including children
  const totalIterations = run.iterations.length;
  const childIterations = run.children.reduce((sum, child) => {
    const countAll = (r: HierarchicalRun): number =>
      r.iterations.length + r.children.reduce((s, c) => s + countAll(c), 0);
    return sum + countAll(child);
  }, 0);

  return (
    <div className="select-none">
      {/* Run header */}
      <div
        className={cn(
          'flex items-center gap-2 py-1.5 px-2 rounded-md cursor-pointer transition-colors',
          'hover:bg-muted/50',
          level > 0 && 'ml-4 border-l-2 border-muted pl-3'
        )}
        style={{ marginLeft: level > 0 ? `${level * 16}px` : 0 }}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Expand/collapse icon */}
        {(hasChildren || hasIterations) ? (
          expanded ? (
            <ChevronDown className="w-4 h-4 text-muted-foreground flex-shrink-0" />
          ) : (
            <ChevronRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />
          )
        ) : (
          <div className="w-4 h-4 flex-shrink-0" />
        )}

        {/* Status icon */}
        {hasFinalAnswer ? (
          <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
        ) : hasError ? (
          <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
        ) : (
          <Play className="w-4 h-4 text-blue-500 flex-shrink-0" />
        )}

        {/* Run label */}
        <span className="font-mono text-xs flex-1 truncate">
          {run.depth === 0 ? 'Root' : `Worker`}
          <span className="text-muted-foreground ml-1">
            ({run.run_id.slice(0, 6)})
          </span>
        </span>

        {/* Badges */}
        <div className="flex items-center gap-1.5">
          {run.depth > 0 && (
            <Badge variant="outline" className="text-[9px] px-1 py-0 h-4">
              depth {run.depth}
            </Badge>
          )}
          <Badge variant="secondary" className="text-[9px] px-1 py-0 h-4">
            {totalIterations} iter
          </Badge>
          {hasChildren && (
            <Badge variant="outline" className="text-[9px] px-1 py-0 h-4">
              <Layers className="w-2.5 h-2.5 mr-0.5" />
              {run.children.length}
            </Badge>
          )}
        </div>
      </div>

      {/* Worker prompt preview */}
      {expanded && run.workerPromptPreview && (
        <div
          className="text-[10px] text-muted-foreground font-mono truncate px-2 py-1 bg-muted/30 rounded mx-2 mb-1"
          style={{ marginLeft: `${(level + 1) * 16 + 8}px` }}
        >
          Task: {run.workerPromptPreview.slice(0, 100)}...
        </div>
      )}

      {/* Iterations */}
      {expanded && hasIterations && (
        <div className="space-y-0.5" style={{ marginLeft: `${(level + 1) * 16}px` }}>
          {run.iterations.map((iter) => {
            const isSelected = selectedIteration?.iteration === iter.iteration &&
              selectedIteration?.run_id === iter.run_id;
            const iterHasError = iter.code_blocks.some(b => b.result?.stderr);
            const iterHasFinal = iter.final_answer != null;

            return (
              <div
                key={`${run.run_id}-${iter.iteration}`}
                className={cn(
                  'flex items-center gap-2 py-1 px-2 rounded cursor-pointer text-xs',
                  'hover:bg-primary/10 transition-colors',
                  isSelected && 'bg-primary/20 border border-primary/30'
                )}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectIteration(iter, run);
                }}
              >
                <div className={cn(
                  'w-1.5 h-1.5 rounded-full flex-shrink-0',
                  iterHasFinal ? 'bg-green-500' : iterHasError ? 'bg-red-500' : 'bg-blue-500'
                )} />
                <span className="font-mono text-muted-foreground">
                  Iter {iter.iteration}
                </span>
                {iter.code_blocks.length > 0 && (
                  <span className="text-muted-foreground/60">
                    ({iter.code_blocks.length} blocks)
                  </span>
                )}
                {iterHasFinal && (
                  <Badge variant="default" className="text-[8px] px-1 py-0 h-3 ml-auto">
                    FINAL
                  </Badge>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Child workers */}
      {expanded && hasChildren && (
        <div className="mt-1">
          {run.children.map((child) => (
            <RunNode
              key={child.run_id}
              run={child}
              onSelectIteration={onSelectIteration}
              selectedIteration={selectedIteration}
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function HierarchicalRunsView({
  rootRun,
  onSelectIteration,
  selectedIteration,
}: HierarchicalRunsViewProps) {
  // Count total runs recursively
  const countRuns = (run: HierarchicalRun): number =>
    1 + run.children.reduce((sum, child) => sum + countRuns(child), 0);

  const totalRuns = countRuns(rootRun);
  const maxDepth = Math.max(
    rootRun.depth,
    ...rootRun.children.map(c => {
      const getMaxDepth = (r: HierarchicalRun): number =>
        Math.max(r.depth, ...r.children.map(getMaxDepth));
      return getMaxDepth(c);
    })
  );

  return (
    <div className="space-y-2">
      {/* Header stats */}
      <div className="flex items-center gap-3 px-2 py-1.5 bg-muted/30 rounded text-xs">
        <span className="text-muted-foreground">Hierarchical Execution</span>
        <Badge variant="outline" className="text-[9px]">
          {totalRuns} run{totalRuns !== 1 ? 's' : ''}
        </Badge>
        {maxDepth > 0 && (
          <Badge variant="outline" className="text-[9px]">
            max depth {maxDepth}
          </Badge>
        )}
      </div>

      {/* Tree view */}
      <div className="space-y-1">
        <RunNode
          run={rootRun}
          onSelectIteration={onSelectIteration}
          selectedIteration={selectedIteration}
          level={0}
        />
      </div>
    </div>
  );
}
