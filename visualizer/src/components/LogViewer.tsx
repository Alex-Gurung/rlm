'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { StatsCard } from './StatsCard';
import { TrajectoryPanel } from './TrajectoryPanel';
import { ExecutionPanel } from './ExecutionPanel';
import { IterationTimeline } from './IterationTimeline';
import { HierarchicalRunsView } from './HierarchicalRunsView';
import { WorkTreeView } from './WorkTreeView';
import { ThemeToggle } from './ThemeToggle';
import { RLMLogFile, RLMIteration, HierarchicalRun } from '@/lib/types';
import { extractContextVariable } from '@/lib/parse-logs';
import { Layers, List, Network } from 'lucide-react';

interface LogViewerProps {
  logFile: RLMLogFile;
  onBack: () => void;
}

export function LogViewer({ logFile, onBack }: LogViewerProps) {
  const [selectedIteration, setSelectedIteration] = useState(0);
  const [selectedRun, setSelectedRun] = useState<HierarchicalRun | null>(null);
  const [viewMode, setViewMode] = useState<'flat' | 'hierarchical' | 'worktree'>('flat');
  const [contextExpanded, setContextExpanded] = useState(false);
  const [answerExpanded, setAnswerExpanded] = useState(false);
  const [contextModalOpen, setContextModalOpen] = useState(false);
  const [contextViewMode, setContextViewMode] = useState<'text' | 'json'>('text');
  const [focusMode, setFocusMode] = useState(false);
  const { iterations, metadata, config, hierarchicalRuns } = logFile;
  const router = useRouter();
  const searchParams = useSearchParams();

  // Check if we have hierarchical data
  const hasHierarchy = hierarchicalRuns && (
    hierarchicalRuns.children.length > 0 ||
    (metadata.totalRuns && metadata.totalRuns > 1)
  );

  const fullContext = useMemo(() => {
    const extracted = extractContextVariable(iterations);
    if (extracted) return extracted;
    const first = iterations[0];
    if (first?.prompt?.length) {
      const userMsg = first.prompt.find(msg => msg.role === 'user');
      if (userMsg?.content) return String(userMsg.content);
    }
    return null;
  }, [iterations]);

  const fullContextJson = useMemo(() => {
    if (!fullContext) return null;
    try {
      const parsed = JSON.parse(fullContext);
      return JSON.stringify(parsed, null, 2);
    } catch (e) {
      return null;
    }
  }, [fullContext]);

  const fullAnswer = useMemo(() => {
    if (metadata.finalAnswer) return String(metadata.finalAnswer);
    const last = iterations[iterations.length - 1];
    if (last?.final_answer) return String(last.final_answer);
    return null;
  }, [iterations, metadata.finalAnswer]);

  // Auto-switch to hierarchical view if hierarchy exists
  useEffect(() => {
    if (hasHierarchy) {
      const viewParam = searchParams.get('view');
      if (!viewParam) {
        setViewMode('hierarchical');
      }
    }
  }, [hasHierarchy, searchParams]);

  const updateParams = useCallback((updates: Record<string, string | number | null | undefined>) => {
    const params = new URLSearchParams(searchParams.toString());
    for (const [key, value] of Object.entries(updates)) {
      if (value === null || value === undefined || value === '') {
        params.delete(key);
      } else {
        params.set(key, String(value));
      }
    }
    const next = params.toString();
    const current = searchParams.toString();
    if (next !== current) {
      router.replace(next ? `?${next}` : '?');
    }
  }, [router, searchParams]);

  const handleViewModeChange = useCallback((mode: 'flat' | 'hierarchical' | 'worktree') => {
    setViewMode(mode);
    updateParams({ view: mode });
  }, [updateParams]);

  // Initialize from URL params (iter, view, run)
  useEffect(() => {
    const iterParam = searchParams.get('iter');
    const viewParam = searchParams.get('view');
    const runParam = searchParams.get('run');

    if (viewParam === 'flat' || viewParam === 'hierarchical' || viewParam === 'worktree') {
      setViewMode(viewParam);
    }

    if (iterParam) {
      const idx = Number(iterParam);
      if (!Number.isNaN(idx)) {
        setSelectedIteration(Math.min(Math.max(0, idx), iterations.length - 1));
      }
    }

    if (runParam && hierarchicalRuns) {
      const stack: HierarchicalRun[] = [hierarchicalRuns];
      while (stack.length) {
        const run = stack.pop()!;
        if (run.run_id === runParam) {
          setSelectedRun(run);
          break;
        }
        stack.push(...run.children);
      }
    }
  }, [hierarchicalRuns, iterations.length, searchParams]);

  useEffect(() => {
    updateParams({ view: viewMode });
  }, [updateParams, viewMode]);

  // Handle iteration selection from hierarchical view
  const handleHierarchicalSelect = useCallback((iteration: RLMIteration, run: HierarchicalRun) => {
    // Find the global index of this iteration
    const globalIndex = iterations.findIndex(
      i => i.iteration === iteration.iteration && i.run_id === iteration.run_id
    );
    if (globalIndex >= 0) {
      setSelectedIteration(globalIndex);
    }
    setSelectedRun(run);
    updateParams({
      iter: globalIndex >= 0 ? globalIndex : 0,
      run: run.run_id,
    });
  }, [iterations, updateParams]);

  const handleIterationSelect = useCallback((index: number) => {
    const clamped = Math.min(Math.max(0, index), iterations.length - 1);
    setSelectedIteration(clamped);
    const iter = iterations[clamped];
    updateParams({
      iter: clamped,
      run: iter?.run_id ?? null,
    });
  }, [iterations, updateParams]);

  // Get current iteration (considering both flat and hierarchical selection)
  const currentIteration = iterations[selectedIteration] || null;

  const goToPrevious = useCallback(() => {
    handleIterationSelect(selectedIteration - 1);
  }, [handleIterationSelect, selectedIteration]);

  const goToNext = useCallback(() => {
    handleIterationSelect(selectedIteration + 1);
  }, [handleIterationSelect, selectedIteration]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey) {
        return;
      }
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return;
      }
      if (e.key === 'ArrowLeft' || e.key === 'j') {
        goToPrevious();
      } else if (e.key === 'ArrowRight' || e.key === 'k') {
        goToNext();
      } else if (e.key === 'Escape') {
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goToPrevious, goToNext, onBack]);

  return (
    <>
      <div className="h-screen flex flex-col overflow-hidden bg-background overflow-x-hidden">
      {/* Top Bar - Compact header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm">
        <div className="px-4 py-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={onBack}
                className="text-muted-foreground hover:text-foreground"
              >
                ← Back
              </Button>
              <div className="h-5 w-px bg-border" />
              <div>
                <h1 className="font-semibold flex items-center gap-2 text-sm">
                  <span className="text-primary">◈</span>
                  {logFile.fileName}
                </h1>
                <p className="text-[10px] text-muted-foreground font-mono mt-0.5">
                  {config.root_model ?? 'Unknown model'} • {config.backend ?? 'Unknown backend'} • {config.environment_type ?? 'Unknown env'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                size="sm"
                className="text-[10px] h-6"
                onClick={() => setFocusMode(prev => !prev)}
              >
                {focusMode ? 'Show chrome' : 'Focus'}
              </Button>
              {metadata.hasErrors && (
                <Badge variant="destructive" className="text-xs">Has Errors</Badge>
              )}
              {metadata.finalAnswer && (
                <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white text-xs">
                  Completed
                </Badge>
              )}
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      {/* Question & Answer + Stats Row */}
      {!focusMode && (
        <div className="border-b border-border bg-muted/30 px-4 py-1">
          <div className="flex gap-3">
          {/* Question & Answer Summary */}
          <Card className="flex-1 bg-gradient-to-r from-primary/5 to-accent/5 border-primary/20">
            <CardContent className="p-1.5">
              <div className="grid md:grid-cols-2 gap-2">
                <div onClick={() => setContextExpanded(prev => !prev)} className="cursor-pointer">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Context / Question
                  </p>
                    <p className="text-sm font-medium line-clamp-1">
                      {metadata.contextQuestion}
                    </p>
                  </div>
                <div onClick={() => setAnswerExpanded(prev => !prev)} className="cursor-pointer">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Final Answer
                  </p>
                    <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400 line-clamp-1">
                      {metadata.finalAnswer || 'Not yet completed'}
                    </p>
                  </div>
              </div>
              {contextExpanded && (
                <div className="mt-2 rounded-md border border-border bg-background/70 p-2 text-xs text-muted-foreground max-h-40 overflow-auto">
                  <div className="flex items-center justify-between mb-1">
                    <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Full Context</div>
                    <div className="flex items-center gap-2">
                      {fullContextJson && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 text-[10px]"
                          onClick={() => setContextViewMode(prev => prev === 'text' ? 'json' : 'text')}
                        >
                          {contextViewMode === 'text' ? 'JSON view' : 'Text view'}
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 text-[10px]"
                        onClick={() => setContextModalOpen(true)}
                      >
                        Expand
                      </Button>
                    </div>
                  </div>
                  <div className="whitespace-pre-wrap break-words">
                    {contextViewMode === 'json'
                      ? (fullContextJson ?? fullContext)
                      : (fullContext ?? 'No context captured in logs.')
                    }
                  </div>
                </div>
              )}
              {answerExpanded && (
                <div className="mt-2 rounded-md border border-border bg-background/70 p-2 text-xs text-muted-foreground max-h-40 overflow-auto">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Full Answer</div>
                  <div className="whitespace-pre-wrap break-words">
                    {fullAnswer ?? 'No final answer recorded yet.'}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <div className="flex gap-2">
            <StatsCard
              label="Iterations"
              value={metadata.totalIterations}
              icon="◎"
              variant="cyan"
              size="compact"
            />
            <StatsCard
              label="Code"
              value={metadata.totalCodeBlocks}
              icon="⟨⟩"
              variant="green"
              size="compact"
            />
            <StatsCard
              label="Sub-LM"
              value={metadata.totalSubLMCalls}
              icon="◇"
              variant="magenta"
              size="compact"
            />
            <StatsCard
              label="Store"
              value={metadata.totalStoreEvents}
              icon="▦"
              variant="cyan"
              subtext={`${metadata.totalStoreEvents} events`}
              size="compact"
            />
            <StatsCard
              label="Batch"
              value={metadata.totalBatchCalls}
              icon="⇉"
              variant="yellow"
              subtext={`${metadata.totalBatchPrompts} prompts`}
              size="compact"
            />
            <StatsCard
              label="Commits"
              value={metadata.totalCommitEvents}
              icon="⇄"
              variant="magenta"
              subtext={`${metadata.totalCommitEvents} events`}
              size="compact"
            />
            {hasHierarchy && (
              <StatsCard
                label="Runs"
                value={metadata.totalRuns || 1}
                icon="⬡"
                variant="cyan"
                subtext={`depth ${metadata.maxDepth || 0}`}
                size="compact"
              />
            )}
            <StatsCard
              label="Exec"
              value={`${metadata.totalExecutionTime.toFixed(2)}s`}
              icon="⏱"
              variant="yellow"
              size="compact"
            />
          </div>
        </div>
        </div>
      )}

      {/* Iteration Timeline / Hierarchical View */}
      {!focusMode && (
        <div className="border-b border-border">
        {/* View mode toggle (only show if hierarchy exists) */}
        {hasHierarchy && (
          <div className="flex items-center gap-2 px-4 py-1.5 bg-muted/20 border-b border-border">
            <span className="text-xs text-muted-foreground">View:</span>
            <Button
              variant={viewMode === 'flat' ? 'secondary' : 'ghost'}
              size="sm"
              className="h-6 text-xs px-2"
              onClick={() => handleViewModeChange('flat')}
            >
              <List className="w-3 h-3 mr-1" />
              Flat
            </Button>
            <Button
              variant={viewMode === 'hierarchical' ? 'secondary' : 'ghost'}
              size="sm"
              className="h-6 text-xs px-2"
              onClick={() => handleViewModeChange('hierarchical')}
            >
              <Layers className="w-3 h-3 mr-1" />
              Hierarchical
            </Button>
            <Button
              variant={viewMode === 'worktree' ? 'secondary' : 'ghost'}
              size="sm"
              className="h-6 text-xs px-2"
              onClick={() => handleViewModeChange('worktree')}
            >
              <Network className="w-3 h-3 mr-1" />
              Work Tree
            </Button>
            {selectedRun && viewMode === 'hierarchical' && (
              <Badge variant="outline" className="text-[9px] ml-2">
                Viewing: {selectedRun.run_id.slice(0, 8)} (depth {selectedRun.depth})
              </Badge>
            )}
          </div>
        )}

        {/* Show appropriate view based on mode */}
        {viewMode === 'hierarchical' && hierarchicalRuns ? (
          <div className="max-h-32 overflow-y-auto px-4 py-1.5">
            <HierarchicalRunsView
              rootRun={hierarchicalRuns}
              onSelectIteration={handleHierarchicalSelect}
              selectedIteration={currentIteration}
            />
          </div>
        ) : viewMode === 'worktree' && hierarchicalRuns ? (
          <div className="max-h-[220px] overflow-y-auto px-4 py-1.5 bg-muted/20">
            <WorkTreeView
              iterations={iterations}
              rootRun={hierarchicalRuns}
              onSelectIteration={handleHierarchicalSelect}
              selectedIteration={currentIteration}
            />
          </div>
        ) : (
          <IterationTimeline
            iterations={iterations}
            selectedIteration={selectedIteration}
            onSelectIteration={handleIterationSelect}
          />
        )}
        </div>
      )}

      {/* Main Content - Split View */}
      <div className="flex-1 min-h-0 min-w-0">
        <div className="h-full w-full min-w-0">
          <div className="grid h-full w-full min-w-0 min-h-0 grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            {/* Left Panel - Prompt & Response */}
            <div className="h-full min-w-0 min-h-0 border-r border-border">
            <div className="h-full min-w-0 min-h-0 overflow-hidden">
              <TrajectoryPanel
                iterations={iterations}
                selectedIteration={selectedIteration}
                onSelectIteration={handleIterationSelect}
              />
            </div>
          </div>

          {/* Right Panel - Code Execution & Sub-LM Calls */}
          <div className="h-full min-w-0 min-h-0 bg-background">
            <div className="h-full min-w-0 min-h-0 overflow-hidden">
              <ExecutionPanel
                iteration={iterations[selectedIteration] || null}
                iterations={iterations}
              />
            </div>
          </div>
          </div>
        </div>
      </div>

      {/* Keyboard hint footer */}
      {!focusMode && (
        <div className="border-t border-border bg-muted/30 px-4 py-1">
          <div className="flex items-center justify-center gap-6 text-[10px] text-muted-foreground">
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">←</kbd>
              <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">→</kbd>
              Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">Esc</kbd>
              Back
            </span>
          </div>
        </div>
      )}
      </div>
      {contextModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
          <div className="w-[92vw] max-w-5xl max-h-[80vh] rounded-lg border border-border bg-card shadow-xl flex flex-col">
            <div className="flex items-center justify-between border-b border-border px-4 py-2">
              <div className="text-sm font-semibold">Full Context</div>
              <div className="flex items-center gap-2">
                {fullContextJson && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => setContextViewMode(prev => prev === 'text' ? 'json' : 'text')}
                  >
                    {contextViewMode === 'text' ? 'JSON view' : 'Text view'}
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={() => setContextModalOpen(false)}
                >
                  Close
                </Button>
              </div>
            </div>
            <div className="px-4 py-3 overflow-auto text-sm text-muted-foreground whitespace-pre-wrap break-words">
              {contextViewMode === 'json'
                ? (fullContextJson ?? fullContext)
                : (fullContext ?? 'No context captured in logs.')
              }
            </div>
          </div>
        </div>
      )}
    </>
  );
}
