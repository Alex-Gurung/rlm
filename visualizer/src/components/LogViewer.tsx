'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { StatsCard } from './StatsCard';
import { TrajectoryPanel } from './TrajectoryPanel';
import { ExecutionPanel } from './ExecutionPanel';
import { IterationTimeline } from './IterationTimeline';
import { HierarchicalRunsView } from './HierarchicalRunsView';
import { ThemeToggle } from './ThemeToggle';
import { RLMLogFile, RLMIteration, HierarchicalRun } from '@/lib/types';
import { Layers, List } from 'lucide-react';

interface LogViewerProps {
  logFile: RLMLogFile;
  onBack: () => void;
}

export function LogViewer({ logFile, onBack }: LogViewerProps) {
  const [selectedIteration, setSelectedIteration] = useState(0);
  const [selectedRun, setSelectedRun] = useState<HierarchicalRun | null>(null);
  const [viewMode, setViewMode] = useState<'flat' | 'hierarchical'>('flat');
  const { iterations, metadata, config, hierarchicalRuns } = logFile;

  // Check if we have hierarchical data
  const hasHierarchy = hierarchicalRuns && (
    hierarchicalRuns.children.length > 0 ||
    (metadata.totalRuns && metadata.totalRuns > 1)
  );

  // Auto-switch to hierarchical view if hierarchy exists
  useEffect(() => {
    if (hasHierarchy) {
      setViewMode('hierarchical');
    }
  }, [hasHierarchy]);

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
  }, [iterations]);

  // Get current iteration (considering both flat and hierarchical selection)
  const currentIteration = iterations[selectedIteration] || null;

  const goToPrevious = useCallback(() => {
    setSelectedIteration(prev => Math.max(0, prev - 1));
  }, []);

  const goToNext = useCallback(() => {
    setSelectedIteration(prev => Math.min(iterations.length - 1, prev + 1));
  }, [iterations.length]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
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
    <div className="h-screen flex flex-col overflow-hidden bg-background">
      {/* Top Bar - Compact header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm">
        <div className="px-6 py-3">
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
      <div className="border-b border-border bg-muted/30 px-6 py-4">
        <div className="flex gap-6">
          {/* Question & Answer Summary */}
          <Card className="flex-1 bg-gradient-to-r from-primary/5 to-accent/5 border-primary/20">
            <CardContent className="p-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Context / Question
                  </p>
                  <p className="text-sm font-medium line-clamp-2">
                    {metadata.contextQuestion}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Final Answer
                  </p>
                  <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400 line-clamp-2">
                    {metadata.finalAnswer || 'Not yet completed'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <div className="flex gap-2">
            <StatsCard
              label="Iterations"
              value={metadata.totalIterations}
              icon="◎"
              variant="cyan"
            />
            <StatsCard
              label="Code"
              value={metadata.totalCodeBlocks}
              icon="⟨⟩"
              variant="green"
            />
            <StatsCard
              label="Sub-LM"
              value={metadata.totalSubLMCalls}
              icon="◇"
              variant="magenta"
            />
            <StatsCard
              label="Store"
              value={metadata.totalStoreEvents}
              icon="▦"
              variant="cyan"
              subtext={`${metadata.totalStoreEvents} events`}
            />
            <StatsCard
              label="Batch"
              value={metadata.totalBatchCalls}
              icon="⇉"
              variant="yellow"
              subtext={`${metadata.totalBatchPrompts} prompts`}
            />
            <StatsCard
              label="Commits"
              value={metadata.totalCommitEvents}
              icon="⇄"
              variant="magenta"
              subtext={`${metadata.totalCommitEvents} events`}
            />
            {hasHierarchy && (
              <StatsCard
                label="Runs"
                value={metadata.totalRuns || 1}
                icon="⬡"
                variant="cyan"
                subtext={`depth ${metadata.maxDepth || 0}`}
              />
            )}
            <StatsCard
              label="Exec"
              value={`${metadata.totalExecutionTime.toFixed(2)}s`}
              icon="⏱"
              variant="yellow"
            />
          </div>
        </div>
      </div>

      {/* Iteration Timeline / Hierarchical View */}
      <div className="border-b border-border">
        {/* View mode toggle (only show if hierarchy exists) */}
        {hasHierarchy && (
          <div className="flex items-center gap-2 px-4 py-2 bg-muted/20 border-b border-border">
            <span className="text-xs text-muted-foreground">View:</span>
            <Button
              variant={viewMode === 'flat' ? 'secondary' : 'ghost'}
              size="sm"
              className="h-6 text-xs px-2"
              onClick={() => setViewMode('flat')}
            >
              <List className="w-3 h-3 mr-1" />
              Flat
            </Button>
            <Button
              variant={viewMode === 'hierarchical' ? 'secondary' : 'ghost'}
              size="sm"
              className="h-6 text-xs px-2"
              onClick={() => setViewMode('hierarchical')}
            >
              <Layers className="w-3 h-3 mr-1" />
              Hierarchical
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
          <div className="max-h-48 overflow-y-auto px-4 py-2">
            <HierarchicalRunsView
              rootRun={hierarchicalRuns}
              onSelectIteration={handleHierarchicalSelect}
              selectedIteration={currentIteration}
            />
          </div>
        ) : (
          <IterationTimeline
            iterations={iterations}
            selectedIteration={selectedIteration}
            onSelectIteration={setSelectedIteration}
          />
        )}
      </div>

      {/* Main Content - Resizable Split View */}
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup orientation="horizontal">
          {/* Left Panel - Prompt & Response */}
          <ResizablePanel defaultSize={50} minSize={20} maxSize={80}>
            <div className="h-full border-r border-border">
              <TrajectoryPanel
                iterations={iterations}
                selectedIteration={selectedIteration}
                onSelectIteration={setSelectedIteration}
              />
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle className="bg-border hover:bg-primary/30 transition-colors" />

          {/* Right Panel - Code Execution & Sub-LM Calls */}
          <ResizablePanel defaultSize={50} minSize={20} maxSize={80}>
            <div className="h-full bg-background">
              <ExecutionPanel
                iteration={iterations[selectedIteration] || null}
                iterations={iterations}
              />
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Keyboard hint footer */}
      <div className="border-t border-border bg-muted/30 px-6 py-1.5">
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
    </div>
  );
}
