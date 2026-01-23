'use client';

import { useMemo, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { RLMChatCompletion, RLMIteration, HierarchicalRun } from '@/lib/types';

interface WorkTreeViewProps {
  iterations: RLMIteration[];
  rootRun: HierarchicalRun;
  onSelectIteration: (iteration: RLMIteration, run: HierarchicalRun) => void;
  selectedIteration?: RLMIteration | null;
}

type StoreNode = {
  id: string;
  type: string;
  description: string;
  parents: string[];
  children: string[];
  ts?: number;
  tags?: string[];
};

function buildStoreTree(iterations: RLMIteration[]) {
  const nodes = new Map<string, StoreNode>();

  const ensureNode = (id: string) => {
    if (!nodes.has(id)) {
      nodes.set(id, {
        id,
        type: 'unknown',
        description: '(missing)',
        parents: [],
        children: [],
      });
    }
    return nodes.get(id)!;
  };

  const addChild = (parentId: string, childId: string) => {
    const parent = ensureNode(parentId);
    if (!parent.children.includes(childId)) {
      parent.children.push(childId);
    }
  };

  iterations.forEach(iter => {
    iter.code_blocks.forEach(block => {
      const events = block.result?.store_events || [];
      events.forEach(event => {
        if (event.op !== 'create') return;
        const node = ensureNode(event.id);
        node.type = event.type || node.type;
        node.description = event.description || node.description;
        node.parents = event.parents || [];
        node.ts = event.ts;
        node.tags = event.tags || node.tags;
        (event.parents || []).forEach(parentId => addChild(parentId, event.id));
      });
    });
  });

  const roots: StoreNode[] = [];
  for (const node of nodes.values()) {
    if (!node.parents || node.parents.length === 0) {
      roots.push(node);
    }
  }

  const sortByTs = (a: StoreNode, b: StoreNode) => (b.ts || 0) - (a.ts || 0);
  roots.sort(sortByTs);
  for (const node of nodes.values()) {
    node.children.sort((a, b) => {
      const aNode = nodes.get(a);
      const bNode = nodes.get(b);
      return (bNode?.ts || 0) - (aNode?.ts || 0);
    });
  }

  return { nodes, roots };
}

function countSubcalls(iterations: RLMIteration[]) {
  return iterations.reduce((acc, iter) => {
    const calls = iter.code_blocks.reduce(
      (sum, block) => sum + (block.result?.rlm_calls?.length || 0),
      0
    );
    return acc + calls;
  }, 0);
}

function countStoreEvents(iterations: RLMIteration[]) {
  return iterations.reduce((acc, iter) => {
    const events = iter.code_blocks.reduce(
      (sum, block) => sum + (block.result?.store_events?.length || 0),
      0
    );
    return acc + events;
  }, 0);
}

export function WorkTreeView({ iterations, rootRun, onSelectIteration, selectedIteration }: WorkTreeViewProps) {
  const [expandedRuns, setExpandedRuns] = useState<Record<string, boolean>>({});
  const [expandedStore, setExpandedStore] = useState<Record<string, boolean>>({});
  const [expandedSubcalls, setExpandedSubcalls] = useState<Record<string, boolean>>({});
  const [expandedStoreNodes, setExpandedStoreNodes] = useState<Record<string, boolean>>({});

  const runMap = useMemo(() => {
    const map = new Map<string, RLMIteration[]>();
    iterations.forEach(iter => {
      const runId = iter.run_id ?? 'root';
      if (!map.has(runId)) map.set(runId, []);
      map.get(runId)!.push(iter);
    });
    return map;
  }, [iterations]);

  const toggleRun = (runId: string) => {
    setExpandedRuns(prev => ({ ...prev, [runId]: !prev[runId] }));
  };

  const toggleStore = (runId: string) => {
    setExpandedStore(prev => ({ ...prev, [runId]: !prev[runId] }));
  };

  const toggleSubcalls = (runId: string) => {
    setExpandedSubcalls(prev => ({ ...prev, [runId]: !prev[runId] }));
  };

  const toggleStoreNode = (nodeId: string) => {
    setExpandedStoreNodes(prev => ({ ...prev, [nodeId]: !prev[nodeId] }));
  };

  const formatPrompt = (prompt: RLMChatCompletion['prompt']) => {
    if (typeof prompt === 'string') return prompt;
    try {
      return JSON.stringify(prompt, null, 2);
    } catch {
      return '[structured prompt]';
    }
  };

  const renderStoreTree = (nodes: Map<string, StoreNode>, nodeId: string, depth: number) => {
    const node = nodes.get(nodeId);
    if (!node) return null;
    const nodeExpanded = expandedStoreNodes[nodeId] ?? false;
    return (
      <div key={nodeId} className="space-y-1">
        <div className="flex items-start gap-2 rounded-lg border border-border bg-muted/30 px-3 py-2" style={{ marginLeft: depth * 12 }}>
          <Badge variant="outline" className="text-[10px] font-mono">
            {node.type}
          </Badge>
          <div className="flex-1 min-w-0">
            <div className="text-xs font-mono text-muted-foreground">{node.id}</div>
            <div className="text-xs">{node.description}</div>
            {node.tags && node.tags.length > 0 && (
              <div className="mt-1 flex flex-wrap gap-1">
                {node.tags.map(tag => (
                  <Badge key={tag} variant="outline" className="text-[9px] font-mono">
                    #{tag}
                  </Badge>
                ))}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            {node.children.length > 0 && (
              <Badge variant="outline" className="text-[10px]">{node.children.length}</Badge>
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-[10px]"
              onClick={() => toggleStoreNode(nodeId)}
            >
              {nodeExpanded ? 'Hide' : 'Inspect'}
            </Button>
          </div>
        </div>
        {nodeExpanded && (
          <div className="rounded-lg border border-border bg-background/60 px-3 py-2 text-[11px] text-muted-foreground" style={{ marginLeft: depth * 12 }}>
            <div>Parents: {node.parents.length > 0 ? node.parents.join(', ') : 'none'}</div>
            <div>Children: {node.children.length > 0 ? node.children.join(', ') : 'none'}</div>
          </div>
        )}
        {node.children.length > 0 && (
          <div className="space-y-1">
            {node.children.map(childId => renderStoreTree(nodes, childId, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  const renderRun = (run: HierarchicalRun, level: number) => {
    const runIters = runMap.get(run.run_id) || [];
    const subcalls = countSubcalls(runIters);
    const storeEvents = countStoreEvents(runIters);
    const storeTree = buildStoreTree(runIters);
    const isExpanded = expandedRuns[run.run_id] ?? level === 0;
    const storeExpanded = expandedStore[run.run_id] ?? false;
    const subcallsExpanded = expandedSubcalls[run.run_id] ?? false;

    return (
      <div key={run.run_id} className="space-y-2">
        <Card className="border-primary/20 bg-gradient-to-br from-primary/5 via-background to-accent/5">
          <CardHeader className="py-3 px-4">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm flex items-center gap-2">
                <span className="text-primary">⬡</span>
                Run {run.run_id.slice(0, 8)} (depth {run.depth})
              </CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-[10px] font-mono">{runIters.length} iters</Badge>
                <Badge variant="outline" className="text-[10px] font-mono">{subcalls} subcalls</Badge>
                <Badge variant="outline" className="text-[10px] font-mono">{storeEvents} store</Badge>
                <Button variant="ghost" size="sm" className="h-6 text-[10px]" onClick={() => toggleRun(run.run_id)}>
                  {isExpanded ? 'Collapse' : 'Expand'}
                </Button>
              </div>
            </div>
            {(run.workerPromptPreview || run.resultSummary) && (
              <div className="mt-2 grid gap-2 text-[11px] text-muted-foreground">
                {run.workerPromptPreview && (
                  <div className="rounded-md border border-border bg-muted/30 px-2 py-1">
                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Worker Prompt</div>
                    <div className="line-clamp-2">{run.workerPromptPreview}</div>
                  </div>
                )}
                {run.resultSummary && (
                  <div className="rounded-md border border-border bg-muted/30 px-2 py-1">
                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Result Summary</div>
                    <div className="line-clamp-2">{run.resultSummary}</div>
                  </div>
                )}
              </div>
            )}
          </CardHeader>
          {isExpanded && (
            <CardContent className="px-4 pb-4 space-y-3">
              <div className="space-y-2">
                <div className="text-xs text-muted-foreground font-medium">Iterations</div>
                <div className="flex flex-wrap gap-2">
                  {runIters.map(iter => (
                    <Button
                      key={`${run.run_id}-${iter.iteration}`}
                      variant={selectedIteration?.iteration === iter.iteration && selectedIteration?.run_id === iter.run_id ? 'secondary' : 'outline'}
                      size="sm"
                      className="h-6 text-[10px]"
                      onClick={() => onSelectIteration(iter, run)}
                    >
                      iter {iter.iteration}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-muted-foreground font-medium">Store Tree</div>
                  <Button variant="ghost" size="sm" className="h-6 text-[10px]" onClick={() => toggleStore(run.run_id)}>
                    {storeExpanded ? 'Hide' : 'Show'}
                  </Button>
                </div>
                {storeExpanded && (
                  <div className="space-y-2">
                    {storeTree.roots.length > 0 ? (
                      storeTree.roots.map(root => renderStoreTree(storeTree.nodes, root.id, 0))
                    ) : (
                      <div className="text-xs text-muted-foreground">No store objects recorded for this run.</div>
                    )}
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-muted-foreground font-medium">Sub‑LM Calls</div>
                  <Button variant="ghost" size="sm" className="h-6 text-[10px]" onClick={() => toggleSubcalls(run.run_id)}>
                    {subcallsExpanded ? 'Hide' : 'Show'}
                  </Button>
                </div>
                {subcallsExpanded && (
                  <div className="space-y-2">
                    {runIters.map(iter => {
                      const calls = iter.code_blocks.flatMap(block => block.result?.rlm_calls || []);
                      if (calls.length === 0) return null;
                      return (
                        <div key={`${run.run_id}-${iter.iteration}`} className="rounded-lg border border-border bg-muted/20 p-2">
                          <div className="text-[10px] text-muted-foreground mb-1">iter {iter.iteration} • {calls.length} call{calls.length !== 1 ? 's' : ''}</div>
                          <Accordion type="multiple" className="space-y-1">
                            {calls.map((call, idx) => {
                              const callKey = `${run.run_id}-${iter.iteration}-${idx}`;
                              return (
                                <AccordionItem key={callKey} value={callKey} className="border border-border rounded-md">
                                  <AccordionTrigger className="px-2 py-1 text-xs hover:no-underline">
                                    <div className="flex flex-1 items-center justify-between gap-2">
                                      <span className="font-mono text-muted-foreground">call {idx + 1}</span>
                                      <span className="text-[10px] text-muted-foreground">
                                        {(call.root_model ?? 'model')} • {call.execution_time.toFixed(2)}s
                                      </span>
                                    </div>
                                  </AccordionTrigger>
                                  <AccordionContent className="px-2 pb-2">
                                    <div className="space-y-2">
                                      <div className="rounded-md border border-border bg-background/60 p-2">
                                        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Prompt</div>
                                        <pre className="text-[11px] whitespace-pre-wrap break-words">{formatPrompt(call.prompt)}</pre>
                                      </div>
                                      <div className="rounded-md border border-border bg-background/60 p-2">
                                        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Response</div>
                                        <pre className="text-[11px] whitespace-pre-wrap break-words">{call.response}</pre>
                                      </div>
                                      {(call.prompt_tokens || call.completion_tokens) && (
                                        <div className="flex gap-2 text-[10px] text-muted-foreground">
                                          <span>prompt {call.prompt_tokens ?? 0}</span>
                                          <span>completion {call.completion_tokens ?? 0}</span>
                                        </div>
                                      )}
                                    </div>
                                  </AccordionContent>
                                </AccordionItem>
                              );
                            })}
                          </Accordion>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </CardContent>
          )}
        </Card>

        {run.children.map(child => (
          <div
            key={child.run_id}
            className="border-l border-dashed border-primary/30 pl-4 ml-2"
            style={{ marginLeft: Math.min(24, (level + 1) * 12) }}
          >
            {renderRun(child, level + 1)}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="h-full">
      <ScrollArea className="h-full">
        <div className="p-4 space-y-4">
          {renderRun(rootRun, 0)}
        </div>
      </ScrollArea>
    </div>
  );
}
