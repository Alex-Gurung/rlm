'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { CodeBlock } from './CodeBlock';
import { RLMIteration } from '@/lib/types';

interface ExecutionPanelProps {
  iteration: RLMIteration | null;
  iterations?: RLMIteration[];
}

export function ExecutionPanel({ iteration, iterations }: ExecutionPanelProps) {
  if (!iteration) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-muted/30 border border-border flex items-center justify-center">
            <span className="text-3xl opacity-50">◇</span>
          </div>
          <p className="text-muted-foreground text-sm">
            Select an iteration to view execution details
          </p>
        </div>
      </div>
    );
  }

  const totalSubCalls = iteration.code_blocks.reduce(
    (acc, block) => acc + (block.result?.rlm_calls?.length || 0), 
    0
  );
  const storeEvents = iteration.code_blocks.flatMap(
    block => block.result?.store_events || []
  );
  const batchCalls = iteration.code_blocks.flatMap(
    block => block.result?.batch_calls || []
  );
  const allStoreEvents = (iterations || []).flatMap(iter =>
    iter.code_blocks.flatMap(block =>
      (block.result?.store_events || []).map(event => ({
        ...event,
        iteration: iter.iteration,
        iteration_ts: iter.timestamp,
      }))
    )
  );
  const allBatchCalls = (iterations || []).flatMap(iter =>
    iter.code_blocks.flatMap(block =>
      (block.result?.batch_calls || []).map(call => ({
        ...call,
        iteration: iter.iteration,
        iteration_ts: iter.timestamp,
      }))
    )
  );

  const [storeScope, setStoreScope] = useState<'iteration' | 'all'>('iteration');
  const [batchScope, setBatchScope] = useState<'iteration' | 'all'>('iteration');
  const displayedStoreEvents = storeScope === 'all' ? allStoreEvents : storeEvents;
  const displayedBatchCalls = batchScope === 'all' ? allBatchCalls : batchCalls;
  const totalStoreEvents = iterations ? allStoreEvents.length : storeEvents.length;
  const totalBatchCalls = iterations ? allBatchCalls.length : batchCalls.length;

  return (
    <div className="h-full flex flex-col overflow-hidden bg-background">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-border bg-muted/30">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
              <span className="text-emerald-500 text-sm">⟨⟩</span>
            </div>
            <div>
              <h2 className="font-semibold text-sm">Code & Sub-LM Calls</h2>
              <p className="text-[11px] text-muted-foreground">
                Iteration {iteration.iteration} • {new Date(iteration.timestamp).toLocaleString()}
              </p>
            </div>
          </div>
        </div>
        
        {/* Quick stats */}
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline" className="text-xs">
            {iteration.code_blocks.length} code block{iteration.code_blocks.length !== 1 ? 's' : ''}
          </Badge>
          {totalSubCalls > 0 && (
            <Badge className="bg-fuchsia-500/15 text-fuchsia-600 dark:text-fuchsia-400 border-fuchsia-500/30 text-xs">
              {totalSubCalls} sub-LM call{totalSubCalls !== 1 ? 's' : ''}
            </Badge>
          )}
          {storeEvents.length > 0 && (
            <Badge className="bg-cyan-500/15 text-cyan-600 dark:text-cyan-400 border-cyan-500/30 text-xs">
              {storeEvents.length}{iterations ? `/${totalStoreEvents}` : ''} store event{storeEvents.length !== 1 ? 's' : ''}
            </Badge>
          )}
          {batchCalls.length > 0 && (
            <Badge className="bg-indigo-500/15 text-indigo-600 dark:text-indigo-400 border-indigo-500/30 text-xs">
              {batchCalls.length}{iterations ? `/${totalBatchCalls}` : ''} batch call{batchCalls.length !== 1 ? 's' : ''}
            </Badge>
          )}
          {iteration.final_answer && (
            <Badge className="bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30 text-xs">
              Has Final Answer
            </Badge>
          )}
        </div>
      </div>

      {/* Tabs - Code Execution and Sub-LM Calls only */}
      <Tabs defaultValue="code" className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-shrink-0 px-4 pt-3">
          <TabsList className="w-full grid grid-cols-4">
            <TabsTrigger value="code" className="text-xs">
              Code Execution
            </TabsTrigger>
            <TabsTrigger value="sublm" className="text-xs">
              Sub-LM Calls ({totalSubCalls})
            </TabsTrigger>
            <TabsTrigger value="store" className="text-xs">
              Store ({storeEvents.length})
            </TabsTrigger>
            <TabsTrigger value="batch" className="text-xs">
              Batch ({batchCalls.length})
            </TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-hidden">
          <TabsContent value="code" className="h-full m-0 data-[state=active]:flex data-[state=active]:flex-col">
            <ScrollArea className="flex-1 h-full">
              <div className="p-4 space-y-4">
                {iteration.code_blocks.length > 0 ? (
                  iteration.code_blocks.map((block, idx) => (
                    <CodeBlock key={idx} block={block} index={idx} />
                  ))
                ) : (
                  <Card className="border-dashed">
                    <CardContent className="p-8 text-center">
                      <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-muted/30 border border-border flex items-center justify-center">
                        <span className="text-xl opacity-50">⟨⟩</span>
                      </div>
                      <p className="text-muted-foreground text-sm">
                        No code was executed in this iteration
                      </p>
                      <p className="text-muted-foreground text-xs mt-1">
                        The model didn&apos;t write any code blocks
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="sublm" className="h-full m-0 data-[state=active]:flex data-[state=active]:flex-col">
            <ScrollArea className="flex-1 h-full">
              <div className="p-4 space-y-4">
                {totalSubCalls > 0 ? (
                  iteration.code_blocks.flatMap((block, blockIdx) =>
                    (block.result?.rlm_calls || []).map((call, callIdx) => (
                      <Card 
                        key={`${blockIdx}-${callIdx}`}
                        className="border-fuchsia-500/30 bg-fuchsia-500/5 dark:border-fuchsia-400/30 dark:bg-fuchsia-400/5"
                      >
                        <CardHeader className="py-3 px-4">
                          <div className="flex items-center justify-between flex-wrap gap-2">
                            <CardTitle className="text-sm flex items-center gap-2">
                              <span className="w-2 h-2 rounded-full bg-fuchsia-500 dark:bg-fuchsia-400" />
                              llm_query() from Block #{blockIdx + 1}
                            </CardTitle>
                            <div className="flex gap-2">
                              {(() => {
                                const usage = call.usage_summary?.model_usage_summaries?.[call.root_model ?? ''];
                                const inputTokens = call.prompt_tokens ?? usage?.total_input_tokens;
                                const outputTokens = call.completion_tokens ?? usage?.total_output_tokens;
                                return (
                                  <>
                                    <Badge variant="outline" className="text-[10px] font-mono">
                                      {inputTokens != null ? `${inputTokens} in` : 'n/a in'}
                                    </Badge>
                                    <Badge variant="outline" className="text-[10px] font-mono">
                                      {outputTokens != null ? `${outputTokens} out` : 'n/a out'}
                                    </Badge>
                                  </>
                                );
                              })()}
                              <Badge variant="outline" className="text-[10px] font-mono">
                                {call.execution_time.toFixed(2)}s
                              </Badge>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent className="px-4 pb-4 space-y-3">
                          <div>
                            <p className="text-xs text-muted-foreground mb-1.5 font-medium uppercase tracking-wider">
                              Prompt
                            </p>
                            <div className="bg-muted/50 rounded-lg p-3 max-h-40 overflow-y-auto border border-border">
                              <pre className="text-xs whitespace-pre-wrap font-mono">
                                {typeof call.prompt === 'string' 
                                  ? call.prompt 
                                  : JSON.stringify(call.prompt, null, 2)}
                              </pre>
                            </div>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground mb-1.5 font-medium uppercase tracking-wider">
                              Response
                            </p>
                            <div className="bg-fuchsia-500/10 dark:bg-fuchsia-400/10 rounded-lg p-3 max-h-56 overflow-y-auto border border-fuchsia-500/20 dark:border-fuchsia-400/20">
                              <pre className="text-xs whitespace-pre-wrap font-mono text-fuchsia-700 dark:text-fuchsia-300">
                                {call.response}
                              </pre>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )
                ) : (
                  <Card className="border-dashed">
                    <CardContent className="p-8 text-center">
                      <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-muted/30 border border-border flex items-center justify-center">
                        <span className="text-xl opacity-50">⊘</span>
                      </div>
                      <p className="text-muted-foreground text-sm">
                        No sub-LM calls were made in this iteration
                      </p>
                      <p className="text-muted-foreground text-xs mt-1">
                        Sub-LM calls appear when using llm_query() in the REPL
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="store" className="h-full m-0 data-[state=active]:flex data-[state=active]:flex-col">
            <ScrollArea className="flex-1 h-full">
              <div className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-muted-foreground">
                    Scope
                  </div>
                  <div className="flex gap-2">
                    <Badge
                      variant={storeScope === 'iteration' ? 'default' : 'outline'}
                      className="cursor-pointer text-[10px]"
                      onClick={() => setStoreScope('iteration')}
                    >
                      Iteration
                    </Badge>
                    <Badge
                      variant={storeScope === 'all' ? 'default' : 'outline'}
                      className="cursor-pointer text-[10px]"
                      onClick={() => setStoreScope('all')}
                    >
                      All
                    </Badge>
                  </div>
                </div>
                {displayedStoreEvents.length > 0 ? (
                  displayedStoreEvents.map((event, idx) => (
                    <Card key={idx} className="border-cyan-500/30 bg-cyan-500/5 dark:border-cyan-400/30 dark:bg-cyan-400/5">
                      <CardHeader className="py-3 px-4">
                        <div className="flex items-center justify-between flex-wrap gap-2">
                          <CardTitle className="text-sm flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-cyan-500 dark:bg-cyan-400" />
                            {event.op}
                          </CardTitle>
                          <div className="flex gap-2">
                            <Badge variant="outline" className="text-[10px] font-mono">
                              {event.type}
                            </Badge>
                            <Badge variant="outline" className="text-[10px] font-mono">
                              {event.id}
                            </Badge>
                            {'iteration' in event && (
                              <Badge variant="outline" className="text-[10px] font-mono">
                                iter {event.iteration}
                              </Badge>
                            )}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="px-4 pb-4 space-y-2">
                        <div className="text-xs text-muted-foreground">Description</div>
                        <div className="bg-muted/50 rounded-lg p-3 border border-border text-xs">
                          {event.description}
                        </div>
                        {'parents' in event && event.parents && event.parents.length > 0 && (
                          <div className="text-[10px] text-muted-foreground">
                            Parents: {event.parents.join(', ')}
                          </div>
                        )}
                        {'tags' in event && event.tags && event.tags.length > 0 && (
                          <div className="text-[10px] text-muted-foreground">
                            Tags: {event.tags.join(', ')}
                          </div>
                        )}
                        {'backrefs_count' in event && (
                          <div className="text-[10px] text-muted-foreground">
                            Backrefs: {event.backrefs_count ?? 0}
                          </div>
                        )}
                        <div className="text-[10px] text-muted-foreground">
                          {new Date(event.ts * 1000).toLocaleString()}
                        </div>
                      </CardContent>
                    </Card>
                  ))
                ) : (
                  <Card className="border-dashed">
                    <CardContent className="p-8 text-center">
                      <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-muted/30 border border-border flex items-center justify-center">
                        <span className="text-xl opacity-50">∅</span>
                      </div>
                      <p className="text-muted-foreground text-sm">
                        No store events recorded in this iteration
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="batch" className="h-full m-0 data-[state=active]:flex data-[state=active]:flex-col">
            <ScrollArea className="flex-1 h-full">
              <div className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-muted-foreground">
                    Scope
                  </div>
                  <div className="flex gap-2">
                    <Badge
                      variant={batchScope === 'iteration' ? 'default' : 'outline'}
                      className="cursor-pointer text-[10px]"
                      onClick={() => setBatchScope('iteration')}
                    >
                      Iteration
                    </Badge>
                    <Badge
                      variant={batchScope === 'all' ? 'default' : 'outline'}
                      className="cursor-pointer text-[10px]"
                      onClick={() => setBatchScope('all')}
                    >
                      All
                    </Badge>
                  </div>
                </div>
                {displayedBatchCalls.length > 0 ? (
                  displayedBatchCalls.map((call, idx) => (
                    <Card key={idx} className="border-indigo-500/30 bg-indigo-500/5 dark:border-indigo-400/30 dark:bg-indigo-400/5">
                      <CardHeader className="py-3 px-4">
                        <div className="flex items-center justify-between flex-wrap gap-2">
                          <CardTitle className="text-sm flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-indigo-500 dark:bg-indigo-400" />
                            Batch call
                          </CardTitle>
                          <div className="flex gap-2">
                            <Badge variant="outline" className="text-[10px] font-mono">
                              {call.prompts_count} prompts
                            </Badge>
                            <Badge variant="outline" className="text-[10px] font-mono">
                              {call.model ?? 'default'}
                            </Badge>
                            {'iteration' in call && (
                              <Badge variant="outline" className="text-[10px] font-mono">
                                iter {call.iteration}
                              </Badge>
                            )}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="px-4 pb-4 space-y-2">
                        <div className="text-xs text-muted-foreground">
                          Execution time
                        </div>
                        <div className="bg-muted/50 rounded-lg p-3 border border-border text-xs font-mono">
                          {call.execution_time.toFixed(2)}s
                        </div>
                        <div className="text-[10px] text-muted-foreground">
                          {new Date(call.ts * 1000).toLocaleString()}
                        </div>
                      </CardContent>
                    </Card>
                  ))
                ) : (
                  <Card className="border-dashed">
                    <CardContent className="p-8 text-center">
                      <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-muted/30 border border-border flex items-center justify-center">
                        <span className="text-xl opacity-50">∅</span>
                      </div>
                      <p className="text-muted-foreground text-sm">
                        No batched calls recorded in this iteration
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </ScrollArea>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
