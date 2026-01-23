'use client';

import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import type { MouseEvent } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { FileUploader } from './FileUploader';
import { LogViewer } from './LogViewer';
import { AsciiRLM } from './AsciiGlobe';
import { ThemeToggle } from './ThemeToggle';
import { parseLogFile, extractContextVariable } from '@/lib/parse-logs';
import { RLMLogFile } from '@/lib/types';
import { cn } from '@/lib/utils';

interface DemoLogInfo {
  fileName: string;
  contextPreview: string | null;
  hasFinalAnswer: boolean;
  iterations: number;
  taskName: string | null;
  storeMode: string | null;
  rootModel: string | null;
  backend: string | null;
  environment: string | null;
  subcalls: number;
  storeEvents: number;
  commitEvents: number;
  executionTime: number;
  hasErrors: boolean;
}

export function Dashboard() {
  const [logFiles, setLogFiles] = useState<RLMLogFile[]>([]);
  const [selectedLog, setSelectedLog] = useState<RLMLogFile | null>(null);
  const [demoLogs, setDemoLogs] = useState<DemoLogInfo[]>([]);
  const [loadingDemos, setLoadingDemos] = useState(true);
  const [recentLogs, setRecentLogs] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [taskFilter, setTaskFilter] = useState('all');
  const [modelFilter, setModelFilter] = useState('all');
  const historyPrimed = useRef(false);
  const router = useRouter();
  const searchParams = useSearchParams();

  const updateParams = useCallback((updates: Record<string, string | null | undefined>) => {
    const params = new URLSearchParams(searchParams.toString());
    for (const [key, value] of Object.entries(updates)) {
      if (value === null || value === undefined || value === '') {
        params.delete(key);
      } else {
        params.set(key, value);
      }
    }
    const next = params.toString();
    const current = searchParams.toString();
    if (next !== current) {
      router.replace(next ? `?${next}` : '?');
    }
  }, [router, searchParams]);

  const pushParams = useCallback((updates: Record<string, string | null | undefined>) => {
    const params = new URLSearchParams(searchParams.toString());
    for (const [key, value] of Object.entries(updates)) {
      if (value === null || value === undefined || value === '') {
        params.delete(key);
      } else {
        params.set(key, value);
      }
    }
    const next = params.toString();
    router.push(next ? `?${next}` : '?');
  }, [router, searchParams]);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem('rlm_recent_logs');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          setRecentLogs(parsed);
        }
      }
    } catch (e) {
      console.warn('Failed to load recent logs', e);
    }
  }, []);

  const persistRecentLogs = useCallback((nextLogs: string[]) => {
    setRecentLogs(nextLogs);
    try {
      window.localStorage.setItem('rlm_recent_logs', JSON.stringify(nextLogs));
    } catch (e) {
      console.warn('Failed to save recent logs', e);
    }
  }, []);

  const markRecent = useCallback((fileName: string) => {
    const next = [fileName, ...recentLogs.filter(name => name !== fileName)].slice(0, 10);
    persistRecentLogs(next);
  }, [persistRecentLogs, recentLogs]);

  // Load demo log previews on mount - fetches latest 10 from API
  useEffect(() => {
    async function loadDemoPreviews() {
      try {
        // Fetch list of log files from API
        const listResponse = await fetch('/api/logs');
        if (!listResponse.ok) {
          throw new Error('Failed to fetch log list');
        }
        const { files } = await listResponse.json();
        
        const previews: DemoLogInfo[] = [];
        
        for (const fileName of files) {
          try {
            const response = await fetch(`/logs/${fileName}`);
            if (!response.ok) continue;
            const content = await response.text();
            const parsed = parseLogFile(fileName, content);
            const contextVar = extractContextVariable(parsed.iterations);
            
            previews.push({
              fileName,
              contextPreview: contextVar,
              hasFinalAnswer: !!parsed.metadata.finalAnswer,
              iterations: parsed.metadata.totalIterations,
              taskName: parsed.config.task_name ?? null,
              storeMode: parsed.config.store_mode ?? null,
              rootModel: parsed.config.root_model ?? null,
              backend: parsed.config.backend ?? null,
              environment: parsed.config.environment_type ?? null,
              subcalls: parsed.metadata.totalSubLMCalls,
              storeEvents: parsed.metadata.totalStoreEvents,
              commitEvents: parsed.metadata.totalCommitEvents,
              executionTime: parsed.metadata.totalExecutionTime,
              hasErrors: parsed.metadata.hasErrors,
            });
          } catch (e) {
            console.error('Failed to load demo preview:', fileName, e);
          }
        }
        
        setDemoLogs(previews);
      } catch (e) {
        console.error('Failed to load demo logs:', e);
      } finally {
        setLoadingDemos(false);
      }
    }
    
    loadDemoPreviews();
  }, []);

  const handleFileLoaded = useCallback((fileName: string, content: string) => {
    const parsed = parseLogFile(fileName, content);
    setLogFiles(prev => {
      if (prev.some(f => f.fileName === fileName)) {
        return prev.map(f => f.fileName === fileName ? parsed : f);
      }
      return [...prev, parsed];
    });
    setSelectedLog(parsed);
    pushParams({ log: fileName });
    markRecent(fileName);
  }, [markRecent, pushParams]);

  const loadDemoLog = useCallback(async (fileName: string) => {
    try {
      const response = await fetch(`/logs/${fileName}`);
      if (!response.ok) throw new Error('Failed to load demo log');
      const content = await response.text();
      handleFileLoaded(fileName, content);
    } catch (error) {
      console.error('Error loading demo log:', error);
      alert('Failed to load demo log. Make sure the log files are in the public/logs folder.');
    }
  }, [handleFileLoaded]);

  const openLog = useCallback((fileName: string) => {
    const existing = logFiles.find(log => log.fileName === fileName);
    if (existing) {
      setSelectedLog(existing);
      pushParams({ log: fileName });
      markRecent(fileName);
      return;
    }
    loadDemoLog(fileName);
  }, [loadDemoLog, logFiles, markRecent, pushParams]);

  useEffect(() => {
    const logParam = searchParams.get('log');
    if (!logParam) return;
    if (
      !historyPrimed.current
      && typeof window !== 'undefined'
      && (window.history.length <= 1 || document.referrer === '' || document.referrer === 'about:blank')
    ) {
      historyPrimed.current = true;
      router.replace('?');
      router.push(getLogHref(logParam));
      return;
    }
    if (selectedLog?.fileName === logParam) return;
    const existing = logFiles.find(log => log.fileName === logParam);
    if (existing) {
      setSelectedLog(existing);
      markRecent(existing.fileName);
      return;
    }
    loadDemoLog(logParam);
  }, [loadDemoLog, logFiles, markRecent, router, searchParams, selectedLog]);

  const filteredDemoLogs = useMemo(() => {
    if (!searchQuery.trim()) return demoLogs;
    const query = searchQuery.toLowerCase();
    return demoLogs.filter(item =>
      item.fileName.toLowerCase().includes(query) ||
      (item.contextPreview ?? '').toLowerCase().includes(query)
    );
  }, [demoLogs, searchQuery]);

  const filteredLoadedLogs = useMemo(() => {
    if (!searchQuery.trim()) return logFiles;
    const query = searchQuery.toLowerCase();
    return logFiles.filter(item =>
      item.fileName.toLowerCase().includes(query) ||
      (item.metadata.contextQuestion ?? '').toLowerCase().includes(query) ||
      (item.config.task_name ?? '').toLowerCase().includes(query) ||
      (item.config.root_model ?? '').toLowerCase().includes(query)
    );
  }, [logFiles, searchQuery]);

  const tableRows = useMemo(() => {
    const byName = new Map<string, DemoLogInfo>();
    for (const demo of demoLogs) {
      byName.set(demo.fileName, demo);
    }
    for (const log of logFiles) {
      byName.set(log.fileName, {
        fileName: log.fileName,
        contextPreview: extractContextVariable(log.iterations),
        hasFinalAnswer: !!log.metadata.finalAnswer,
        iterations: log.metadata.totalIterations,
        taskName: log.config.task_name ?? null,
        storeMode: log.config.store_mode ?? null,
        rootModel: log.config.root_model ?? null,
        backend: log.config.backend ?? null,
        environment: log.config.environment_type ?? null,
        subcalls: log.metadata.totalSubLMCalls,
        storeEvents: log.metadata.totalStoreEvents,
        commitEvents: log.metadata.totalCommitEvents,
        executionTime: log.metadata.totalExecutionTime,
        hasErrors: log.metadata.hasErrors,
      });
    }
    const rows = Array.from(byName.values());
    if (!searchQuery.trim()) return rows;
    const query = searchQuery.toLowerCase();
    return rows.filter(row =>
      row.fileName.toLowerCase().includes(query) ||
      (row.contextPreview ?? '').toLowerCase().includes(query) ||
      (row.taskName ?? '').toLowerCase().includes(query) ||
      (row.rootModel ?? '').toLowerCase().includes(query)
    );
  }, [demoLogs, logFiles, searchQuery]);

  const filteredTableRows = useMemo(() => {
    return tableRows.filter(row => {
      if (taskFilter !== 'all' && row.taskName !== taskFilter) return false;
      if (modelFilter !== 'all' && row.rootModel !== modelFilter) return false;
      return true;
    });
  }, [modelFilter, tableRows, taskFilter]);

  const uniqueTasks = useMemo(() => {
    const tasks = new Set<string>();
    tableRows.forEach(row => {
      if (row.taskName) tasks.add(row.taskName);
    });
    return Array.from(tasks).sort();
  }, [tableRows]);

  const uniqueModels = useMemo(() => {
    const models = new Set<string>();
    tableRows.forEach(row => {
      if (row.rootModel) models.add(row.rootModel);
    });
    return Array.from(models).sort();
  }, [tableRows]);

  const getRunKind = (row: DemoLogInfo) => {
    if (row.taskName && row.taskName.toLowerCase().includes('benchmark')) return 'benchmark';
    return 'interactive';
  };

  const getLogHref = (fileName: string) => `?log=${encodeURIComponent(fileName)}`;

  const getAbsoluteLogHref = (fileName: string) => {
    if (typeof window === 'undefined') return getLogHref(fileName);
    const url = new URL(window.location.href);
    url.searchParams.set('log', fileName);
    url.searchParams.delete('iter');
    url.searchParams.delete('view');
    url.searchParams.delete('run');
    return url.toString();
  };

  const handleRowClick = (event: MouseEvent, fileName: string) => {
    const href = getAbsoluteLogHref(fileName);
    if (event.metaKey || event.ctrlKey || event.button === 1) {
      window.open(href, "_blank", "noopener");
      return;
    }
    if (event.shiftKey || event.altKey) {
      return;
    }
    event.preventDefault();
    openLog(fileName);
  };

  if (selectedLog) {
    return (
      <LogViewer 
        logFile={selectedLog} 
        onBack={() => {
          setSelectedLog(null);
          updateParams({ log: null, iter: null, view: null, run: null });
        }} 
      />
    );
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 grid-pattern opacity-30 dark:opacity-15" />
      <div className="absolute top-0 left-1/3 w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary/3 rounded-full blur-3xl" />
      
      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-border">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold tracking-tight">
                  <span className="text-primary">RLM</span>
                  <span className="text-muted-foreground ml-2 font-normal">Visualizer</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                  Debug recursive language model execution traces
                </p>
              </div>
              <div className="flex items-center gap-4">
                <ThemeToggle />
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-mono">
                  <span className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                    READY
                  </span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-6 py-8 space-y-10">
          <div className="grid lg:grid-cols-2 gap-10">
            {/* Left Column - Upload & ASCII Art */}
            <div className="space-y-8">
              {/* Upload Section */}
              <div>
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">01</span>
                  Upload Log File
                </h2>
                <FileUploader onFileLoaded={handleFileLoaded} />
              </div>
              
              {/* ASCII Architecture Diagram */}
              <div className="hidden lg:block">
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">◈</span>
                  RLM Architecture
                </h2>
                <div className="bg-muted/50 border border-border rounded-lg p-4 overflow-x-auto">
                  <AsciiRLM />
                </div>
              </div>
            </div>

            {/* Right Column - Demo Logs & Loaded Files */}
            <div className="space-y-8">
              {/* Search */}
              <div>
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">00</span>
                  Search Traces
                </h2>
                <div className="rounded-lg border border-border bg-card/70 p-3">
                  <input
                    value={searchQuery}
                    onChange={(event) => setSearchQuery(event.target.value)}
                    placeholder="Search by filename or context…"
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                  />
                </div>
              </div>

              {/* Recent Logs */}
              {recentLogs.length > 0 && (
                <div>
                  <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                    <span className="text-primary font-mono">01</span>
                    Recently Viewed
                  </h2>
                  <div className="grid gap-2">
                    {recentLogs.map((fileName) => (
                      <Card
                        key={fileName}
                        onClick={() => openLog(fileName)}
                        className={cn(
                          'cursor-pointer transition-all hover:scale-[1.01]',
                          'hover:border-primary/50 hover:bg-primary/5'
                        )}
                      >
                        <CardContent className="p-3">
                          <div className="flex items-center gap-3">
                            <div className="relative flex-shrink-0">
                              <div className="w-2.5 h-2.5 rounded-full bg-primary/60" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <span className="font-mono text-xs text-foreground/80">
                                  {fileName}
                                </span>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {/* Demo Logs Section */}
              <div>
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">02</span>
                  Recent Traces
                  <span className="text-[10px] text-muted-foreground/60 ml-1">(latest 10)</span>
                </h2>
                
                {loadingDemos ? (
                  <Card>
                    <CardContent className="p-6 text-center">
                      <div className="animate-pulse flex items-center justify-center gap-2 text-muted-foreground text-sm">
                        Loading traces...
                      </div>
                    </CardContent>
                  </Card>
                ) : filteredDemoLogs.length === 0 ? (
                  <Card className="border-dashed">
                    <CardContent className="p-6 text-center text-muted-foreground text-sm">
                      No log files found in /public/logs/
                    </CardContent>
                  </Card>
                ) : (
                  <ScrollArea className="h-[320px]">
                    <div className="space-y-2 pr-4">
                      {filteredDemoLogs.map((demo) => (
                        <Card
                          key={demo.fileName}
                          onClick={() => openLog(demo.fileName)}
                          className={cn(
                            'cursor-pointer transition-all hover:scale-[1.01]',
                            'hover:border-primary/50 hover:bg-primary/5'
                          )}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center gap-3">
                              {/* Status indicator */}
                              <div className="relative flex-shrink-0">
                                <div className={cn(
                                  'w-2.5 h-2.5 rounded-full',
                                  demo.hasFinalAnswer 
                                    ? 'bg-primary' 
                                    : 'bg-muted-foreground/30'
                                )} />
                                {demo.hasFinalAnswer && (
                                  <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-primary animate-ping opacity-50" />
                                )}
                              </div>
                              
                              {/* Content */}
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-mono text-xs text-foreground/80">
                                    {demo.fileName}
                                  </span>
                                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4">
                                    {demo.iterations} iter
                                  </Badge>
                                </div>
                                {demo.contextPreview && (
                                  <p className="text-[11px] font-mono text-muted-foreground truncate">
                                    {demo.contextPreview.length > 80 
                                      ? demo.contextPreview.slice(0, 80) + '...'
                                      : demo.contextPreview}
                                  </p>
                                )}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </div>

              {/* Loaded Files Section */}
              {logFiles.length > 0 && (
                <div>
                  <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                    <span className="text-primary font-mono">03</span>
                    Loaded Files
                  </h2>
                  <ScrollArea className="h-[200px]">
                    <div className="space-y-2 pr-4">
                      {filteredLoadedLogs.map((log) => (
                        <Card
                          key={log.fileName}
                          className={cn(
                            'cursor-pointer transition-all hover:scale-[1.01]',
                            'hover:border-primary/50 hover:bg-primary/5'
                          )}
                          onClick={() => {
                            openLog(log.fileName);
                          }}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center gap-3">
                              <div className="relative flex-shrink-0">
                                <div className={cn(
                                  'w-2.5 h-2.5 rounded-full',
                                  log.metadata.finalAnswer 
                                    ? 'bg-primary' 
                                    : 'bg-muted-foreground/30'
                                )} />
                                {log.metadata.finalAnswer && (
                                  <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-primary animate-ping opacity-50" />
                                )}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-mono text-xs truncate text-foreground/80">
                                    {log.fileName}
                                  </span>
                                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4">
                                    {log.metadata.totalIterations} iter
                                  </Badge>
                                </div>
                                <p className="text-[11px] text-muted-foreground truncate">
                                  {log.metadata.contextQuestion}
                                </p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              )}

            </div>
          </div>

          {/* Trace Table - Full Width */}
          <div>
            <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
              <span className="text-primary font-mono">04</span>
              Trace Table
            </h2>
            <div className="rounded-lg border border-border bg-card/70">
              <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
                <select
                  value={taskFilter}
                  onChange={(event) => setTaskFilter(event.target.value)}
                  className="rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground"
                >
                  <option value="all">All tasks</option>
                  {uniqueTasks.map(task => (
                    <option key={task} value={task}>{task}</option>
                  ))}
                </select>
                <select
                  value={modelFilter}
                  onChange={(event) => setModelFilter(event.target.value)}
                  className="rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground"
                >
                  <option value="all">All models</option>
                  {uniqueModels.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
                <span className="text-[10px] text-muted-foreground">
                  {filteredTableRows.length} traces
                </span>
              </div>
              <div className="max-h-[520px] overflow-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-muted/80 text-[10px] uppercase tracking-wide text-muted-foreground">
                    <tr>
                      <th className="px-3 py-2 text-left">File</th>
                      <th className="px-3 py-2 text-left">Task</th>
                      <th className="px-3 py-2 text-left">Run</th>
                      <th className="px-3 py-2 text-left">Store</th>
                      <th className="px-3 py-2 text-left">Model</th>
                      <th className="px-3 py-2 text-left">Env</th>
                      <th className="px-3 py-2 text-right">Iters</th>
                      <th className="px-3 py-2 text-right">Sub‑LM</th>
                      <th className="px-3 py-2 text-right">Store</th>
                      <th className="px-3 py-2 text-right">Commits</th>
                      <th className="px-3 py-2 text-right">Time (s)</th>
                      <th className="px-3 py-2 text-left">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTableRows.map(row => (
                      <tr
                        key={row.fileName}
                        className="border-b border-border/60 hover:bg-primary/5 cursor-pointer"
                        onClick={(event) => handleRowClick(event, row.fileName)}
                      >
                        <td className="px-3 py-2 font-mono text-[11px]">
                          <a
                            href={getLogHref(row.fileName)}
                            onClick={(event) => {
                              event.stopPropagation();
                              if (event.metaKey || event.ctrlKey || event.button === 1) {
                                return;
                              }
                              event.preventDefault();
                              openLog(row.fileName);
                            }}
                            className="hover:underline"
                          >
                            {row.fileName}
                          </a>
                        </td>
                        <td className="px-3 py-2">{row.taskName ?? '—'}</td>
                        <td className="px-3 py-2">{getRunKind(row)}</td>
                        <td className="px-3 py-2">{row.storeMode ?? '—'}</td>
                        <td className="px-3 py-2">{row.rootModel ?? '—'}</td>
                        <td className="px-3 py-2">{row.environment ?? '—'}</td>
                        <td className="px-3 py-2 text-right">{row.iterations}</td>
                        <td className="px-3 py-2 text-right">{row.subcalls}</td>
                        <td className="px-3 py-2 text-right">{row.storeEvents}</td>
                        <td className="px-3 py-2 text-right">{row.commitEvents}</td>
                        <td className="px-3 py-2 text-right">{row.executionTime.toFixed(2)}</td>
                        <td className="px-3 py-2">
                          {row.hasErrors ? (
                            <Badge variant="destructive" className="text-[9px]">error</Badge>
                          ) : row.hasFinalAnswer ? (
                            <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white text-[9px]">done</Badge>
                          ) : (
                            <Badge variant="outline" className="text-[9px]">running</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                    {filteredTableRows.length === 0 && (
                      <tr>
                        <td className="px-3 py-4 text-xs text-muted-foreground" colSpan={12}>
                          No traces match the current filters.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-border mt-8">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <p className="text-[10px] text-muted-foreground font-mono">
              RLM Visualizer • Recursive Language Models
            </p>
            <p className="text-[10px] text-muted-foreground font-mono">
              Prompt → [LM ↔ REPL] → Answer
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}
