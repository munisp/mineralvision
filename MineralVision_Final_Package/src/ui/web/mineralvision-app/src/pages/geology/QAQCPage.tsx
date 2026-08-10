import { useEffect, useMemo, useState } from 'react';
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  RefreshCw,
  Loader2,
} from 'lucide-react';
import { projectsApi, qaqcApi, Project, QAQCResult } from '../../services/api';
import api from '../../services/api';

interface QAQCSummaryCategory {
  total: number;
  pass: number;
  fail: number;
  passRate: number;
}

interface QAQCSummary {
  standards: QAQCSummaryCategory;
  blanks: QAQCSummaryCategory;
  duplicates: QAQCSummaryCategory;
  umpire: QAQCSummaryCategory;
}

type LoadState = 'loading' | 'ready' | 'error';

/**
 * QA/QC Analysis — wired to the real backend:
 *   GET /api/qaqc                        (QA/QC result records)
 *   GET /api/qaqc/summary/{project_id}   (pass/fail summary)
 * Alerts below are derived ONLY from real failed QAQC records. When there is
 * no data (or no failures) the page says so explicitly — the previous version
 * rendered fabricated control charts and a fake blank-failure alert.
 */
export default function QAQCPage() {
  const [project, setProject] = useState<Project | null>(null);
  const [results, setResults] = useState<QAQCResult[]>([]);
  const [summary, setSummary] = useState<QAQCSummary | null>(null);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const load = async () => {
    setLoadState('loading');
    setErrorMessage(null);
    try {
      const projects = (await projectsApi.list()).data;
      const first = projects[0] ?? null;
      setProject(first);
      const [resultsResp, summaryResp] = await Promise.all([
        qaqcApi.list(first?.id),
        first
          ? api.get<QAQCSummary>(`/api/qaqc/summary/${first.id}`)
          : Promise.resolve({ data: null }),
      ]);
      setResults(Array.isArray(resultsResp.data) ? resultsResp.data : []);
      const s = summaryResp.data as { summary?: QAQCSummary } | QAQCSummary | null;
      setSummary(s && 'summary' in s ? (s.summary ?? null) : (s as QAQCSummary | null));
      setLoadState('ready');
    } catch (err) {
      setLoadState('error');
      setErrorMessage(err instanceof Error ? err.message : 'Failed to load QA/QC data');
    }
  };

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Alerts are derived exclusively from real failed QA/QC records. */
  const alerts = useMemo(
    () =>
      results
        .filter((r) => r.status === 'failed' || r.status === 'fail')
        .map((r) => ({
          id: r.id,
          message: `${r.type} check failed: value ${r.value} vs expected ${r.expectedValue} (deviation ${r.deviation})`,
          time: r.timestamp,
        })),
    [results],
  );

  const summaryRows = useMemo(() => {
    if (!summary) return [];
    const rows: Array<{ type: string } & QAQCSummaryCategory> = [];
    if (summary.standards) rows.push({ type: 'Standards', ...summary.standards });
    if (summary.blanks) rows.push({ type: 'Blanks', ...summary.blanks });
    if (summary.duplicates) rows.push({ type: 'Field Duplicates', ...summary.duplicates });
    if (summary.umpire) rows.push({ type: 'Umpire Assays', ...summary.umpire });
    return rows;
  }, [summary]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">QA/QC Analysis</h1>
          <p className="text-muted-foreground">
            Quality control metrics from the live QA/QC register
            {project ? ` — project: ${project.name}` : ''}
          </p>
        </div>
        <button
          onClick={() => void load()}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {loadState === 'loading' && (
        <div className="bg-card border border-border rounded-xl p-10 flex items-center justify-center gap-2">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <span className="text-sm text-foreground">Loading QA/QC data…</span>
        </div>
      )}

      {loadState === 'error' && (
        <div className="bg-destructive/10 border border-destructive/40 text-destructive rounded-xl p-4 text-sm flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Failed to load QA/QC data</p>
            <p>{errorMessage}</p>
          </div>
        </div>
      )}

      {loadState === 'ready' && (
        <>
          {/* Summary cards — real pass/fail counts */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {summaryRows.length === 0 && (
              <p className="text-sm text-muted-foreground col-span-full">
                No QA/QC summary data recorded yet for this project.
              </p>
            )}
            {summaryRows.map((item) => (
              <div key={item.type} className="bg-card border border-border rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-muted-foreground">{item.type}</h3>
                  {item.fail === 0 ? (
                    <CheckCircle className="h-5 w-5 text-green-500" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 text-yellow-500" />
                  )}
                </div>
                <p className="text-2xl font-bold text-foreground">
                  {(item.passRate * (item.passRate <= 1 ? 100 : 1)).toFixed(1)}%
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {item.pass}/{item.total} passed · {item.fail} failed
                </p>
              </div>
            ))}
          </div>

          {/* Alerts — derived from real failed records only */}
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">QA/QC Alerts</h2>
            {alerts.length === 0 ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground py-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                {results.length === 0
                  ? 'No QA/QC records yet — alerts will appear here when checks fail.'
                  : 'No failed QA/QC checks. All recorded results passed.'}
              </div>
            ) : (
              <div className="space-y-2">
                {alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className="flex items-start gap-3 p-3 rounded-lg bg-red-500/10"
                  >
                    <XCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                    <div>
                      <p className="text-sm text-foreground">{alert.message}</p>
                      <p className="text-xs text-muted-foreground mt-1">{alert.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Records table */}
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">
              QA/QC Records ({results.length})
            </h2>
            {results.length === 0 ? (
              <p className="text-sm text-muted-foreground py-2">
                No QA/QC records in the register. Insert standards/blanks/duplicates via the QA/QC
                API to populate this view.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-muted-foreground border-b border-border">
                      <th className="py-2 pr-4">Type</th>
                      <th className="py-2 pr-4">Status</th>
                      <th className="py-2 pr-4">Value</th>
                      <th className="py-2 pr-4">Expected</th>
                      <th className="py-2 pr-4">Deviation</th>
                      <th className="py-2">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r) => (
                      <tr key={r.id} className="border-b border-border/50">
                        <td className="py-2 pr-4 text-foreground">{r.type}</td>
                        <td className="py-2 pr-4">
                          <span
                            className={
                              r.status === 'failed' || r.status === 'fail'
                                ? 'text-red-500'
                                : 'text-green-500'
                            }
                          >
                            {r.status}
                          </span>
                        </td>
                        <td className="py-2 pr-4 text-foreground">{r.value}</td>
                        <td className="py-2 pr-4 text-foreground">{r.expectedValue}</td>
                        <td className="py-2 pr-4 text-foreground">{r.deviation}</td>
                        <td className="py-2 text-muted-foreground">{r.timestamp}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
