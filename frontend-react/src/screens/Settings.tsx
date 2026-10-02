import { api } from '@/api/client';
import { Panel } from '@/components/Panel';
import { ErrorNote, Loading } from '@/components/Widgets';
import { OfflinePill } from '@/components/Shell';
import { DASH, fmtFixed, fmtInt, fmtPct } from '@/fmt';
import { useApi } from '@/hooks/useApi';
import { useOfflineStatus } from '@/hooks/useOfflineStatus';
import { useStore } from '@/state/store';

const SUB = { fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase' } as const;

/** Held-out model accuracy. Kept off the analyst's home screen on purpose: it is what an evaluator needs, not an operator. */
function ModelPerformance() {
  const m = useApi((s) => api.metrics(s), []);
  const cm = m.data?.change_model, det = m.data?.detector, dota = det?.dota_val, xv = det?.xview_test;
  return (
    <Panel title="System / Model performance">
      {m.error ? <ErrorNote error={m.error} onRetry={m.reload} /> : !m.data ? <Loading rows={8} /> : (
        <div className="settings-perf">
          <section aria-label="Change detection model">
            <div className="dim" style={SUB}>Change detection · {cm?.name ?? DASH}</div>
            {cm?.available ? (
              <>
                <div className="perf-big">
                  <div><b className="mono">{fmtPct(cm.f1, 1)}</b><span>F1 at the deployed threshold {fmtFixed(cm.operating_threshold, 2)}</span></div>
                  <div><b className="mono">{fmtPct(cm.precision, 1)}</b><span>precision</span></div>
                  <div><b className="mono">{fmtPct(cm.recall, 1)}</b><span>recall</span></div>
                </div>
                <dl className="kv" style={{ marginTop: 8 }}>
                  <dt>IoU</dt><dd>{fmtPct(cm.iou, 1)}</dd>
                  <dt>F1 at the neutral 0.50 threshold</dt><dd>{fmtPct(cm.f1_at_0_50, 1)}</dd>
                  <dt>Validation F1 (threshold-selection split)</dt><dd>{fmtPct(cm.validation_f1, 1)}</dd>
                  <dt>Evaluated on</dt><dd>{cm.dataset ?? DASH}</dd>
                </dl>
                {cm.caveat && <div className="warnbox" style={{ marginTop: 8 }}>{cm.caveat}</div>}
                <div className="faint" style={{ fontSize: 10.5, marginTop: 6 }}>Source: {cm.source}</div>
              </>
            ) : <div className="empty">Change-model evaluation card not found.</div>}
          </section>

          <section aria-label="Object detection model">
            <div className="dim" style={SUB}>Object detection</div>
            {dota ? (
              <>
                <div className="perf-big">
                  <div><b className="mono">{fmtFixed(dota.small_vehicle_ap50, 3)}</b><span>small-vehicle AP50</span></div>
                  <div><b className="mono">{fmtFixed(dota.ground_vehicles_ap50, 3)}</b><span>ground vehicles AP50</span></div>
                  <div><b className="mono">{fmtFixed(dota.all_classes_ap50, 3)}</b><span>all classes AP50</span></div>
                </div>
                <dl className="kv" style={{ marginTop: 8 }}>
                  <dt>Ground vehicles 95% CI</dt><dd>{dota.ground_vehicles_ap50_ci?.map((v) => v.toFixed(3)).join(' – ') ?? DASH}</dd>
                  <dt>Evaluated on</dt><dd>{dota.dataset} ({fmtInt(dota.n_images)} images)</dd>
                  {xv && <><dt>xView independent test · small-vehicle AP50</dt><dd>{fmtFixed(xv.small_vehicle_ap50, 3)}</dd>
                    <dt>xView independent test · large-vehicle AP50</dt><dd>{fmtFixed(xv.large_vehicle_ap50, 3)}</dd></>}
                </dl>
                {det?.caveat && <div className="warnbox" style={{ marginTop: 8 }}>{det.caveat}</div>}
                <div className="faint" style={{ fontSize: 10.5, marginTop: 6 }}>Source: {dota.source}{xv ? ` · ${xv.source ?? xv.dataset}` : ''}</div>
              </>
            ) : <div className="empty">Detector evaluation results not found.</div>}
          </section>
        </div>
      )}
    </Panel>
  );
}

function SystemStatus() {
  const s = useOfflineStatus();
  const m = useApi((x) => api.metrics(x), []);
  const h = useApi((x) => api.health(x), []);
  return (
    <Panel title="System status">
      <dl className="kv">
        <dt>Network posture</dt><dd><OfflinePill /></dd>
        <dt>External requests observed</dt><dd>{fmtInt(s.externalRequests.length)}</dd>
        <dt>Content-security-policy violations</dt><dd>{fmtInt(s.cspViolations)}</dd>
        <dt>Backend</dt><dd>{h.data ? h.data.status : h.error ? 'unreachable' : DASH}</dd>
        <dt>Searchable vectors</dt><dd>{fmtInt(h.data?.vectors)}</dd>
        <dt>Catalog generated</dt><dd>{m.data?.generated_at ? new Date(m.data.generated_at).toLocaleString('en-GB', { timeZone: 'Asia/Kolkata', hour12: false }) + ' IST' : DASH}</dd>
        <dt>Change pipeline run on</dt><dd>{m.data?.change_pipeline_aoi ?? DASH}</dd>
        <dt>Acquisition dates in the catalog</dt><dd>{m.data ? fmtInt(m.data.observation_dates.length) : DASH}</dd>
      </dl>
    </Panel>
  );
}

function AnalystProfile() {
  const { analyst, setAnalyst } = useStore();
  return (
    <Panel title="Analyst profile">
      <label className="field">Name recorded on review decisions
        <input className="input" value={analyst} maxLength={40} onChange={(e) => setAnalyst(e.target.value)} aria-label="Analyst name" />
      </label>
      <div className="faint" style={{ fontSize: 11, marginTop: 6 }}>
        Stored only in this browser. Every confirm / reject / reopen carries this name in the append-only audit trail.
      </div>
    </Panel>
  );
}

export function Settings() {
  return (
    <div className="settings-grid">
      <ModelPerformance />
      <div className="col">
        <SystemStatus />
        <AnalystProfile />
      </div>
    </div>
  );
}
