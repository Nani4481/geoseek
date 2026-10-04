import { Panel } from '@/components/Panel';
import { NAV } from '@/components/Shell';
import { href, type Route } from '@/router';

/** Shown for any hash that is not a screen. It names what was asked for and substitutes nothing: a mistyped link must not
 *  quietly render some other screen and look like success. */
export function NotFound({ route }: { route: Route }) {
  return (
    <div data-testid="route-not-found">
      <Panel title="Page not found">
        <p style={{ marginTop: 0 }}>
          Unrecognised route <code className="mono" data-testid="route-not-found-name">#/{route.raw}</code>.
          No screen is registered under that address, so none is shown in its place.
        </p>
        <p className="dim" style={{ marginBottom: 6 }}>Available screens:</p>
        <ul style={{ columnWidth: 160, margin: 0, paddingLeft: 18 }}>
          {NAV.map((n) => <li key={n.id}><a href={href(n.id)}>{n.label}</a> <span className="faint mono">#/{n.id}</span></li>)}
        </ul>
      </Panel>
    </div>
  );
}
