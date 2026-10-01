import { Suspense, lazy } from 'react';
import { Rail, TopBar } from '@/components/Shell';
import { Loading } from '@/components/Widgets';
import { useRoute } from '@/router';
import { Dashboard } from '@/screens/Dashboard';

// Heavier / secondary screens are code-split; everything still ships in the local build.
const Search = lazy(() => import('@/screens/Search').then((m) => ({ default: m.Search })));
const Changes = lazy(() => import('@/screens/Changes').then((m) => ({ default: m.Changes })));
const Detect = lazy(() => import('@/screens/Detect').then((m) => ({ default: m.Detect })));
const Discovery = lazy(() => import('@/screens/Discovery').then((m) => ({ default: m.Discovery })));
const Fingerprints = lazy(() => import('@/screens/Fingerprints').then((m) => ({ default: m.Fingerprints })));
const Briefing = lazy(() => import('@/screens/Briefing').then((m) => ({ default: m.Briefing })));
const Roadmap = lazy(() => import('@/screens/Roadmap').then((m) => ({ default: m.Roadmap })));

const TITLES: Record<string, string> = {
  dashboard: 'Situation dashboard', search: 'Archive search', changes: 'Change review queue', detect: 'Object detection',
  discovery: 'Discovery & clustering', fingerprints: 'Structural fingerprint gallery', briefing: 'Briefing mode', roadmap: 'Roadmap mockups',
};

export function App() {
  const route = useRoute();
  const name = TITLES[route.name] ? route.name : 'dashboard';

  if (name === 'briefing') {
    return <Suspense fallback={<div style={{ padding: 30 }}><Loading rows={3} /></div>}><Briefing /></Suspense>;
  }
  return (
    <div className="app">
      <Rail active={name} />
      <TopBar title={TITLES[name]} />
      <main className="main" id="main">
        <Suspense fallback={<Loading rows={5} />}>
          {name === 'dashboard' && <Dashboard />}
          {name === 'search' && <Search />}
          {name === 'changes' && <Changes id={route.param} />}
          {name === 'detect' && <Detect />}
          {name === 'discovery' && <Discovery />}
          {name === 'fingerprints' && <Fingerprints seed={route.param} />}
          {name === 'roadmap' && <Roadmap panel={route.param} />}
        </Suspense>
      </main>
    </div>
  );
}
