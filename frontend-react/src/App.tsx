import { Suspense, lazy, useState } from 'react';
import { Intro, shouldShowIntro } from '@/components/Intro';
import { Rail, TopBar } from '@/components/Shell';
import { Loading } from '@/components/Widgets';
import { useRoute } from '@/router';
import { Dashboard } from '@/screens/Dashboard';
import { NotFound } from '@/screens/NotFound';

// Heavier / secondary screens are code-split; everything still ships in the local build.
const Search = lazy(() => import('@/screens/Search').then((m) => ({ default: m.Search })));
const Changes = lazy(() => import('@/screens/Changes').then((m) => ({ default: m.Changes })));
const Pipeline = lazy(() => import('@/screens/Pipeline').then((m) => ({ default: m.Pipeline })));
const Detect = lazy(() => import('@/screens/Detect').then((m) => ({ default: m.Detect })));
const Discovery = lazy(() => import('@/screens/Discovery').then((m) => ({ default: m.Discovery })));
const Fingerprints = lazy(() => import('@/screens/Fingerprints').then((m) => ({ default: m.Fingerprints })));
const Briefing = lazy(() => import('@/screens/Briefing').then((m) => ({ default: m.Briefing })));
const Temporal = lazy(() => import('@/screens/Temporal').then((m) => ({ default: m.Temporal })));
const Data = lazy(() => import('@/screens/Data').then((m) => ({ default: m.Data })));
const Settings = lazy(() => import('@/screens/Settings').then((m) => ({ default: m.Settings })));

const TITLES: Record<string, string> = {
  dashboard: 'Situation dashboard', search: 'Archive search', changes: 'Change review queue', pipeline: 'False-alarm suppression pipeline', detect: 'Object detection',
  temporal: 'Temporal view', discovery: 'Discovery & clustering', fingerprints: 'Structural fingerprint gallery', briefing: 'Briefing mode', data: 'Data management', settings: 'Settings',
};

export function App() {
  const route = useRoute();
  const known = Object.prototype.hasOwnProperty.call(TITLES, route.name);   // never fall back to another screen
  const name = known ? route.name : '';
  const [intro, setIntro] = useState(shouldShowIntro);   // decided once, from where the analyst landed

  if (name === 'briefing') {
    return <Suspense fallback={<div style={{ padding: 30 }}><Loading rows={3} /></div>}><Briefing /></Suspense>;
  }
  return (
    <>
    {intro && <Intro onDone={() => setIntro(false)} />}
    <div className="app">
      <Rail active={name} />
      <TopBar title={known ? TITLES[name] : 'Page not found'} />
      <main className="main" id="main">
        {!known && <NotFound route={route} />}
        <Suspense fallback={<Loading rows={5} />}>
          {name === 'dashboard' && <Dashboard />}
          {name === 'search' && <Search />}
          {name === 'changes' && <Changes id={route.param} />}
          {name === 'pipeline' && <Pipeline />}
          {name === 'temporal' && <Temporal mode={route.param} />}
          {name === 'detect' && <Detect />}
          {name === 'discovery' && <Discovery />}
          {name === 'fingerprints' && <Fingerprints seed={route.param} />}
          {name === 'data' && <Data />}
          {name === 'settings' && <Settings />}
        </Suspense>
      </main>
    </div>
    </>
  );
}
