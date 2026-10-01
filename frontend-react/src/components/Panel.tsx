import type { CSSProperties, ReactNode } from 'react';

export type Tier = 'live' | 'surfaced' | 'roadmap';

export const TIER_TEXT: Record<Tier, string> = {
  live: 'LIVE',
  surfaced: 'LIVE UI · EXISTING BACKEND',
  roadmap: 'ROADMAP — NOT IMPLEMENTED',
};

export function TierBadge({ tier, className = '' }: { tier: Tier; className?: string }) {
  return <span className={`tier-badge ${tier} ${className}`} data-tier={tier}><i />{TIER_TEXT[tier]}</span>;
}

interface PanelProps {
  title: string;
  tier: Tier;
  /** roadmap panels: why it is not implemented, shown in the panel chrome (never a tooltip or footnote) */
  note?: string;
  actions?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  className?: string;
  style?: CSSProperties;
  bodyStyle?: CSSProperties;
}

/**
 * Every screen section is a Panel and carries its tier in its own header, so the live / surfaced / roadmap state
 * is visible on the element itself. Roadmap panels additionally get a dashed amber frame, a hatched header, a
 * persistent badge, a "requires …" strip and a diagonal watermark (see components.css).
 */
export function Panel({ title, tier, note, actions, children, flush, className = '', style, bodyStyle }: PanelProps) {
  return (
    <section className={`panel t-${tier} ${className}`} style={style} data-tier={tier} aria-label={`${title} (${TIER_TEXT[tier]})`}>
      <header>
        <h3>{title}</h3>
        <span className="grow" />
        {actions}
        <TierBadge tier={tier} />
      </header>
      {tier === 'roadmap' && note && (
        <div className="roadmap-note" role="note"><b>Roadmap — not implemented</b><span>{note}</span></div>
      )}
      <div className={`body ${flush ? 'flush' : ''}`} style={bodyStyle}>{children}</div>
    </section>
  );
}
