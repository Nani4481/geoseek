import type { CSSProperties, ReactNode } from 'react';

interface PanelProps {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  /** take the leftover height of a stacked column (equal-height rows) */
  grow?: boolean;
  /** lay the body out as a flex column (children may flex-grow) */
  stack?: boolean;
  fill?: boolean;
  className?: string;
  style?: CSSProperties;
  bodyStyle?: CSSProperties;
}

/** Every screen section is a Panel: a titled card with optional header actions. */
export function Panel({ title, actions, children, flush, grow, stack, fill, className = '', style, bodyStyle }: PanelProps) {
  return (
    <section className={`panel ${grow ? 'grow' : ''} ${stack ? 'stack' : ''} ${className}`.replace(/\s+/g, ' ').trim()} style={style} aria-label={title}>
      <header>
        <h3>{title}</h3>
        <span className="grow" />
        {actions}
      </header>
      <div className={`body ${flush ? 'flush' : ''} ${fill ? 'fill' : ''}`.trim()} style={bodyStyle}>{children}</div>
    </section>
  );
}
