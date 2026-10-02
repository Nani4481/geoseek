import { TemporalArchiveMode } from '@/components/TemporalArchive';
import { TemporalUploadMode } from '@/components/TemporalUpload';
import { href } from '@/router';
import { useUploads } from '@/lib/uploads';

/** Two modes that are never drawn together: pipeline output on the archive's real dates, and a plain spectral profile of dropped files. */
export function Temporal({ mode }: { mode: string | null }) {
  const upload = mode === 'upload';
  const files = useUploads().filter((u) => u.header).length;
  return (
    <div className="col temporal" data-testid="temporal">
      <div className="tc-tabs" role="tablist" aria-label="Temporal modes">
        <a role="tab" aria-selected={!upload} href={href('temporal')} className={`tc-tab archive ${!upload ? 'on' : ''}`} data-testid="tab-archive">
          <b>Archive</b><span>Change-pipeline output across the archive’s real acquisition dates</span>
        </a>
        <a role="tab" aria-selected={upload} href={href('temporal', 'upload')} className={`tc-tab upload ${upload ? 'on' : ''}`} data-testid="tab-upload">
          <b>Uploads</b><span>Spectral profile of files dropped on the Data screen · not change detection{files ? ` · ${files} loaded` : ''}</span>
        </a>
      </div>
      {upload ? <TemporalUploadMode /> : <TemporalArchiveMode />}
    </div>
  );
}
