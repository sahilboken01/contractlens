import type { Citation } from "@/lib/types";

export function SourceCitation({ source }: { source: Citation }) {
  const page = source.page_number === null ? null : `Page ${source.page_number}`;
  const label = [page, source.source_locator].filter(Boolean).join(" · ");

  return (
    <details className="source-citation">
      <summary aria-label={`View source ${label || source.chunk_id}`}>
        <span className="source-icon">↗</span>
        <span>{label || "Source location unavailable"}</span>
        <span className="source-open">View source</span>
      </summary>
      <div className="source-detail">
        <div><span>Chunk</span><code>{source.chunk_id}</code></div>
        {source.page_number !== null && <div><span>Page</span><strong>{source.page_number}</strong></div>}
        {source.source_locator && <div><span>Section</span><strong>{source.source_locator}</strong></div>}
        {Number.isFinite(source.similarity) && <div><span>Match</span><strong>{Math.round(source.similarity * 100)}%</strong></div>}
        <p>Source text is not included in this API response. Use the locator above to review the contract excerpt.</p>
      </div>
    </details>
  );
}
