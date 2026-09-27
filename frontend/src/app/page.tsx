"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { SourceCitation } from "@/components/source-citation";
import { ApiError, contractApi } from "@/lib/api";
import type { AskResponse, ContractAnalysis, DocumentUpload, GroundedFinding, RecentDocument } from "@/lib/types";

type Tab = "overview" | "ask" | "analysis" | "sources";
type ChatMessage = { role: "user" | "assistant"; text: string; sources?: AskResponse["sources"]; insufficient?: boolean };

const MAX_UPLOAD_BYTES = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB || 20) * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["pdf", "docx", "txt"]);

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return "Something went wrong. Please try again.";
}

function Icon({ name }: { name: "logo" | "file" | "upload" | "spark" | "chat" | "bars" | "book" | "plus" | "arrow" | "check" | "clock" | "send" }) {
  const common = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true as const };
  const paths: Record<typeof name, React.ReactNode> = {
    logo: <><path d="M5 4.5h9l5 5V20H5z"/><path d="M14 4.5v5h5M8 14h8M8 17h5"/><path d="m8 10 1.2 1.2L12 8.5"/></>,
    file: <><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v5h4M9 13h6M9 17h6"/></>,
    upload: <><path d="M12 16V4m-4 4 4-4 4 4"/><path d="M5 14v5h14v-5"/></>,
    spark: <><path d="m12 3 1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5z"/><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7z"/></>,
    chat: <><path d="M20 11.5a7.5 7.5 0 0 1-8 7.5 9 9 0 0 1-4-.9L4 20l1-3.5a7.2 7.2 0 0 1-1-3.5 7.5 7.5 0 0 1 8-7.5 7.5 7.5 0 0 1 8 6z"/><path d="M8 12h.01M12 12h.01M16 12h.01"/></>,
    bars: <><path d="M4 5h16M4 12h16M4 19h16"/><path d="M8 5v.01M8 12v.01M8 19v.01"/></>,
    book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 0 4 22z"/><path d="M4 5.5v14A2.5 2.5 0 0 1 6.5 17H20M8 7h8M8 10h8"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    send: <><path d="m21 3-7.5 18-3.5-7-7-3.5z"/><path d="M21 3 10 14"/></>,
  };
  return <svg {...common}>{paths[name]}</svg>;
}

function Brand() {
  return <div className="brand"><span className="brand-mark"><Icon name="logo" /></span><span>Contract<span>Lens</span></span></div>;
}

function FindingCard({ title, finding }: { title: string; finding: GroundedFinding }) {
  return <article className="finding-card">
    <div className="finding-top"><h4>{title}</h4><span className={`basis ${finding.basis}`}>{finding.basis}</span></div>
    <p>{finding.value}</p>
    <div className="finding-sources">{finding.sources.map((source) => <SourceCitation key={source.chunk_id} source={source} />)}</div>
  </article>;
}

function FindingsGroup({ title, entries }: { title: string; entries: Array<[string, GroundedFinding | null | undefined]> }) {
  const findings = entries.filter((entry): entry is [string, GroundedFinding] => Boolean(entry[1]));
  if (!findings.length) return null;
  return <section className="analysis-group"><div className="section-label"><h3>{title}</h3><span>{findings.length}</span></div><div className="finding-list">{findings.map(([label, item]) => <FindingCard key={`${label}-${item.value}`} title={label} finding={item} />)}</div></section>;
}

function arrayEntries(label: string, items: GroundedFinding[]): Array<[string, GroundedFinding]> {
  return items.map((item, index) => [items.length > 1 ? `${label} ${index + 1}` : label, item]);
}

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [current, setCurrent] = useState<DocumentUpload | null>(null);
  const [history, setHistory] = useState<RecentDocument[]>([]);
  const [processing, setProcessing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<ContractAnalysis | null>(null);
  const [analysisError, setAnalysisError] = useState("");
  const [globalError, setGlobalError] = useState("");
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [selectingDocument, setSelectingDocument] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const chatBottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void contractApi.listDocuments().then((response) => {
      setHistory(response.documents.map((doc) => ({
        document_id: doc.document_id, filename: doc.filename, size_bytes: doc.size_bytes,
        processing_status: doc.processing_status, created_at: doc.created_at,
      })));
    }).catch((error: unknown) => setGlobalError(getErrorMessage(error)))
      .finally(() => setHistoryLoading(false));
  }, []);

  useEffect(() => {
    chatBottom.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages, asking]);

  const processedCurrent = current?.processing_status === "processed";
  const currentExtension = useMemo(() => current?.filename.split(".").pop()?.toUpperCase() || "DOC", [current]);

  async function refreshHistory() {
    const response = await contractApi.listDocuments();
    setHistory(response.documents.map((doc) => ({
      document_id: doc.document_id, filename: doc.filename, size_bytes: doc.size_bytes,
      processing_status: doc.processing_status, created_at: doc.created_at,
    })));
  }

  async function handleUpload(file: File | undefined) {
    if (!file) return;
    setGlobalError("");
    const extension = file.name.split(".").pop()?.toLowerCase() || "";
    if (!ACCEPTED_TYPES.has(extension)) {
      setGlobalError("Choose a PDF, DOCX, or TXT file.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setGlobalError(`This file is larger than the ${Math.floor(MAX_UPLOAD_BYTES / (1024 * 1024))} MB upload limit.`);
      return;
    }
    setUploading(true);
    try {
      const uploaded = await contractApi.upload(file);
      setCurrent(uploaded);
      setActiveTab("overview");
      setAnalysis(null);
      setAnalysisError("");
      setMessages([]);
      await refreshHistory();
    } catch (error) {
      setGlobalError(getErrorMessage(error));
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function handleProcess() {
    if (!current) return;
    setProcessing(true);
    setCurrent((previous) => previous ? { ...previous, processing_status: "processing" } : previous);
    setGlobalError("");
    try {
      const result = await contractApi.process(current.document_id);
      setCurrent((previous) => previous ? { ...previous, processing_status: result.processing_status || "processed" } : previous);
      await refreshHistory();
    } catch (error) {
      setGlobalError(getErrorMessage(error));
      try {
        const latest = await contractApi.getDocument(current.document_id);
        setCurrent((previous) => previous ? { ...previous, processing_status: latest.processing_status } : previous);
        await refreshHistory();
      } catch { /* Keep the original processing error visible. */ }
    } finally {
      setProcessing(false);
    }
  }

  async function selectRecent(item: RecentDocument) {
    setSelectingDocument(true);
    setGlobalError("");
    try {
      const [metadata, content] = await Promise.all([
        contractApi.getDocument(item.document_id), contractApi.getDocumentText(item.document_id),
      ]);
      setCurrent({ ...metadata, text: content.text, segments: content.segments });
    } catch (error) {
      setGlobalError(getErrorMessage(error));
    } finally { setSelectingDocument(false); }
    setAnalysis(null);
    setMessages([]);
    setGlobalError("");
    setAnalysisError("");
    setActiveTab("overview");
  }

  async function runAnalysis() {
    if (!current || !processedCurrent) return;
    setActiveTab("analysis");
    setAnalyzing(true);
    setAnalysisError("");
    try {
      setAnalysis(await contractApi.analyze(current.document_id));
    } catch (error) {
      setAnalysisError(getErrorMessage(error));
    } finally {
      setAnalyzing(false);
    }
  }

  async function submitQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!current || !processedCurrent || !question.trim() || asking) return;
    const asked = question.trim();
    setQuestion("");
    setMessages((previous) => [...previous, { role: "user", text: asked }]);
    setAsking(true);
    try {
      const response = await contractApi.ask(current.document_id, asked);
      const insufficient = response.answer.toLowerCase().includes("does not contain enough information");
      setMessages((previous) => [...previous, {
        role: "assistant",
        text: insufficient ? "ContractLens could not find enough information in this contract to answer that question." : response.answer,
        sources: response.sources,
        insufficient,
      }]);
    } catch (error) {
      setMessages((previous) => [...previous, { role: "assistant", text: getErrorMessage(error), insufficient: true }]);
    } finally {
      setAsking(false);
    }
  }

  const analysisOverview: Array<[string, GroundedFinding | null]> = analysis ? [
    ["Contract type", analysis.contract_type], ["Effective date", analysis.effective_date], ["Expiration date", analysis.expiration_date], ["Governing law", analysis.governing_law],
  ] : [];
  const analysisCommercial = analysis ? [
    ...arrayEntries("Payment terms", analysis.payment_terms), ...arrayEntries("Important deadline", analysis.important_deadlines),
    ...arrayEntries("Renewal terms", analysis.renewal_terms), ...arrayEntries("Notice period", analysis.notice_periods),
  ] : [];
  const analysisClauses = analysis ? [
    ...arrayEntries("Termination", analysis.termination_conditions), ...arrayEntries("Confidentiality", analysis.confidentiality_obligations),
    ...arrayEntries("Intellectual property", analysis.intellectual_property_provisions), ...arrayEntries("Non-compete / non-solicitation", analysis.non_compete_non_solicitation),
    ...arrayEntries("Dispute resolution", analysis.dispute_resolution), ...arrayEntries("Other clause", analysis.other_important_clauses),
  ] : [];

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-top"><Brand /><span className="workspace-tag">WORKSPACE</span></div>
        <nav className="primary-nav" aria-label="Main navigation">
          <button className="nav-item selected" onClick={() => setActiveTab("overview")}><Icon name="file" />Documents</button>
          <button className="nav-item" disabled={!current || !processedCurrent} onClick={() => setActiveTab("analysis")}><Icon name="spark" />Analysis</button>
          <button className="nav-item" disabled={!current || !processedCurrent} onClick={() => setActiveTab("ask")}><Icon name="chat" />Q&amp;A</button>
        </nav>
        <div className="sidebar-section"><div className="sidebar-heading"><span>RECENT DOCUMENTS</span><button aria-label="Upload a contract" onClick={() => fileInput.current?.click()}><Icon name="plus" /></button></div>
          {history.length ? <div className="recent-list">{history.map((item) => <button key={item.document_id} className={`recent-document ${current?.document_id === item.document_id ? "active" : ""}`} onClick={() => void selectRecent(item)} disabled={selectingDocument}><span className="recent-file"><Icon name="file" /></span><span className="recent-copy"><b>{item.filename}</b><small>{item.processing_status === "processed" ? "Ready for review" : item.processing_status === "failed" ? "Processing failed" : item.processing_status === "processing" ? "Processing" : "Needs processing"}</small></span></button>)}</div> : <p className="empty-history">{historyLoading ? "Loading saved documents…" : "Your uploaded contracts will appear here."}</p>}
        </div>
        <div className="sidebar-bottom"><div className="secure-note"><span className="secure-dot"/><span>Document workspace<br/><small>History saved to the configured API</small></span></div><div className="profile"><span className="avatar">CL</span><span><b>ContractLens</b><small>Document intelligence</small></span><span className="profile-menu">···</span></div></div>
      </aside>

      <section className="main-column">
        <header className="topbar"><div className="breadcrumb"><span>Workspace</span><span>/</span><strong>{current?.filename || "Documents"}</strong></div><div className="topbar-right"><span className="api-status"><i/>Contract workspace</span><button className="button button-primary button-small" onClick={() => fileInput.current?.click()}><Icon name="plus" />New contract</button></div></header>
        <input ref={fileInput} aria-label="Upload a contract file" className="visually-hidden" type="file" accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain" onChange={(event) => void handleUpload(event.target.files?.[0])} />
        <div className="page-content">
          {!current ? <>
            <div className="welcome-header"><div><div className="eyebrow"><span className="eyebrow-line"/>CONTRACT INTELLIGENCE</div><h1>Clarity in every<br/><em>agreement.</em></h1><p>Understand the details that matter. Upload a contract to get a grounded summary, explore key terms, and ask questions with sources.</p></div><div className="welcome-art" aria-hidden="true"><div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/><div className="art-document"><span className="art-logo"><Icon name="logo"/></span><i/><i/><i/><i/><div className="art-highlight"/><span className="art-check"><Icon name="check"/></span></div><div className="art-sparkle">✦</div></div></div>
            {globalError && <div className="alert error-alert" role="alert">{globalError}<button onClick={() => setGlobalError("")}>Dismiss</button></div>}
            <div className="welcome-grid"><section className="upload-card"><div className="card-kicker"><span className="kicker-icon"><Icon name="upload"/></span><span>START WITH A CONTRACT</span></div><h2>Bring your contract into focus</h2><p>Upload a document and ContractLens will extract its text, then prepare it for cited analysis and Q&amp;A.</p><button className="drop-zone" onClick={() => fileInput.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); void handleUpload(event.dataTransfer.files[0]); }} disabled={uploading}><span className="upload-circle"><Icon name="upload"/></span><strong>{uploading ? "Uploading your document…" : "Drop a contract here, or browse"}</strong><small>PDF, DOCX or TXT · Up to {Math.floor(MAX_UPLOAD_BYTES / 1024 / 1024)} MB</small><span className="button button-outline">{uploading ? "Please wait" : "Choose a file"}</span></button></section>
              <div className="right-stack"><section className="how-card"><div className="card-kicker"><span className="kicker-icon soft"><Icon name="spark"/></span><span>HOW IT WORKS</span></div><h3>From document to understanding</h3><div className="workflow"><div><span>01</span><i><Icon name="upload"/></i><b>Upload</b><small>Your document stays yours</small></div><div><span>02</span><i><Icon name="bars"/></i><b>Process</b><small>Structure and index clauses</small></div><div><span>03</span><i><Icon name="chat"/></i><b>Explore</b><small>Ask with cited answers</small></div></div></section><section className="trust-card"><span className="trust-icon">✳</span><div><strong>Grounded in your document</strong><p>Every answer points back to contract sources. No outside facts are used.</p></div></section></div></div>
          </> : <>
            <div className="document-heading"><div className="document-title-wrap"><span className={`file-badge ${currentExtension.toLowerCase()}`}>{currentExtension}</span><div><div className="eyebrow">DOCUMENT WORKSPACE</div><h1>{current.filename}</h1><p>{formatBytes(current.size_bytes)} <span>·</span> ID {current.document_id}</p></div></div><div className={`status-pill ${processedCurrent ? "ready" : "pending"}`}><i/>{current.processing_status}</div></div>
            {globalError && <div className="alert error-alert" role="alert">{globalError}<button onClick={() => setGlobalError("")}>Dismiss</button></div>}
            <div className="tabs" role="tablist" aria-label="Document views">
              <button role="tab" aria-selected={activeTab === "overview"} className={activeTab === "overview" ? "active" : ""} onClick={() => setActiveTab("overview")}>Overview</button>
              <button role="tab" aria-selected={activeTab === "ask"} className={activeTab === "ask" ? "active" : ""} disabled={!processedCurrent} onClick={() => setActiveTab("ask")}><Icon name="chat"/>Ask contract</button>
              <button role="tab" aria-selected={activeTab === "analysis"} className={activeTab === "analysis" ? "active" : ""} disabled={!processedCurrent} onClick={() => setActiveTab("analysis")}><Icon name="spark"/>Analysis</button>
              <button role="tab" aria-selected={activeTab === "sources"} className={activeTab === "sources" ? "active" : ""} onClick={() => setActiveTab("sources")}><Icon name="book"/>Sources</button>
            </div>

            {activeTab === "overview" && <div className="overview-layout"><div className="overview-main"><section className="summary-card"><div className="card-heading"><div><span className="card-kicker-text">CONTRACT OVERVIEW</span><h2>Your document, ready for review</h2></div><span className="overview-icon"><Icon name="file"/></span></div><p className="summary-body">{current.text ? `${current.text.slice(0, 470)}${current.text.length > 470 ? "…" : ""}` : "Extracted text is currently unavailable."}</p><div className="summary-footer"><span><Icon name="clock"/>{new Date(history.find((item) => item.document_id === current.document_id)?.created_at || Date.now()).toLocaleDateString()}</span><span>{current.character_count ? `${current.character_count.toLocaleString()} characters extracted` : "Text preview unavailable"}</span></div></section>
                <section className="process-card"><div className="process-icon"><Icon name={processedCurrent ? "check" : "bars"}/></div><div className="process-copy"><h3>{processedCurrent ? "Document processed" : current.processing_status === "failed" ? "Processing failed" : "Prepare this contract"}</h3><p>{processedCurrent ? "Text is chunked and indexed. Analysis and contract Q&A are ready." : "Processing extracts source-aware sections and builds the document index. This can take a moment."}</p>{processedCurrent && <span className="processed-label"><i/>Ready for analysis and Q&amp;A</span>}</div>{!processedCurrent && <button className="button button-primary" onClick={() => void handleProcess()} disabled={processing}>{processing ? <><span className="spinner"/>Processing…</> : <>Process document <Icon name="arrow"/></>}</button>}</section>
                {current.text && <section className="extracted-card"><div className="section-label"><h3>Extracted text</h3><span>{current.segments.length} source segment{current.segments.length === 1 ? "" : "s"}</span></div><div className="document-text">{current.segments.map((segment, index) => <div key={`${segment.locator}-${index}`}><span>{segment.locator}</span><p>{segment.text}</p></div>)}</div></section>}</div>
                <aside className="overview-aside"><section className="details-card"><div className="section-label"><h3>Document details</h3></div><dl><div><dt>File name</dt><dd>{current.filename}</dd></div><div><dt>File type</dt><dd>{currentExtension}</dd></div><div><dt>Size</dt><dd>{formatBytes(current.size_bytes)}</dd></div><div><dt>Document ID</dt><dd className="id-value" title={current.document_id}>{current.document_id}</dd></div><div><dt>Status</dt><dd><span className={`inline-status ${processedCurrent ? "ready" : "pending"}`}>{processedCurrent ? "Processed" : "Awaiting process"}</span></dd></div></dl></section><section className="next-card"><div className="card-kicker-text">NEXT STEPS</div><button disabled={!processedCurrent} onClick={() => setActiveTab("analysis")}><span className="next-icon"><Icon name="spark"/></span><span><b>Review the analysis</b><small>Find key facts and clauses</small></span><Icon name="arrow"/></button><button disabled={!processedCurrent} onClick={() => setActiveTab("ask")}><span className="next-icon"><Icon name="chat"/></span><span><b>Ask your contract</b><small>Get answers with citations</small></span><Icon name="arrow"/></button></section></aside></div>}

            {activeTab === "ask" && <section className="chat-panel"><div className="chat-header"><div><span className="card-kicker-text">DOCUMENT Q&amp;A</span><h2>Ask this contract</h2><p>Answers are grounded in retrieved sections from {current.filename}.</p></div><span className="chat-header-icon"><Icon name="chat"/></span></div><div className="chat-messages" aria-live="polite">{messages.length === 0 && <div className="chat-empty"><span className="chat-empty-icon"><Icon name="spark"/></span><h3>What would you like to understand?</h3><p>Ask about payment terms, renewal, obligations, or any specific clause.</p><div className="suggestions"><button onClick={() => setQuestion("Who are the parties to this agreement?")}>Who are the parties?</button><button onClick={() => setQuestion("What are the termination conditions?")}>How can it be terminated?</button><button onClick={() => setQuestion("What are the payment terms?")}>What are the payment terms?</button></div></div>}{messages.map((message, index) => <div key={`${message.role}-${index}`} className={`message-row ${message.role}`}><span className={`message-avatar ${message.role}`}>{message.role === "assistant" ? "CL" : "You"}</span><div className={`message-bubble ${message.insufficient ? "insufficient" : ""}`}><span className="message-label">{message.role === "assistant" ? "CONTRACTLENS" : "YOU"}</span><p>{message.text}</p>{message.sources && message.sources.length > 0 && <div className="answer-sources"><strong>Supporting sources</strong>{message.sources.map((source) => <SourceCitation key={source.chunk_id} source={source}/>)}</div>}</div></div>)}{asking && <div className="message-row assistant"><span className="message-avatar assistant">CL</span><div className="message-bubble loading-bubble"><span className="message-label">CONTRACTLENS</span><span className="typing-indicator"><i/><i/><i/></span><small>Searching this contract…</small></div></div>}<div ref={chatBottom}/></div><form className="chat-compose" onSubmit={(event) => void submitQuestion(event)}><label className="visually-hidden" htmlFor="contract-question">Ask a question about this contract</label><textarea id="contract-question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about this contract…" rows={2} maxLength={4000} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }}/><div className="compose-footer"><span>Answers are generated from this document only</span><button className="button button-primary send-button" type="submit" disabled={!question.trim() || asking}>{asking ? <span className="spinner"/> : <><span>Send question</span><Icon name="send"/></>}</button></div></form></section>}

            {activeTab === "analysis" && <div className="analysis-view"><div className="analysis-heading"><div><span className="card-kicker-text">STRUCTURED REVIEW</span><h2>Contract analysis</h2><p>Key information extracted from the processed document, with source references.</p></div>{analysis && <button className="button button-outline" onClick={() => void runAnalysis()} disabled={analyzing}>{analyzing ? <><span className="spinner dark"/>Analyzing…</> : <><Icon name="spark"/>Refresh analysis</>}</button>}</div>{analysisError && <div className="alert error-alert" role="alert">{analysisError}<button onClick={() => setAnalysisError("")}>Dismiss</button></div>}{analyzing && <div className="analysis-loading"><span className="spinner dark"/><div><b>Reviewing the contract</b><p>Retrieving relevant clauses and mapping source citations.</p></div></div>}{!analyzing && !analysis && !analysisError && <div className="analysis-empty"><span className="empty-icon"><Icon name="spark"/></span><h3>Build a structured view of this contract</h3><p>Run the analysis to extract key facts, commercial terms, obligations, and notable clauses.</p><button className="button button-primary" onClick={() => void runAnalysis()}><Icon name="spark"/>Run analysis</button></div>}{analysis && <>{analysis.status === "insufficient_context" && <div className="alert info-alert">The retrieved sections did not contain enough information for a complete analysis. Empty fields are not proof that a clause is absent.</div>}<div className="analysis-grid"><div className="analysis-main"><FindingsGroup title="Contract overview" entries={[...analysisOverview, ...arrayEntries("Party", analysis.parties)]}/><FindingsGroup title="Commercial terms" entries={analysisCommercial}/><FindingsGroup title="Important clauses" entries={analysisClauses}/><FindingsGroup title="Obligations" entries={arrayEntries("Obligation", analysis.major_obligations)}/><FindingsGroup title="Attention items" entries={arrayEntries("Notable provision", analysis.attention_items)}/>{!analysisOverview.some((item) => item[1]) && !analysisCommercial.length && !analysisClauses.length && !analysis.major_obligations.length && !analysis.attention_items.length && <div className="analysis-empty compact"><h3>No supported findings to show</h3><p>ContractLens did not receive citable findings from the retrieved sections.</p></div>}</div><aside className="analysis-note"><span className="note-mark">i</span><b>Source-led review</b><p>Each finding includes its source chunk and locator. Review the cited language in the original contract before relying on a summary.</p><small>Informational summary · Not legal advice</small></aside></div></>}</div>}

            {activeTab === "sources" && <section className="sources-view"><div className="analysis-heading"><div><span className="card-kicker-text">DOCUMENT CONTENT</span><h2>Sources &amp; extracted text</h2><p>Review text stored by the backend and source locators used in answers.</p></div></div>{current.segments.length ? <div className="source-segment-list">{current.segments.map((segment, index) => <article className="source-segment" key={`${segment.locator}-${index}`}><div className="source-segment-title"><span className="segment-number">{String(index + 1).padStart(2, "0")}</span><span>{segment.locator}</span></div><p>{segment.text}</p></article>)}</div> : <div className="analysis-empty compact"><span className="empty-icon"><Icon name="book"/></span><h3>Extracted text is not available</h3><p>The backend returned no extracted source segments for this document.</p></div>}</section>}
          </>}
          <footer className="page-footer"><span>ContractLens</span><span>Document intelligence, grounded in your sources.</span><span>Informational use · Not legal advice</span></footer>
        </div>
      </section>
    </main>
  );
}
