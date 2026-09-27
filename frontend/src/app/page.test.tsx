import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import HomePage from "@/app/page";
import { contractApi } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {},
  contractApi: {
    listDocuments: vi.fn(),
    getDocument: vi.fn(),
    getDocumentText: vi.fn(),
    upload: vi.fn(),
    process: vi.fn(),
    ask: vi.fn(),
    analyze: vi.fn(),
  },
}));

const uploadResponse = {
  document_id: "doc-123",
  filename: "services-agreement.txt",
  content_type: "text/plain",
  size_bytes: 136,
  character_count: 136,
  text: "Services Agreement between Acme and Northwind. Payment is due within 30 days.",
  segments: [{ locator: "Document", text: "Services Agreement between Acme and Northwind. Payment is due within 30 days." }],
  processing_status: "uploaded" as const,
};

const source = { chunk_id: "chunk-1", source_locator: "Section 4 - Fees", page_number: 2, similarity: 0.88 };

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.mocked(contractApi.listDocuments).mockResolvedValue({ documents: [], total: 0 });
});

async function uploadAndProcess(user: ReturnType<typeof userEvent.setup>) {
  vi.mocked(contractApi.upload).mockResolvedValue(uploadResponse);
  vi.mocked(contractApi.process).mockResolvedValue({ document_id: "doc-123", chunk_count: 3, embedding_model: "bge", embedding_dimensions: 384 });
  const file = new File([uploadResponse.text], "services-agreement.txt", { type: "text/plain" });
  const input = document.querySelector<HTMLInputElement>('input[type="file"]');
  if (!input) throw new Error("Upload input was not rendered");
  await user.upload(input, file);
  await screen.findByText("Prepare this contract");
  await user.click(screen.getByRole("button", { name: /process document/i }));
  await screen.findByText("Document processed");
}

describe("ContractLens workspace", () => {
  it("renders the upload interface and empty state", () => {
    render(<HomePage />);
    expect(screen.getByRole("heading", { name: /clarity in every agreement/i })).toBeInTheDocument();
    expect(screen.getByText(/pdf, docx or txt/i)).toBeInTheDocument();
  });

  it("loads persisted document history and restores its text after selection", async () => {
    const user = userEvent.setup();
    vi.mocked(contractApi.listDocuments).mockResolvedValue({ documents: [{
      document_id: "old-doc", filename: "previous.txt", content_type: "text/plain", size_bytes: 40,
      character_count: 40, processing_status: "processed", chunk_count: 2, embedding_model: "bge",
      embedding_dimensions: 384, created_at: "2025-01-01T00:00:00Z", updated_at: "2025-01-01T00:01:00Z",
    }], total: 1 });
    vi.mocked(contractApi.getDocument).mockResolvedValue({
      document_id: "old-doc", filename: "previous.txt", content_type: "text/plain", size_bytes: 40,
      character_count: 40, processing_status: "processed", chunk_count: 2, embedding_model: "bge",
      embedding_dimensions: 384, created_at: "2025-01-01T00:00:00Z", updated_at: "2025-01-01T00:01:00Z",
    });
    vi.mocked(contractApi.getDocumentText).mockResolvedValue({
      document_id: "old-doc", text: "Stored contract text", segments: [{ locator: "Section 2", text: "Stored contract text" }],
    });
    render(<HomePage />);
    await user.click(await screen.findByRole("button", { name: /previous.txt/i }));
    expect((await screen.findAllByText("Stored contract text")).length).toBeGreaterThan(0);
    expect(screen.getByText("processed")).toBeInTheDocument();
  });

  it("uploads a file and only marks it processed after API confirmation", async () => {
    const user = userEvent.setup();
    vi.mocked(contractApi.upload).mockResolvedValue(uploadResponse);
    vi.mocked(contractApi.process).mockResolvedValue({ document_id: "doc-123", chunk_count: 3, embedding_model: "bge", embedding_dimensions: 384 });
    render(<HomePage />);
    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    if (!input) throw new Error("Upload input was not rendered");
    await user.upload(input, new File([uploadResponse.text], "services-agreement.txt", { type: "text/plain" }));
    expect(await screen.findByText("Prepare this contract")).toBeInTheDocument();
    expect(screen.getByText("uploaded")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /process document/i }));
    expect(await screen.findByText("Document processed")).toBeInTheDocument();
    expect(contractApi.process).toHaveBeenCalledWith("doc-123");
  });

  it("renders Q&A answers with clickable citation metadata and abstention text", async () => {
    const user = userEvent.setup();
    render(<HomePage />);
    await uploadAndProcess(user);
    await user.click(screen.getByRole("tab", { name: /ask contract/i }));
    vi.mocked(contractApi.ask).mockResolvedValue({ document_id: "doc-123", question: "When is payment due?", answer: "Payment is due within 30 days.", sources: [source] });
    await user.type(screen.getByLabelText(/ask a question/i), "When is payment due?");
    await user.click(screen.getByRole("button", { name: /send question/i }));
    expect(await screen.findByText("Payment is due within 30 days.")).toBeInTheDocument();
    expect(screen.getByText(/Page 2 · Section 4 - Fees/)).toBeInTheDocument();
    expect(screen.getByText("chunk-1")).toBeInTheDocument();
    vi.mocked(contractApi.ask).mockResolvedValue({ document_id: "doc-123", question: "What is the renewal period?", answer: "The provided contract does not contain enough information to answer this question.", sources: [] });
    await user.type(screen.getByLabelText(/ask a question/i), "What is the renewal period?");
    await user.click(screen.getByRole("button", { name: /send question/i }));
    expect(await screen.findByText(/could not find enough information/i)).toBeInTheDocument();
  });

  it("shows grounded analysis findings and citation source controls", async () => {
    const user = userEvent.setup();
    render(<HomePage />);
    await uploadAndProcess(user);
    vi.mocked(contractApi.analyze).mockResolvedValue({
      document_id: "doc-123", status: "complete", contract_type: null, parties: [], effective_date: null, expiration_date: null,
      renewal_terms: [], notice_periods: [], payment_terms: [{ value: "Invoices are due within 30 days.", basis: "explicit", sources: [source] }],
      termination_conditions: [], confidentiality_obligations: [], intellectual_property_provisions: [], non_compete_non_solicitation: [],
      governing_law: null, dispute_resolution: [], major_obligations: [], important_deadlines: [], other_important_clauses: [], attention_items: [],
    });
    await user.click(screen.getByRole("tab", { name: /analysis/i }));
    await user.click(screen.getByRole("button", { name: /run analysis/i }));
    expect(await screen.findByText("Invoices are due within 30 days.")).toBeInTheDocument();
    expect(screen.getByText("explicit")).toBeInTheDocument();
    expect(screen.getByText(/Page 2 · Section 4 - Fees/)).toBeInTheDocument();
  });

  it("shows API errors and loading state", async () => {
    const user = userEvent.setup();
    render(<HomePage />);
    vi.mocked(contractApi.upload).mockRejectedValue(new Error("API offline"));
    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    if (!input) throw new Error("Upload input was not rendered");
    await user.upload(input, new File(["contract"], "contract.txt", { type: "text/plain" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong");
  });

  it("shows an upload loading state while the API request is pending", async () => {
    const user = userEvent.setup();
    render(<HomePage />);
    vi.mocked(contractApi.upload).mockImplementation(() => new Promise(() => undefined));
    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    if (!input) throw new Error("Upload input was not rendered");
    await user.upload(input, new File(["contract"], "contract.txt", { type: "text/plain" }));
    expect(await screen.findByText(/uploading your document/i)).toBeInTheDocument();
  });

  it("rejects unsupported file types on the client", async () => {
    const user = userEvent.setup();
    render(<HomePage />);
    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    if (!input) throw new Error("Upload input was not rendered");
    input.removeAttribute("accept");
    await user.upload(input, new File(["nope"], "contract.exe", { type: "application/octet-stream" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Choose a PDF, DOCX, or TXT file");
    expect(contractApi.upload).not.toHaveBeenCalled();
  });
});
