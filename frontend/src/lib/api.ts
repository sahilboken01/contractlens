import type { AskResponse, ContractAnalysis, DocumentListResponse, DocumentMetadata, DocumentTextResponse, DocumentUpload, ProcessResult } from "@/lib/types";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_BASE_URL) throw new ApiError("Set NEXT_PUBLIC_API_URL to the ContractLens backend URL.", 0);
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError("Could not connect to ContractLens API. Check that the backend is running.", 0);
  }

  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null && "detail" in body) {
        const detail = (body as { detail: unknown }).detail;
        if (typeof detail === "string") message = detail;
      }
    } catch {
      // Keep the safe status message when an error body is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export const contractApi = {
  listDocuments(): Promise<DocumentListResponse> {
    return request<DocumentListResponse>("/api/documents");
  },

  getDocument(documentId: string): Promise<DocumentMetadata> {
    return request<DocumentMetadata>(`/api/documents/${encodeURIComponent(documentId)}`);
  },

  getDocumentText(documentId: string): Promise<DocumentTextResponse> {
    return request<DocumentTextResponse>(`/api/documents/${encodeURIComponent(documentId)}/text`);
  },
  upload(file: File): Promise<DocumentUpload> {
    const body = new FormData();
    body.append("file", file);
    return request<DocumentUpload>("/api/documents", { method: "POST", body });
  },

  process(documentId: string): Promise<ProcessResult> {
    return request<ProcessResult>(`/api/documents/${encodeURIComponent(documentId)}/process`, {
      method: "POST",
    });
  },

  ask(documentId: string, question: string): Promise<AskResponse> {
    return request<AskResponse>(`/api/documents/${encodeURIComponent(documentId)}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  },

  analyze(documentId: string): Promise<ContractAnalysis> {
    return request<ContractAnalysis>(`/api/documents/${encodeURIComponent(documentId)}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
  },
};
