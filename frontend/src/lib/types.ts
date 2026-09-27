export interface TextSegment {
  locator: string;
  text: string;
}

export interface DocumentUpload {
  document_id: string;
  filename: string;
  content_type: string | null;
  size_bytes: number;
  character_count: number;
  text: string;
  segments: TextSegment[];
  processing_status: DocumentStatus;
}

export type DocumentStatus = "uploaded" | "processing" | "processed" | "failed";

export interface DocumentMetadata {
  document_id: string;
  filename: string;
  content_type: string | null;
  size_bytes: number;
  character_count: number;
  processing_status: DocumentStatus;
  chunk_count: number;
  embedding_model: string | null;
  embedding_dimensions: number | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  documents: DocumentMetadata[];
  total: number;
}

export interface DocumentTextResponse {
  document_id: string;
  text: string;
  segments: TextSegment[];
}

export interface ProcessResult {
  document_id: string;
  chunk_count: number;
  embedding_model: string;
  embedding_dimensions: number;
  processing_status?: "processed";
}

export interface Citation {
  chunk_id: string;
  source_locator: string;
  page_number: number | null;
  similarity: number;
}

export interface AskResponse {
  document_id: string;
  question: string;
  answer: string;
  sources: Citation[];
}

export interface GroundedFinding {
  value: string;
  basis: "explicit" | "inference";
  sources: Citation[];
}

export interface ContractAnalysis {
  document_id: string;
  status: "complete" | "insufficient_context";
  contract_type: GroundedFinding | null;
  parties: GroundedFinding[];
  effective_date: GroundedFinding | null;
  expiration_date: GroundedFinding | null;
  renewal_terms: GroundedFinding[];
  notice_periods: GroundedFinding[];
  payment_terms: GroundedFinding[];
  termination_conditions: GroundedFinding[];
  confidentiality_obligations: GroundedFinding[];
  intellectual_property_provisions: GroundedFinding[];
  non_compete_non_solicitation: GroundedFinding[];
  governing_law: GroundedFinding | null;
  dispute_resolution: GroundedFinding[];
  major_obligations: GroundedFinding[];
  important_deadlines: GroundedFinding[];
  other_important_clauses: GroundedFinding[];
  attention_items: GroundedFinding[];
}

export interface RecentDocument {
  document_id: string;
  filename: string;
  size_bytes: number;
  processing_status: DocumentStatus;
  created_at: string;
}

export const NOT_FOUND_ANSWER =
  "The provided contract does not contain enough information to answer this question.";
