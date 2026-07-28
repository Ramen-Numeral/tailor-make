export type Constraints = {
  max_items?: number | null; min_items?: number | null;
  max_words?: number | null; min_words?: number | null;
  max_sentences?: number | null;
  max_bullets_per_item?: number | null; min_bullets_per_item?: number | null;
  max_words_per_bullet?: number | null; max_skill_categories?: number | null;
  max_skills_per_category?: number | null; min_skills_per_category?: number | null;
  max_courses?: number | null; max_technologies?: number | null;
  show_gpa?: boolean | null; show_coursework?: boolean | null;
  require_metrics?: boolean | null; required_keywords?: string[];
  forbidden_phrases?: string[]; style?: string | null;
};

export type Item = Record<string, unknown> & { id?: string };
export type Section = { heading: string; items: Item[]; constraints: Constraints };
export type Resume = {
  candidate: Record<string, string | null>;
  summary: Section; skills: Section; work_experience: Section;
  education: Section; projects: Section | null; research: Section | null;
};
export type Trace = {
  event_type: string; section?: string; attempt?: number; title: string;
  summary: string; observations: string[]; action?: string;
  decision?: string; decision_reasons: string[];
  evaluation?: {
    detection: {
      ai_probability: number; threshold: number; feedback: string[];
      components: Array<{
        model_name: string; ai_probability: number; error?: string | null;
        base_value?: number | null; explanation_note?: string | null;
        token_attributions: Array<{ token: string; attribution: number; direction: string }>;
        term_contributions: Array<{ term: string; tfidf_value: number; coefficient: number; contribution: number; direction: string }>;
      }>;
      feature_evidence: Array<{ label: string; description: string; direction: string; observed_value: number; shap_value: number; importance_rank: number }>;
      rubric_axes: Array<{ label: string; score: number; interpretation: string; contribution?: number | null; direction?: string | null }>;
      ensemble_explanation?: {
        method: string; weighted_sum: number; weight_total: number; combined_probability: number;
        minimum_probability: number; maximum_probability: number; spread: number;
        standard_deviation: number; agreement: string;
        components: Array<{ model_name: string; probability: number; weight: number; weighted_value: number; normalized_contribution: number }>;
      } | null;
      scoring_status: string;
    };
    constraints: Array<{ label: string; expected: string; observed: string; passed: boolean; severity: string }>;
    decision: { average_rubric_score?: number; reasons: string[] };
    counterfactual?: {
      before_probability: number; after_probability: number; delta: number;
      components: Array<{ model_name: string; before: number; after: number; delta: number }>;
    } | null;
  };
  score?: number;
  job_requirements?: Array<{
    id: string; text: string; kind: string; importance: "critical" | "important" | "supporting";
    required: boolean; source_text?: string | null;
  }>;
  coverage_plan?: {
    requirement_matches: Array<{
      requirement_id: string; requirement_text: string; requirement_kind: string;
      importance: "critical" | "important" | "supporting";
      support: "supported" | "partial" | "unsupported";
      adjudication_reason?: string | null;
      matches: Array<{ evidence: { section: string; field: string; text: string }; judge_reason?: string | null }>;
    }>;
  };
  match_score?: {
    stage: "initial" | "final"; score: number; evidence_coverage_score: number;
    holistic_fit_score?: number | null; holistic_summary?: string | null;
    supported: number; partial: number; unsupported: number;
    breakdown: Array<{ kind: string; score: number; supported: number; partial: number; unsupported: number }>;
    rubric_observations: string[]; largest_gaps: string[];
  };
};
export type DiffToken = { operation: "equal" | "add" | "remove"; text: string };
export type Diff = {
  section: string; item_id: string; item_label: string; field: string;
  change_type: string; original: string | string[] | null;
  final: string | string[] | null; tokens: DiffToken[]; reason: string;
};
export type Usage = {
  calls: number; prompt_tokens: number; completion_tokens: number;
  total_tokens: number; tokens_are_estimated: boolean;
};
export type RunResult = {
  resume: Resume; diffs: Diff[];
  state: {
    status: string; total_rewrites: number; events: Trace[];
    supplemental_evidence?: string | null;
    bullet_quality?: Array<{
      section: string; item_index: number; bullet_index: number; text: string;
      score: number; passed_dimensions: string[]; improvement_dimensions: string[];
    }>;
    recruiter_evaluation?: {
      axes: Array<{ axis: string; score: number; reason: string }>;
      strengths: string[]; gaps: string[]; recommendations: string[];
      summary: string; ready: boolean;
    } | null;
  };
  page_check: { final_pages: number; maximum_pages: number; passed: boolean };
};
