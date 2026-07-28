import type { Resume } from "./types";

async function request(path: string, init?: RequestInit) {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response;
}

export async function parsePdf(file: File) {
  const body = new FormData();
  body.append("file", file);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 45_000);
  try {
    return (await request("/api/profile/parse-pdf", {
      method: "POST", body, signal: controller.signal,
    })).json();
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(
        "PDF extraction timed out after 45 seconds. The parsing provider may be rate-limited; please retry."
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function startRun(resume: Resume, job_listing: string, maximum_pages: number, include_summary: boolean, human_in_the_loop: boolean) {
  return (await request("/api/runs", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume, job_listing, maximum_pages, include_summary, human_in_the_loop }),
  })).json() as Promise<{ run_id: string }>;
}

export async function provideHumanInput(runId: string, notes: string) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15_000);
  try {
    await request(`/api/runs/${runId}/human-input`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes }), signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(
        "The server did not acknowledge your input within 15 seconds. The rewrite has not been confirmed as resumed."
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function editRun(runId: string, instruction: string) {
  await request(`/api/runs/${runId}/edits`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction }),
  });
}

export async function updateRunResume(runId: string, resume: Resume) {
  return (await request(`/api/runs/${runId}/resume`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume }),
  })).json();
}

export async function downloadPdf(runId: string, resume: Resume) {
  const response = await request(`/api/runs/${runId}/pdf`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmed: true, resume }),
  });
  return response.blob();
}
