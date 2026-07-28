import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight, BriefcaseBusiness, Check, ChevronRight, CircleDot, Download,
  FileText, Gauge, GraduationCap, Layers3, LoaderCircle, Menu, Plus, Settings2,
  Sparkles, Upload, UserRound, WandSparkles, X,
} from "lucide-react";
import { downloadPdf, editRun, parsePdf, provideHumanInput, startRun, updateRunResume } from "./api";
import { defaultResume } from "./defaultResume";
import type { Constraints, Diff, Item, Resume, RunResult, Trace, Usage } from "./types";

type Tab = "profile" | "constraints" | "resume";
const sections = ["summary", "skills", "work_experience", "education", "projects", "research"] as const;
const clone = <T,>(value: T): T => structuredClone(value);
const valueText = (v: unknown) => Array.isArray(v) ? v.join("\n") : String(v ?? "");

function TailorMakeLogo() {
  return <div className="tailormake-logo">
    <svg viewBox="0 0 1000 1000" role="img" aria-label="TailorMake">
      <rect width="1000" height="1000" fill="#FFFFFF" />
      <path className="logo-t" d="M 135 80 L 585 80 L 585 135 L 215 135 L 460 435 L 405 435 L 405 915 L 310 770 L 310 365 L 135 365 Z" />
      <path className="logo-page" d="M 585 80 L 670 80 L 815 225 L 815 625 L 460 970 L 405 915 L 405 435 L 460 435 L 460 890 L 760 600 L 760 225 L 670 225 L 670 135 L 585 135 Z" />
    </svg>
    <span><b>Tailor</b>Make</span>
  </div>;
}

function SplashPage({ enter }: { enter: () => void }) {
  return <main className="landing">
    <header className="landing-header"><TailorMakeLogo /><span>Evidence-led resume tailoring</span></header>
    <div className="splash-hero">
      <div className="splash-copy">
        <span className="splash-kicker">TailorMake · The candidate profile</span>
        <h1>“Your experience<br />already has<br />the story.”</h1>
        <p>Start with Jack Doe, edit the example, or import your own PDF. Every claim stays tied to candidate evidence.</p>
        <button className="landing-enter" onClick={enter}>Enter workspace <ArrowRight /></button>
        <div className="splash-key"><span>Profile</span><i /><span>Evidence</span><i /><span>Rewrite</span></div>
      </div>
      <div className="splash-art" aria-hidden="true">
        <span className="float-book"><i /></span>
        <span className="float-glasses"><i /><b /></span>
        <div className="typewriter">
          <div className="type-paper"><i /><i /><i /></div>
          <div className="type-top" />
          <div className="type-body"><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
          <div className="type-space" />
        </div>
        <span className="float-note"><i /><i /><i /></span>
        <span className="art-plus">+</span>
      </div>
    </div>
  </main>;
}

function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [tab, setTab] = useState<Tab>("profile");
  const [resume, setResume] = useState<Resume>(clone(defaultResume));
  const [sourceResume, setSourceResume] = useState<Resume | null>(null);
  const [job, setJob] = useState("");
  const [traces, setTraces] = useState<Trace[]>([]);
  const [result, setResult] = useState<RunResult | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [edit, setEdit] = useState("");
  const [pages, setPages] = useState(1);
  const [includeSummary, setIncludeSummary] = useState(true);
  const [verboseOutput, setVerboseOutput] = useState(false);
  const [humanInTheLoop, setHumanInTheLoop] = useState(false);
  const [humanNotes, setHumanNotes] = useState("");
  const [humanInputPhase, setHumanInputPhase] = useState<"idle" | "submitting" | "resuming">("idle");
  const [sidebar, setSidebar] = useState(true);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const streamRef = useRef<EventSource | null>(null);
  const saveTimerRef = useRef<number | null>(null);
  const editVersionRef = useRef(0);

  const canRun = Boolean(resume.candidate.name?.trim() && job.trim() && !running);
  const lastHumanEvent = traces.filter(
    trace => trace.event_type === "human_input_requested"
      || trace.event_type === "human_input_received"
  ).at(-1);
  const awaitingHumanInput = lastHumanEvent?.event_type === "human_input_requested" && running;

  function connect(id: string) {
    streamRef.current?.close();
    const stream = new EventSource(`/api/runs/${id}/events`);
    streamRef.current = stream;
    stream.addEventListener("trace", (event) => {
      const trace = JSON.parse((event as MessageEvent).data) as Trace;
      setTraces((old) => [...old, trace]);
      if (trace.event_type === "human_input_requested") setHumanInputPhase("idle");
      if (trace.event_type === "human_input_received") setHumanInputPhase("resuming");
    });
    stream.addEventListener("completed", (event) => {
      const payload = JSON.parse((event as MessageEvent).data);
      setResult(payload.result);
      setUsage(payload.usage);
      setSaveStatus("idle");
      setRunning(false);
      setHumanInputPhase("idle");
      setTab("resume");
      stream.close();
    });
    stream.addEventListener("failed", (event) => {
      setError(JSON.parse((event as MessageEvent).data).message);
      setRunning(false);
      setHumanInputPhase("idle");
      stream.close();
    });
  }

  async function begin() {
    setError(""); setTraces([]); setResult(null); setUsage(null);
    setHumanInputPhase("idle");
    setSourceResume(clone(resume)); setRunning(true); setTab("resume");
    try {
      const { run_id } = await startRun(resume, job, pages, includeSummary, humanInTheLoop);
      setRunId(run_id); connect(run_id);
    } catch (e) { setError((e as Error).message); setRunning(false); }
  }

  async function sendHumanInput(notes: string) {
    if (!runId || humanInputPhase !== "idle") return;
    setHumanInputPhase("submitting"); setError("");
    try {
      await provideHumanInput(runId, notes);
      setHumanNotes("");
      setHumanInputPhase("resuming");
    } catch (e) {
      setError((e as Error).message);
      setHumanInputPhase("idle");
    }
  }

  async function applyEdit() {
    if (!runId || !edit.trim()) return;
    setError(""); setRunning(true); setTraces([]);
    try { await editRun(runId, edit); setEdit(""); connect(runId); }
    catch (e) { setError((e as Error).message); setRunning(false); }
  }

  async function exportPdf() {
    if (!runId || !window.confirm("Save this reviewed version as a PDF?")) return;
    try {
      if (!result) return;
      const blob = await downloadPdf(runId, result.resume);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url; link.download = `${result.resume.candidate.name || "resume"}-resume.pdf`;
      link.click(); URL.revokeObjectURL(url);
    } catch (e) { setError((e as Error).message); }
  }

  function updateTailored(next: Resume) {
    if (!runId || !result) return;
    setResult({ ...result, resume: next });
    setSaveStatus("saving");
    const version = ++editVersionRef.current;
    if (saveTimerRef.current != null) window.clearTimeout(saveTimerRef.current);
    saveTimerRef.current = window.setTimeout(async () => {
      try {
        const payload = await updateRunResume(runId, next);
        if (version === editVersionRef.current) {
          setResult(payload.result);
          setSaveStatus("saved");
        }
      } catch (e) {
        if (version === editVersionRef.current) {
          setSaveStatus("error");
          setError((e as Error).message);
        }
      }
    }, 600);
  }

  if (showSplash) {
    return <SplashPage enter={() => setShowSplash(false)} />;
  }

  return (<>
    <div className={`shell ${sidebar ? "" : "sidebar-closed"} ${awaitingHumanInput ? "input-frozen" : ""}`}>
      <aside>
        <div className="brand"><TailorMakeLogo /></div>
        <div className="aside-copy">
          <span className="eyebrow">Tailoring brief</span>
          <h2>What role are we aiming for?</h2>
          <p>Paste the complete listing. The agent will map only supported candidate evidence.</p>
        </div>
        <textarea className="job-input" value={job} onChange={(e) => setJob(e.target.value)}
          placeholder="Paste the job listing here…" aria-label="Job listing" />
        <div className="page-control">
          <span>Maximum length</span>
          <div>{[1, 2, 3].map(n => <button className={pages === n ? "active" : ""} onClick={() => setPages(n)} key={n}>{n} page{n > 1 ? "s" : ""}</button>)}</div>
        </div>
        <label className="summary-toggle human-toggle"><span><UserRound /> Ask for input</span><input type="checkbox" checked={humanInTheLoop} disabled={running} onChange={e => setHumanInTheLoop(e.target.checked)} /><i /></label>
        <label className="summary-toggle"><span><FileText /> Include summary</span><input type="checkbox" checked={includeSummary} onChange={e => setIncludeSummary(e.target.checked)} /><i /></label>
        <label className="summary-toggle"><span><Layers3 /> Verbose output</span><input type="checkbox" checked={verboseOutput} onChange={e => setVerboseOutput(e.target.checked)} /><i /></label>
        <button className="primary run" disabled={!canRun || awaitingHumanInput} onClick={begin}>
          {running ? <LoaderCircle className="spin" /> : <WandSparkles />}
          {running ? "Tailoring…" : "Tailor this resume"} <ArrowRight />
        </button>
        <div className="privacy"><Check /> Facts stay anchored to your source profile.</div>
      </aside>

      <main>
        <header>
          <button className="icon mobile-menu" onClick={() => setSidebar(!sidebar)}><Menu /></button>
          <nav>
            <TabButton active={tab === "profile"} onClick={() => setTab("profile")} icon={<UserRound />}>Candidate profile</TabButton>
            <TabButton active={tab === "constraints"} onClick={() => setTab("constraints")} icon={<Settings2 />}>Constraints</TabButton>
            <TabButton active={tab === "resume"} onClick={() => setTab("resume")} icon={<FileText />}>Resume</TabButton>
          </nav>
          <div className={`status ${running ? "working" : result ? "done" : ""}`}>
            <CircleDot /> {running ? "Agent working" : result ? "Review ready" : "Draft"}
          </div>
        </header>
        {error && <div className="error"><span>{error}</span><button onClick={() => setError("")}><X /></button></div>}
        <section className="workspace">
          {tab === "profile" && <ProfileEditor resume={resume} setResume={setResume} />}
          {tab === "constraints" && <ConstraintEditor resume={resume} setResume={setResume} />}
          {tab === "resume" && <ResumeWorkspace running={running} traces={traces} result={result}
            source={sourceResume} usage={usage} edit={edit} setEdit={setEdit}
            applyEdit={applyEdit} exportPdf={exportPdf} updateTailored={updateTailored}
            saveStatus={saveStatus} verboseOutput={verboseOutput} />}
        </section>
      </main>
    </div>
    {awaitingHumanInput && <div className="input-modal-backdrop">
      <section className="input-modal" role="dialog" aria-modal="true" aria-labelledby="input-modal-title">
        {humanInputPhase === "resuming" ? <>
          <div className="input-modal-state active"><span /><b>Input accepted</b></div>
          <span className="eyebrow">Rewrite resumed</span>
          <h2 id="input-modal-title">Adding your evidence to the plan.</h2>
          <p>The workflow is running again. This window will close automatically when the run completes.</p>
          <div className="input-resuming"><LoaderCircle className="spin" /><span>Validating claims and rewriting supported sections…</span></div>
        </> : <>
          <div className="input-modal-state"><span /><b>{humanInputPhase === "submitting" ? "Sending input" : "Workflow paused"}</b></div>
          <span className="eyebrow">Candidate evidence checkpoint</span>
          <h2 id="input-modal-title">Do you have experience with these gaps?</h2>
          <p>The rewrite is frozen until you respond. Add only factual experience that was omitted from your profile.</p>
          {error && <div className="inline-error input-modal-error">{error}</div>}
          <div className="input-gap-list">{lastHumanEvent?.observations.map((item, index) => <div key={index}><span>{String(index + 1).padStart(2, "0")}</span><b>{item}</b></div>)}</div>
          <label><span>Supplemental evidence</span><textarea autoFocus disabled={humanInputPhase === "submitting"} value={humanNotes} onChange={e => setHumanNotes(e.target.value)} placeholder="Where and how did you use these skills? Include employers, projects, or outcomes when relevant." /></label>
          <div className="input-modal-actions">
            <button className="skip-input" disabled={humanInputPhase === "submitting"} onClick={() => sendHumanInput("")}>Continue without notes</button>
            <button className="primary" disabled={humanInputPhase === "submitting" || !humanNotes.trim()} onClick={() => sendHumanInput(humanNotes)}>{humanInputPhase === "submitting" ? <LoaderCircle className="spin" /> : <Check />} {humanInputPhase === "submitting" ? "Adding…" : "Add to rewrite plan"}</button>
          </div>
        </>}
      </section>
    </div>}
  </>
  );
}

function TabButton({ active, onClick, icon, children }: any) {
  return <button className={active ? "active" : ""} onClick={onClick}>{icon}{children}</button>;
}

function ProfileEditor({ resume, setResume }: { resume: Resume; setResume: (r: Resume) => void }) {
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const showingExample = JSON.stringify(resume) === JSON.stringify(defaultResume);
  const updateCandidate = (key: string, value: string) => {
    const next = clone(resume); next.candidate[key] = value; setResume(next);
  };
  async function upload(file?: File) {
    if (!file) return;
    setUploading(true); setUploadError("");
    try { const parsed = await parsePdf(file); setResume(parsed.resume); }
    catch (e) { setUploadError((e as Error).message); }
    finally { setUploading(false); }
  }
  return <div className={`content ${showingExample ? "example-profile" : ""}`}>
    <div className="title-row"><div><span className="eyebrow">Source of truth</span><h1>Candidate profile</h1>
      <p>Everything the agent may use begins here. Parsed content stays fully editable.</p></div>
      <button className="secondary" onClick={() => fileRef.current?.click()} disabled={uploading}>
        {uploading ? <LoaderCircle className="spin" /> : <Upload />} {uploading ? "Extracting…" : "Import PDF"}
      </button><input ref={fileRef} hidden type="file" accept=".pdf,application/pdf" onChange={e => upload(e.target.files?.[0])} />
    </div>
    {uploadError && <div className="inline-error">{uploadError}</div>}
    {showingExample && <div className="example-note"><Sparkles /> Example profile loaded. Every value is real editable source data—change anything or import your PDF.</div>}
    <div className="card identity"><div className="card-title"><UserRound /><div><h3>Identity & links</h3><p>Contact details shown in the resume header.</p></div></div>
      <div className="form-grid">
        {Object.entries(resume.candidate).map(([key, val]) =>
          <label key={key}><span>{key.replaceAll("_", " ")}</span><input value={val ?? ""} onChange={e => updateCandidate(key, e.target.value)} placeholder={key === "name" ? "Full name" : ""} /></label>)}
      </div>
    </div>
    {sections.map(name => <SectionEditor key={name} name={name} resume={resume} setResume={setResume} />)}
  </div>;
}

const itemTemplates: Record<string, Item> = {
  summary: { content: "" }, skills: { name: "", skills: [] },
  work_experience: { title: "", company: "", start_date: "", end_date: "", location: "", bullets: [] },
  education: { degree: "", institution: "", graduation_date: "", location: "", gpa: "", coursework: [], honors: [] },
  projects: { name: "", description: "", technologies: [], bullets: [], url: "" },
  research: { name: "", description: "", technologies: [], bullets: [], url: "" },
};

function SectionEditor({ name, resume, setResume }: { name: typeof sections[number]; resume: Resume; setResume: (r: Resume) => void }) {
  const section = resume[name];
  const actual = section ?? { heading: name.replaceAll("_", " "), items: [], constraints: {} };
  const icon = name === "education" ? <GraduationCap /> : name === "skills" ? <Layers3 /> : <BriefcaseBusiness />;
  function mutate(fn: (items: Item[]) => void) {
    const next = clone(resume); const target = next[name] ?? clone(actual); fn(target.items); (next as any)[name] = target; setResume(next);
  }
  return <div className="card section-card"><div className="card-title">{icon}<div><h3>{actual.heading}</h3><p>{actual.items.length} source entr{actual.items.length === 1 ? "y" : "ies"}</p></div>
    <button className="text-button" onClick={() => mutate(items => items.push(clone(itemTemplates[name])))}><Plus /> Add entry</button></div>
    {actual.items.length === 0 && <div className="empty-row">No entries yet. Add one or import a PDF.</div>}
    {actual.items.map((item, index) => <div className="item-editor" key={item.id ?? index}>
      <div className="item-number">{String(index + 1).padStart(2, "0")}</div>
      <div className="item-fields">{Object.entries(item).filter(([key]) => key !== "id").map(([key, val]) =>
        <label key={key} className={Array.isArray(val) || key === "content" || key === "description" ? "wide" : ""}>
          <span>{key.replaceAll("_", " ")}</span>
          {Array.isArray(val) || key === "content" || key === "description"
            ? <textarea value={valueText(val)} onChange={e => mutate(items => { items[index][key] = Array.isArray(val) ? e.target.value.split("\n").filter(Boolean) : e.target.value; })} placeholder={Array.isArray(val) ? "One item per line" : ""} />
            : <input value={valueText(val)} onChange={e => mutate(items => { items[index][key] = e.target.value; })} />}
        </label>)}</div>
      <button className="icon remove" onClick={() => mutate(items => items.splice(index, 1))}><X /></button>
    </div>)}
  </div>;
}

const sectionConstraintFields: Record<typeof sections[number], {
  numeric: string[];
  toggles: string[];
}> = {
  summary: {
    numeric: ["min_words", "max_words", "max_sentences"],
    toggles: [],
  },
  skills: {
    numeric: [
      "min_items",
      "max_skill_categories",
      "min_skills_per_category",
      "max_skills_per_category",
    ],
    toggles: [],
  },
  work_experience: {
    numeric: [
      "min_bullets_per_item",
      "max_bullets_per_item",
      "max_words_per_bullet",
    ],
    toggles: ["require_metrics"],
  },
  education: {
    numeric: ["max_courses"],
    toggles: ["show_gpa", "show_coursework"],
  },
  projects: {
    numeric: [
      "min_items",
      "max_items",
      "min_bullets_per_item",
      "max_bullets_per_item",
      "max_words_per_bullet",
      "max_technologies",
    ],
    toggles: ["require_metrics"],
  },
  research: {
    numeric: [
      "min_items",
      "max_items",
      "min_bullets_per_item",
      "max_bullets_per_item",
      "max_words_per_bullet",
      "max_technologies",
    ],
    toggles: ["require_metrics"],
  },
};

function ConstraintEditor({ resume, setResume }: { resume: Resume; setResume: (r: Resume) => void }) {
  const [selected, setSelected] = useState<typeof sections[number]>("summary");
  const section = resume[selected] ?? { heading: selected, items: [], constraints: {} };
  const visibleFields = sectionConstraintFields[selected];
  const update = (key: string, value: unknown) => {
    const next = clone(resume); const target = next[selected] ?? clone(section);
    (target.constraints as any)[key] = value; (next as any)[selected] = target; setResume(next);
  };
  return <div className="content"><div className="title-row"><div><span className="eyebrow">Guardrails</span><h1>Writing constraints</h1>
    <p>Set deterministic limits by section. These are checked after every rewrite.</p></div></div>
    <div className="settings-layout"><div className="section-nav">{sections.map(s => <button className={selected === s ? "active" : ""} onClick={() => setSelected(s)} key={s}><span>{s.replaceAll("_", " ")}</span><ChevronRight /></button>)}</div>
      <div className="card constraint-card"><div className="card-title"><Gauge /><div><h3>{section.heading}</h3><p>Empty values use the application default.</p></div></div>
        {(selected === "education" || selected === "work_experience") && <div className="locked-note"><Check /> All entries are canonical and will always be preserved. Item-count limits do not apply.</div>}
        <div className="form-grid constraint-grid">{visibleFields.numeric.map(key => <label key={key}><span>{key.replaceAll("_", " ")}</span>
          <input type="number" min="0" value={(section.constraints as any)[key] ?? ""} onChange={e => update(key, e.target.value === "" ? null : Number(e.target.value))} /></label>)}</div>
        {visibleFields.toggles.length > 0 && <div className="toggles">{visibleFields.toggles.map(key => <label key={key}><input type="checkbox" checked={Boolean((section.constraints as any)[key])} onChange={e => update(key, e.target.checked)} /><span>{key.replaceAll("_", " ")}</span></label>)}</div>}
        <label className="block-label"><span>Required keywords · comma separated</span><input value={(section.constraints.required_keywords ?? []).join(", ")} onChange={e => update("required_keywords", e.target.value.split(",").map(x => x.trim()).filter(Boolean))} /></label>
        <label className="block-label"><span>Forbidden phrases · comma separated</span><input value={(section.constraints.forbidden_phrases ?? []).join(", ")} onChange={e => update("forbidden_phrases", e.target.value.split(",").map(x => x.trim()).filter(Boolean))} /></label>
        <label className="block-label"><span>Style instruction</span><textarea value={section.constraints.style ?? ""} onChange={e => update("style", e.target.value)} /></label>
      </div></div>
  </div>;
}

function ResumeWorkspace({ running, traces, result, source, usage, edit, setEdit, applyEdit, exportPdf, updateTailored, saveStatus, verboseOutput }: {
  running: boolean; traces: Trace[]; result: RunResult | null; source: Resume | null; usage: Usage | null;
  edit: string; setEdit: (v: string) => void; applyEdit: () => void; exportPdf: () => void;
  updateTailored: (resume: Resume) => void; saveStatus: "idle" | "saving" | "saved" | "error";
  verboseOutput: boolean;
}) {
  const [view, setView] = useState<"diff" | "trace" | "edit">("trace");
  useEffect(() => {
    if (running) setView("trace");
  }, [running]);
  const grouped = useMemo(
    () => (result?.diffs ?? []).reduce<Record<string, Diff[]>>((groups, diff) => {
      (groups[diff.section] ??= []).push(diff);
      return groups;
    }, {}),
    [result],
  );
  if (!running && !result) return <div className="start-state"><div className="orb"><Sparkles /></div><span className="eyebrow">Ready when you are</span><h1>Your tailored resume will appear here.</h1><p>Complete the candidate profile, paste a job listing, and start the agent from the sidebar.</p></div>;
  return <div className="content resume-content">
    <div className="title-row"><div><span className="eyebrow">{running ? "Agent in progress" : "Review & refine"}</span><h1>{running ? "Building your tailored draft" : "Tailored resume"}</h1>
      <p>{running ? "Decisions appear as each stage completes." : "Every change remains traceable to the source profile."}</p></div>
      {result && <button className="primary" disabled={saveStatus === "saving"} onClick={exportPdf}><Download /> {saveStatus === "saving" ? "Saving edits…" : "Confirm & download PDF"}</button>}
    </div>
    <div className="run-layout">
      <div className="result-pane">
        <div className="segmented"><button className={view === "trace" ? "active" : ""} onClick={() => setView("trace")}>Decision trace <span>{traces.length}</span></button><button className={view === "diff" ? "active" : ""} onClick={() => setView("diff")}>Changes</button>{result && <button className={view === "edit" ? "active" : ""} onClick={() => setView("edit")}>Edit</button>}</div>
        {view === "trace" ? <TraceList traces={traces} running={running} verbose={verboseOutput} /> :
          view === "edit" && result ? <TailoredEditor resume={result.resume} setResume={updateTailored} saveStatus={saveStatus} /> :
          result ? <div className="diff-list">{Object.entries(grouped).map(([section, diffs]) => <div key={section}><h3>{section.replaceAll("_", " ")}</h3>{diffs!.map((d, i) => <DiffCard diff={d} key={`${d.item_id}-${d.field}-${i}`} />)}</div>)}
            {(result.diffs.length === 0) && <div className="empty-row">No source fields changed.</div>}</div>
          : <TraceList traces={traces} running={running} verbose={verboseOutput} />}
      </div>
      <div className="run-summary">
        {running ? <div className="summary-card live"><LoaderCircle className="spin" /><h3>Working through the plan</h3><p>{traces.at(-1)?.summary ?? "Preparing candidate evidence…"}</p></div> :
        result && <><div className="summary-card"><span className="success"><Check /></span><h3>Run complete</h3><p>{result.state.total_rewrites} rewrites · {result.page_check.final_pages}/{result.page_check.maximum_pages} pages</p>
          <div className="metrics"><div><strong>{usage?.calls ?? 0}</strong><span>model calls</span></div><div><strong>{usage?.total_tokens.toLocaleString() ?? 0}</strong><span>est. tokens</span></div></div>
          <button className="secondary view-changes" onClick={() => setView("diff")}>View changes <ArrowRight /></button></div>
          {result.state.recruiter_evaluation && <div className={`summary-card recruiter-summary ${result.state.recruiter_evaluation.ready ? "ready" : "review"}`}><span className="eyebrow">Recruiter check</span><h3>{result.state.recruiter_evaluation.ready ? "Positioning reads clearly" : "Review before sending"}</h3><p>{result.state.recruiter_evaluation.summary}</p>
            <div className="recruiter-axes">{result.state.recruiter_evaluation.axes.map(axis => <div key={axis.axis}><span>{axis.axis.replaceAll("_", " ")}</span><strong>{axis.score}/5</strong></div>)}</div>
            {result.state.recruiter_evaluation.recommendations.length > 0 && <ul>{result.state.recruiter_evaluation.recommendations.map((item, index) => <li key={index}>{item}</li>)}</ul>}
          </div>}
          {(result.state.bullet_quality?.length ?? 0) > 0 && <div className="summary-card"><span className="eyebrow">Bullet quality</span><h3>{Math.round((result.state.bullet_quality ?? []).reduce((sum, item) => sum + item.score, 0) / (result.state.bullet_quality?.length || 1))}/100 average</h3><p>Action clarity, implementation, context, outcomes, concision, and credibility.</p></div>}
          <div className="summary-card feedback"><span className="eyebrow">One more pass?</span><h3>Ask for a global edit</h3><p>Give a plain-language direction. The same factual and page-limit checks run again.</p>
            <textarea value={edit} onChange={e => setEdit(e.target.value)} placeholder="e.g. Make the tone more direct and emphasize backend systems…" />
            <button className="primary" disabled={!edit.trim()} onClick={applyEdit}><Sparkles /> Apply feedback</button></div></>}
      </div>
    </div>
  </div>;
}

function TailoredEditor({ resume, setResume, saveStatus }: { resume: Resume; setResume: (resume: Resume) => void; saveStatus: string }) {
  const updateCandidate = (key: string, value: string) => {
    const next = clone(resume); next.candidate[key] = value; setResume(next);
  };
  return <div className="tailored-editor">
    <div className="edit-banner"><div><strong>Edit the tailored copy</strong><span>The canonical Candidate Profile will not change.</span></div><span className={`save-state ${saveStatus}`}>{saveStatus === "saving" ? "Saving…" : saveStatus === "saved" ? "Saved" : saveStatus === "error" ? "Save failed" : "Autosave on"}</span></div>
    <div className="card identity"><div className="card-title"><UserRound /><div><h3>Header</h3><p>Changes apply to this export only.</p></div></div>
      <div className="form-grid">{Object.entries(resume.candidate).map(([key, val]) =>
        <label key={key}><span>{key.replaceAll("_", " ")}</span><input value={val ?? ""} onChange={e => updateCandidate(key, e.target.value)} /></label>)}</div>
    </div>
    {sections.map(name => <SectionEditor key={name} name={name} resume={resume} setResume={setResume} />)}
  </div>;
}

function TraceList({ traces, running, verbose }: { traces: Trace[]; running: boolean; verbose: boolean }) {
  const displayTraces = useMemo(() => {
    if (verbose) return traces;
    const latestEvaluation = new Map<string, number>();
    traces.forEach((trace, index) => {
      if (trace.evaluation) latestEvaluation.set(trace.section ?? "workflow", index);
    });
    return traces.filter((trace, index) => {
      if (trace.event_type === "rewrite_started") return false;
      if (trace.evaluation) return latestEvaluation.get(trace.section ?? "workflow") === index;
      return !["attempt_accepted", "best_attempt_selected"].includes(trace.event_type);
    });
  }, [traces, verbose]);
  return <div className={`trace-list ${verbose ? "verbose" : "compact"}`}>{displayTraces.map((t, i) => <div className="trace-item" key={`${t.event_type}-${t.section ?? "workflow"}-${i}`}><div className="trace-rail"><span className={i === displayTraces.length - 1 && running ? "pulse" : ""}>{i + 1}</span></div>
    <div><div className="trace-meta">{t.section?.replaceAll("_", " ") || "workflow"} {t.attempt != null && `· attempt ${t.attempt}`}</div><h3>{t.title}</h3><p>{t.summary}</p>
      {t.match_score && !verbose && <div className="compact-trace-summary"><strong>{t.match_score.score}/100 match</strong><span>{t.match_score.supported} supported · {t.match_score.partial} partial · {t.match_score.unsupported} gaps</span></div>}
      {t.match_score && verbose && <div className="match-panel"><div className="match-score">
        <div className="score-ring" style={{"--score": `${t.match_score.score * 3.6}deg`} as React.CSSProperties}><span>{t.match_score.score}</span><small>/100</small></div>
        <div><strong>{t.title}</strong><span>{t.match_score.score >= 75 ? "Strong evidence alignment" : t.match_score.score >= 50 ? "Moderate evidence alignment" : "Material evidence gaps remain"}</span></div>
      </div><div className="score-components"><div><span>Evidence coverage</span><strong>{t.match_score.evidence_coverage_score}</strong></div><div><span>Holistic fit</span><strong>{t.match_score.holistic_fit_score ?? "—"}</strong></div></div>
        <div className="rubric-boxes">{t.match_score.breakdown.map(box => <div key={box.kind}><span>{box.kind.replaceAll("_", " ")}</span><strong>{box.score}/100</strong><small>{box.supported} supported · {box.partial} partial · {box.unsupported} gaps</small></div>)}</div>
        {t.match_score.rubric_observations.length > 0 && <details><summary>Holistic rubric</summary><ul>{t.match_score.rubric_observations.map((o,j)=><li key={j}>{o}</li>)}</ul></details>}
      </div>}
      {t.job_requirements && t.job_requirements.length > 0 && !verbose && <div className="compact-trace-summary"><strong>{t.job_requirements.length} requirements extracted</strong><span>{t.job_requirements.filter(req => req.required).length} required</span></div>}
      {t.job_requirements && t.job_requirements.length > 0 && verbose && <details className="trace-accordion"><summary>All {t.job_requirements.length} extracted requirements</summary><div className="requirement-list">{t.job_requirements.map(req => {
        const tier = req.required ? "required" : req.importance === "supporting" ? "bonus" : "preferred";
        return <div className={`requirement-row ${tier}`} key={req.id}><span className="tier-symbol">{tier === "required" ? "×" : tier === "preferred" ? "△" : "◇"}</span><div><b>{req.text}</b><small>{req.kind}</small></div><em>{tier}</em></div>;
      })}</div></details>}
      {t.coverage_plan && !verbose && <div className="compact-trace-summary"><strong>Evidence plan</strong><span>{t.coverage_plan.requirement_matches.filter(match => match.support === "supported").length} supported · {t.coverage_plan.requirement_matches.filter(match => match.support === "partial").length} partial · {t.coverage_plan.requirement_matches.filter(match => match.support === "unsupported").length} unsupported</span></div>}
      {t.coverage_plan && verbose && <div className="coverage-grid">{t.coverage_plan.requirement_matches.map(match => <div className={`coverage-box ${match.support}`} key={match.requirement_id}><div><span>{match.requirement_kind}</span><em>{match.importance}</em></div><strong>{match.requirement_text}</strong><small>{match.adjudication_reason || (match.support === "supported" ? "Direct source evidence found." : "No confirmed source evidence.")}</small><div className="support-label">{match.support === "supported" ? <Check /> : match.support === "partial" ? <span>≈</span> : <X />}{match.support}</div></div>)}</div>}
      {verbose && t.observations.length > 0 && !t.match_score && !t.coverage_plan && !(t.job_requirements?.length) && <ul>{t.observations.map((o, j) => <li key={j}>{o}</li>)}</ul>}
      {t.evaluation && <div className="evaluation">
        {t.evaluation.detection.scoring_status === "completed" && <><div className="score-strip">
          <span><strong>{Math.round(t.evaluation.detection.ai_probability * 100)}%</strong> AI likeness <em>debug</em></span>
          <span><strong>{t.evaluation.decision.average_rubric_score?.toFixed(1) ?? "—"}</strong> / 5 writing</span>
        </div>
        {verbose && t.evaluation.detection.components.length > 0 && <div className="rubric-boxes">
          {t.evaluation.detection.components.map(component => <div key={component.model_name}><span>{component.model_name.replaceAll("_", " ")}</span><strong>{component.error ? "unavailable" : `${Math.round(component.ai_probability * 100)}%`}</strong><small>{component.error || "AI-likeness component score"}</small></div>)}
        </div>}
        {verbose && t.evaluation.detection.ensemble_explanation && <details open><summary>Ensemble calculation · {t.evaluation.detection.ensemble_explanation.agreement} agreement</summary>
          <ul>{t.evaluation.detection.ensemble_explanation.components.map(component => <li key={component.model_name}><b>{component.model_name.replaceAll("_", " ")}</b>: {component.probability.toFixed(3)} × {component.weight.toFixed(2)} = {component.weighted_value.toFixed(3)}</li>)}</ul>
          <p>({t.evaluation.detection.ensemble_explanation.weighted_sum.toFixed(3)} ÷ {t.evaluation.detection.ensemble_explanation.weight_total.toFixed(2)}) = <b>{t.evaluation.detection.ensemble_explanation.combined_probability.toFixed(3)}</b> · spread {t.evaluation.detection.ensemble_explanation.spread.toFixed(3)} · σ {t.evaluation.detection.ensemble_explanation.standard_deviation.toFixed(3)}</p>
        </details>}
        {verbose && t.evaluation.detection.rubric_axes.length > 0 && <div className="axis-grid">
          {t.evaluation.detection.rubric_axes.map(axis => <div key={axis.label}><span>{axis.label}</span><strong>{axis.score}/5</strong><p>{axis.interpretation}{axis.contribution != null && <> · contribution {axis.contribution >= 0 ? "+" : ""}{axis.contribution.toFixed(3)} ({axis.direction?.replaceAll("_", " ")})</>}</p></div>)}
        </div>}
        {verbose && t.evaluation.detection.feature_evidence.length > 0 && <details><summary>Signals uncovered</summary>
          <ul>{t.evaluation.detection.feature_evidence.map(feature => <li key={feature.label}><b>#{feature.importance_rank} {feature.label}</b> · {feature.direction.replaceAll("_", " ")} · observed {feature.observed_value.toFixed(3)} · SHAP {feature.shap_value >= 0 ? "+" : ""}{feature.shap_value.toFixed(3)} — {feature.description}</li>)}</ul>
        </details>}
        {verbose && t.evaluation.detection.components.map(component => <Fragment key={`${component.model_name}-explanation`}>
          {component.model_name === "distilbert" && component.token_attributions.length > 0 && <details><summary>DistilBERT Integrated Gradients · all tokens</summary><div className="token-evidence">{component.token_attributions.map((token, j) => <span className={token.direction} key={j} title={`${token.attribution >= 0 ? "+" : ""}${token.attribution.toFixed(5)}`}>{token.token} </span>)}</div></details>}
          {component.model_name === "tfidf_svm" && component.term_contributions.length > 0 && <details><summary>TF-IDF phrase signals</summary><ul>{component.term_contributions.slice(0, 12).map(term => <li key={term.term}><b>{term.term}</b> · {term.contribution >= 0 ? "+" : ""}{term.contribution.toFixed(3)} ({term.direction.replaceAll("_", " ")})</li>)}</ul>{component.explanation_note && <p>{component.explanation_note}</p>}</details>}
        </Fragment>)}
        {verbose && t.evaluation.counterfactual && <details open><summary>Rewrite counterfactual · {t.evaluation.counterfactual.delta <= 0 ? "improved" : "regressed"} {Math.abs(t.evaluation.counterfactual.delta * 100).toFixed(1)} points</summary><ul>{t.evaluation.counterfactual.components.map(component => <li key={component.model_name}><b>{component.model_name.replaceAll("_", " ")}</b>: {component.before.toFixed(3)} → {component.after.toFixed(3)} ({component.delta >= 0 ? "+" : ""}{component.delta.toFixed(3)})</li>)}</ul></details>}</>}
        {!verbose && t.evaluation.constraints.length > 0 && <div className="compact-trace-summary"><strong>{t.evaluation.constraints.filter(c => c.passed).length}/{t.evaluation.constraints.length} checks passed</strong><span>{t.evaluation.constraints.filter(c => !c.passed).length ? "Review remaining constraints" : "All configured constraints satisfied"}</span></div>}
        {verbose && t.evaluation.constraints.length > 0 && <details><summary>Constraint checklist · {t.evaluation.constraints.filter(c => c.passed).length}/{t.evaluation.constraints.length} passed</summary>
          <div className="checklist">{t.evaluation.constraints.map((check, j) => <div className={check.passed ? "passed" : "failed"} key={`${check.label}-${j}`}>
            <span>{check.passed ? <Check /> : <X />}</span><div><b>{check.label}</b><small>{check.observed} · expected {check.expected}</small></div>
          </div>)}</div>
        </details>}
      </div>}
      {t.decision && <span className={`decision ${t.decision}`}>{t.decision.replaceAll("_", " ")}</span>}</div></div>)}
    {running && <div className="trace-item ghost"><div className="trace-rail"><span><LoaderCircle className="spin" /></span></div><div><h3>Next decision</h3><p>Waiting for the agent…</p></div></div>}
  </div>;
}

function DiffCard({ diff }: { diff: Diff }) {
  return <article className="diff-card"><div className="diff-head"><div><span>{diff.item_label}</span><strong>{diff.field === "__item__" ? "Entry" : diff.field.replaceAll("_", " ")}</strong></div><span className={`change ${diff.change_type}`}>{diff.change_type.replaceAll("_", " ")}</span></div>
    <div className="inline-diff">{diff.tokens.map((token, i) => <span key={i} className={token.operation}>{token.text}</span>)}</div>
    <p className="why"><Sparkles /> {diff.reason}</p></article>;
}

export default App;
