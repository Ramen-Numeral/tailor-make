import type { Resume } from "./types";

const constraints = {};

/** Editable example data used as the application's initial source resume. */
export const defaultResume: Resume = {
  candidate: {
    name: "Jack Doe",
    email: "jack.doe@example.com",
    phone: "(555) 014-2026",
    location: "New York, NY",
    target_title: "Product Manager",
    github: "github.com/jackdoe",
    linkedin: "linkedin.com/in/jackdoe",
    website: "jackdoe.dev",
  },
  summary: {
    heading: "Professional Summary",
    items: [{
      content: "Product manager with experience translating customer research and product data into practical roadmap decisions for web-based software.",
    }],
    constraints: { ...constraints },
  },
  skills: {
    heading: "Skills",
    items: [
      { name: "Product", skills: ["Roadmapping", "User research", "Agile", "A/B testing"] },
      { name: "Analytics", skills: ["SQL", "Google Analytics", "Looker", "Excel"] },
      { name: "Tools", skills: ["Jira", "Figma", "Git", "Notion"] },
    ],
    constraints: { ...constraints },
  },
  work_experience: {
    heading: "Work Experience",
    items: [
      {
        title: "Product Manager",
        company: "Northstar Software",
        start_date: "Jan 2023",
        end_date: "Present",
        location: "New York, NY",
        bullets: [
          "Owned the roadmap for a customer onboarding platform used by 18,000 monthly users.",
          "Partnered with engineering, design, and support to reduce onboarding abandonment by 17%.",
          "Used product analytics and customer interviews to prioritize quarterly releases.",
        ],
      },
      {
        title: "Product Operations Associate",
        company: "Beacon Digital",
        start_date: "Jun 2020",
        end_date: "Dec 2022",
        location: "Brooklyn, NY",
        bullets: [
          "Maintained the product backlog and coordinated releases across three client-facing teams.",
          "Built SQL and Looker reports that cut weekly reporting time by six hours.",
        ],
      },
    ],
    constraints: { ...constraints },
  },
  education: {
    heading: "Education",
    items: [{
      degree: "Bachelor of Science, Information Systems",
      institution: "Example State University",
      graduation_date: "May 2020",
      location: "Albany, NY",
      gpa: "3.7",
      coursework: ["Database Systems", "Product Design", "Business Analytics"],
      honors: ["Dean's List"],
    }],
    constraints: {
      ...constraints,
      show_gpa: true,
      show_coursework: true,
      max_courses: 5,
    },
  },
  projects: {
    heading: "Projects",
    items: [{
      name: "Neighborhood Events Finder",
      description: "Designed and shipped a searchable community-events prototype based on interviews with local organizers.",
      technologies: ["React", "Figma", "Firebase"],
      bullets: [
        "Tested the prototype with 12 users and revised event discovery around their feedback.",
      ],
      url: "github.com/jackdoe/events-finder",
    }],
    constraints: { ...constraints },
  },
  research: {
    heading: "Research",
    items: [],
    constraints: { ...constraints },
  },
};
