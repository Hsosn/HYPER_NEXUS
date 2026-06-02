/**
 * MCP Servers — Drawer overlay for managing Model Context Protocol servers.
 * Same UX pattern as Integrations and Triggers drawers.
 */
import { api } from '../utils.js';
import { toast } from '../enhancements.js';

// ── MCP Server logos by name/id keywords ──────────────────────────────────────
const MCP_LOGOS = {
  filesystem:  `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="1.5"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>`,
  github:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>`,
  postgres:    `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#336791" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`,
  postgresql:  `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#336791" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`,
  sqlite:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#003b57" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18"/></svg>`,
  brave:       `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fb542b" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>`,
  search:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fb542b" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>`,
  puppeteer:   `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>`,
  browser:     `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>`,
  slack:       `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#E01E5A" stroke-width="1.5"><path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313z"/></svg>`,
  memory:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="1.5"><path d="M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7z"/><circle cx="12" cy="9" r="2.5"/></svg>`,
  sentry:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#362d59" stroke-width="1.5"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M16 10H8v4h8z"/></svg>`,
  google_maps: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34a853" stroke-width="1.5"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/><circle cx="12" cy="9" r="2.5"/></svg>`,
  maps:        `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34a853" stroke-width="1.5"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/><circle cx="12" cy="9" r="2.5"/></svg>`,
  discord:     `<svg width="18" height="18" viewBox="0 0 24 24"><path fill="#5865F2" d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03z"/></svg>`,
  docker:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2496ED" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18"/></svg>`,
  kubernetes:  `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#326CE5" stroke-width="1.5"><path d="M12 2L2 12h3v9h5v-6h2v6h5v-9h3z"/><circle cx="12" cy="12" r="3"/></svg>`,
  notion:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 7h10v2H7zM7 11h10v2H7zM7 15h6v2H7z"/></svg>`,
  airtable:    `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#18BFFF" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>`,
  shopify:     `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#96BF48" stroke-width="1.5"><path d="M12 2C7 2 3 6 3 10c0 5 4 8 9 12 5-4 9-7 9-12 0-4-4-8-9-8z"/></svg>`,
  clickup:     `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7B68EE" stroke-width="1.5"><path d="M5 17l4-4 3 3 7-9v7l-7 6-3-3z"/></svg>`,
  linear:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#5E6AD2" stroke-width="1.5"><path d="M3 12l7-10 7 10H3z"/><path d="M3 12l7 10 7-10" opacity="0.5"/></svg>`,
  jira:        `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0052CC" stroke-width="1.5"><path d="M11.571 11.513H0a5.218 5.218 0 0 0 5.232 5.215h2.13v2.056A5.215 5.215 0 0 0 12.575 24V12.518a1.005 1.005 0 0 0-1.004-1.005z"/></svg>`,
  confluence:  `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#172B4D" stroke-width="1.5"><path d="M4 14c2-3.5 5-6 8-7 3 1 6 3.5 8 7-2 3.5-5 6-8 7-3-1-6-3.5-8-7z"/></svg>`,
  trello:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0079BF" stroke-width="1.5"><rect x="2" y="4" width="6" height="16" rx="1"/><rect x="10" y="8" width="6" height="12" rx="1"/></svg>`,
  asana:       `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F06A6A" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="7" cy="12" r="3"/><circle cx="17" cy="12" r="3"/></svg>`,
  gitlab:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FC6D26" stroke-width="1.5"><path d="m21.435 9.365-9.436 13.04L2.565 9.365l2.818-8.68 2.056 6.334h9.122l2.056-6.334z"/></svg>`,
  bitbucket:   `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2684FF" stroke-width="1.5"><path d="M2 4l2.18 16.06L12 22l7.82-1.94L22 4H2z"/></svg>`,
  salesforce:  `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00A1E0" stroke-width="1.5"><circle cx="12" cy="12" r="10"/></svg>`,
  hubspot:     `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF7A59" stroke-width="1.5"><path d="M18.164 7.931V5.085a2.198 2.198 0 1 0-1.672 0v2.846a6.413 6.413 0 0 0-2.914 1.433l-7.52-5.854a2.481 2.481 0 1 0-1.04 1.326l7.313 5.691a6.413 6.413 0 0 0-.876 3.261c0 1.054.253 2.047.706 2.921l-2.303 2.303a2.195 2.195 0 1 0 1.18 1.18l2.283-2.283a6.41 6.41 0 0 0 3.173.838 6.434 6.434 0 0 0 1.67-12.616z"/></svg>`,
  zendesk:     `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#03363D" stroke-width="1.5"><path d="M12 2L2 20h10V2z"/><path d="M22 4H12v16l10-16z" opacity="0.5"/></svg>`,
  twilio:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F22F46" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="9" cy="9" r="2"/><circle cx="15" cy="9" r="2"/><circle cx="9" cy="15" r="2"/><circle cx="15" cy="15" r="2"/></svg>`,
  email:       `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EA580C" stroke-width="1.5"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 5 9-5"/></svg>`,
  openai:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10A37A" stroke-width="1.5"><path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91 6.046 6.046 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a5.985 5.985 0 0 0-3.998 2.9 6.046 6.046 0 0 0 .743 7.097 5.98 5.98 0 0 0 .51 4.911 6.051 6.051 0 0 0 6.515 2.9A5.985 5.985 0 0 0 13.26 24a6.056 6.056 0 0 0 5.772-4.206 5.99 5.99 0 0 0 3.997-2.9 6.056 6.056 0 0 0-.747-7.073z"/></svg>`,
  anthropic:   `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#D4A574" stroke-width="1.5"><path d="M2 20L9 4h2l-7 16H2zm11 0L20 4h2l-7 16h-2z"/></svg>`,
  cloudflare:  `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F38020" stroke-width="1.5"><path d="M12 2C8 2 4 4 4 7c0 1.5.5 2.8 1.3 3.8A4 4 0 0 0 5 15a4 4 0 0 0 6.8 2.3A5 5 0 0 0 15 14a5 5 0 0 0 3.7-1.7A4 4 0 0 0 20 9c0-2.2-1.8-4-4-4-.5 0-1 .1-1.5.3C14.2 3.5 13.2 2.5 12 2z"/></svg>`,
  vercel:      `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5"><polygon points="12,2 24,22 0,22"/></svg>`,
  aws:         `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF9900" stroke-width="1.5"><path d="M12 2L4 7l8 5 8-5-8-5z"/><path d="M4 12l8 5 8-5"/></svg>`,
  gcp:         `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4285F4" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/></svg>`,
  azure:       `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0078D4" stroke-width="1.5"><path d="M3 7l9-3 9 3v10l-9 3-9-3z"/></svg>`,
  firebase:    `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFCA28" stroke-width="1.5"><path d="M4 12l4-9h12l-4 9-4 7z"/></svg>`,
  supabase:    `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3ECF8E" stroke-width="1.5"><path d="M12 2L3 9v6l9 7 9-7V9L12 2z"/></svg>`,
  mongodb:     `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#47A248" stroke-width="1.5"><path d="M12 2C10 2 8 5 8 10c0 5 2 10 4 12 2-2 4-7 4-12s-2-8-4-8z"/></svg>`,
  redis:       `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#DC382D" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 8h10M7 12h10M7 16h6"/></svg>`,
  mysql:       `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4479A1" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`,
  http:        `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8"/></svg>`,
  api:         `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 8h10M7 12h10M7 16h6"/></svg>`,
  graphql:     `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e535ab" stroke-width="1.5"><circle cx="12" cy="4" r="2"/><circle cx="5" cy="8" r="2"/><circle cx="19" cy="8" r="2"/><circle cx="5" cy="16" r="2"/><circle cx="19" cy="16" r="2"/><circle cx="12" cy="20" r="2"/><path d="M12 6L5 8M12 6l7 2M5 10l7 10M19 10l-7 10M5 16h14"/></svg>`,
  rss:         `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f97316" stroke-width="1.5"><circle cx="6" cy="18" r="2"/><path d="M4 12a8 8 0 0 1 8 8"/><path d="M4 6a14 14 0 0 1 14 14"/></svg>`,
  webhook:     `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M8 8l4 4-4 4M16 8l-4 4 4 4"/></svg>`,
};

/**
 * Resolve the best logo SVG for an MCP server based on its name or id.
 */
function _getMCPIcon(server) {
  // If the server has an explicit icon from a template, use it
  if (server.icon) return server.icon;

  // Try to match by name or id against known logos
  const name = (server.name || server.id || '').toLowerCase();
  for (const [key, svg] of Object.entries(MCP_LOGOS)) {
    if (name.includes(key)) return svg;
  }

  // Default MCP grid icon
  return null;
}

// ── Pre-defined MCP server templates ──────────────────────────────────────────
const MCP_TEMPLATES = [
  {
    id: 'mcp_tmpl_filesystem',
    name: 'Filesystem',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@anthropic/mcp-server-filesystem', '/workspace'],
    description: 'Read, write, and manage files on the local filesystem',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="1.5"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
  },
  {
    id: 'mcp_tmpl_github',
    name: 'GitHub',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@anthropic/mcp-server-github'],
    env_keys: ['GITHUB_PERSONAL_ACCESS_TOKEN'],
    description: 'GitHub API — repos, issues, PRs, code search',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>',
  },
  {
    id: 'mcp_tmpl_postgres',
    name: 'PostgreSQL',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-postgres', 'postgresql://localhost/mydb'],
    description: 'Query PostgreSQL databases with natural language',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#336791" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
  },
  {
    id: 'mcp_tmpl_sqlite',
    name: 'SQLite',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-sqlite', '--db-path', '/workspace/data.db'],
    description: 'Query SQLite databases',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#003b57" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18"/></svg>',
  },
  {
    id: 'mcp_tmpl_brave',
    name: 'Brave Search',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@anthropic/mcp-server-brave-search'],
    env_keys: ['BRAVE_API_KEY'],
    description: 'Web search via Brave Search API',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fb542b" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>',
  },
  {
    id: 'mcp_tmpl_puppeteer',
    name: 'Puppeteer',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@anthropic/mcp-server-puppeteer'],
    description: 'Browser automation — screenshots, clicks, form fills',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
  },
  {
    id: 'mcp_tmpl_slack',
    name: 'Slack',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@anthropic/mcp-server-slack'],
    env_keys: ['SLACK_BOT_TOKEN'],
    description: 'Send messages, search channels, manage Slack workspace',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#E01E5A" stroke-width="1.5"><path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313z"/></svg>',
  },
  {
    id: 'mcp_tmpl_memory',
    name: 'Memory',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@anthropic/mcp-server-memory'],
    description: 'Persistent knowledge graph memory for the agent',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="1.5"><path d="M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7z"/><circle cx="12" cy="9" r="2.5"/></svg>',
  },
  {
    id: 'mcp_tmpl_sentry',
    name: 'Sentry',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@anthropic/mcp-server-sentry'],
    env_keys: ['SENTRY_AUTH_TOKEN'],
    description: 'Error tracking, issue management, and release monitoring',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#362d59" stroke-width="1.5"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M16 10H8v4h8z"/></svg>',
  },
  {
    id: 'mcp_tmpl_google_maps',
    name: 'Google Maps',
    cat: 'Built-in',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@anthropic/mcp-server-google-maps'],
    env_keys: ['GOOGLE_MAPS_API_KEY'],
    description: 'Places, directions, geocoding, and maps data',
    icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34a853" stroke-width="1.5"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/><circle cx="12" cy="9" r="2.5"/></svg>',
  },
];

const MCP_CATEGORIES = ['All', 'Built-in', 'Custom', 'Connected'];

let mcpServers = [];
let mcpActiveCat = 'All';
let mcpSearchQ = '';

// ── Import the drawer helper from integrations ─────────────────────────────────
// We re-create the drawer helper here to avoid circular imports
function createDrawer(title, icon, buildContent) {
  const root = document.getElementById('modal-root') || document.body;
  const backdrop = document.createElement('div');
  backdrop.className = 'drw-backdrop';
  const drawer = document.createElement('div');
  drawer.className = 'drw';
  drawer.innerHTML = `
    <div class="drw-header">
      <div class="drw-title">${icon}<span>${title}</span></div>
      <button class="drw-close" title="Close">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
    <div class="drw-body" id="drw-body-inner"></div>`;
  backdrop.appendChild(drawer);
  root.appendChild(backdrop);
  requestAnimationFrame(() => backdrop.classList.add('drw-open'));
  const close = () => {
    backdrop.classList.remove('drw-open');
    setTimeout(() => backdrop.remove(), 260);
  };
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });
  drawer.querySelector('.drw-close').addEventListener('click', close);
  buildContent(drawer.querySelector('#drw-body-inner'), close);
  return { backdrop, drawer, close };
}

// ── MCP Drawer Init ────────────────────────────────────────────────────────────
export async function initMCP() {
  const btn = document.getElementById('mcp-trigger');
  if (!btn) return;

  await refreshMCPServers();

  btn.addEventListener('click', () => {
    createDrawer(
      'MCP Servers',
      `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="2" y="2" width="8" height="8" rx="2"/><rect x="14" y="2" width="8" height="8" rx="2"/><rect x="2" y="14" width="8" height="8" rx="2"/><rect x="14" y="14" width="8" height="8" rx="2"/></svg>`,
      (body) => buildMCPBody(body)
    );
  });

  // Listen for WebSocket events to refresh MCP state
  window.addEventListener('ws:event', e => {
    const msg = e.detail;
    if (msg.kind && msg.kind.startsWith('mcp_')) {
      refreshMCPServers();
    }
  });
}

async function refreshMCPServers() {
  try {
    const data = await api('/api/mcp/servers');
    mcpServers = data || [];
  } catch {
    mcpServers = [];
  }
}

function buildMCPBody(body) {
  // Info banner
  const info = document.createElement('div');
  info.className = 'drw-mcp-info';
  info.innerHTML = `
    <div class="drw-mcp-info-icon">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>
    </div>
    <div>
      <div style="font-weight:600;font-size:12px;margin-bottom:2px">Model Context Protocol</div>
      <div style="font-size:11px;color:var(--tx3);line-height:1.4">Connect MCP servers to extend the agent with specialized tools. The agent becomes aware of all connected servers automatically across sessions.</div>
    </div>`;
  body.appendChild(info);

  // Connected count summary
  const connectedCount = mcpServers.filter(s => s.connected).length;
  const summary = document.createElement('div');
  summary.className = 'drw-trig-summary';
  summary.innerHTML = `
    <div class="drw-trig-stat">
      <span class="drw-trig-stat-num">${mcpServers.length}</span>
      <span class="drw-trig-stat-label">Total</span>
    </div>
    <div class="drw-trig-stat">
      <span class="drw-trig-stat-num" style="color:#10b981">${connectedCount}</span>
      <span class="drw-trig-stat-label">Connected</span>
    </div>
    <div class="drw-trig-stat">
      <span class="drw-trig-stat-num" style="color:#f59e0b">${mcpServers.reduce((sum, s) => sum + (s.tools?.length || 0), 0)}</span>
      <span class="drw-trig-stat-label">Tools</span>
    </div>`;
  body.appendChild(summary);

  // Search
  const search = document.createElement('div');
  search.className = 'drw-search-row';
  search.innerHTML = `
    <div class="drw-search-wrap">
      <svg class="drw-search-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <input type="text" class="drw-search" placeholder="Search MCP servers…" autocomplete="off"/>
    </div>`;
  body.appendChild(search);

  // Category tabs
  const cats = document.createElement('div');
  cats.className = 'drw-cats';
  MCP_CATEGORIES.forEach(cat => {
    const b = document.createElement('button');
    b.className = 'drw-cat' + (cat === mcpActiveCat ? ' active' : '');
    b.textContent = cat;
    b.addEventListener('click', () => {
      mcpActiveCat = cat;
      cats.querySelectorAll('.drw-cat').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      renderMCPGrid(grid);
    });
    cats.appendChild(b);
  });
  body.appendChild(cats);

  const grid = document.createElement('div');
  grid.className = 'drw-int-grid';
  body.appendChild(grid);

  // Add Custom MCP Server button
  const addCustomBtn = document.createElement('button');
  addCustomBtn.className = 'drw-add-custom-btn';
  addCustomBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg> Add Custom MCP Server`;
  addCustomBtn.style.cssText = 'width:100%;margin-top:8px;padding:10px;border:1px dashed var(--bd2);border-radius:8px;background:var(--bg1);color:var(--tx2);cursor:pointer;font-size:13px;display:flex;align-items:center;justify-content:center;gap:6px;transition:all .15s';
  addCustomBtn.addEventListener('mouseenter', () => { addCustomBtn.style.borderColor = '#06b6d4'; addCustomBtn.style.color = '#06b6d4'; });
  addCustomBtn.addEventListener('mouseleave', () => { addCustomBtn.style.borderColor = 'var(--bd2)'; addCustomBtn.style.color = 'var(--tx2)'; });
  addCustomBtn.addEventListener('click', () => openCustomMCPServerModal(grid));
  body.appendChild(addCustomBtn);

  // Ask Agent button
  const askAgentBtn = document.createElement('button');
  askAgentBtn.className = 'drw-add-custom-btn';
  askAgentBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg> Ask Agent to Connect`;
  askAgentBtn.style.cssText = 'width:100%;margin-top:4px;padding:10px;border:1px solid #8b5cf640;border-radius:8px;background:linear-gradient(135deg,#8b5cf610,#06b6d410);color:#8b5cf6;cursor:pointer;font-size:13px;display:flex;align-items:center;justify-content:center;gap:6px;transition:all .15s';
  askAgentBtn.addEventListener('mouseenter', () => { askAgentBtn.style.borderColor = '#8b5cf6'; askAgentBtn.style.background = 'linear-gradient(135deg,#8b5cf620,#06b6d420)'; });
  askAgentBtn.addEventListener('mouseleave', () => { askAgentBtn.style.borderColor = '#8b5cf640'; askAgentBtn.style.background = 'linear-gradient(135deg,#8b5cf610,#06b6d410)'; });
  askAgentBtn.addEventListener('click', () => {
    // Open chat and pre-fill a message for the agent
    const input = document.getElementById('input');
    if (input) {
      input.value = 'I want to connect to an MCP server. Can you help me set it up? Just tell me what you need — a URL, service name, or I can auto-discover it for you.';
      input.focus();
      input.dispatchEvent(new Event('input'));
    }
    // Close the drawer
    const backdrop = body.closest('.drw-backdrop');
    if (backdrop) {
      backdrop.classList.remove('drw-open');
      setTimeout(() => backdrop.remove(), 260);
    }
    toast('Tell the agent what MCP server you need — it will handle the setup for you!', 'info');
  });
  body.appendChild(askAgentBtn);

  search.querySelector('.drw-search').addEventListener('input', e => {
    mcpSearchQ = e.target.value.toLowerCase();
    renderMCPGrid(grid);
  });

  renderMCPGrid(grid);
}

function renderMCPGrid(grid) {
  grid.innerHTML = '';

  // Merge templates with existing servers
  const existingIds = new Set(mcpServers.map(s => s.id));

  // Build the full list: existing servers + templates not yet added
  const allItems = [];

  // Add existing servers
  mcpServers.forEach(s => {
    allItems.push({
      id: s.id,
      name: s.name || s.id,
      cat: s.category === 'auto-discovered' ? 'Built-in' : (s.category || 'Custom'),
      bg: s.connected ? '#10b98110' : '#06b6d410',
      isExisting: true,
      connected: s.connected,
      status: s.status,
      tools: s.tools || [],
      transport: s.transport,
      description: s.description || '',
      command: s.command,
      auto_connect: s.auto_connect,
      icon: null,
    });
  });

  // Add templates that are not yet configured
  MCP_TEMPLATES.forEach(tmpl => {
    if (!existingIds.has(tmpl.id)) {
      allItems.push({
        id: tmpl.id,
        name: tmpl.name,
        cat: 'Built-in',
        bg: '#06b6d410',
        isExisting: false,
        connected: false,
        status: 'available',
        tools: [],
        transport: tmpl.transport,
        description: tmpl.description,
        command: tmpl.command,
        auto_connect: false,
        icon: tmpl.icon,
        _template: tmpl,
      });
    }
  });

  // Filter
  const visible = allItems.filter(item => {
    const catOk = mcpActiveCat === 'All'
      || item.cat === mcpActiveCat
      || (mcpActiveCat === 'Connected' && item.connected)
      || (mcpActiveCat === 'Custom' && item.cat === 'Custom');
    const termOk = !mcpSearchQ || item.name.toLowerCase().includes(mcpSearchQ) || (item.description || '').toLowerCase().includes(mcpSearchQ);
    return catOk && termOk;
  });

  visible.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'drw-int-card' + (item.connected ? ' connected' : '');
    if (item.status === 'error') card.classList.add('error');
    card.style.animationDelay = idx * 28 + 'ms';

    const statusDot = item.connected
      ? '<span style="width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block;margin-right:4px"></span>'
      : item.status === 'error'
        ? '<span style="width:8px;height:8px;border-radius:50%;background:#ef4444;display:inline-block;margin-right:4px"></span>'
        : '';

    const icon = _getMCPIcon(item) || `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="${item.connected ? '#10b981' : '#06b6d4'}" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>`;

    const toolCount = item.tools?.length || 0;
    const toolBadge = toolCount > 0 ? `<span style="font-size:9px;color:var(--tx3);background:var(--bg1);padding:1px 5px;border-radius:4px;margin-left:4px">${toolCount} tools</span>` : '';

    card.innerHTML = `
      <div class="drw-int-logo" style="background:${item.bg}">${icon}</div>
      <div class="drw-int-info">
        <div class="drw-int-name">${statusDot}${item.name}${toolBadge}</div>
        <div class="drw-int-cat">${item.cat} · ${item.transport || 'stdio'}${item.auto_connect ? ' · <span style="color:#8b5cf6">auto</span>' : ''}</div>
      </div>
      <button class="drw-int-btn ${item.connected ? 'manage' : ''}">${item.connected ? 'Manage' : item.isExisting ? 'Connect' : 'Add'}</button>`;

    card.querySelector('.drw-int-btn').addEventListener('click', () => {
      if (item.isExisting) {
        openMCPServerModal(item, grid);
      } else {
        // Template — add from template
        openMCPTemplateModal(item, grid);
      }
    });
    grid.appendChild(card);
  });

  if (!visible.length) {
    grid.innerHTML = '<div class="drw-empty">No MCP servers match</div>';
  }
}

function openMCPServerModal(server, grid) {
  const root = document.getElementById('modal-root') || document.body;
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';

  const isExisting = server.connected;

  const resolvedIcon = _getMCPIcon(server);
  const logo = resolvedIcon || `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="${server.connected ? '#10b981' : '#06b6d4'}" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>`;

  // Build tools list HTML
  let toolsHtml = '';
  if (server.tools && server.tools.length > 0) {
    const toolItems = server.tools.map(t => {
      const tname = typeof t === 'string' ? t : (t.name || JSON.stringify(t));
      const tdesc = typeof t === 'object' ? (t.description || '') : '';
      return `<div class="drw-mcp-tool-item">
        <span class="drw-mcp-tool-name">${tname}</span>
        ${tdesc ? `<span class="drw-mcp-tool-desc">${tdesc}</span>` : ''}
      </div>`;
    }).join('');
    toolsHtml = `
      <div style="margin-top:12px">
        <div style="font-size:11px;font-weight:600;color:var(--tx3);margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px">Discovered Tools (${server.tools.length})</div>
        <div class="drw-mcp-tools-list">${toolItems}</div>
      </div>`;
  }

  backdrop.innerHTML = `
    <div class="modal" style="max-width:480px">
      <div class="modal-header">
        <div style="display:flex;align-items:center;gap:10px">
          <div style="width:32px;height:32px;border-radius:8px;background:${server.bg};display:flex;align-items:center;justify-content:center">${logo}</div>
          <div>
            <div style="font-weight:700;font-size:14px">${server.name}</div>
            <div style="font-size:11px;color:var(--tx3)">${server.cat} · ${server.transport} · ${server.status}</div>
          </div>
        </div>
        <button class="modal-close" id="mcp-modal-close">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="modal-body">
        <div style="font-size:12px;color:var(--tx2);line-height:1.5;margin-bottom:12px">${server.description || 'MCP Server'}</div>
        ${server.command ? `<div class="modal-field"><label>Command</label><input type="text" value="${server.command}" disabled style="opacity:0.6"/></div>` : ''}
        ${server.url ? `<div class="modal-field"><label>Server URL</label><input type="text" value="${server.url}" disabled style="opacity:0.6"/></div>` : ''}
        ${toolsHtml}
      </div>
      <div class="modal-footer">
        ${isExisting ? `<button class="btn danger" id="mcp-disconnect">Disconnect</button>` : `<button class="btn primary" id="mcp-connect">Connect</button>`}
        <button class="btn ghost" id="mcp-cancel">Cancel</button>
        ${isExisting ? '' : `<button class="btn primary" id="mcp-connect" style="margin-left:auto">Connect</button>`}
        <button class="btn danger-outline" id="mcp-delete" title="Remove server" style="padding:6px 10px;font-size:11px">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>
        </button>
      </div>
    </div>`;

  document.body.appendChild(backdrop);
  requestAnimationFrame(() => backdrop.classList.add('visible'));

  const close = () => { backdrop.classList.remove('visible'); setTimeout(() => backdrop.remove(), 200); };
  backdrop.querySelector('#mcp-modal-close').addEventListener('click', close);
  backdrop.querySelector('#mcp-cancel').addEventListener('click', close);
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });

  // Connect button
  const connectBtn = backdrop.querySelector('#mcp-connect');
  if (connectBtn) {
    connectBtn.addEventListener('click', async () => {
      connectBtn.textContent = 'Connecting...';
      connectBtn.disabled = true;
      try {
        await api(`/api/mcp/servers/${encodeURIComponent(server.id)}/connect`, { method: 'POST' });
        toast(`${server.name} connected successfully`, 'success');
        await refreshMCPServers();
        close();
        renderMCPGrid(grid);
      } catch (e) {
        toast(`Connection failed: ${e.message}`, 'error');
        connectBtn.textContent = 'Retry';
        connectBtn.disabled = false;
      }
    });
  }

  // Disconnect button
  const disconnectBtn = backdrop.querySelector('#mcp-disconnect');
  if (disconnectBtn) {
    disconnectBtn.addEventListener('click', async () => {
      try {
        await api(`/api/mcp/servers/${encodeURIComponent(server.id)}/disconnect`, { method: 'POST' });
        toast(`${server.name} disconnected`, 'success');
        await refreshMCPServers();
        close();
        renderMCPGrid(grid);
      } catch (e) {
        toast(`Failed: ${e.message}`, 'error');
      }
    });
  }

  // Delete button
  const deleteBtn = backdrop.querySelector('#mcp-delete');
  if (deleteBtn) {
    deleteBtn.addEventListener('click', async () => {
      try {
        await api(`/api/mcp/servers/${encodeURIComponent(server.id)}`, { method: 'DELETE' });
        toast(`${server.name} removed`, 'success');
        await refreshMCPServers();
        close();
        renderMCPGrid(grid);
      } catch (e) {
        toast(`Failed: ${e.message}`, 'error');
      }
    });
  }
}

function openMCPTemplateModal(item, grid) {
  const tmpl = item._template;
  if (!tmpl) return;

  const root = document.getElementById('modal-root') || document.body;
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';

  // Build env fields
  let envHtml = '';
  if (tmpl.env_keys && tmpl.env_keys.length > 0) {
    envHtml = tmpl.env_keys.map(key => `
      <div class="modal-field">
        <label>${key.replace(/_/g, ' ')}</label>
        <input type="password" data-env-key="${key}" placeholder="Enter ${key.replace(/_/g, ' ')}..." autocomplete="off"/>
      </div>
    `).join('');
  }

  // Build args display
  const argsStr = tmpl.args.join(' ');

  backdrop.innerHTML = `
    <div class="modal" style="max-width:440px">
      <div class="modal-header">
        <div style="display:flex;align-items:center;gap:10px">
          <div style="width:32px;height:32px;border-radius:8px;background:${tmpl.cat === 'Built-in' ? '#06b6d410' : '#8b5cf610'};display:flex;align-items:center;justify-content:center">${tmpl.icon}</div>
          <div>
            <div style="font-weight:700;font-size:14px">Add ${tmpl.name}</div>
            <div style="font-size:11px;color:var(--tx3)">${tmpl.cat} MCP Server</div>
          </div>
        </div>
        <button class="modal-close" id="tmpl-modal-close">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="modal-body">
        <div style="font-size:12px;color:var(--tx2);line-height:1.5;margin-bottom:12px">${tmpl.description}</div>
        <div class="modal-field">
          <label>Transport</label>
          <input type="text" value="${tmpl.transport}" disabled style="opacity:0.6"/>
        </div>
        <div class="modal-field">
          <label>Command</label>
          <input type="text" value="${tmpl.command}" disabled style="opacity:0.6"/>
        </div>
        <div class="modal-field">
          <label>Arguments</label>
          <input type="text" value="${argsStr}" disabled style="opacity:0.6;font-family:monospace;font-size:11px"/>
        </div>
        ${envHtml}
        <div class="modal-field" style="margin-top:8px">
          <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
            <input type="checkbox" id="tmpl-auto-connect" checked style="accent-color:#06b6d4"/>
            Auto-connect on startup
          </label>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn ghost" id="tmpl-cancel">Cancel</button>
        <button class="btn primary" id="tmpl-add">Add & Connect</button>
      </div>
    </div>`;

  document.body.appendChild(backdrop);
  requestAnimationFrame(() => backdrop.classList.add('visible'));

  const close = () => { backdrop.classList.remove('visible'); setTimeout(() => backdrop.remove(), 200); };
  backdrop.querySelector('#tmpl-modal-close').addEventListener('click', close);
  backdrop.querySelector('#tmpl-cancel').addEventListener('click', close);
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });

  backdrop.querySelector('#tmpl-add').addEventListener('click', async () => {
    const addBtn = backdrop.querySelector('#tmpl-add');
    addBtn.textContent = 'Adding...';
    addBtn.disabled = true;

    // Collect env vars
    const env = {};
    backdrop.querySelectorAll('[data-env-key]').forEach(inp => {
      if (inp.value) env[inp.dataset.envKey] = inp.value;
    });
    const autoConnect = backdrop.querySelector('#tmpl-auto-connect').checked;

    try {
      await api('/api/mcp/servers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: tmpl.name,
          transport: tmpl.transport,
          command: tmpl.command,
          args: tmpl.args,
          env: env,
          description: tmpl.description,
          category: 'builtin',
          auto_connect: autoConnect,
        }),
      });

      // Connect immediately
      try {
        await api(`/api/mcp/servers/${encodeURIComponent(tmpl.id)}/connect`, { method: 'POST' });
        toast(`${tmpl.name} added and connected successfully`, 'success');
      } catch (e) {
        toast(`${tmpl.name} added but connection failed: ${e.message}`, 'warn');
      }

      await refreshMCPServers();
      close();
      renderMCPGrid(grid);
    } catch (e) {
      toast(`Failed to add: ${e.message}`, 'error');
      addBtn.textContent = 'Retry';
      addBtn.disabled = false;
    }
  });
}

function openCustomMCPServerModal(grid) {
  const root = document.getElementById('modal-root') || document.body;
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';

  backdrop.innerHTML = `
    <div class="modal" style="max-width:480px">
      <div class="modal-header">
        <div style="display:flex;align-items:center;gap:10px">
          <div style="width:32px;height:32px;border-radius:8px;background:#8b5cf610;display:flex;align-items:center;justify-content:center">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
          </div>
          <div>
            <div style="font-weight:700;font-size:14px">Add Custom MCP Server</div>
            <div style="font-size:11px;color:var(--tx3)">Connect to any MCP-compatible server</div>
          </div>
        </div>
        <button class="modal-close" id="custom-modal-close">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="modal-body">
        <div class="modal-field">
          <label>Name *</label>
          <input type="text" id="custom-name" placeholder="My Custom MCP Server" autocomplete="off"/>
        </div>
        <div class="modal-field">
          <label>Transport</label>
          <select id="custom-transport" style="width:100%;padding:8px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg-2);color:var(--text);font-size:12px">
            <option value="sse">SSE (Remote HTTP Server)</option>
            <option value="stdio">stdio (Local Process)</option>
          </select>
        </div>
        <div class="modal-field" id="custom-url-field">
          <label>Server URL (for SSE)</label>
          <input type="text" id="custom-url" placeholder="http://localhost:3000" autocomplete="off"/>
        </div>
        <div id="custom-stdio-fields" style="display:none">
          <div class="modal-field">
            <label>Command</label>
            <input type="text" id="custom-command" placeholder="npx" autocomplete="off"/>
          </div>
          <div class="modal-field">
            <label>Arguments (comma separated)</label>
            <input type="text" id="custom-args" placeholder="-y, @anthropic/mcp-server-filesystem" autocomplete="off"/>
          </div>
          <div class="modal-field">
            <label>Environment Variables (JSON)</label>
            <textarea id="custom-env" rows="2" placeholder='{"API_KEY": "your-key"}'></textarea>
          </div>
        </div>
        <div class="modal-field">
          <label>Description</label>
          <input type="text" id="custom-desc" placeholder="What does this server provide?" autocomplete="off"/>
        </div>
        <div class="modal-field">
          <label>Category</label>
          <input type="text" id="custom-cat" placeholder="custom" value="custom" autocomplete="off"/>
        </div>
        <div class="modal-field" style="margin-top:4px">
          <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
            <input type="checkbox" id="custom-auto-connect" checked style="accent-color:#06b6d4"/>
            Auto-connect on startup
          </label>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn ghost" id="custom-cancel">Cancel</button>
        <button class="btn primary" id="custom-add">Add & Connect</button>
      </div>
    </div>`;

  document.body.appendChild(backdrop);
  requestAnimationFrame(() => backdrop.classList.add('visible'));

  // Toggle transport fields
  const transportSelect = backdrop.querySelector('#custom-transport');
  transportSelect.addEventListener('change', () => {
    const isSSE = transportSelect.value === 'sse';
    backdrop.querySelector('#custom-url-field').style.display = isSSE ? '' : 'none';
    backdrop.querySelector('#custom-stdio-fields').style.display = isSSE ? 'none' : '';
  });

  const close = () => { backdrop.classList.remove('visible'); setTimeout(() => backdrop.remove(), 200); };
  backdrop.querySelector('#custom-modal-close').addEventListener('click', close);
  backdrop.querySelector('#custom-cancel').addEventListener('click', close);
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });

  backdrop.querySelector('#custom-add').addEventListener('click', async () => {
    const addBtn = backdrop.querySelector('#custom-add');
    const name = backdrop.querySelector('#custom-name').value.trim();
    if (!name) {
      toast('Please enter a name', 'error');
      return;
    }

    addBtn.textContent = 'Adding...';
    addBtn.disabled = true;

    const transport = transportSelect.value;
    const url = backdrop.querySelector('#custom-url').value.trim();
    const command = backdrop.querySelector('#custom-command').value.trim();
    const argsStr = backdrop.querySelector('#custom-args').value.trim();
    const envStr = backdrop.querySelector('#custom-env').value.trim();
    const description = backdrop.querySelector('#custom-desc').value.trim();
    const category = backdrop.querySelector('#custom-cat').value.trim() || 'custom';
    const autoConnect = backdrop.querySelector('#custom-auto-connect').checked;

    let args = [];
    if (argsStr) {
      args = argsStr.split(',').map(a => a.trim()).filter(Boolean);
    }

    let env = {};
    if (envStr) {
      try {
        env = JSON.parse(envStr);
      } catch {
        toast('Invalid JSON for environment variables', 'error');
        addBtn.textContent = 'Add & Connect';
        addBtn.disabled = false;
        return;
      }
    }

    try {
      const createResp = await api('/api/mcp/servers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name, transport, url, command, args, env,
          description, category, auto_connect,
        }),
      });

      const serverId = createResp?.server?.id;
      if (serverId) {
        try {
          await api(`/api/mcp/servers/${encodeURIComponent(serverId)}/connect`, { method: 'POST' });
          toast(`${name} added and connected`, 'success');
        } catch (e) {
          toast(`${name} added but connection failed: ${e.message}`, 'warn');
        }
      }

      await refreshMCPServers();
      close();
      renderMCPGrid(grid);
    } catch (e) {
      toast(`Failed: ${e.message}`, 'error');
      addBtn.textContent = 'Add & Connect';
      addBtn.disabled = false;
    }
  });
}
