/**
 * Integrations & Triggers — Drawer overlays
 * Opens as side-drawers from topbar buttons in chat panel.
 */
import { el, api } from '../utils.js';
import { toast } from '../enhancements.js';

// ── Brand SVG logos ───────────────────────────────────────────────────────────
const LOGOS = {
  slack:        `<svg width="20" height="20" viewBox="0 0 24 24"><g fill="#E01E5A"><path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312z"/></g><g fill="#36C5F0"><path d="M18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z"/></g></svg>`,
  discord:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#5865F2" d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03z"/></svg>`,
  email:        `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><rect x="3" y="5" width="18" height="14" rx="2" fill="#EA580C"/><path d="M3 7l9 5 9-5" stroke="white" stroke-width="1.5" fill="none"/></svg>`,
  github:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#fff" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z"/><circle fill="#ff6D6D" cx="16" cy="8" r="3"/></svg>`,
  gitlab:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FC6D26" d="m21.435 9.365-9.436 13.04L2.565 9.365l2.818-8.68 2.056 6.334h9.122l2.056-6.334z"/></svg>`,
  notion:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#fff" d="M4.459 4.208c.746.606 1.026.56 2.428.466l13.215-.793c.28 0 .047-.28-.046-.326L17.86 1.968c-.42-.326-.981-.7-2.055-.607L3.01 2.295c-.466.046-.56.28-.374.466zm.793 3.08v13.904c0 .747.373 1.027 1.214.98l14.523-.84c.841-.046.935-.56.935-1.167V6.354c0-.606-.233-.933-.748-.887l-15.177.887c-.56.047-.747.327-.747.933zm14.337.745c.093.42 0 .84-.42.888l-.7.14v10.264c-.608.327-1.168.514-1.635.514-.748 0-.935-.234-1.495-.933l-4.577-7.186v6.952L12.21 19s0 .84-1.168.84l-3.222.186c-.093-.186 0-.653.327-.746l.84-.233V9.854L7.822 9.76c-.094-.42.14-1.026.793-1.073l3.456-.233 4.764 7.279v-6.44l-1.215-.14c-.093-.514.28-.887.747-.933z"/></svg>`,
  jira:         `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#0052CC" d="M11.571 11.513H0a5.218 5.218 0 0 0 5.232 5.215h2.13v2.056A5.215 5.215 0 0 0 12.575 24V12.518a1.005 1.005 0 0 0-1.004-1.005zm5.723-5.756H5.757a5.215 5.215 0 0 0 5.214 5.214h2.129v2.058a5.218 5.218 0 0 0 5.215 5.214V6.762a1.005 1.005 0 0 0-1.021-1.005zM23.013 0H11.475a5.215 5.215 0 0 0 5.215 5.215h2.129v2.057A5.215 5.215 0 0 0 24.019 12.49V1.005A1.005 1.005 0 0 0 23.013 0z"/></svg>`,
  linear:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#5E6AD2" d="M3 12l7-10 7 10H3z"/><path fill="#8A8AAA" d="M3 12l7 10 7-10" opacity="0.5"/></svg>`,
  trello:       `<svg width="20" height="20" viewBox="0 0 24 24"><rect fill="#0079BF" x="2" y="4" width="6" height="16" rx="1"/><rect fill="#0079BF" x="10" y="8" width="6" height="12" rx="1"/><rect fill="#0079BF" x="18" y="2" width="6" height="18" rx="1"/></svg>`,
  asana:        `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#F06A6A" cx="12" cy="12" r="10"/><circle fill="#fff" cx="7" cy="12" r="3"/><circle fill="#fff" cx="17" cy="12" r="3"/></svg>`,
  vercel:       `<svg width="20" height="20" viewBox="0 0 24 24"><polygon fill="#fff" points="12,2 24,22 0,22"/></svg>`,
  google_drive: `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#0F9D58" d="M6.28 2.4L2.1 9.6h5.6L11.88 2.4H6.28z"/><path fill="#4285F4" d="M21.9 9.6L17.72 2.4H12.12L16.3 9.6H21.9z"/><path fill="#FBBC05" d="M2.1 14.4l4.18 7.2h11.64l4.18-7.2H2.1z"/><path fill="#EA4335" d="M12.12 14.4l4.18 7.2h5.58l4.18-7.2h-14"/></svg>`,
  dropbox:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#0061FF" d="M6 2l6 4-6 4 6-4-6 4zM0 10l6-4 6 4-6-4-6 4zM12 22l6-4 6 4-6-4-6 4zM18 12l6-4 6 4-6-4-6 4z"/><path fill="#0061FF" d="M12 6l-6 4 6 4 6-4z" opacity="0.5"/></svg>`,
  aws_s3:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FF9900" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#252F3E" d="M12 8l-4 2.5v5L12 20l4-4.5v-5z"/><path fill="#E6E6E6" d="M8 12h8v1H8z"/></svg>`,
  airtable:     `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#18BFFF" d="M11.97 2.273 2.277 6.026a.546.546 0 0 0 0 1.01l9.695 3.758a.546.546 0 0 0 .39 0l9.694-3.756a.546.546 0 0 0 0-1.01L12.36 2.272a.546.546 0 0 0-.39 0z"/><path fill="#fff" d="M4 10h4v4H4zM10 8h4v8h-4zM16 6h4v12h-4z" opacity="0.8"/></svg>`,
  google_sheets: `<svg width="20" height="20" viewBox="0 0 24 24"><rect fill="#34A853" x="3" y="3" width="18" height="18" rx="2"/><rect fill="#fff" x="6" y="7" width="12" height="2"/><rect fill="#fff" x="6" y="11" width="12" height="2"/><rect fill="#fff" x="6" y="15" width="8" height="2"/></svg>`,
  hubspot:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FF7A59" d="M18.164 7.931V5.085a2.198 2.198 0 1 0-1.672 0v2.846a6.413 6.413 0 0 0-2.914 1.433l-7.52-5.854a2.481 2.481 0 1 0-1.04 1.326l7.313 5.691a6.413 6.413 0 0 0-.876 3.261c0 1.054.253 2.047.706 2.921l-2.303 2.303a2.195 2.195 0 1 0 1.18 1.18l2.283-2.283a6.41 6.41 0 0 0 3.173.838 6.434 6.434 0 0 0 1.67-12.616z"/></svg>`,
  salesforce:   `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#00A1E0" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M7 10c0-2.76 2.24-5 5-5s5 2.24 5 5c0 2.21-1.44 4.08-3.39 4.72L12 16l-2-1.28A4.97 4.97 0 0 1 7 10z"/></svg>`,
  openai:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#10A37A" d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91 6.046 6.046 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a5.985 5.985 0 0 0-3.998 2.9 6.046 6.046 0 0 0 .743 7.097 5.98 5.98 0 0 0 .51 4.911 6.051 6.051 0 0 0 6.515 2.9A5.985 5.985 0 0 0 13.26 24a6.056 6.056 0 0 0 5.772-4.206 5.99 5.99 0 0 0 3.997-2.9 6.056 6.056 0 0 0-.747-7.073z"/></svg>`,
  zapier:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FF4A00" d="M12 2L2 12l10 10 10-10L12 2z"/><path fill="#FF4A00" d="M17 7l-5 5 5 5M7 7l5 5-5 5" stroke="#fff" stroke-width="2"/></svg>`,
  grafana:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#F46800" d="M12 2L2 7v10l10 5 10-5V7L12 2z"/><rect fill="#F46800" x="7" y="10" width="10" height="5" rx="1"/><rect x="4" y="17" width="4" height="5" rx="1" fill="#F46800"/><rect x="16" y="17" width="4" height="5" rx="1" fill="#F46800"/></svg>`,
  pagerduty:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#06AC38" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><rect fill="#fff" x="9" y="8" width="6" height="8" rx="3"/></svg>`,
  posthog:     `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#FF9885" cx="9" cy="9" r="5"/><circle fill="#FFDE00" cx="15" cy="9" r="5"/><circle fill="#FF9885" cx="9" cy="15" r="5"/><circle fill="#FFDE00" cx="15" cy="15" r="5"/><circle fill="#FF5C00" cx="12" cy="12" r="3"/></svg>`,
  amplitude:   `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#8257F5" d="M12 2L4 8v8l8 6 8-6V8L12 2z"/><path fill="#fff" d="M12 6v4l4 3-4 3v4l-4-3 4-3V6l4 3-4-3z" opacity="0.3"/><path fill="#fff" d="M12 10l-2 2 2 2 2-2z"/></svg>`,
  mixpanel:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#7B2FF5" d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z"/><circle fill="#fff" cx="7" cy="7" r="2"/><circle fill="#fff" cx="17" cy="17" r="2"/></svg>`,
  heap:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#F52E5A" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M8 10h8v1H8zM8 13h8v1H8zM8 16h4v1H8z"/></svg>`,
  hotjar:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FFDE00" d="M12 2L2 12l10 10 10-10L12 2z"/><path fill="#F52E5A" d="M8 8l8 8M16 8l-8 8M8 16l8-8" stroke="#F52E5A" stroke-width="2"/><circle fill="#fff" cx="12" cy="12" r="3"/></svg>`,
  terraform:  `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#7B42BC" d="M12 2L2 12l10 10 10-10-5-5v10l5 5 10-10-10-10z"/><path fill="#fff" d="M7 7l5-5 5 5-5 5z" opacity="0.5"/></svg>`,
  ansible:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#EE0000" d="M12 2L6 7v10l6 5 6-5V7l-6-5z"/><path fill="#fff" d="M12 6l-3 2.5v3L12 15l3-3.5v-3z"/><path fill="#fff" d="M9 10h6M9 13h3"/></svg>`,
  kubernetes: `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#326CE5" d="M12 2L2 12h3v9h5v-6h2v6h5v-9h3z"/><circle fill="#fff" cx="12" cy="12" r="3"/><circle fill="#326CE5" cx="12" cy="12" r="2"/><circle fill="#fff" cx="6" cy="19" r="1"/><circle fill="#fff" cx="18" cy="19" r="1"/></svg>`,
  docker:     `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#2496ED" d="M12 2L2 12h3v9h5v-6h2v6h5v-9h3z"/><path fill="#fff" d="M7 8h2v2H7zM11 8h2v2h-2zM15 8h2v2h-2z" opacity="0.8"/></svg>`,
  harbor:     `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#0F1689" d="M12 2L2 8v8c0 5.52 4.48 10 10 10s10-4.48 10-10V8L12 2z"/><path fill="#fff" d="M12 6v4l3 2-3 2V18l-4-3 4-3v-4z"/><circle fill="#fff" cx="12" cy="12" r="2"/></svg>`,
  portainer:  `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#15A0BF" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M7 7h4v4H7zM13 7h4v4h-4zM7 13h4v4H7zM13 13h4v4h-4z"/></svg>`,
  cloudflare:  `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#F38020" d="M12 2C8 2 4 4 4 7c0 1.5.5 2.8 1.3 3.8A4 4 0 0 0 5 15a4 4 0 0 0 6.8 2.3A5 5 0 0 0 15 14a5 5 0 0 0 3.7-1.7A4 4 0 0 0 20 9c0-2.2-1.8-4-4-4-.5 0-1 .1-1.5.3C14.2 3.5 13.2 2.5 12 2z"/><path fill="#FAae40" d="M12 6c-2 0-3 1-3 2.5S10 11 12 11s3-1 3-2.5S14 6 12 6z"/></svg>`,
  route53:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FF9900" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#252F3E" d="M8 9h8v1H8zM8 12h8v1H8zM8 15h3v1H8z"/></svg>`,
  godaddy:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FFB32D" d="M12 2L3 7v10l9 5 9-5V7l-9-5z"/><path fill="#fff" d="M12 6l-4 2v4l4-2 4 2V8l-4-2z"/><path fill="#fff" d="M12 10l-2 1.5v3l2-1.5 2 1.5v-3z"/></svg>`,
  namecheap:  `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#0085FF" d="M12 2C8 2 5 4 5 7v10c0 3 3 5 7 5s7-2 7-5V7c0-3-3-5-7-5z"/><path fill="#fff" d="M8 8h8v2H8zM8 12h8v1H8zM8 15h4v1H8z"/></svg>`,
  letsencrypt: `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FF5A00" d="M12 2L6 7v10l6 5 6-5V7l-6-5z"/><path fill="#fff" d="M12 6v4l3 2-3 2v4l-3-2 3-2V6l-3 2 3-2z"/></svg>`,
  zerossl:   `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#4CAF50" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#fff" d="M9 10l2 2 4-4-2 2H9z"/><path fill="#fff" d="M10 13l-2 4 2-2h4z" opacity="0.7"/></svg>`,
  aws:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FF9900" d="M12 2L4 7l8 5 8-5-8-5z"/><path fill="#252F3E" d="M8 9l4 2.5L16 9v2l-8 5-8-5z"/><path fill="#E6E6E6" d="M9 11h6v1H9zM9 14h4v1H9z" opacity="0.5"/></svg>`,
  gcp:       `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#EA4335" cx="7" cy="9" r="4"/><circle fill="#FBBC05" cx="12" cy="9" r="4"/><circle fill="#34A853" cx="10" cy="14" r="4"/><circle fill="#4285F4" cx="14" cy="14" r="4"/></svg>`,
  azure:     `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#0078D4" d="M3 7l9-3 9 3v10l-9 3-9-3z"/><path fill="#fff" d="M12 10v4M9 7l3 5 3-5"/></svg>`,
  vercel:    `<svg width="20" height="20" viewBox="0 0 24 24"><polygon fill="#fff" points="12,2 24,22 0,22"/></polygon><polygon fill="#000" points="12,6 20,18 4,18"/></svg>`,
  netlify:  `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#00C2C2" d="M12 2L2 22h20L12 2z"/><path fill="#fff" d="M9 10h6v8H9z"/></svg>`,
  supabase: `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#3ECF8E" d="M12 2L3 9v6l9 7 9-7V9L12 2z"/><path fill="#1A1A1A" d="M12 6l-4 3v6l4 3 4-3v-6z"/></svg>`,
  firebase: `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FFCA28" d="M4 12l4-9h12l-4 9-4 7z"/><path fill="#F4511E" d="M14 10l-4 4"/><path fill="#039BE5" d="M14 10l4-7"/></svg>`,
  dockerhub: `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#2496ED" d="M12 2L2 12h3v9h5v-6h2v6h5v-9h3z"/><path fill="#fff" d="M7 8h2v2H7zM11 8h2v2h-2z"/></svg>`,
  ghcr:     `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#fff" d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2z"/><path fill="#000" d="M8 7h8v2H8zM8 11h8v2H8zM8 15h4v2H8z"/></svg>`,
  appstore: `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#fff" d="M18 5h-6l-4-4-4 4H2c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2z"/><path fill="#000" d="M15 8l-3 3v5h2l-2 3-3-3v-5l3-3z" opacity="0.6"/></svg>`,
  googleplay:`<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FA5000" d="M12 2L4 6l8 8 8-8-8-4z"/><path fill="#34A853" d="M4 10l8 8 8-8z"/><path fill="#0050DD" d="M4 18l8-8 8 8z"/><path fill="#F50100" d="M12 10l8 8z"/></svg>`,
  sentry:   `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#362D59" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M16 10H8v4h8zM8 16h8v2H8z"/></svg>`,
  schedule:  `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#8b5cf6" cx="12" cy="12" r="10"/><path fill="#fff" d="M12 6v6l4 2" stroke="#fff" stroke-width="2" stroke-linecap="round"/><circle fill="#fff" cx="12" cy="12" r="2"/></svg>`,
  webhook:   `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#06b6d4" cx="12" cy="12" r="10"/><path fill="#fff" d="M12 2c-2.8 0-5 2.2-5 5v3h3M12 22c2.8 0 5-2.2 5-5v-3h-3M2 12c0-2.8 2.2-5 5-5h3v3M22 12c0 2.8-2.2 5-5 5h-3v-3"/></svg>`,
  schedule_hr:  `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#f59e0b" cx="12" cy="12" r="10"/><path fill="#fff" d="M12 6v6l3 2" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>`,
  schedule_daily: `<svg width="20" height="20" viewBox="0 0 24 24"><rect fill="#10b981" x="2" y="4" width="20" height="16" rx="2"/><path fill="#fff" d="M2 8h20M7 2v4M17 2v4M8 13h3v3H8zM14 13h3v3h-3z" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/></svg>`,
  youtube:     `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FF0000" d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>`,
  // ── Missing Messaging logos ──────────────────────────────────
  telegram:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#26A5E4" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M5.43 11.78l10.07-3.78c.47-.17.88.11.73.83l-1.71 8.06c-.12.54-.44.67-.89.42l-2.46-1.81-1.19 1.14c-.13.13-.24.24-.5.24l.18-2.5 4.57-4.13c.2-.18-.04-.28-.31-.1l-5.65 3.56-2.44-.76c-.53-.17-.54-.53.11-.78z"/></svg>`,
  microsoft_teams:`<svg width="20" height="20" viewBox="0 0 24 24"><rect fill="#6264A7" x="2" y="6" width="14" height="14" rx="2"/><path fill="#fff" d="M6 10h6v1H6zM6 13h4v1H6z"/><circle fill="#6264A7" cx="19" cy="8" r="3"/><rect fill="#7B83EB" x="16" y="13" width="6" height="7" rx="2"/></svg>`,
  // ── Missing Dev logos ───────────────────────────────────────
  bitbucket:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#2684FF" d="M2 4l2.18 16.06L12 22l7.82-1.94L22 4H2zm14.64 12.62H7.36l-1.28-7.5h11.84l-1.28 7.5z"/><path fill="#2684FF" d="M10.22 13.36l-.4-4.24h4.36l-.4 4.24H10.22z" opacity="0.6"/></svg>`,
  circleci:       `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#161621" cx="12" cy="12" r="10"/><circle fill="none" stroke="#fff" stroke-width="2" cx="12" cy="12" r="7"/><circle fill="#06B6D4" cx="15" cy="12" r="2.5"/></svg>`,
  jenkins:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#D33833" d="M12 2C8 2 5 4.5 5 7.5V12l3.5 2.5L12 22l3.5-7.5L19 12V7.5C19 4.5 16 2 12 2z"/><path fill="#fff" d="M10 7h4v1h-4zM10 10h4v1h-4z" opacity="0.8"/></svg>`,
  // ── Missing Productivity logos ───────────────────────────────
  confluence:     `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#172B4D" d="M4 14c2-3.5 5-6 8-7 3 1 6 3.5 8 7-2 3.5-5 6-8 7-3-1-6-3.5-8-7z"/><path fill="#fff" d="M8 12l4-3 4 3-4 3z" opacity="0.7"/></svg>`,
  clickup:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#7B68EE" d="M5 17l4-4 3 3 7-9v7l-7 6-3-3z"/><path fill="#7B68EE" d="M5 10l4-4 3 3 7-6v2l-7 6-3-3z" opacity="0.6"/></svg>`,
  monday:         `<svg width="20" height="20" viewBox="0 0 24 24"><rect fill="#FF3D57" x="2" y="4" width="4" height="16" rx="1"/><rect fill="#FF6B35" x="8" y="8" width="4" height="12" rx="1"/><rect fill="#FFB800" x="14" y="6" width="4" height="14" rx="1"/><rect fill="#00CA72" x="20" y="10" width="2" height="10" rx="1"/></svg>`,
  basecamp:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#1D2D35" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#57D163" d="M6 16c1.5-5 4-8 6-8s4.5 3 6 8" fill="none" stroke="#57D163" stroke-width="2"/><path fill="none" stroke="#fff" d="M6 16c2-3 4.5-5 6-5s4 2 6 5" stroke-width="1.5" opacity="0.5"/></svg>`,
  // ── Missing Storage logos ────────────────────────────────────
  backblaze:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#E21E26" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#fff" d="M9 9l3-2 3 2-3 6z" opacity="0.8"/></svg>`,
  wasabi:         `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#56B847" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#fff" d="M9 10h6v4H9z" opacity="0.8"/><circle fill="#fff" cx="12" cy="8" r="1.5"/></svg>`,
  // ── Missing CRM logos ────────────────────────────────────────
  pipedrive:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#2D4E74" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M7 10l5-4 5 4-5 8z"/></svg>`,
  zoho:           `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#D62828" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#fff" d="M8 8l4-2 4 2v6l-4 4-4-4z" opacity="0.8"/><path fill="#D62828" d="M10 10l2-1 2 1v2l-2 2-2-2z"/></svg>`,
  close:          `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#2B5AED" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M8 8l8 8M16 8l-8 8" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/></svg>`,
  // ── Missing AI logos ─────────────────────────────────────────
  anthropic:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#D4A574" d="M2 20L9 4h2l-7 16H2zm11 0L20 4h2l-7 16h-2z"/><path fill="#D4A574" d="M9 4l1.5 4-2 5L6 20H2L9 4z" opacity="0.5"/></svg>`,
  cohere:         `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#39594D" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M8 8h3v8H8zM13 8h3v8h-3z" opacity="0.9"/></svg>`,
  google_ai:      `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#EA4335" cx="6" cy="6" r="3"/><circle fill="#FBBC05" cx="12" cy="6" r="3"/><circle fill="#34A853" cx="18" cy="6" r="3"/><circle fill="#4285F4" cx="9" cy="12" r="3"/><circle fill="#EA4335" cx="15" cy="12" r="3"/><circle fill="#FBBC05" cx="12" cy="18" r="3"/></svg>`,
  huggingface:    `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#FFD21E" cx="12" cy="12" r="10"/><circle fill="#1a1a1a" cx="9" cy="10" r="1.5"/><circle fill="#1a1a1a" cx="15" cy="10" r="1.5"/><path fill="none" stroke="#1a1a1a" stroke-width="1.5" d="M8 14c1 1.5 3 2.5 4 2.5s3-1 4-2.5"/><path fill="#FF9D00" d="M6 7l3-2v3zM18 7l-3-2v3z" opacity="0.6"/></svg>`,
  replicate:      `<svg width="20" height="20" viewBox="0 0 24 24"><rect fill="#000" x="2" y="2" width="9" height="9" rx="2"/><rect fill="#000" x="13" y="2" width="9" height="9" rx="2" opacity="0.6"/><rect fill="#000" x="2" y="13" width="9" height="9" rx="2" opacity="0.6"/><rect fill="#000" x="13" y="13" width="9" height="9" rx="2" opacity="0.3"/></svg>`,
  stabilityai:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#A0A0A0" d="M12 2L2 12l10 10 10-10L12 2z"/><path fill="#fff" d="M12 6l-4 4h8z"/><path fill="#fff" d="M8 14h8l-4 4z" opacity="0.6"/></svg>`,
  // ── Missing Automation logos ─────────────────────────────────
  make:           `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#6D00CC" d="M12 2L2 7v10l10 5 10-5V7L12 2z"/><path fill="#fff" d="M12 6l4 2.5v5L12 16l-4-2.5v-5z"/></svg>`,
  n8n:            `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#EA4B71" d="M4 8a4 4 0 1 1 4 4H4V8z"/><path fill="#EA4B71" d="M12 12a4 4 0 1 1 4 4h-4v-4z" opacity="0.7"/><circle fill="#EA4B71" cx="18" cy="6" r="3" opacity="0.5"/></svg>`,
  ifttt:          `<svg width="20" height="20" viewBox="0 0 24 24"><rect fill="#000" x="2" y="4" width="6" height="16" rx="1"/><rect fill="#000" x="9" y="4" width="6" height="16" rx="1"/><rect fill="#000" x="16" y="4" width="6" height="16" rx="1"/><path fill="#fff" d="M4 8h2v8H4zM11 8h2v8h-2zM18 8h2v8h-2z" opacity="0.3"/></svg>`,
  // ── Missing Monitoring logos ─────────────────────────────────
  datadog:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#632CA6" d="M5 7l7-4 7 4v9l-7 5-7-5V7z"/><path fill="#fff" d="M9 9h6v5H9z" opacity="0.8"/><path fill="#632CA6" d="M10 10h4v3h-4z" opacity="0.5"/></svg>`,
  newrelic:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#1CE783" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#0D1B2A" d="M8 8h8v8H8z"/><path fill="#1CE783" d="M10 10h4v4h-4z"/></svg>`,
  pingdom:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#00B74A" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#fff" d="M12 7l3 5-3 5-3-5z"/></svg>`,
  uptimerobot:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#2ECC71" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M7 12l3 3 7-7" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>`,
  statuspage:     `<svg width="20" height="20" viewBox="0 0 24 24"><rect fill="#1F2937" x="2" y="4" width="20" height="16" rx="3"/><circle fill="#10b981" cx="7" cy="12" r="2"/><rect fill="#fff" x="11" y="10" width="8" height="4" rx="1" opacity="0.7"/></svg>`,
  // ── Missing Analytics logos ──────────────────────────────────
  segment:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#52BD95" d="M3 6l9-3 9 3v2l-9 3-9-3V6z"/><path fill="#52BD95" d="M3 12l9 3 9-3v2l-9 3-9-3v-2z" opacity="0.7"/><path fill="#52BD95" d="M3 18l9 3 9-3" opacity="0.4" stroke="#52BD95" stroke-width="2" fill="none"/></svg>`,
  plausible:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#5850EC" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M7 14l3-4 3 2 4-5" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>`,
  // ── Missing Domain logos ─────────────────────────────────────
  porkbun:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#EF8547" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><circle fill="#fff" cx="9" cy="10" r="2" opacity="0.8"/><circle fill="#fff" cx="15" cy="10" r="2" opacity="0.8"/><circle fill="#fff" cx="12" cy="15" r="1.5" opacity="0.6"/></svg>`,
  // ── Missing Hosting logos ────────────────────────────────────
  hostinger:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#6C3BA8" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#fff" d="M9 8l3-2 3 2-3 8z" opacity="0.8"/></svg>`,
  railway:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#0B0D0E" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#fff" d="M8 9h8l-4 8z" opacity="0.9"/></svg>`,
  render:         `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#46E3B7" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#000" d="M9 9l3-2 3 2v4l-3 4-3-4z" opacity="0.3"/></svg>`,
  flyio:          `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#7B3BE2" d="M12 2L2 12l10 10 10-10L12 2z"/><path fill="#fff" d="M12 6l-4 4h8z" opacity="0.8"/><path fill="#fff" d="M8 14h8l-4 4z" opacity="0.5"/></svg>`,
  heroku:         `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#430098" d="M4 2h16a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/><path fill="#fff" d="M7 17l4-5-4-5h3l4 5-4 5H7zM14 17l4-5-4-5h2l3 5-3 5z" opacity="0.9"/></svg>`,
  digitalocean:   `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#0080FF" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#fff" d="M6 12h4v4H6zM12 6h4v4h-4z" opacity="0.7"/><path fill="#fff" d="M12 14h6v2h-6z" opacity="0.5"/></svg>`,
  cloudways:      `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#2B8FDE" d="M4 8c0-3 3-5 7-5s7 2 7 5c0 2-1 3-2 4 1 0 2 2 2 4 0 3-3 5-7 5s-7-2-7-5c0-1.5.5-3 2-4-1-1-2-2-2-4z"/><path fill="#fff" d="M9 9h6v2H9zM9 13h6v2H9z" opacity="0.8"/></svg>`,
  // ── Missing Database logos ───────────────────────────────────
  planetable:     `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#000" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><ellipse fill="none" stroke="#fff" stroke-width="1.5" cx="12" cy="12" rx="5" ry="10" opacity="0.6"/><line fill="none" stroke="#fff" stroke-width="1" x1="2" y1="12" x2="22" y2="12" opacity="0.4"/></svg>`,
  // ── Missing Communication logos ──────────────────────────────
  twilio:         `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#F22F46" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><circle fill="#fff" cx="9" cy="9" r="2"/><circle fill="#fff" cx="15" cy="9" r="2"/><circle fill="#fff" cx="9" cy="15" r="2"/><circle fill="#fff" cx="15" cy="15" r="2"/></svg>`,
  sendgrid:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#1A82E2" d="M2 8l10-4 10 4v2l-10 4L2 10V8z"/><path fill="#1A82E2" d="M2 14l10 4 10-4" opacity="0.6" stroke="#1A82E2" stroke-width="2" fill="none"/><path fill="#1A82E2" d="M2 18l10 4 10-4" opacity="0.3" stroke="#1A82E2" stroke-width="2" fill="none"/></svg>`,
  mailgun:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#2185C4" d="M12 2L4 7v10l8 5 8-5V7l-8-5z"/><path fill="#fff" d="M9 9h6v6H9z" opacity="0.8"/><path fill="#2185C4" d="M10 10h4v4h-4z" opacity="0.4"/></svg>`,
  postmark:       `<svg width="20" height="20" viewBox="0 0 24 24"><rect fill="#000" x="3" y="5" width="18" height="14" rx="2"/><circle fill="#fff" cx="12" cy="12" r="4" opacity="0.8"/><path fill="#000" d="M10 12l2-2 2 2-2 2z"/></svg>`,
  // ── Missing Payments logos ───────────────────────────────────
  // ── Missing E-commerce logos ─────────────────────────────────
  shopify:        `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#96BF48" d="M12 2C7 2 3 6 3 10c0 5 4 8 9 12 5-4 9-7 9-12 0-4-4-8-9-8z"/><path fill="#fff" d="M9 8h6l-1 6H10z" opacity="0.8"/><circle fill="#fff" cx="10" cy="16" r="1"/><circle fill="#fff" cx="14" cy="16" r="1"/></svg>`,
  woocommerce:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#96588A" d="M3 6h18v10a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V6z"/><circle fill="#fff" cx="9" cy="12" r="3" opacity="0.8"/><path fill="#96588A" d="M9 10l2 2-2 2z"/><circle fill="#fff" cx="16" cy="12" r="2" opacity="0.6"/></svg>`,
  bigcommerce:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#34313F" d="M4 4h16v16H4z"/><path fill="#fff" d="M8 8l4 4-4 4M13 16h4" stroke="#fff" stroke-width="2" stroke-linecap="round" fill="none"/></svg>`,
  // ── Missing Payments logos ─────────────────────────────────
  stripe:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#6772E5" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><rect fill="#fff" x="7" y="9" width="10" height="6" rx="1"/><path fill="#6772E5" d="M9.5 10.5h5v1h-5zM9.5 12.5h5v1h-5z"/></svg>`,
  lemon_squeezy: `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#FFD700" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path fill="#1a1a1a" d="M7 10l5-3 5 3-5 8z"/></svg>`,
  paypal:       `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#003087" d="M7.076 21.337H2.47a.641.641 0 0 1-.633-.74L4.944.901C5.026.382 5.474 0 5.998 0h7.46c2.57 0 4.578.543 5.69 1.81 1.01 1.15 1.304 2.42 1.012 4.287-.023.143-.047.288-.077.437-.983 5.05-4.349 6.797-8.647 6.797h-2.19c-.524 0-.968.382-1.05.9l-1.12 7.106z"/><path fill="#009FE3" d="M14.536 4.31c.873.234 1.473.87 1.473 2.498 0 .537-.074 1.041-.22 1.509-.58 1.867-1.91 2.515-3.904 2.515h-2.21c-.524 0-.968.382-1.05.9L6.82 19.034H8.96c.528 0 .97-.38 1.053-.896l.022-.208.52-3.29.034-.127c.08-.515.524-.895 1.05-.895h.662c2.05 0 3.655-.828 4.13-3.227.215-1.09.104-1.957-.526-2.588-.14-.14-.303-.26-.478-.362.104-.17.178-.363.215-.572.129-.64.006-1.155-.28-1.544z"/></svg>`,
  planetable:  `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#000" cx="12" cy="12" r="10"/><path fill="#fff" d="M7 7l10 10M7 7h5v-5M12 2v5l-5 5H2"/><path fill="#2B6666" d="M12 2v5l5 5" opacity="0.7"/></svg>`,
  // ── Missing Media logos ─────────────────────────────────

  // ── Trigger logos ────────────────────────────────────────────
  new_email:      `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><rect x="3" y="5" width="18" height="14" rx="2" fill="#EA580C"/><path d="M3 7l9 5 9-5" stroke="white" stroke-width="1.5" fill="none"/><circle fill="#10b981" cx="18" cy="5" r="4"/><path fill="#fff" d="M16 5h4" stroke="#fff" stroke-width="1.5"/></svg>`,
  new_message:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#E01E5A" d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52z"/><circle fill="#10b981" cx="18" cy="5" r="4"/><path fill="#fff" d="M16 4h4v2h-4z"/></svg>`,
  new_commit:     `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#fff" cx="12" cy="12" r="10"/><path fill="#000" d="M12 6C8.69 6 6 8.69 6 12s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm0 9a3 3 0 1 1 0-6 3 3 0 0 1 0 6z"/><circle fill="#10b981" cx="18" cy="5" r="4"/><path fill="#fff" d="M16.5 4.5h3v1h-3z"/></svg>`,
  new_issue:      `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#fff" cx="12" cy="12" r="10"/><path fill="#000" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.84.004 1.69.113 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z"/><circle fill="#ef4444" cx="18" cy="5" r="4"/><path fill="#fff" d="M16 4h4v2h-4z"/></svg>`,
  file_changed:   `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#0F9D58" d="M6.28 2.4L2.1 9.6h5.6L11.88 2.4H6.28z"/><path fill="#4285F4" d="M21.9 9.6L17.72 2.4H12.12L16.3 9.6H21.9z"/><circle fill="#f59e0b" cx="18" cy="5" r="4"/><path fill="#fff" d="M16.5 4.5h3v1h-3z"/></svg>`,
  discord_msg:    `<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#5865F2" d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03z"/><circle fill="#10b981" cx="18" cy="4" r="3.5"/><path fill="#fff" d="M16.5 3.5h3v1h-3z"/></svg>`,
  posthog_event:  `<svg width="20" height="20" viewBox="0 0 24 24"><circle fill="#FF9885" cx="9" cy="9" r="5"/><circle fill="#FFDE00" cx="15" cy="9" r="5"/><circle fill="#FF9885" cx="9" cy="15" r="5"/><circle fill="#FFDE00" cx="15" cy="15" r="5"/><circle fill="#FF5C00" cx="12" cy="12" r="3"/><circle fill="#10b981" cx="19" cy="4" r="3.5"/><path fill="#fff" d="M17.5 3.5h3v1h-3z"/></svg>`,
  amplitude_event:`<svg width="20" height="20" viewBox="0 0 24 24"><path fill="#8257F5" d="M12 2L4 8v8l8 6 8-6V8L12 2z"/><circle fill="#10b981" cx="19" cy="4" r="3.5"/><path fill="#fff" d="M17.5 3.5h3v1h-3z"/></svg>`,
};

const INTEGRATIONS = [
  // ── Messaging ───────────────────────────────────────────
  { id: 'slack',        name: 'Slack',        cat: 'Messaging',    bg: '#611f6910', fields: ['webhook_url','bot_token','channel'] },
  { id: 'discord',      name: 'Discord',      cat: 'Messaging',    bg: '#5865F210', fields: ['webhook_url','bot_token'] },
  { id: 'email',        name: 'Email / SMTP', cat: 'Messaging',    bg: '#EA580C10', fields: ['smtp_host','smtp_port','username','password','from_email'], oauthFields: ['oauth_client_id','oauth_client_secret'] },
  { id: 'telegram',     name: 'Telegram',     cat: 'Messaging',    bg: '#26A5E410', fields: ['bot_token','chat_id'] },
  { id: 'microsoft_teams', name: 'MS Teams',  cat: 'Messaging',    bg: '#6264A710', fields: ['webhook_url','tenant_id'] },
  // ── Dev ─────────────────────────────────────────────────
  { id: 'github',       name: 'GitHub',       cat: 'Dev',          bg: '#ffffff08', fields: ['api_key','repo'] },
  { id: 'gitlab',       name: 'GitLab',       cat: 'Dev',          bg: '#FC6D2610', fields: ['api_key','base_url'] },
  { id: 'bitbucket',   name: 'Bitbucket',   cat: 'Dev',          bg: '#2684FF10', fields: ['api_key','workspace'] },
  { id: 'jira',         name: 'Jira',         cat: 'Dev',          bg: '#0052CC10', fields: ['api_key','base_url','email'] },
  { id: 'linear',       name: 'Linear',        cat: 'Dev',          bg: '#5E6AD210', fields: ['api_key','team_id'] },
  { id: 'trello',       name: 'Trello',       cat: 'Dev',          bg: '#0079BF10', fields: ['api_key','token'] },
  { id: 'asana',        name: 'Asana',         cat: 'Dev',          bg: '#F06A6A10', fields: ['api_key','workspace_id'] },
  { id: 'circleci',   name: 'CircleCI',    cat: 'Dev',          bg: '#16162110', fields: ['api_token','org_slug'] },
  { id: 'jenkins',    name: 'Jenkins',     cat: 'Dev',          bg: '#D3383310', fields: ['url','username','api_token'] },
  // ── Productivity ────────────────────────────────────────
  { id: 'notion',       name: 'Notion',       cat: 'Productivity', bg: '#ffffff08', fields: ['api_key','database_id'] },
  { id: 'confluence',  name: 'Confluence',  cat: 'Productivity', bg: '#172B4D10', fields: ['api_key','base_url','space_key'] },
  { id: 'clickup',     name: 'ClickUp',     cat: 'Productivity', bg: '#7B68EE10', fields: ['api_key','team_id'] },
  { id: 'monday',      name: 'Monday.com',  cat: 'Productivity', bg: '#FF3D5710', fields: ['api_key','board_id'] },
  { id: 'basecamp',    name: 'Basecamp',    cat: 'Productivity', bg: '#1D2D3510', fields: ['api_key','account_id'] },
  // ── Storage & Data ──────────────────────────────────────
  { id: 'google_drive', name: 'Google Drive', cat: 'Storage',     bg: '#4285F410', fields: ['api_key','credentials_json'] },
  { id: 'dropbox',      name: 'Dropbox',      cat: 'Storage',     bg: '#0061FF10', fields: ['api_key','refresh_token'] },
  { id: 'aws_s3',       name: 'AWS S3',       cat: 'Storage',     bg: '#FF990010', fields: ['access_key','secret_key','region','bucket'] },
  { id: 'backblaze',   name: 'Backblaze B2', cat: 'Storage',     bg: '#E21E2610', fields: ['api_key','bucket_id'] },
  { id: 'wasabi',      name: 'Wasabi',      cat: 'Storage',     bg: '#56B84710', fields: ['access_key','secret_key','region','bucket'] },
  { id: 'airtable',     name: 'Airtable',     cat: 'Data',         bg: '#18BFFF10', fields: ['api_key','base_id'] },
  { id: 'google_sheets',name: 'Google Sheets',cat: 'Data',        bg: '#34A85310', fields: ['api_key','spreadsheet_id'] },
  // ── CRM ─────────────────────────────────────────────────
  { id: 'hubspot',      name: 'HubSpot',      cat: 'CRM',          bg: '#FF7A5910', fields: ['api_key','portal_id'] },
  { id: 'salesforce',   name: 'Salesforce',  cat: 'CRM',          bg: '#00A1E010', fields: ['client_id','client_secret','instance_url'] },
  { id: 'pipedrive',   name: 'Pipedrive',   cat: 'CRM',          bg: '#2D4E7410', fields: ['api_key','company_domain'] },
  { id: 'zoho',        name: 'Zoho',        cat: 'CRM',          bg: '#D6282810', fields: ['api_key','domain'] },
  { id: 'close',       name: 'Close.io',    cat: 'CRM',          bg: '#2B5AED10', fields: ['api_key'] },
  // ── AI ──────────────────────────────────────────────────
  { id: 'openai',       name: 'OpenAI',       cat: 'AI',           bg: '#10A37A10', fields: ['api_key','org_id'] },
  { id: 'anthropic',   name: 'Anthropic',   cat: 'AI',           bg: '#D4A57410', fields: ['api_key'] },
  { id: 'cohere',      name: 'Cohere',      cat: 'AI',           bg: '#39594D10', fields: ['api_key'] },
  { id: 'google_ai',   name: 'Google AI',   cat: 'AI',           bg: '#EA433510', fields: ['api_key','project_id'] },
  { id: 'huggingface', name: 'Hugging Face', cat: 'AI',          bg: '#FFD21E10', fields: ['api_key'] },
  { id: 'replicate',   name: 'Replicate',   cat: 'AI',           bg: '#00000010', fields: ['api_token'] },
  { id: 'stabilityai', name: 'Stability AI', cat: 'AI',          bg: '#A0A0A010', fields: ['api_key'] },
  // ── Automation ──────────────────────────────────────────
  { id: 'zapier',       name: 'Zapier',       cat: 'Automation',   bg: '#FF4A0010', fields: ['api_key','webhook_url'] },
  { id: 'make',        name: 'Make',        cat: 'Automation',   bg: '#6D00CC10', fields: ['api_key','scenario_id'] },
  { id: 'n8n',         name: 'n8n',         cat: 'Automation',   bg: '#EA4B7110', fields: ['api_key','base_url'] },
  { id: 'ifttt',       name: 'IFTTT',       cat: 'Automation',   bg: '#00000010', fields: ['webhook_url'] },
  // ── Monitoring ──────────────────────────────────────────
  { id: 'grafana',      name: 'Grafana',      cat: 'Monitoring',  bg: '#F4680010', fields: ['api_key','url'] },
  { id: 'pagerduty',    name: 'PagerDuty',   cat: 'Monitoring',  bg: '#06AC3808', fields: ['api_key','routing_key'] },
  { id: 'datadog',     name: 'Datadog',     cat: 'Monitoring',  bg: '#632CA610', fields: ['api_key','app_key','site'] },
  { id: 'newrelic',    name: 'New Relic',   cat: 'Monitoring',  bg: '#1CE78310', fields: ['api_key','account_id'] },
  { id: 'pingdom',     name: 'Pingdom',     cat: 'Monitoring',  bg: '#00B74A10', fields: ['api_key'] },
  { id: 'uptimerobot', name: 'UptimeRobot', cat: 'Monitoring',  bg: '#2ECC7110', fields: ['api_key'] },
  { id: 'statuspage',  name: 'StatusPage',  cat: 'Monitoring',  bg: '#1F293710', fields: ['api_key','page_id'] },
  { id: 'sentry',      name: 'Sentry',      cat: 'Monitoring',  bg: '#362D5910', fields: ['api_key','org_slug'] },
  // ── Analytics ───────────────────────────────────────────
  { id: 'posthog',      name: 'PostHog',     cat: 'Analytics',  bg: '#FF988510', fields: ['api_key','project_id'] },
  { id: 'amplitude',    name: 'Amplitude',   cat: 'Analytics',  bg: '#8257F510', fields: ['api_key','project_id'] },
  { id: 'mixpanel',    name: 'Mixpanel',   cat: 'Analytics',  bg: '#7B2FF510', fields: ['api_key','project_id'] },
  { id: 'segment',     name: 'Segment',    cat: 'Analytics',  bg: '#52BD9510', fields: ['write_key'] },
  { id: 'plausible',   name: 'Plausible',  cat: 'Analytics',  bg: '#5850EC10', fields: ['api_key','site_id'] },
  { id: 'hotjar',      name: 'Hotjar',      cat: 'Analytics',  bg: '#FFDE0010', fields: ['site_id','api_secret'] },
  { id: 'heap',        name: 'Heap',        cat: 'Analytics',  bg: '#F52E5A10', fields: ['api_key','app_id'] },
  // ── Infrastructure ──────────────────────────────────────
  { id: 'terraform',   name: 'Terraform',   cat: 'Infra',      bg: '#7B42BC10', fields: ['api_token','organization'] },
  { id: 'ansible',     name: 'Ansible',     cat: 'Infra',      bg: '#EE000010', fields: ['api_token','hostname','username'] },
  { id: 'kubernetes', name: 'Kubernetes', cat: 'Infra',      bg: '#326CE510', fields: ['api_token','cluster_url','namespace'] },
  { id: 'docker',     name: 'Docker',      cat: 'Infra',      bg: '#2496ED10', fields: ['api_token','registry_url'] },
  { id: 'harbor',      name: 'Harbor',       cat: 'Infra',      bg: '#0F168910', fields: ['api_key','registry_url'] },
  { id: 'portainer',  name: 'Portainer',   cat: 'Infra',      bg: '#15A0BF10', fields: ['api_key','url'] },
  // ── Domain & SSL ────────────────────────────────────────
  { id: 'cloudflare', name: 'Cloudflare', cat: 'Domain',    bg: '#FAAE4010', fields: ['api_key','email','zone_id'] },
  { id: 'route53',    name: 'Route53',     cat: 'Domain',    bg: '#FF990010', fields: ['access_key','secret_key','region'] },
  { id: 'godaddy',    name: 'GoDaddy',     cat: 'Domain',    bg: '#FFB32D10', fields: ['api_key','secret','domain'] },
  { id: 'namecheap',  name: 'Namecheap',   cat: 'Domain',    bg: '#0085FF10', fields: ['api_key','username'] },
  { id: 'letsencrypt',name: 'Let\'s Encrypt', cat: 'Domain',  bg: '#FF5A0010', fields: ['email'] },
  { id: 'zerossl',    name: 'ZeroSSL',    cat: 'Domain',    bg: '#4CAF5010', fields: ['api_key','email'] },
  { id: 'porkbun',   name: 'Porkbun',    cat: 'Domain',    bg: '#EF854710', fields: ['api_key','secret'] },
  // ── Cloud ───────────────────────────────────────────────
  { id: 'aws',        name: 'AWS',         cat: 'Cloud',     bg: '#FF990010', fields: ['access_key','secret_key','region'] },
  { id: 'gcp',        name: 'GCP',         cat: 'Cloud',     bg: '#4285F410', fields: ['api_key','project_id'] },
  { id: 'azure',      name: 'Azure',       cat: 'Cloud',     bg: '#0078D410', fields: ['client_id','client_secret','tenant_id'] },
  // ── Hosting ─────────────────────────────────────────────
  { id: 'vercel',     name: 'Vercel',      cat: 'Hosting',   bg: '#ffffff08', fields: ['api_key','team_id'] },
  { id: 'netlify',    name: 'Netlify',     cat: 'Hosting',   bg: '#00C2C210', fields: ['api_key','site_id'] },
  { id: 'hostinger',  name: 'Hostinger',   cat: 'Hosting',   bg: '#6C3BA810', fields: ['api_key'] },
  { id: 'railway',    name: 'Railway',     cat: 'Hosting',   bg: '#0B0D0E10', fields: ['api_token'] },
  { id: 'render',     name: 'Render',      cat: 'Hosting',   bg: '#46E3B710', fields: ['api_key','service_id'] },
  { id: 'flyio',      name: 'Fly.io',      cat: 'Hosting',   bg: '#7B3BE210', fields: ['api_token','org_slug'] },
  { id: 'heroku',     name: 'Heroku',      cat: 'Hosting',   bg: '#43009810', fields: ['api_key','app_name'] },
  { id: 'digitalocean', name: 'DigitalOcean', cat: 'Hosting', bg: '#0080FF10', fields: ['api_token'] },
  { id: 'cloudways',  name: 'Cloudways',   cat: 'Hosting',   bg: '#2B8FDE10', fields: ['api_key','server_id'] },
  // ── Database ────────────────────────────────────────────
  { id: 'supabase',   name: 'Supabase',    cat: 'Database', bg: '#3ECF8E10', fields: ['api_key','url'] },
  { id: 'firebase',  name: 'Firebase',   cat: 'Database', bg: '#FFCA2810', fields: ['api_key','project_id'] },
    // ── Registry ────────────────────────────────────────────
  { id: 'dockerhub', name: 'Docker Hub',  cat: 'Registry', bg: '#2496ED10', fields: ['username','password'] },
  { id: 'ghcr',      name: 'GHCR',        cat: 'Registry', bg: '#ffffff08', fields: ['token'] },
  // ── Communication ───────────────────────────────────────
  { id: 'twilio',     name: 'Twilio',     cat: 'Communication', bg: '#F22F4610', fields: ['account_sid','auth_token','from_number'] },
  { id: 'sendgrid',  name: 'SendGrid',   cat: 'Communication', bg: '#1A82E210', fields: ['api_key','from_email'] },
  { id: 'mailgun',   name: 'Mailgun',    cat: 'Communication', bg: '#2185C410', fields: ['api_key','domain'] },
  { id: 'postmark',  name: 'Postmark',   cat: 'Communication', bg: '#00000010', fields: ['api_key','from_email'] },
  // ── E-commerce ──────────────────────────────────────────
  { id: 'shopify',     name: 'Shopify',     cat: 'E-commerce', bg: '#96BF4810', fields: ['api_key','shop_domain','password'] },
  { id: 'woocommerce', name: 'WooCommerce', cat: 'E-commerce', bg: '#96588A10', fields: ['api_key','url','consumer_secret'] },
  { id: 'bigcommerce', name: 'BigCommerce', cat: 'E-commerce', bg: '#34313F10', fields: ['api_key','store_hash'] },
  // ── Distribution ────────────────────────────────────────
  { id: 'appstore',   name: 'App Store',    cat: 'Distribution', bg: '#00000010', fields: ['api_key','issuer_id'] },
  { id: 'googleplay',name: 'Google Play', cat: 'Distribution', bg: '#FA500010', fields: ['api_key','package_name'] },

  // ── Payments ──────────────────────────────────────────
  { id: 'paypal',       name: 'PayPal',       cat: 'Payments',    bg: '#00308710', fields: ['api_key','client_id','client_secret'] },
  { id: 'lemon_squeezy', name: 'Lemon Squeezy', cat: 'Payments', bg: '#FFD70010', fields: ['api_key','store_id'] },
  { id: 'stripe',       name: 'Stripe',       cat: 'Payments',    bg: '#6772E510', fields: ['api_key'] },
  // ── Database (continued) ───────────────────────────────
  { id: 'planetable',   name: 'PlanetScale',  cat: 'Database',    bg: '#00000010', fields: ['api_token','org','database'] },
  // ── Media ─────────────────────────────────────────────
  { id: 'youtube',     name: 'YouTube',      cat: 'Media',      bg: '#FF000010', fields: ['api_key','oauth_client_id','oauth_client_secret','channel_id'] },
];

const CATEGORIES = ['All', ...new Set([...INTEGRATIONS.map(i => i.cat), 'Custom'])];

const TRIGGERS_DEF = [
  { id: 'new_email',    name: 'New Email',        icon: 'EML',  intg: 'email',        desc: 'Fires when a new email arrives in your inbox.' },
  { id: 'new_message',  name: 'New Slack Message', icon: 'SLK',  intg: 'slack',        desc: 'Fires on every new message in a channel.' },
  { id: 'new_commit',   name: 'New Commit',        icon: 'GIT',  intg: 'github',       desc: 'Fires when a commit is pushed to a repo.' },
  { id: 'new_issue',    name: 'New Issue',         icon: 'BUG',  intg: 'github',       desc: 'Fires when a GitHub issue is opened.' },
  { id: 'file_changed', name: 'File Changed',      icon: 'DRV',  intg: 'google_drive', desc: 'Fires when a Drive file is modified.' },
  { id: 'schedule',     name: 'Schedule / CRON',   icon: 'CRN',  intg: 'schedule',     desc: 'Time-based recurring trigger.' },
  { id: 'webhook',      name: 'Webhook Received',  icon: 'WKH',  intg: 'webhook',      desc: 'Fires on any incoming HTTP webhook.' },
  { id: 'discord_msg',  name: 'Discord Message',   icon: 'DSC',  intg: 'discord',      desc: 'Fires on a new Discord channel message.' },
  { id: 'posthog_event', name: 'PostHog Event',   icon: 'PHG',  intg: 'posthog',     desc: 'Fires on tracked PostHog events.' },
  { id: 'amplitude_event', name: 'Amplitude Event', icon: 'AMP', intg: 'amplitude',  desc: 'Fires on tracked Amplitude events.' },
  { id: 'schedule_hr',  name: 'Every Hour',        icon: '1HR',  intg: 'schedule_hr',  desc: 'Fires once every hour.' },
  { id: 'schedule_daily', name: 'Daily at 9am',   icon: 'DAY',  intg: 'schedule_daily', desc: 'Fires every day at 9:00 AM.' },
];

const TRIGGER_BUILTINS = new Set(['schedule', 'webhook', 'schedule_hr', 'schedule_daily']);

let connectedMap = {};
let customIntegrations = [];  // Custom API integrations from DB
let activeTriggers = new Set(JSON.parse(localStorage.getItem('nexus.triggers') || '[]'));
let activeCat = 'All';
let intSearchQ = '';

// ── Drawer helper ─────────────────────────────────────────────────────────────
function createDrawer(title, icon, buildContent) {
  const root = document.getElementById('modal-root') || document.body;

  // Backdrop
  const backdrop = document.createElement('div');
  backdrop.className = 'drw-backdrop';

  // Drawer
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

  // Animate open
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

// ── Integrations Drawer ───────────────────────────────────────────────────────
export async function initIntegrations() {
  // Load current credentials once
  await refreshConnectedMap();

  const btn = document.getElementById('integrations-trigger');
  if (!btn) return;

  btn.addEventListener('click', () => {
    createDrawer(
      'Integrations',
      `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>`,
      (body) => buildIntgBody(body)
    );
  });

  // Listen for WebSocket events to refresh integration state
  window.addEventListener('ws:event', e => {
    const msg = e.detail;
    if (msg.kind === 'integration_toggled' || msg.kind === 'integration_config_updated' || msg.kind === 'integration_deleted') {
      refreshConnectedMap();
    }
  });
}

async function refreshConnectedMap() {
  try {
    const creds = await api('/api/integrations');
    connectedMap = {};
    (creds || []).forEach(c => { connectedMap[c.id || c.type] = c; });
  } catch { /* server may not have integrations yet */ }
  // Also load custom API integrations from the database
  try {
    const custom = await api('/api/integrations/custom/list');
    customIntegrations = (custom || [])
      .filter(c => !INTEGRATIONS.some(i => i.id === c.id))  // Skip duplicates
      .map(c => ({
        id: c.id,
        name: c.name || c.id.replace(/^custom_api_/, '').replace(/_/g, ' '),
        cat: (c.category || 'custom').charAt(0).toUpperCase() + (c.category || 'custom').slice(1),
        bg: '#8b5cf610',
        fields: ['api_key', 'base_url'],
        isCustom: true,
        description: c.description || '',
      }));
  } catch { /* custom endpoint may not exist yet */ }
}

function buildIntgBody(body) {
  // Search + cats
  const search = document.createElement('div');
  search.className = 'drw-search-row';
  search.innerHTML = `
    <div class="drw-search-wrap">
      <svg class="drw-search-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <input type="text" class="drw-search" placeholder="Search integrations…" autocomplete="off"/>
    </div>`;
  body.appendChild(search);

  const cats = document.createElement('div');
  cats.className = 'drw-cats';
  CATEGORIES.forEach(cat => {
    const b = document.createElement('button');
    b.className = 'drw-cat' + (cat === activeCat ? ' active' : '');
    b.textContent = cat;
    b.addEventListener('click', () => {
      activeCat = cat;
      cats.querySelectorAll('.drw-cat').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      renderIntgGrid(grid);
    });
    cats.appendChild(b);
  });
  body.appendChild(cats);

  const grid = document.createElement('div');
  grid.className = 'drw-int-grid';
  body.appendChild(grid);

  // Add Custom API button
  const addCustomBtn = document.createElement('button');
  addCustomBtn.className = 'drw-add-custom-btn';
  addCustomBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg> Add Custom API`;
  addCustomBtn.style.cssText = 'width:100%;margin-top:8px;padding:10px;border:1px dashed var(--bd2);border-radius:8px;background:var(--bg1);color:var(--tx2);cursor:pointer;font-size:13px;display:flex;align-items:center;justify-content:center;gap:6px;transition:all .15s';
  addCustomBtn.addEventListener('mouseenter', () => { addCustomBtn.style.borderColor = '#8b5cf6'; addCustomBtn.style.color = '#8b5cf6'; });
  addCustomBtn.addEventListener('mouseleave', () => { addCustomBtn.style.borderColor = 'var(--bd2)'; addCustomBtn.style.color = 'var(--tx2)'; });
  addCustomBtn.addEventListener('click', () => openCreateCustomApiModal(grid));
  body.appendChild(addCustomBtn);

  search.querySelector('.drw-search').addEventListener('input', e => {
    intSearchQ = e.target.value.toLowerCase();
    renderIntgGrid(grid);
  });

  renderIntgGrid(grid);
}

function renderIntgGrid(grid) {
  grid.innerHTML = '';
  // Merge built-in integrations with custom API integrations from the database
  const allIntegrations = [...INTEGRATIONS, ...customIntegrations];
  const visible = allIntegrations.filter(i => {
    const catOk  = activeCat === 'All' || i.cat === activeCat || (activeCat === 'Custom' && i.isCustom);
    const termOk = !intSearchQ || i.name.toLowerCase().includes(intSearchQ);
    return catOk && termOk;
  });

  visible.forEach((intg, idx) => {
    const connected = !!connectedMap[intg.id];
    const card = document.createElement('div');
    card.className = 'drw-int-card' + (connected ? ' connected' : '');
    card.style.animationDelay = idx * 28 + 'ms';
    const logo = LOGOS[intg.id] || (intg.isCustom
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="1.5"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/></svg>`);
    card.innerHTML = `
      <div class="drw-int-logo" style="background:${intg.bg}">${logo}</div>
      <div class="drw-int-info">
        <div class="drw-int-name">${intg.name}${intg.isCustom ? ' <span style="font-size:9px;color:#8b5cf6;font-weight:600">CUSTOM</span>' : ''}</div>
        <div class="drw-int-cat">${intg.cat}</div>
      </div>
      <button class="drw-int-btn ${connected ? 'manage' : ''}">${connected ? 'Manage' : 'Connect'}</button>`;
    card.querySelector('.drw-int-btn').addEventListener('click', () => openIntgModal(intg, connected, grid));
    grid.appendChild(card);
  });

  if (!visible.length) {
    grid.innerHTML = `<div class="drw-empty">No integrations match</div>`;
  }
}

function openIntgModal(intg, isConnected, grid) {
  const root = document.getElementById('modal-root') || document.body;
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';
  const creds = connectedMap[intg.id] || {};

  const fieldDefs = {
    api_key:          { label: 'API Key',           type: 'password' },
    api_token:        { label: 'API Token',         type: 'password' },
    token:            { label: 'Token',             type: 'password' },
    bot_token:        { label: 'Bot Token',         type: 'password' },
    webhook_url:      { label: 'Webhook URL',       type: 'text' },
    base_url:         { label: 'Base URL',          type: 'text' },
    url:              { label: 'URL',               type: 'text' },
    chat_id:          { label: 'Chat ID',           type: 'text' },
    tenant_id:        { label: 'Tenant ID',         type: 'text' },
    workspace:        { label: 'Workspace',         type: 'text' },
    org_slug:         { label: 'Org Slug',          type: 'text' },
    org:              { label: 'Organization',      type: 'text' },
    space_key:        { label: 'Space Key',         type: 'text' },
    smtp_host:        { label: 'SMTP Host',         type: 'text' },
    smtp_port:        { label: 'SMTP Port',         type: 'text' },
    username:         { label: 'Username',          type: 'text' },
    password:         { label: 'Password',          type: 'password' },
    from_email:       { label: 'From Email',        type: 'email' },
    from_number:      { label: 'From Number',       type: 'text' },
    email:            { label: 'Email',             type: 'email' },
    channel:          { label: 'Default Channel',   type: 'text' },
    repo:             { label: 'Repository (owner/repo)', type: 'text' },
    database_id:      { label: 'Database ID',       type: 'text' },
    database:         { label: 'Database Name',     type: 'text' },
    org_id:           { label: 'Organization ID',   type: 'text' },
    portal_id:        { label: 'Portal ID',         type: 'text' },
    base_id:          { label: 'Base ID',           type: 'text' },
    credentials_json: { label: 'Credentials JSON',  type: 'textarea' },
    access_key:       { label: 'Access Key',        type: 'password' },
    secret_key:       { label: 'Secret Key',        type: 'password' },
    secret:           { label: 'Secret',            type: 'password' },
    auth_token:       { label: 'Auth Token',        type: 'password' },
    account_sid:      { label: 'Account SID',       type: 'text' },
    region:           { label: 'Region',            type: 'text' },
    bucket:           { label: 'Bucket',            type: 'text' },
    bucket_id:        { label: 'Bucket ID',         type: 'text' },
    refresh_token:    { label: 'Refresh Token',     type: 'password' },
    client_id:        { label: 'Client ID',         type: 'text' },
    client_secret:    { label: 'Client Secret',     type: 'password' },
    instance_url:     { label: 'Instance URL',      type: 'text' },
    team_id:          { label: 'Team ID',           type: 'text' },
    workspace_id:     { label: 'Workspace ID',      type: 'text' },
    spreadsheet_id:   { label: 'Spreadsheet ID',    type: 'text' },
    routing_key:      { label: 'Routing Key',       type: 'text' },
    app_key:          { label: 'App Key',           type: 'password' },
    site:             { label: 'Site',              type: 'text' },
    site_id:          { label: 'Site ID',           type: 'text' },
    page_id:          { label: 'Page ID',           type: 'text' },
    project_id:       { label: 'Project ID',        type: 'text' },
    account_id:       { label: 'Account ID',        type: 'text' },
    write_key:        { label: 'Write Key',         type: 'password' },
    api_secret:       { label: 'API Secret',        type: 'password' },
    app_id:           { label: 'App ID',            type: 'text' },
    service_id:       { label: 'Service ID',        type: 'text' },
    server_id:        { label: 'Server ID',         type: 'text' },
    shop_domain:      { label: 'Shop Domain',       type: 'text' },
    consumer_secret:  { label: 'Consumer Secret',   type: 'password' },
    store_hash:       { label: 'Store Hash',        type: 'text' },
    company_domain:   { label: 'Company Domain',    type: 'text' },
    domain:           { label: 'Domain',            type: 'text' },
    scenario_id:      { label: 'Scenario ID',       type: 'text' },
    board_id:         { label: 'Board ID',          type: 'text' },
    zone_id:          { label: 'Zone ID',           type: 'text' },
    issuer_id:        { label: 'Issuer ID',         type: 'text' },
    package_name:     { label: 'Package Name',      type: 'text' },
    app_name:         { label: 'App Name',          type: 'text' },
    cluster_url:      { label: 'Cluster URL',       type: 'text' },
    namespace:        { label: 'Namespace',          type: 'text' },
    registry_url:     { label: 'Registry URL',      type: 'text' },
    method:           { label: 'HTTP Method',       type: 'text' },
    path:             { label: 'API Path',          type: 'text' },
    headers:          { label: 'Custom Headers (JSON)', type: 'textarea' },
    oauth_client_id:     { label: 'OAuth Client ID',     type: 'text' },
    oauth_client_secret: { label: 'OAuth Client Secret', type: 'password' },
  };

  const logo = LOGOS[intg.id] || (intg.isCustom
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="1.5"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>`
    : '');
  // For custom integrations, derive fields from existing config keys in the DB
  let fieldsList = intg.fields || [];
  if (intg.isCustom && creds) {
    const configKeys = Object.keys(creds.config || creds).filter(k =>
      !['id', 'name', 'category', 'description', 'connected', 'created_at', 'updated_at'].includes(k)
    );
    if (configKeys.length > 0) fieldsList = configKeys;
  }
  let fieldsHtml = '';

  // For email integration, show OAuth buttons + OAuth config fields + SMTP fields
  if (intg.id === 'email') {
    const oa_id = creds['oauth_client_id'] || '';
    const oa_sec = creds['oauth_client_secret'] || '';
    const oa_masked_sec = oa_sec ? '\u2022'.repeat(Math.min(oa_sec.length, 12)) : '';
    fieldsHtml += `
      <div style="margin-bottom:16px;padding:14px;border-radius:8px;background:var(--bg2);border:1px solid var(--bd2)">
        <div style="font-size:12px;font-weight:600;margin-bottom:10px;color:var(--tx2)">OAuth Quick Connect</div>
        <div style="display:flex;flex-direction:column;gap:6px;margin-bottom:10px">
          <button class="email-oauth-btn google" data-provider="gmail" style="display:flex;align-items:center;justify-content:center;gap:10px;padding:8px 14px;border-radius:8px;border:1px solid var(--bd2);background:var(--bg1);color:var(--tx1);cursor:pointer;font-size:12px;font-weight:600"
            ><svg width="16" height="16" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
            Sign in with Google (Gmail)</button>
          <button class="email-oauth-btn microsoft" data-provider="outlook" style="display:flex;align-items:center;justify-content:center;gap:10px;padding:8px 14px;border-radius:8px;border:1px solid var(--bd2);background:var(--bg1);color:var(--tx1);cursor:pointer;font-size:12px;font-weight:600"
            ><svg width="16" height="16" viewBox="0 0 24 24"><rect fill="#0078D4" x="1" y="4" width="22" height="16" rx="2"/><path fill="#fff" d="M1 10l6 4-6 4V10zM23 10l-6 4 6 4V10zM1 10l6 4 8-2 4 2V6l-6 4-6 4-6-4z"/></svg>
            Sign in with Microsoft (Outlook)</button>
        </div>
        <div style="margin-bottom:8px;padding:8px 10px;border-radius:6px;background:var(--bg1);border:1px solid var(--bd2);font-size:11px;color:var(--tx3)">
          Enter your OAuth Client ID and Secret below, then click <strong>Save</strong> before using the OAuth buttons above.
          <a href="https://console.cloud.google.com/apis/credentials" target="_blank" style="color:#818cf8">Get Gmail credentials</a>
          &middot;
          <a href="https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade" target="_blank" style="color:#818cf8">Get Outlook credentials</a>
        </div>
        <div style="display:flex;flex-direction:column;gap:6px">
          <div class="modal-field" style="margin:0">
            <label>OAuth Client ID</label>
            <input type="text" data-key="oauth_client_id" value="${oa_id}" autocomplete="off"/>
          </div>
          <div class="modal-field" style="margin:0">
            <label>OAuth Client Secret</label>
            <input type="password" data-key="oauth_client_secret" value="${oa_masked_sec}" autocomplete="off"/>
          </div>
        </div>
      </div>
      <div style="margin-bottom:14px;padding:12px;border-radius:8px;background:var(--bg2);border:1px solid var(--bd2);font-size:11px;color:var(--tx3);line-height:1.6">
        <strong style="color:var(--tx2)">Or use SMTP directly</strong><br>
        Gmail: <code>smtp.gmail.com:587</code> + App Password &middot;
        Outlook: <code>smtp.office365.com:587</code> + password
      </div>`;
  }

  fieldsHtml += fieldsList.map(f => {
    const def = fieldDefs[f] || { label: f.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()), type: 'text' };
    const val = creds[f] || '';
    const masked = val && def.type === 'password' ? '\u2022'.repeat(Math.min(val.length, 12)) : val;
    if (def.type === 'textarea') {
      return `<div class="modal-field">
        <label>${def.label}</label>
        <textarea data-key="${f}" rows="3">${masked}</textarea>
      </div>`;
    }
    return `<div class="modal-field">
      <label>${def.label}</label>
      <input type="${def.type}" data-key="${f}" value="${masked}" autocomplete="off"/>
    </div>`;
  }).join('');



  backdrop.innerHTML = `
    <div class="modal" style="max-width:440px">
      <div class="modal-header">
        <div style="display:flex;align-items:center;gap:10px">
          <div style="width:32px;height:32px;border-radius:8px;background:${intg.bg};display:flex;align-items:center;justify-content:center">${logo}</div>
          <div>
            <div style="font-weight:700;font-size:14px">${intg.name}</div>
            <div style="font-size:11px;color:var(--tx3)">${intg.cat}</div>
          </div>
        </div>
        <button class="modal-close" id="int-modal-close">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="modal-body">
        ${fieldsHtml}
      </div>
      <div class="modal-footer">
        ${isConnected ? `<button class="btn danger" id="int-disconnect">Disconnect</button>` : ''}
        <button class="btn ghost" id="int-cancel">Cancel</button>
        <button class="btn primary" id="int-save">${isConnected ? 'Update' : 'Connect'}</button>
      </div>
    </div>`;

  document.body.appendChild(backdrop);
  requestAnimationFrame(() => backdrop.classList.add('visible'));

  // OAuth button click handlers
  setTimeout(() => {
    backdrop.querySelectorAll('.email-oauth-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const provider = btn.dataset.provider;
        try {
          btn.disabled = true;
          btn.textContent = 'Connecting...';
          const res = await api(`/api/integrations/email/oauth/start?provider=${provider}`);
          if (res.authorization_url) {
            window.open(res.authorization_url, 'oauth-popup', 'width=600,height=700');
          } else {
            toast(res.error || 'OAuth failed — save your Client ID and Secret first', 'warn', 5000);
          }
        } catch (e) {
          toast('OAuth error: ' + e.message, 'error');
        } finally {
          btn.disabled = false;
          btn.innerHTML = btn.dataset.provider === 'gmail'
            ? '<svg width="16" height="16" viewBox="0 0 24 24">...</svg> Sign in with Google (Gmail)'
            : '<svg width="16" height="16" viewBox="0 0 24 24">...</svg> Sign in with Microsoft (Outlook)';
        }
      });
    });
  }, 50);

  // Listen for OAuth success message from popup
  const _oauthMsgHandler = (event) => {
    if (event.data && event.data.type === 'email_oauth_success') {
      toast('Email connected successfully', 'success');
      window.removeEventListener('message', _oauthMsgHandler);
      refreshConnectedMap().then(() => { close(); renderIntgGrid(grid); });
    }
  };
  window.addEventListener('message', _oauthMsgHandler);

  const close = () => { backdrop.classList.remove('visible'); setTimeout(() => backdrop.remove(), 200); };
  backdrop.querySelector('#int-modal-close').addEventListener('click', close);
  backdrop.querySelector('#int-cancel').addEventListener('click', close);
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });

  backdrop.querySelector('#int-save').addEventListener('click', async () => {
    const configData = {};
    backdrop.querySelectorAll('[data-key]').forEach(inp => {
      const val = inp.value;
      // Skip masked password fields that weren't changed
      if (inp.type === 'password' && val && val.startsWith('\u2022')) return;
      configData[inp.dataset.key] = val;
    });
    try {
      await api(`/api/integrations/${encodeURIComponent(intg.id)}/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ config: configData }) });
      connectedMap[intg.id] = { id: intg.id, ...configData, connected: true };
      toast(`${intg.name} ${isConnected ? 'updated' : 'connected'} ✓`, 'success');
      close();
      renderIntgGrid(grid);
    } catch (e) { toast('Failed: ' + e.message, 'error'); }
  });

  backdrop.querySelector('#int-disconnect')?.addEventListener('click', async () => {
    try {
      await api(`/api/integrations/${encodeURIComponent(intg.id)}`, { method: 'DELETE' });
      delete connectedMap[intg.id];
      toast(`${intg.name} disconnected`, 'success');
      close();
      renderIntgGrid(grid);
    } catch (e) { toast('Failed: ' + e.message, 'error'); }
  });
}

// ── Triggers Drawer ───────────────────────────────────────────────────────────
let _selectedTriggerContext = null;  // Context set when user clicks a trigger card

export async function initTriggers() {
  // Sync triggers from backend
  try {
    const saved = await api('/api/triggers');
    if (saved && Array.isArray(saved)) {
      saved.forEach(t => {
        if (t.enabled) activeTriggers.add(t.id);
        else activeTriggers.delete(t.id);
      });
      localStorage.setItem('nexus.triggers', JSON.stringify([...activeTriggers]));
    }
  } catch { /* backend may not be available yet */ }

  const btn = document.getElementById('triggers-trigger');
  if (!btn) return;

  btn.addEventListener('click', () => {
    createDrawer(
      'Triggers',
      `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
      (body, close) => buildTriggersBody(body, close)
    );
  });

  // Listen for trigger events to re-sync active triggers
  window.addEventListener('ws:event', e => {
    const msg = e.detail;
    if (msg.kind === 'trigger_updated' || msg.kind === 'trigger_fired') {
      // Re-fetch trigger state from backend
      api('/api/triggers').then(saved => {
        if (saved && Array.isArray(saved)) {
          activeTriggers.clear();
          saved.forEach(t => {
            if (t.enabled) activeTriggers.add(t.id);
          });
          localStorage.setItem('nexus.triggers', JSON.stringify([...activeTriggers]));
        }
      }).catch(() => {});
    }
  });
}

/**
 * Get the human-readable integration name from a trigger's intg id.
 */
function _getIntgName(intgId) {
  const found = INTEGRATIONS.find(i => i.id === intgId);
  if (found) return found.name;
  // Builtin types
  const builtinNames = {
    schedule: 'Schedule',
    webhook: 'Webhook',
    schedule_hr: 'Hourly',
    schedule_daily: 'Daily',
  };
  return builtinNames[intgId] || intgId.replace(/_/g, ' ');
}

/**
 * Resolve the integration logo SVG for a trigger.
 */
function _getTrigLogo(t) {
  // First check for trigger-specific logo
  if (t.id && LOGOS[t.id]) return LOGOS[t.id];
  // Then fall back to the integration logo
  if (t.intg && LOGOS[t.intg]) return LOGOS[t.intg];
  return null;
}

/**
 * Select a trigger — called when user clicks a trigger card.
 * Closes the drawer, checks integration credentials, and dispatches an event
 * so chat.js can populate the input with the appropriate context message.
 */
function selectTrigger(trigger, closeFn) {
  const intgId = trigger.intg;
  const intgName = _getIntgName(intgId);
  const isBuiltin = !intgId || TRIGGER_BUILTINS.has(intgId);
  const isConnected = !!connectedMap[intgId];

  // Close the drawer immediately
  if (closeFn) closeFn();

  // Build the context message depending on credential state
  let contextMsg;
  if (!isBuiltin && !isConnected) {
    // Integration not connected — warn user
    contextMsg = `⚠️ ${intgName} credentials not found. Please connect ${intgName} first before setting up the ${trigger.name} trigger.`;
    toast(`${intgName} not connected — please add credentials first`, 'warn', 4000);
  } else {
    // Integration connected or builtin — ask user what they want
    contextMsg = `⚡ Setting up ${trigger.name} trigger${!isBuiltin ? ' for ' + intgName : ''}. What would you like this trigger to do?`;
    toast(`${trigger.name} selected — describe what it should do`, 'success', 3000);
  }

  _selectedTriggerContext = contextMsg;

  // Dispatch event that chat.js listens to
  window.dispatchEvent(new CustomEvent('trigger:selected', {
    detail: { trigger, context: _selectedTriggerContext }
  }));
}

/**
 * Build the triggers drawer body with clickable cards (no toggle switches).
 */
function buildTriggersBody(body, closeFn) {
  // Info banner at top
  const banner = document.createElement('div');
  banner.className = 'drw-trig-banner';
  banner.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;color:#f59e0b;margin-top:1px">
      <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
    </svg>
    <span>Click a trigger to configure it via chat. The agent will guide you through setup.</span>`;
  body.appendChild(banner);

  // Stats row
  const stats = document.createElement('div');
  stats.className = 'drw-trig-stats';
  const connectedCount = TRIGGERS_DEF.filter(t => {
    if (!t.intg || TRIGGER_BUILTINS.has(t.intg)) return true;
    return !!connectedMap[t.intg];
  }).length;
  stats.innerHTML = `
    <div class="drw-trig-stat">
      <span class="drw-trig-stat-num">${TRIGGERS_DEF.length}</span>
      <span class="drw-trig-stat-label">Available</span>
    </div>
    <div class="drw-trig-stat">
      <span class="drw-trig-stat-num">${activeTriggers.size}</span>
      <span class="drw-trig-stat-label">Active</span>
    </div>
    <div class="drw-trig-stat">
      <span class="drw-trig-stat-num">${connectedCount}</span>
      <span class="drw-trig-stat-label">Ready</span>
    </div>`;
  body.appendChild(stats);

  // Trigger cards
  const list = document.createElement('div');
  list.className = 'drw-trig-list';
  body.appendChild(list);

  TRIGGERS_DEF.forEach((t, i) => {
    const isOn = activeTriggers.has(t.id);
    const isBuiltin = !t.intg || TRIGGER_BUILTINS.has(t.intg);
    const isConnected = t.intg ? !!connectedMap[t.intg] : true;
    const intgName = _getIntgName(t.intg);
    const logoSvg = _getTrigLogo(t);

    const card = document.createElement('div');
    card.className = 'drw-trig-card';
    card.style.animationDelay = i * 30 + 'ms';
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-label', `Set up ${t.name} trigger`);

    // Build status dot: green = connected, red = not connected, grey = builtin
    let statusClass = 'builtin';
    let statusTitle = 'Built-in — no credentials needed';
    if (!isBuiltin && isConnected) {
      statusClass = 'connected';
      statusTitle = `${intgName} connected`;
    } else if (!isBuiltin && !isConnected) {
      statusClass = 'disconnected';
      statusTitle = `${intgName} not connected`;
    }

    // Build icon badge
    let iconHtml;
    if (logoSvg) {
      iconHtml = `<div class="drw-trig-icon-badge">${logoSvg}</div>`;
    } else {
      // Use text icon badge
      iconHtml = `<div class="drw-trig-icon-badge drw-trig-icon-text">${t.icon}</div>`;
    }

    // Integration tag
    let intgTag = '';
    if (t.intg && !isBuiltin) {
      intgTag = `<span class="drw-trig-intg-tag ${isConnected ? 'ok' : 'warn'}">${intgName}</span>`;
    } else if (isBuiltin) {
      intgTag = `<span class="drw-trig-intg-tag builtin">Built-in</span>`;
    }

    // Active indicator
    let activeDot = '';
    if (isOn) {
      activeDot = `<span class="drw-trig-active-dot"></span>`;
    }

    card.innerHTML = `
      <div class="drw-trig-left">
        ${iconHtml}
        <span class="drw-trig-status-dot ${statusClass}" title="${statusTitle}"></span>
      </div>
      <div class="drw-trig-info">
        <div class="drw-trig-name">${t.name} ${activeDot}</div>
        <div class="drw-trig-desc">${t.desc}</div>
        ${intgTag}
      </div>
      <svg class="drw-trig-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="9 18 15 12 9 6"/>
      </svg>`;

    // Click handler — selects the trigger
    card.addEventListener('click', () => selectTrigger(t, closeFn));
    // Keyboard support
    card.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        selectTrigger(t, closeFn);
      }
    });

    list.appendChild(card);
  });
}

// ── Context helpers (used by chat.js) ─────────────────────────────────────────
export function getIntegrationContext() {
  const keys = Object.keys(connectedMap);
  return keys.length ? '[Connected integrations: ' + keys.join(', ') + ']' : '';
}

/**
 * Returns the trigger context message if a trigger was selected.
 * This is consumed by chat.js to prepend context to the user's message.
 * After being read once, the context is cleared to avoid duplicate injection.
 */
export function getTriggerContext() {
  const ctx = _selectedTriggerContext;
  _selectedTriggerContext = null;  // One-shot: clear after read
  return ctx || '';
}

// ── Automation Drawer (file watches / web monitors / schedules) ──
// Self-improvement is fully automatic via the heartbeat (nexus/heartbeat.py)
// every 30 minutes — no UI surface needed.
//
// Wires up the previously-unused API endpoints:
//   GET  /api/watches
//   POST /api/watches
//   DELETE /api/watches/{watch_id}
//   GET  /api/monitors
//   POST /api/monitors
//   DELETE /api/monitors/{monitor_id}
//   GET  /api/schedules
//   DELETE /api/schedules/{schedule_id}
// (Self-improvement endpoints still exist on the backend — they are
// driven by the heartbeat and don't need a UI surface.)

const AUTOMATION_SECTIONS = [
  {
    id: 'watches',
    title: 'File Watches',
    desc: 'Watch files / directories for changes. The agent is notified on every create / modify / delete.',
    list:   '/api/watches',
    delete: (id) => `/api/watches/${id}`,
    add:    {
      endpoint: '/api/watches',
      method: 'POST',
      body: { path: '', label: '' },
      fields: [
        { id: 'path',  label: 'Path (absolute or workspace-relative)', placeholder: 'C:\\Users\\me\\Documents or data/workspace/myapp' },
        { id: 'label', label: 'Label (optional)',                        placeholder: 'My Documents' },
      ],
    },
    emptyMsg: 'No file watches yet',
  },
  {
    id: 'monitors',
    title: 'Web Monitors',
    desc: 'Poll a URL on a schedule. The agent is notified when its content hash changes.',
    list:   '/api/monitors',
    delete: (id) => `/api/monitors/${id}`,
    add:    {
      endpoint: '/api/monitors',
      method: 'POST',
      body: { url: '', interval_seconds: 3600, label: '' },
      fields: [
        { id: 'url',             label: 'URL',                placeholder: 'https://example.com/page' },
        { id: 'interval_seconds',label: 'Interval (seconds)', placeholder: '3600', type: 'number' },
        { id: 'label',           label: 'Label (optional)',   placeholder: 'Competitor pricing page' },
      ],
    },
    emptyMsg: 'No web monitors yet',
  },
  {
    id: 'schedules',
    title: 'Scheduled Tasks',
    desc: 'NL-defined scheduled tasks (e.g. "every morning at 9am, run X"). Managed by the agent via the `schedule` tool.',
    list:   '/api/schedules',
    delete: (id) => `/api/schedules/${id}`,
    add: null,  // Schedules are created by the agent, not the user
    emptyMsg: 'No scheduled tasks yet — ask the agent to schedule one',
  },
  // Self-improvement section intentionally removed: it runs automatically
  // every 30 min via the heartbeat (nexus/heartbeat.py:178) and needs no UI.
];

export async function initAutomation() {
  const btn = document.getElementById('automation-trigger');
  if (!btn) return;
  btn.addEventListener('click', () => {
    createDrawer(
      'Automation',
      `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`,
      (body, close) => buildAutomationBody(body, close)
    );
  });
}

function buildAutomationBody(body, closeFn) {
  body.innerHTML = '';
  body.classList.add('drw-auto-body');

  // Render each section asynchronously
  AUTOMATION_SECTIONS.forEach(section => {
    const sec = document.createElement('div');
    sec.className = 'drw-auto-section';
    sec.innerHTML = `
      <div class="drw-auto-header">
        <h3 class="drw-auto-title">${section.title}</h3>
        <p class="drw-auto-desc">${section.desc}</p>
      </div>
      <div class="drw-auto-list" id="auto-list-${section.id}">
        <div class="drw-auto-loading">Loading…</div>
      </div>
      <div class="drw-auto-actions" id="auto-actions-${section.id}"></div>
    `;
    body.appendChild(sec);
    loadSection(section, sec);
  });
}

async function loadSection(section, secEl) {
  const listEl   = secEl.querySelector(`#auto-list-${section.id}`);
  const actionsEl= secEl.querySelector(`#auto-actions-${section.id}`);

  // ── Load list ────────────────────────────────────────────────
  let items = [];
  try {
    const res = await api(section.list);
    items = Array.isArray(res) ? res : [];
  } catch (e) {
    listEl.innerHTML = `<div class="drw-auto-error">Failed to load: ${e.message}</div>`;
    return;
  }

  // Self-improvement section was removed — sections here all use the
  // standard list+add form path.
  renderList(section, items, listEl);
  if (section.add) renderAddForm(section, actionsEl, () => loadSection(section, secEl));
}

function renderList(section, items, listEl) {
  if (!items.length) {
    listEl.innerHTML = `<div class="drw-auto-empty">${section.emptyMsg}</div>`;
    return;
  }
  listEl.innerHTML = '';
  items.forEach(it => {
    const row = document.createElement('div');
    row.className = 'drw-auto-row';
    const id = it.id ?? it.watch_id ?? it.monitor_id ?? it.schedule_id;
    const label = (it.label || it.name || it.url || it.path || '').toString();
    const enabled = it.enabled !== false;
    const meta = [];
    if (it.path)        meta.push(`<code>${escapeHtml(it.path)}</code>`);
    if (it.url)         meta.push(`<a href="${escapeAttr(it.url)}" target="_blank" rel="noreferrer">${escapeHtml(it.url)}</a>`);
    if (it.interval_seconds) meta.push(`every ${formatInterval(it.interval_seconds)}`);
    if (it.last_checked) meta.push(`checked ${it.last_checked}`);
    if (it.next_run)    meta.push(`next ${it.next_run}`);
    if (it.expression)  meta.push(`<code>${escapeHtml(it.expression)}</code>`);
    row.innerHTML = `
      <div class="drw-auto-row-main">
        <div class="drw-auto-row-title">
          <span class="drw-auto-dot ${enabled ? 'on' : 'off'}" title="${enabled ? 'Active' : 'Disabled'}"></span>
          ${escapeHtml(label || `(${id})`)}
        </div>
        <div class="drw-auto-row-meta">${meta.join(' &middot; ')}</div>
      </div>
      ${section.delete ? `<button class="drw-auto-del" title="Remove" data-id="${id}">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/></svg>
      </button>` : ''}
    `;
    if (section.delete) {
      row.querySelector('.drw-auto-del').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!confirm(`Remove this ${section.id.replace(/s$/,'')}?`)) return;
        try {
          await api(section.delete(id), { method: 'DELETE' });
          toast('Removed', 'success');
          loadSection(section, listEl.parentElement);
        } catch (err) { toast('Failed: ' + err.message, 'error'); }
      });
    }
    listEl.appendChild(row);
  });
}

function renderAddForm(section, actionsEl, onAdded) {
  const form = document.createElement('form');
  form.className = 'drw-auto-form';
  form.innerHTML = `
    <div class="drw-auto-form-fields">
      ${section.add.fields.map(f => `
        <input
          type="${f.type || 'text'}"
          name="${f.id}"
          placeholder="${escapeAttr(f.label + (f.placeholder ? ' — ' + f.placeholder : ''))}"
          required
        />`).join('')}
    </div>
    <button type="submit" class="drw-auto-add">
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      Add
    </button>
  `;
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const params = new URLSearchParams();
    for (const f of section.add.fields) {
      let v = fd.get(f.id);
      if (v == null || v === '') continue;
      if (f.type === 'number') v = parseInt(v, 10);
      params.set(f.id, v);
    }
    try {
      // API uses query parameters (not JSON body) for these POSTs
      await api(`${section.add.endpoint}?${params.toString()}`, { method: 'POST' });
      toast(`${section.title.replace(/s$/, '')} added`, 'success');
      form.reset();
      onAdded();
    } catch (err) { toast('Failed: ' + err.message, 'error'); }
  });
  actionsEl.appendChild(form);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function escapeAttr(s) { return escapeHtml(s); }
function formatInterval(sec) {
  if (sec < 60)   return `${sec}s`;
  if (sec < 3600) return `${Math.round(sec/60)}m`;
  if (sec < 86400)return `${Math.round(sec/3600)}h`;
  return `${Math.round(sec/86400)}d`;
}

// ── Create Custom API Integration ──────────────────────────────────────────────
function openCreateCustomApiModal(grid) {
  const root = document.getElementById('modal-root') || document.body;
  const backdrop = document.createElement('div');
  backdrop.className = 'modal-backdrop';

  backdrop.innerHTML = `
    <div class="modal" style="max-width:480px">
      <div class="modal-header">
        <div style="display:flex;align-items:center;gap:10px">
          <div style="width:32px;height:32px;border-radius:8px;background:#8b5cf610;display:flex;align-items:center;justify-content:center">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="1.5"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
          </div>
          <div>
            <div style="font-weight:700;font-size:14px">Add Custom API</div>
            <div style="font-size:11px;color:var(--tx3)">Connect any HTTP API endpoint</div>
          </div>
        </div>
        <button class="modal-close" id="custom-modal-close">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <div class="modal-body">
        <div class="modal-field">
          <label>Integration Name</label>
          <input type="text" id="custom-name" placeholder="e.g. My Internal API" autocomplete="off"/>
        </div>
        <div class="modal-field">
          <label>Base URL</label>
          <input type="text" id="custom-base-url" placeholder="https://api.example.com/v1" autocomplete="off"/>
        </div>
        <div class="modal-field">
          <label>API Key</label>
          <input type="password" id="custom-api-key" placeholder="Your API key" autocomplete="off"/>
        </div>
        <div class="modal-field">
          <label>Auth Type</label>
          <select id="custom-auth-type" style="width:100%;padding:8px;border:1px solid var(--bd2);border-radius:6px;background:var(--bg1);color:var(--tx1);font-size:13px">
            <option value="bearer">Bearer Token</option>
            <option value="apikey">API Key Header</option>
            <option value="basic">Basic Auth</option>
            <option value="none">No Auth</option>
          </select>
        </div>
        <div class="modal-field">
          <label>Default HTTP Method</label>
          <select id="custom-method" style="width:100%;padding:8px;border:1px solid var(--bd2);border-radius:6px;background:var(--bg1);color:var(--tx1);font-size:13px">
            <option value="GET">GET</option>
            <option value="POST">POST</option>
          </select>
        </div>
        <div class="modal-field">
          <label>Custom Headers (JSON, optional)</label>
          <textarea id="custom-headers" rows="2" placeholder='{"X-Custom-Header": "value"}'></textarea>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn ghost" id="custom-cancel">Cancel</button>
        <button class="btn primary" id="custom-save">Create</button>
      </div>
    </div>`;

  document.body.appendChild(backdrop);
  requestAnimationFrame(() => backdrop.classList.add('visible'));

  const close = () => { backdrop.classList.remove('visible'); setTimeout(() => backdrop.remove(), 200); };
  backdrop.querySelector('#custom-modal-close').addEventListener('click', close);
  backdrop.querySelector('#custom-cancel').addEventListener('click', close);
  backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });

  backdrop.querySelector('#custom-save').addEventListener('click', async () => {
    const name = backdrop.querySelector('#custom-name').value.trim();
    const baseUrl = backdrop.querySelector('#custom-base-url').value.trim();
    const apiKey = backdrop.querySelector('#custom-api-key').value.trim();
    const authType = backdrop.querySelector('#custom-auth-type').value;
    const method = backdrop.querySelector('#custom-method').value;
    const headersRaw = backdrop.querySelector('#custom-headers').value.trim();

    if (!name) { toast('Name is required', 'error'); return; }
    if (!baseUrl) { toast('Base URL is required', 'error'); return; }

    const intgId = 'custom_api_' + name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/_+$/, '');
    const configData = { base_url: baseUrl, method: method, auth_type: authType };
    if (apiKey) configData.api_key = apiKey;
    if (headersRaw) {
      try { configData.headers = JSON.parse(headersRaw); }
      catch { toast('Invalid JSON for headers', 'error'); return; }
    }

    try {
      await api(`/api/integrations/${encodeURIComponent(intgId)}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: configData }),
      });
      toast(`Custom API "${name}" created`, 'success');
      await refreshConnectedMap();
      close();
      renderIntgGrid(grid);
    } catch (e) { toast('Failed: ' + e.message, 'error'); }
  });
}

// ── VM Pull & Start ───────────────────────────────────────────────────────────
export function initVM() {
  const btn = document.getElementById('vm-trigger');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>';
    
    try {
      const status = await api('/api/vm/status');
      
      if (status.running) {
        toast('VM is already running', 'info');
        btn.disabled = false;
        btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg> Computer';
        return;
      }
      
      await api('/api/vm/start', { method: 'POST' });
      toast('VM started successfully', 'success');
    } catch (e) {
      if (e.message?.includes('404') || e.message?.includes('not found') || e.message?.includes('does not exist')) {
        try {
          toast('Pulling VM image...', 'info');
          await api('/api/vm/execute', {
            method: 'POST',
            body: JSON.stringify({ command: 'docker pull ubuntu:22.04', timeout: 300 }),
          });
          await api('/api/vm/start', { method: 'POST' });
          toast('VM pulled and started', 'success');
        } catch (pullErr) {
          toast('Failed to pull/start VM: ' + pullErr.message, 'error');
        }
      } else {
        toast('Failed to start VM: ' + e.message, 'error');
      }
    }
    
    btn.disabled = false;
    btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg> Computer';
  });
}
