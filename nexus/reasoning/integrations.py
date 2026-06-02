"""
Integration catalog and summaries for the Nexus reasoning engine.
"""
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

from ..memory import db


async def _get_connected_integrations_summary() -> str:


    """Get a summary of all connected integrations for the system prompt."""


    try:


        integrations = await db.list_connected_integrations(connected_only=True)


    except Exception:


        return "No integrations connected."


    if not integrations:


        return "No integrations connected."


    try:


        from ..tools.builtin.integration_tools import _ADAPTER_ACTIONS, _GENERIC_ENDPOINT_ROUTES


    except Exception:


        _ADAPTER_ACTIONS = {}


        _GENERIC_ENDPOINT_ROUTES = {}


    lines = ["Available integrations and their capabilities:"]


    for intg in integrations:


        sid = intg.get("id", "")


        name = intg.get("name", sid)


        desc = intg.get("description", "")


        # Check typed adapter actions first, then generic endpoint routes


        actions = _ADAPTER_ACTIONS.get(sid, {})


        if not actions:


            generic_routes = _GENERIC_ENDPOINT_ROUTES.get(sid, {})


            actions = {action: route.path for action, route in generic_routes.items()}


        actions_str = ", ".join(actions.keys()) if actions else "basic operations"


        line = f"- {name} ({sid}): {desc}"


        if actions:


            line += f"\n  Actions: {actions_str}"


        lines.append(line)


    return "\n".join(lines)


#  -  -  Full integration catalog (module-level, used by prompt builder and summary)  -  - 


_ALL_KNOWN = {


    'slack': ('Slack', 'Messaging', 'Send messages, channels, threads'),


    'discord': ('Discord', 'Messaging', 'Channels, DMs, webhooks'),


    'email': ('Email', 'Messaging', 'Send and read emails'),


    'github': ('GitHub', 'Dev', 'Repos, issues, PRs, commits'),


    'gitlab': ('GitLab', 'Dev', 'Projects, merge requests'),


    'bitbucket': ('Bitbucket', 'Dev', 'Repos, pipelines'),


    'vercel': ('Vercel', 'Hosting', 'Deployments, previews'),


    'netlify': ('Netlify', 'Hosting', 'Static hosting, forms'),


    'hostinger': ('Hostinger', 'Hosting', 'VPS, domains, DNS - NOTE: No website content API, only server management'),


    'railway': ('Railway', 'Hosting', 'Full-stack deploy'),


    'render': ('Render', 'Hosting', 'Web services, workers'),


    'flyio': ('Fly.io', 'Hosting', 'Global app hosting'),


    'heroku': ('Heroku', 'Hosting', 'Cloud platform'),


    'digitalocean': ('DigitalOcean', 'Hosting', 'Droplets, apps'),


    'cloudways': ('Cloudways', 'Hosting', 'Managed cloud hosting'),


    'notion': ('Notion', 'Productivity', 'Pages, databases'),


    'jira': ('Jira', 'Productivity', 'Issues, sprints'),


    'linear': ('Linear', 'Productivity', 'Issues, cycles'),


    'trello': ('Trello', 'Productivity', 'Boards, cards'),


    'asana': ('Asana', 'Productivity', 'Tasks, projects'),


    'clickup': ('ClickUp', 'Productivity', 'Tasks, docs, goals'),


    'monday': ('Monday.com', 'Productivity', 'Work management'),


    'airtable': ('Airtable', 'Data', 'Spreadsheets, databases'),


    'confluence': ('Confluence', 'Productivity', 'Documentation, wikis'),


    'basecamp': ('Basecamp', 'Productivity', 'Project management'),


    'google_drive': ('Google Drive', 'Storage', 'Files, folders'),


    'dropbox': ('Dropbox', 'Storage', 'Cloud storage'),


    'aws_s3': ('AWS S3', 'Storage', 'Object storage'),


    'backblaze': ('Backblaze B2', 'Storage', 'Cloud storage'),


    'wasabi': ('Wasabi', 'Storage', 'Hot cloud storage'),


    'google_sheets': ('Google Sheets', 'Data', 'Spreadsheets'),


    'hubspot': ('HubSpot', 'CRM', 'Contacts, deals'),


    'salesforce': ('Salesforce', 'CRM', 'Leads, opportunities'),


    'pipedrive': ('Pipedrive', 'CRM', 'Sales pipeline'),


    'zoho': ('Zoho', 'CRM', 'Business suite'),


    'close': ('Close.io', 'CRM', 'Sales engagement'),


    'openai': ('OpenAI', 'AI', 'GPT models'),


    'anthropic': ('Anthropic', 'AI', 'Claude models'),


    'cohere': ('Cohere', 'AI', 'Language AI'),


    'google_ai': ('Google AI', 'AI', 'Gemini models'),


    'huggingface': ('Hugging Face', 'AI', 'ML models, spaces'),


    'replicate': ('Replicate', 'AI', 'ML model hosting'),


    'stabilityai': ('Stability AI', 'AI', 'Image generation'),


    'zapier': ('Zapier', 'Automation', 'Workflow automation'),


    'make': ('Make', 'Automation', 'Visual automation'),


    'n8n': ('n8n', 'Automation', 'Workflow automation'),


    'ifttt': ('IFTTT', 'Automation', 'Simple automation'),


    'grafana': ('Grafana', 'Monitoring', 'Dashboards, metrics'),


    'pagerduty': ('PagerDuty', 'Monitoring', 'Incident management'),


    'datadog': ('Datadog', 'Monitoring', 'Observability platform'),


    'newrelic': ('New Relic', 'Monitoring', 'APM, monitoring'),


    'pingdom': ('Pingdom', 'Monitoring', 'Uptime monitoring'),


    'uptimerobot': ('UptimeRobot', 'Monitoring', 'Uptime monitoring'),


    'statuspage': ('StatusPage', 'Monitoring', 'Status pages'),


    'sentry': ('Sentry', 'Monitoring', 'Error tracking'),


    'posthog': ('PostHog', 'Analytics', 'Product analytics'),


    'amplitude': ('Amplitude', 'Analytics', 'Product analytics'),


    'mixpanel': ('Mixpanel', 'Analytics', 'Product analytics'),


    'segment': ('Segment', 'Analytics', 'Data pipeline'),


    'plausible': ('Plausible', 'Analytics', 'Privacy-first analytics'),


    'hotjar': ('Hotjar', 'Analytics', 'Heatmaps'),


    'heap': ('Heap', 'Analytics', 'Product analytics'),


    'terraform': ('Terraform', 'Infra', 'Infrastructure as Code'),


    'ansible': ('Ansible', 'Infra', 'Configuration management'),


    'kubernetes': ('Kubernetes', 'Infra', 'Container orchestration'),


    'docker': ('Docker', 'Infra', 'Containers'),


    'harbor': ('Harbor', 'Infra', 'Container registry'),


    'portainer': ('Portainer', 'Infra', 'Container management'),


    'cloudflare': ('Cloudflare', 'Domain', 'DNS, SSL, CDN'),


    'route53': ('Route53', 'Domain', 'AWS DNS'),


    'godaddy': ('GoDaddy', 'Domain', 'Domain registration'),


    'namecheap': ('Namecheap', 'Domain', 'Domain registration'),


    'letsencrypt': ("Let's Encrypt", 'Domain', 'Free SSL'),


    'zerossl': ('ZeroSSL', 'Domain', 'Free SSL certificates'),


    'porkbun': ('Porkbun', 'Domain', 'Domain registration'),


    'aws': ('AWS', 'Cloud', 'EC2, S3, Lambda'),


    'gcp': ('GCP', 'Cloud', 'Compute, Functions'),


    'azure': ('Azure', 'Cloud', 'Virtual machines'),


    'supabase': ('Supabase', 'Database', 'PostgreSQL, auth'),


    'firebase': ('Firebase', 'Database', 'Firestore, auth'),


    'planetscale': ('PlanetScale', 'Database', 'Serverless MySQL'),


    'dockerhub': ('Docker Hub', 'Registry', 'Container images'),


    'ghcr': ('GHCR', 'Registry', 'GitHub containers'),


    'circleci': ('CircleCI', 'CI/CD', 'Continuous integration'),


    'jenkins': ('Jenkins', 'CI/CD', 'Automation server'),


    'stripe': ('Stripe', 'Payments', 'Payment processing'),


    'paypal': ('PayPal', 'Payments', 'Payment processing'),


    'lemon_squeezy': ('Lemon Squeezy', 'Payments', 'Digital products'),


    'twilio': ('Twilio', 'Communication', 'SMS, voice, video'),


    'sendgrid': ('SendGrid', 'Communication', 'Email delivery'),


    'mailgun': ('Mailgun', 'Communication', 'Email API'),


    'postmark': ('Postmark', 'Communication', 'Email delivery'),


    'telegram': ('Telegram', 'Communication', 'Bot API'),


    'microsoft_teams': ('MS Teams', 'Communication', 'Chat, meetings'),


    'shopify': ('Shopify', 'E-commerce', 'Online store'),


    'woocommerce': ('WooCommerce', 'E-commerce', 'WordPress commerce'),


    'bigcommerce': ('BigCommerce', 'E-commerce', 'Online store'),


    'appstore': ('App Store', 'Distribution', 'iOS app publishing'),


    'googleplay': ('Google Play', 'Distribution', 'Android app publishing'),


}


# #15: System prompt cache  - avoid rebuilding on every call (data changes rarely)


# M7 fix: Made per-session to prevent cross-session prompt leakage.

