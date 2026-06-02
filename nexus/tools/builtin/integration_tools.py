"""
Integration API tools — v19 Complete Integration Coverage
==========================================================

Production-grade integration service mesh with:

1. **IntegrationHub** — centralized service mesh managing all connected services
   with health checking, auto-reconnect, and connection pooling.
2. **OAuth 2.0 Flow** — Authorization Code flow support with token refresh.
3. **Webhook Receiver** — HMAC-SHA256 signed webhook ingestion with event routing.
4. **Per-Service Rate Limiting** — respects HTTP 429 + self-imposed sliding window.
5. **GenericEndpointAdapter** — automatic REST endpoint routing for ALL marketplace
   services, even those without typed adapter classes. Uses endpoint route maps to
   translate action names into proper HTTP method + path + parameter placement.
6. **Full Coverage** — ALL 75+ marketplace services now have both typed adapters
   (for 38 core services) AND generic endpoint routing (for remaining services).
5. **API Versioning** — multi-version negotiation with graceful fallback.
6. **Data Transformation Pipeline** — field mapping, normalization, pagination.
7. **Integration Marketplace** — discoverable service catalog with rich metadata.
8. **Pre-built Service Adapters** — typed adapter classes for 50+ major services.
9. **Event Bus Integration** — broadcasts all integration lifecycle events.
10. **v17 Bug Fixes** — Trello headers preserved, configurable timeouts, error categorization.

v17 bugs fixed:
- Trello headers: no longer throw away custom headers when building auth
  (line 108 in v17 reset `headers = {}` after custom headers were merged)
- Timeout: configurable per-request via `timeout` param (default 30s)
- Error categorization: network, auth, rate-limit, timeout, validation, runtime
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from ...memory import db
from ..registry import tool

__all__ = [
    "IntegrationHub",
    "ServiceAdapter",
    "SlackAdapter",
    "GitHubAdapter",
    "NotionAdapter",
    "JiraAdapter",
    "LinearAdapter",
    "TrelloAdapter",
    "GrafanaAdapter",
    "GoogleDriveAdapter",
    "DiscordAdapter",
    "PostHogAdapter",
    "AmplitudeAdapter",
    "MixpanelAdapter",
    "HotjarAdapter",
    "CloudflareAdapter",
    "KubernetesAdapter",
    "DockerAdapter",
    "HeapAdapter",
    "HarborAdapter",
    "PortainerAdapter",
    "TerraformAdapter",
    "Route53Adapter",
    "GoDaddyAdapter",
    "NamecheapAdapter",
    "LetsEncryptAdapter",
    "ZeroSSLAdapter",
    "AWSAdapter",
    "GCPAdapter",
    "AzureAdapter",
    "VercelAdapter",
    "NetlifyAdapter",
    "SupabaseAdapter",
    "FirebaseAdapter",
    "DockerHubAdapter",
    "GHCRAdapter",
    "AppStoreAdapter",
    "GooglePlayAdapter",
    "SentryAdapter",
    "StripeAdapter",
    "RATE_LIMITER",
]

logger = logging.getLogger("nexus.integrations")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT: float = 30.0
_MAX_RESPONSE_CHARS: int = 6000
_MAX_CONNECTOR_LIMIT: int = 50
_MAX_KEEPALIVE: int = 20

# ---------------------------------------------------------------------------
# Error categorization
# ---------------------------------------------------------------------------

class IntegrationError(Enum):
    """Categories for integration errors."""

    NETWORK = "network"
    AUTH = "auth"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    RUNTIME = "runtime"
    NOT_CONNECTED = "not_connected"
    UNKNOWN = "unknown"


def _categorize_error(exc: Exception) -> IntegrationError:
    """Categorize an exception into an IntegrationError."""
    etype = type(exc).__name__
    if isinstance(exc, httpx.TimeoutException):
        return IntegrationError.TIMEOUT
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return IntegrationError.NETWORK
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 401 or code == 403:
            return IntegrationError.AUTH
        if code == 429:
            return IntegrationError.RATE_LIMITED
        if code == 400:
            return IntegrationError.VALIDATION
    if "timeout" in etype.lower():
        return IntegrationError.TIMEOUT
    if "auth" in etype.lower() or "unauthorized" in str(exc).lower():
        return IntegrationError.AUTH
    if "rate" in str(exc).lower() or "429" in str(exc):
        return IntegrationError.RATE_LIMITED
    return IntegrationError.UNKNOWN


# ---------------------------------------------------------------------------
# EventBus emitter helper (safe no-op when events module not loaded)
# ---------------------------------------------------------------------------

async def _emit(kind: str, **data: Any) -> None:
    """Emit an event to the EventBus, gracefully handling import errors."""
    try:
        from ...events import emit
        await emit(kind, **data)
    except Exception:
        pass  # Events module not available — skip silently


# ---------------------------------------------------------------------------
# Known service base URLs & metadata (marketplace catalog)
# ---------------------------------------------------------------------------

@dataclass
class ServiceMetadata:
    """Rich metadata for a connectable integration service."""

    service_id: str
    name: str
    category: str
    description: str
    auth_type: str  # "api_key" | "oauth2" | "bearer" | "basic" | "token"
    base_url: str  # default; can be overridden per-instance
    docs_url: str
    rate_limit_rpm: int = 60
    capabilities: list[str] = field(default_factory=list)
    api_versions: list[str] = field(default_factory=list)
    config_fields: list[dict[str, str]] = field(default_factory=list)


_MARKETPLACE: dict[str, ServiceMetadata] = {}


def _build_marketplace() -> dict[str, ServiceMetadata]:
    """Build the integration marketplace catalog."""
    services: list[ServiceMetadata] = [
        ServiceMetadata(
            service_id="slack",
            name="Slack",
            category="Communication",
            description="Send messages, list channels, manage conversations",
            auth_type="bearer",
            base_url="https://slack.com/api",
            docs_url="https://api.slack.com/methods",
            rate_limit_rpm=60,
            capabilities=["channels.list", "messages.send", "messages.list", "users.list", "channels.create", "channels.invite", "messages.update", "messages.delete"],
            config_fields=[
                {"key": "bot_token", "label": "Bot Token", "secret": True},
            ],
        ),
        ServiceMetadata(
            service_id="github",
            name="GitHub",
            category="Developer Tools",
            description="Manage repos, issues, pull requests, and files",
            auth_type="bearer",
            base_url="https://api.github.com",
            docs_url="https://docs.github.com/en/rest",
            rate_limit_rpm=5000,
            capabilities=["repos.list", "issues.list", "issues.create", "issues.update", "prs.list", "prs.create", "files.read", "branches.create", "prs.merge"],
            config_fields=[
                {"key": "api_key", "label": "Personal Access Token", "secret": True},
            ],
        ),
        ServiceMetadata(
            service_id="notion",
            name="Notion",
            category="Productivity",
            description="Query databases, manage pages, read block children",
            auth_type="bearer",
            base_url="https://api.notion.com/v1",
            docs_url="https://developers.notion.com/reference",
            rate_limit_rpm=170,
            capabilities=["pages.list", "pages.create", "pages.update", "databases.query", "databases.create", "blocks.children"],
            config_fields=[
                {"key": "token", "label": "Integration Token", "secret": True},
            ],
        ),
        ServiceMetadata(
            service_id="jira",
            name="Jira",
            category="Project Management",
            description="Search issues, create tickets, manage transitions",
            auth_type="bearer",
            base_url="",  # Requires domain in config
            docs_url="https://developer.atlassian.com/cloud/jira/platform/rest/v3/",
            rate_limit_rpm=100,
            capabilities=["projects.list", "issues.search", "issues.create", "issues.update", "issues.transition", "issues.assign", "issues.comment"],
            config_fields=[
                {"key": "domain", "label": "Instance URL", "secret": False},
                {"key": "api_key", "label": "API Token / PAT", "secret": True},
                {"key": "email", "label": "Email (for Basic auth)", "secret": False},
            ],
        ),
        ServiceMetadata(
            service_id="linear",
            name="Linear",
            category="Project Management",
            description="List and create issues, manage teams",
            auth_type="bearer",
            base_url="https://api.linear.app",
            docs_url="https://linear.app/docs/api",
            rate_limit_rpm=200,
            capabilities=["issues.list", "issues.create", "teams.list"],
            config_fields=[
                {"key": "api_key", "label": "API Key", "secret": True},
            ],
        ),
        ServiceMetadata(
            service_id="trello",
            name="Trello",
            category="Project Management",
            description="Manage boards, lists, and cards",
            auth_type="token",
            base_url="https://api.trello.com/1",
            docs_url="https://developer.atlassian.com/cloud/trello/rest/api-group/",
            rate_limit_rpm=120,
            capabilities=["boards.list", "cards.list", "cards.create", "lists.list"],
            config_fields=[
                {"key": "api_key", "label": "API Key", "secret": False},
                {"key": "token", "label": "Member Token", "secret": True},
            ],
        ),
        ServiceMetadata(
            service_id="grafana",
            name="Grafana",
            category="Monitoring",
            description="Query dashboards, create annotations, query datasources",
            auth_type="bearer",
            base_url="",  # Requires url in config
            docs_url="https://grafana.com/docs/grafana/latest/http_api/",
            rate_limit_rpm=300,
            capabilities=["dashboards.list", "annotations.create", "datasource.query"],
            config_fields=[
                {"key": "url", "label": "Grafana URL", "secret": False},
                {"key": "api_key", "label": "API Key / Service Account Token", "secret": True},
            ],
        ),
        ServiceMetadata(
            service_id="google_drive",
            name="Google Drive",
            category="Cloud Storage",
            description="List, upload, and read files from Google Drive",
            auth_type="oauth2",
            base_url="https://www.googleapis.com/drive/v3",
            docs_url="https://developers.google.com/drive/api/v3/reference",
            rate_limit_rpm=1200,
            capabilities=["files.list", "files.upload", "files.read"],
            config_fields=[
                {"key": "access_token", "label": "Access Token (OAuth)", "secret": True},
            ],
        ),
        ServiceMetadata(
            service_id="discord",
            name="Discord",
            category="Communication",
            description="Send messages, list channels, manage guilds",
            auth_type="bearer",
            base_url="https://discord.com/api/v10",
            docs_url="https://discord.com/developers/docs/intro",
            rate_limit_rpm=240,
            capabilities=["channels.list", "channels.create", "messages.send", "messages.delete", "messages.react", "guilds.list"],
            config_fields=[
                {"key": "bot_token", "label": "Bot Token", "secret": True},
            ],
        ),
        # ────────────────────────────────────────────────────────────────
        # Hosting & Deployment
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="hostinger", name="Hostinger VPS & Hosting", category="Hosting & Deployment",
            description="Manage VPS, domains, DNS, billing, backups, firewall, and more on Hostinger",
            auth_type="bearer", base_url="https://developers.hostinger.com",
            docs_url="https://developers.hostinger.com/", rate_limit_rpm=120,
            capabilities=[
                # VPS
                "vps.list", "vps.get", "vps.create", "vps.delete", "vps.start", "vps.stop", "vps.reboot", "vps.recreate",
                "vps.metrics", "vps.backups", "vps.backup.restore", "vps.firewall", "vps.setup",
                "vps.set_hostname", "vps.set_root_password", "vps.set_panel_password",
                "vps.actions", "vps.action_details",
                # SSH Keys
                "ssh_keys.list", "ssh_keys.attach", "ssh_keys.detach",
                # Infrastructure
                "datacenters", "plans.list",
                # Firewall Full Management
                "firewall.list", "firewall.create", "firewall.get", "firewall.delete",
                "firewall.rule.create", "firewall.rule.update", "firewall.rule.delete",
                "firewall.activate", "firewall.deactivate", "firewall.sync",
                # Docker
                "docker.projects", "docker.create_project", "docker.delete_project",
                "docker.start_project", "docker.stop_project", "docker.restart_project",
                # Domains
                "domains.list", "domains.get", "domains.check", "domains.forward",
                # DNS Records (for SSL certs via DNS-01)
                "dns.list", "dns.update", "dns.delete", "dns.reset", "dns.validate", "dns.snapshots", "dns.snapshot.restore",
                # Billing
                "billing.catalog", "billing.catalog_vps", "billing.orders", "billing.orders.create",
                "billing.subscriptions", "billing.payment_methods",
                # Raw API for any other endpoints (database, SSL, etc)
                "raw_api"
            ],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="vercel", name="Vercel", category="Hosting & Deployment",
            description="Deploy and manage serverless functions, previews, and production sites",
            auth_type="bearer", base_url="https://api.vercel.com",
            docs_url="https://vercel.com/docs/rest-api", rate_limit_rpm=60,
            capabilities=["deployments.list", "deployment.create", "projects.list", "projects.create", "domains.list", "envvars.list"],
            config_fields=[{"key": "api_key", "label": "Vercel Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="netlify", name="Netlify", category="Hosting & Deployment",
            description="Deploy static sites, manage DNS, and configure serverless functions",
            auth_type="bearer", base_url="https://api.netlify.com/api/v1",
            docs_url="https://open-api.netlify.com/", rate_limit_rpm=120,
            capabilities=["sites.list", "deploys.list", "deploy.trigger", "dns.list", "functions.list", "forms.list"],
            config_fields=[{"key": "api_key", "label": "Personal Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="railway", name="Railway", category="Hosting & Deployment",
            description="Deploy apps, databases, and services with Railway infrastructure",
            auth_type="bearer", base_url="https://backboard.railway.app/graphql/v2",
            docs_url="https://railway.app/docs", rate_limit_rpm=60,
            capabilities=["projects.list", "deployments.list", "services.list", "variables.list", "triggers.deploy"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="render", name="Render", category="Hosting & Deployment",
            description="Deploy web services, background workers, and databases on Render",
            auth_type="bearer", base_url="https://api.render.com/v1",
            docs_url="https://render.com/docs/api", rate_limit_rpm=60,
            capabilities=["services.list", "deploys.list", "deploy.create", "custom_domains.list", "envvars.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="flyio", name="Fly.io", category="Hosting & Deployment",
            description="Deploy containerized apps close to users with Fly.io edge network",
            auth_type="bearer", base_url="https://api.machines.dev/v1",
            docs_url="https://docs.machines.dev/", rate_limit_rpm=120,
            capabilities=["apps.list", "machines.list", "machines.create", "machines.start", "machines.stop", "volumes.list"],
            config_fields=[{"key": "api_key", "label": "Fly API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="heroku", name="Heroku", category="Hosting & Deployment",
            description="Manage Heroku apps, dynos, add-ons, and releases",
            auth_type="bearer", base_url="https://api.heroku.com",
            docs_url="https://devcenter.heroku.com/categories/platform-api-reference", rate_limit_rpm=120,
            capabilities=["apps.list", "apps.create", "dynos.list", "dynos.restart", "releases.list", "configvars.list"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="digitalocean", name="DigitalOcean", category="Hosting & Deployment",
            description="Manage droplets, Kubernetes, databases, and DNS on DigitalOcean",
            auth_type="bearer", base_url="https://api.digitalocean.com/v2",
            docs_url="https://docs.digitalocean.com/reference/api/", rate_limit_rpm=200,
            capabilities=["droplets.list", "droplets.create", "kubernetes.list", "databases.list", "domains.list", "volumes.list"],
            config_fields=[{"key": "api_key", "label": "Personal Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="cloudways", name="Cloudways", category="Hosting & Deployment",
            description="Managed cloud hosting with server and app management",
            auth_type="bearer", base_url="https://api.cloudways.com/api/v1",
            docs_url="https://developers.cloudways.com/docs/", rate_limit_rpm=60,
            capabilities=["servers.list", "apps.list", "apps.create", "services.list", "domains.manage"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}, {"key": "email", "label": "Email", "secret": False}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Cloud & Infrastructure
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="aws", name="Amazon Web Services", category="Cloud & Infrastructure",
            description="Manage EC2, S3, Lambda, CloudWatch, and more AWS services",
            auth_type="bearer", base_url="https://api.aws.amazon.com",
            docs_url="https://docs.aws.amazon.com/", rate_limit_rpm=100,
            capabilities=["ec2.list", "s3.buckets", "s3.objects", "lambda.list", "cloudwatch.metrics", "rds.list"],
            config_fields=[{"key": "api_key", "label": "Access Key ID", "secret": False}, {"key": "api_secret", "label": "Secret Access Key", "secret": True}, {"key": "region", "label": "AWS Region", "secret": False}],
        ),
        ServiceMetadata(
            service_id="gcp", name="Google Cloud Platform", category="Cloud & Infrastructure",
            description="Manage Compute, Cloud Functions, Cloud Run, and Storage on GCP",
            auth_type="bearer", base_url="https://www.googleapis.com",
            docs_url="https://cloud.google.com/docs", rate_limit_rpm=100,
            capabilities=["compute.list", "functions.list", "run.list", "storage.buckets", "bigquery.query", "pubsub.list"],
            config_fields=[{"key": "api_key", "label": "Service Account Key (JSON)", "secret": True}, {"key": "project_id", "label": "GCP Project ID", "secret": False}],
        ),
        ServiceMetadata(
            service_id="azure", name="Microsoft Azure", category="Cloud & Infrastructure",
            description="Manage VMs, App Services, Containers, and Functions on Azure",
            auth_type="bearer", base_url="https://management.azure.com",
            docs_url="https://learn.microsoft.com/en-us/rest/api/azure/", rate_limit_rpm=120,
            capabilities=["resources.list", "webapps.list", "containers.list", "functions.list", "storage.list", "sql.list"],
            config_fields=[{"key": "api_key", "label": "Bearer Token / Service Principal", "secret": True}, {"key": "subscription_id", "label": "Subscription ID", "secret": False}],
        ),
        ServiceMetadata(
            service_id="cloudflare", name="Cloudflare", category="Cloud & Infrastructure",
            description="Manage DNS, SSL/TLS, CDN caching, and firewall rules",
            auth_type="bearer", base_url="https://api.cloudflare.com/client/v4",
            docs_url="https://developers.cloudflare.com/api/", rate_limit_rpm=1200,
            capabilities=["zones.list", "dns.list", "dns.create", "ssl.list", "cache.purge", "firewall.rules"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="supabase", name="Supabase", category="Cloud & Infrastructure",
            description="Postgres database, Auth, Storage, and Realtime via Supabase",
            auth_type="bearer", base_url="https://api.supabase.com/v1",
            docs_url="https://supabase.com/docs/reference/api", rate_limit_rpm=60,
            capabilities=["table.select", "table.insert", "table.update", "auth.users", "storage.buckets", "projects.list"],
            config_fields=[{"key": "api_key", "label": "Service Role Key", "secret": True}, {"key": "project_url", "label": "Project URL", "secret": False}],
        ),
        ServiceMetadata(
            service_id="firebase", name="Firebase", category="Cloud & Infrastructure",
            description="Manage Firestore, Auth, Hosting, and Cloud Functions for Firebase",
            auth_type="oauth2", base_url="https://firebase.googleapis.com/v1beta1",
            docs_url="https://firebase.google.com/docs/reference/rest", rate_limit_rpm=100,
            capabilities=["projects.list", "auth.users", "firestore.collections", "hosting.list", "functions.list", "storage.list"],
            config_fields=[{"key": "access_token", "label": "OAuth Access Token", "secret": True}, {"key": "project_id", "label": "Firebase Project ID", "secret": False}],
        ),
        ServiceMetadata(
            service_id="planetable", name="PlanetScale", category="Cloud & Infrastructure",
            description="Serverless MySQL database platform with branching and scaling",
            auth_type="bearer", base_url="https://api.planetscale.com/v1",
            docs_url="https://planetscale.com/docs/api-reference", rate_limit_rpm=60,
            capabilities=["databases.list", "databases.create", "branches.list", "branches.create", "backups.list", "deploy_requests.list"],
            config_fields=[{"key": "api_key", "label": "Service Token", "secret": True}, {"key": "organization", "label": "Organization Slug", "secret": False}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Domain & DNS
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="godaddy", name="GoDaddy", category="Domain & DNS",
            description="Register and manage domains, DNS records, and SSL certificates",
            auth_type="api_key", base_url="https://api.godaddy.com",
            docs_url="https://developer.godaddy.com/doc/endpoint", rate_limit_rpm=60,
            capabilities=["domains.list", "dns.records", "dns.create", "domains.available", "certificates.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": False}, {"key": "api_secret", "label": "API Secret", "secret": True}],
        ),
        ServiceMetadata(
            service_id="namecheap", name="Namecheap", category="Domain & DNS",
            description="Manage domain registrations, DNS, and SSL via Namecheap",
            auth_type="token", base_url="https://api.namecheap.com/xml.response",
            docs_url="https://www.namecheap.com/support/api/intro/", rate_limit_rpm=30,
            capabilities=["domains.list", "dns.get", "dns.set", "domains.check", "ssl.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}, {"key": "username", "label": "Username", "secret": False}, {"key": "client_ip", "label": "Whitelisted IP", "secret": False}],
        ),
        ServiceMetadata(
            service_id="route53", name="AWS Route53", category="Domain & DNS",
            description="Manage DNS hosted zones, record sets, and health checks",
            auth_type="bearer", base_url="https://route53.amazonaws.com/2013-04-01",
            docs_url="https://docs.aws.amazon.com/Route53/latest/APIReference/", rate_limit_rpm=300,
            capabilities=["zones.list", "records.list", "records.change", "healthchecks.list", "traffic.policies"],
            config_fields=[{"key": "api_key", "label": "Access Key ID", "secret": False}, {"key": "api_secret", "label": "Secret Access Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="letsencrypt", name="Let's Encrypt", category="Domain & DNS",
            description="ACME protocol for free SSL/TLS certificate issuance and renewal",
            auth_type="bearer", base_url="https://acme-v02.api.letsencrypt.org",
            docs_url="https://letsencrypt.org/docs/", rate_limit_rpm=20,
            capabilities=["order.new", "authz.new", "certificate.issue", "certificate.revoke", "account.status"],
            config_fields=[{"key": "api_key", "label": "ACME Account Key (PEM)", "secret": True}],
        ),
        ServiceMetadata(
            service_id="porkbun", name="Porkbun", category="Domain & DNS",
            description="Manage domains and DNS records via Porkbun API",
            auth_type="api_key", base_url="https://porkbun.com/api/json/v3",
            docs_url="https://porkbun.com/api/json/v3/documentation", rate_limit_rpm=30,
            capabilities=["domains.list", "dns.list", "dns.create", "dns.edit", "dns.delete", "domains.pricing"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}, {"key": "secret_key", "label": "Secret Key", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Communication
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="twilio", name="Twilio", category="Communication",
            description="Send SMS, make voice calls, and manage phone numbers",
            auth_type="basic", base_url="https://api.twilio.com/2010-04-01",
            docs_url="https://www.twilio.com/docs/api", rate_limit_rpm=60,
            capabilities=["sms.send", "calls.make", "numbers.list", "messages.list", "accounts.list", "recordings.list"],
            config_fields=[{"key": "api_key", "label": "Account SID", "secret": False}, {"key": "api_secret", "label": "Auth Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="sendgrid", name="SendGrid", category="Communication",
            description="Send transactional emails, manage contact lists, and templates",
            auth_type="bearer", base_url="https://api.sendgrid.com/v3",
            docs_url="https://docs.sendgrid.com/api-reference", rate_limit_rpm=600,
            capabilities=["mail.send", "contacts.list", "templates.list", "stats.list", "suppressions.list", "api_keys.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="mailgun", name="Mailgun", category="Communication",
            description="Send, receive, and track emails programmatically",
            auth_type="basic", base_url="https://api.mailgun.net/v3",
            docs_url="https://documentation.mailgun.com/en/latest/api_reference.html", rate_limit_rpm=120,
            capabilities=["messages.send", "domains.list", "events.list", "routes.list", "webhooks.list", "stats.list"],
            config_fields=[{"key": "api_key", "label": "Private API Key", "secret": True}, {"key": "domain", "label": "Sending Domain", "secret": False}],
        ),
        ServiceMetadata(
            service_id="postmark", name="Postmark", category="Communication",
            description="Deliver transactional emails with high inbox placement rates",
            auth_type="bearer", base_url="https://api.postmarkapp.com",
            docs_url="https://postmarkapp.com/developer/api/overview", rate_limit_rpm=300,
            capabilities=["email.send", "email.batch", "bounces.list", "templates.list", "stats.list", "domains.list"],
            config_fields=[{"key": "api_key", "label": "Server API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="telegram", name="Telegram Bot", category="Communication",
            description="Send messages, manage chats, and handle bot commands via Telegram",
            auth_type="bearer", base_url="https://api.telegram.org",
            docs_url="https://core.telegram.org/bots/api", rate_limit_rpm=30,
            capabilities=["messages.send", "messages.edit", "chats.list", "updates.get", "stickers.list", "files.send"],
            config_fields=[{"key": "bot_token", "label": "Bot Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="microsoft_teams", name="Microsoft Teams", category="Communication",
            description="Send messages, manage channels, and interact with Teams workspaces",
            auth_type="bearer", base_url="https://graph.microsoft.com/v1.0",
            docs_url="https://learn.microsoft.com/en-us/graph/api/resources/teams-api-overview", rate_limit_rpm=120,
            capabilities=["channels.list", "messages.send", "messages.list", "teams.list", "files.list", "meetings.list"],
            config_fields=[{"key": "access_token", "label": "OAuth Access Token", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Developer Tools
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="gitlab", name="GitLab", category="Developer Tools",
            description="Manage repos, issues, merge requests, CI/CD pipelines",
            auth_type="bearer", base_url="https://gitlab.com/api/v4",
            docs_url="https://docs.gitlab.com/ee/api/rest/", rate_limit_rpm=500,
            capabilities=["projects.list", "issues.list", "issues.create", "merges.list", "pipelines.list", "runners.list"],
            config_fields=[{"key": "api_key", "label": "Private Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="bitbucket", name="Bitbucket", category="Developer Tools",
            description="Manage repos, pull requests, pipelines, and deployments",
            auth_type="bearer", base_url="https://api.bitbucket.org/2.0",
            docs_url="https://developer.atlassian.com/cloud/bitbucket/rest/", rate_limit_rpm=1000,
            capabilities=["repositories.list", "pullrequests.list", "pipelines.list", "deployments.list", "issues.list", "webhooks.list"],
            config_fields=[{"key": "api_key", "label": "App Password", "secret": True}, {"key": "username", "label": "Username", "secret": False}],
        ),
        ServiceMetadata(
            service_id="dockerhub", name="Docker Hub", category="Developer Tools",
            description="Manage container image repositories, tags, and builds",
            auth_type="bearer", base_url="https://hub.docker.com/v2",
            docs_url="https://docs.docker.com/docker-hub/api/latest/", rate_limit_rpm=100,
            capabilities=["repositories.list", "repository.tags", "repository.manifests", "builds.list", "organizations.list", "access_tokens.list"],
            config_fields=[{"key": "api_key", "label": "Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="ghcr", name="GitHub Container Registry", category="Developer Tools",
            description="Manage container packages and versions via GHCR",
            auth_type="bearer", base_url="https://api.github.com",
            docs_url="https://docs.github.com/en/packages/working-with-a-github-packages-registry", rate_limit_rpm=5000,
            capabilities=["packages.list", "package.versions", "container.list", "packages.delete", "packages.restore"],
            config_fields=[{"key": "api_key", "label": "Personal Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="circleci", name="CircleCI", category="Developer Tools",
            description="Trigger builds, manage pipelines, and view workflow status",
            auth_type="bearer", base_url="https://circleci.com/api/v2",
            docs_url="https://circleci.com/docs/api/v2/", rate_limit_rpm=60,
            capabilities=["pipelines.list", "pipeline.trigger", "workflows.list", "jobs.list", "artifacts.list", "contexts.list"],
            config_fields=[{"key": "api_key", "label": "Personal API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="jenkins", name="Jenkins", category="Developer Tools",
            description="Trigger builds, view job status, and manage Jenkins CI",
            auth_type="basic", base_url="",
            docs_url="https://www.jenkins.io/doc/book/using/remote-access-api/", rate_limit_rpm=120,
            capabilities=["jobs.list", "build.trigger", "build.status", "queue.list", "nodes.list", "credentials.list"],
            config_fields=[{"key": "url", "label": "Jenkins URL", "secret": False}, {"key": "username", "label": "Username", "secret": False}, {"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="pagerduty", name="PagerDuty", category="Developer Tools",
            description="Manage incidents, on-call schedules, and escalation policies",
            auth_type="token", base_url="https://api.pagerduty.com",
            docs_url="https://developer.pagerduty.com/api-reference/", rate_limit_rpm=120,
            capabilities=["incidents.list", "incidents.create", "oncalls.list", "services.list", "schedules.list", "escalation.list"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="sentry", name="Sentry", category="Developer Tools",
            description="Track errors, monitor performance, and manage release health",
            auth_type="bearer", base_url="https://sentry.io/api/0",
            docs_url="https://docs.sentry.io/api/", rate_limit_rpm=120,
            capabilities=["issues.list", "events.list", "projects.list", "releases.list", "organizations.list", "teams.list"],
            config_fields=[{"key": "api_key", "label": "Auth Token", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Productivity
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="asana", name="Asana", category="Productivity",
            description="Manage projects, tasks, sections, and tags in Asana",
            auth_type="bearer", base_url="https://app.asana.com/api/1.0",
            docs_url="https://developers.asana.com/docs", rate_limit_rpm=150,
            capabilities=["projects.list", "tasks.list", "tasks.create", "tasks.update", "sections.list", "tags.list", "workspaces.list"],
            config_fields=[{"key": "access_token", "label": "Personal Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="clickup", name="ClickUp", category="Productivity",
            description="Manage spaces, lists, tasks, and time tracking in ClickUp",
            auth_type="bearer", base_url="https://api.clickup.com/api/v2",
            docs_url="https://clickup.com/api", rate_limit_rpm=100,
            capabilities=["spaces.list", "lists.list", "tasks.list", "tasks.create", "tasks.update", "time.list", "goals.list"],
            config_fields=[{"key": "api_key", "label": "Personal API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="monday", name="Monday.com", category="Productivity",
            description="Manage boards, items, groups, and updates on Monday.com",
            auth_type="bearer", base_url="https://api.monday.com/v2",
            docs_url="https://developer.monday.com/api-reference/docs", rate_limit_rpm=60,
            capabilities=["boards.list", "items.list", "items.create", "updates.create", "groups.list", "workspaces.list"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="airtable", name="Airtable", category="Productivity",
            description="Read and write records, manage bases, tables, and views",
            auth_type="bearer", base_url="https://api.airtable.com/v0",
            docs_url="https://airtable.com/developers/web/api/introduction", rate_limit_rpm=300,
            capabilities=["records.list", "records.create", "records.update", "records.delete", "bases.list", "tables.list"],
            config_fields=[{"key": "api_key", "label": "Personal Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="confluence", name="Confluence", category="Productivity",
            description="Create and manage wiki pages, spaces, and content in Confluence",
            auth_type="bearer", base_url="",
            docs_url="https://developer.atlassian.com/cloud/confluence/rest/v2/intro/", rate_limit_rpm=100,
            capabilities=["pages.list", "pages.create", "pages.update", "spaces.list", "attachments.list", "comments.list"],
            config_fields=[{"key": "domain", "label": "Confluence URL", "secret": False}, {"key": "api_key", "label": "API Token", "secret": True}, {"key": "email", "label": "Email", "secret": False}],
        ),
        ServiceMetadata(
            service_id="basecamp", name="Basecamp", category="Productivity",
            description="Manage projects, to-dos, messages, and schedules in Basecamp",
            auth_type="bearer", base_url="https://3.basecampapi.com",
            docs_url="https://github.com/basecamp/bc3-api", rate_limit_rpm=120,
            capabilities=["projects.list", "todos.list", "todos.create", "messages.list", "schedules.list", "campfires.list"],
            config_fields=[{"key": "access_token", "label": "OAuth Access Token", "secret": True}, {"key": "account_id", "label": "Account ID", "secret": False}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Data & Analytics
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="google_sheets", name="Google Sheets", category="Data & Analytics",
            description="Read and write spreadsheet data, manage sheets and formatting",
            auth_type="oauth2", base_url="https://sheets.googleapis.com/v4",
            docs_url="https://developers.google.com/sheets/api/reference/rest", rate_limit_rpm=300,
            capabilities=["spreadsheets.get", "spreadsheets.create", "values.get", "values.update", "values.append", "sheets.list"],
            config_fields=[{"key": "access_token", "label": "OAuth Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="posthog", name="PostHog", category="Data & Analytics",
            description="Capture events, analyze funnels, and manage feature flags",
            auth_type="bearer", base_url="https://app.posthog.com",
            docs_url="https://posthog.com/docs/api", rate_limit_rpm=300,
            capabilities=["events.capture", "events.list", "trends", "funnels.list", "feature_flags.list", "persons.list"],
            config_fields=[{"key": "api_key", "label": "Personal API Key", "secret": True}, {"key": "project_api_key", "label": "Project API Key (public)", "secret": False}],
        ),
        ServiceMetadata(
            service_id="amplitude", name="Amplitude", category="Data & Analytics",
            description="Track user events, build behavioral cohorts, and analyze funnels",
            auth_type="bearer", base_url="https://amplitude.com/api/2",
            docs_url="https://amplitude.com/docs/api", rate_limit_rpm=120,
            capabilities=["track", "users.list", "chart", "events.list", "cohort.list", "funnel.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": False}, {"key": "api_secret", "label": "Secret Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="mixpanel", name="Mixpanel", category="Data & Analytics",
            description="Track events, query data, and manage user profiles",
            auth_type="basic", base_url="https://mixpanel.com/api/2.0",
            docs_url="https://developer.mixpanel.com/reference/overview", rate_limit_rpm=60,
            capabilities=["track", "engage", "export", "funnels.list", "retention.list", "query"],
            config_fields=[{"key": "api_key", "label": "API Secret", "secret": True}, {"key": "api_secret", "label": "Service Account Secret", "secret": True}],
        ),
        ServiceMetadata(
            service_id="segment", name="Segment", category="Data & Analytics",
            description="Collect, unify, and route customer data to destinations",
            auth_type="bearer", base_url="https://api.segment.io/v1",
            docs_url="https://segment.com/docs/api/", rate_limit_rpm=120,
            capabilities=["track", "identify", "page", "group", "sources.list", "destinations.list"],
            config_fields=[{"key": "api_key", "label": "Write Key (public)", "secret": False}, {"key": "api_secret", "label": "Personal Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="plausible", name="Plausible", category="Data & Analytics",
            description="Privacy-friendly web analytics with simple API access",
            auth_type="bearer", base_url="https://plausible.io/api/v1",
            docs_url="https://plausible.io/docs/api", rate_limit_rpm=60,
            capabilities=["stats.aggregate", "stats.timeseries", "stats.breakdown", "sites.list", "goals.list", "events.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="hotjar", name="Hotjar", category="Data & Analytics",
            description="Heatmaps, session recordings, and user feedback analytics",
            auth_type="bearer", base_url="https://api.hotjar.com/v1",
            docs_url="https://help.hotjar.com/hc/en-us/articles/4405109977047", rate_limit_rpm=60,
            capabilities=["sites.list", "heatmaps.list", "recordings.list", "feedback.list", "surveys.list"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # CRM & Sales
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="hubspot", name="HubSpot", category="CRM & Sales",
            description="Manage contacts, deals, companies, and marketing workflows",
            auth_type="bearer", base_url="https://api.hubapi.com",
            docs_url="https://developers.hubspot.com/docs/api/overview", rate_limit_rpm=100,
            capabilities=["contacts.list", "contacts.create", "deals.list", "companies.list", "tickets.list", "pipelines.list"],
            config_fields=[{"key": "access_token", "label": "Private App Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="salesforce", name="Salesforce", category="CRM & Sales",
            description="Query and manage CRM objects, leads, opportunities, and cases",
            auth_type="bearer", base_url="",
            docs_url="https://developer.salesforce.com/docs/atlas.en-us.api_rest.meta/api_rest/", rate_limit_rpm=100,
            capabilities=["contacts.list", "leads.list", "opportunities.list", "cases.list", "accounts.list", "soql.query"],
            config_fields=[{"key": "access_token", "label": "OAuth Access Token", "secret": True}, {"key": "instance_url", "label": "Instance URL", "secret": False}],
        ),
        ServiceMetadata(
            service_id="pipedrive", name="Pipedrive", category="CRM & Sales",
            description="Manage deals, persons, organizations, and activities",
            auth_type="bearer", base_url="https://api.pipedrive.com/v1",
            docs_url="https://developers.pipedrive.com/docs/api/v1", rate_limit_rpm=120,
            capabilities=["deals.list", "persons.list", "organizations.list", "activities.list", "pipelines.list", "products.list"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="zoho", name="Zoho", category="CRM & Sales",
            description="Manage CRM modules, leads, contacts, deals, and tasks",
            auth_type="oauth2", base_url="https://www.zohoapis.com/crm/v2",
            docs_url="https://www.zoho.com/crm/developer/docs/api/", rate_limit_rpm=100,
            capabilities=["leads.list", "contacts.list", "deals.list", "accounts.list", "tasks.list", "modules.list"],
            config_fields=[{"key": "access_token", "label": "OAuth Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="close", name="Close.io", category="CRM & Sales",
            description="Manage leads, contacts, opportunities, and call activities",
            auth_type="basic", base_url="https://api.close.com/api/v1",
            docs_url="https://developer.close.com/", rate_limit_rpm=120,
            capabilities=["leads.list", "contacts.list", "opportunities.list", "activities.list", "tasks.list", "sequences.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Payments
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="stripe", name="Stripe", category="Payments",
            description="Process payments, manage customers, subscriptions, and invoices",
            auth_type="bearer", base_url="https://api.stripe.com/v1",
            docs_url="https://docs.stripe.com/api", rate_limit_rpm=1000,
            capabilities=["customers.list", "charges.list", "payment.create", "invoices.list", "subscriptions.list", "products.list"],
            config_fields=[{"key": "api_key", "label": "Secret Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="paypal", name="PayPal", category="Payments",
            description="Process payments, manage orders, and handle payouts",
            auth_type="bearer", base_url="https://api-m.paypal.com",
            docs_url="https://developer.paypal.com/api/rest/", rate_limit_rpm=120,
            capabilities=["orders.create", "orders.list", "payments.list", "payouts.create", "subscriptions.list", "invoices.list"],
            config_fields=[{"key": "client_id", "label": "Client ID", "secret": False}, {"key": "client_secret", "label": "Client Secret", "secret": True}],
        ),
        ServiceMetadata(
            service_id="lemon_squeezy", name="Lemon Squeezy", category="Payments",
            description="Sell digital products, manage subscriptions, and handle licenses",
            auth_type="bearer", base_url="https://api.lemonsqueezy.com/v1",
            docs_url="https://docs.lemonsqueezy.com/api", rate_limit_rpm=60,
            capabilities=["products.list", "orders.list", "customers.list", "subscriptions.list", "licenses.list", "discounts.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Monitoring
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="datadog", name="Datadog", category="Monitoring",
            description="Monitor infrastructure, APM, logs, and dashboards",
            auth_type="api_key", base_url="https://api.datadoghq.com/api/v1",
            docs_url="https://docs.datadoghq.com/api/latest/", rate_limit_rpm=300,
            capabilities=["metrics.list", "metrics.query", "dashboards.list", "monitors.list", "events.list", "hosts.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": False}, {"key": "application_key", "label": "Application Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="newrelic", name="New Relic", category="Monitoring",
            description="APM, infrastructure monitoring, and log management via NerdGraph",
            auth_type="bearer", base_url="https://api.newrelic.com/graphql",
            docs_url="https://docs.newrelic.com/docs/apis/nerdgraph/get-started/intro-new-relic-nerdgraph/", rate_limit_rpm=120,
            capabilities=["accounts.list", "apps.list", "alerts.list", "deployments.list", "nrql.query", "dashboards.list"],
            config_fields=[{"key": "api_key", "label": "User API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="pingdom", name="Pingdom", category="Monitoring",
            description="Uptime monitoring, page speed, and transaction checks",
            auth_type="bearer", base_url="https://api.pingdom.com/api/3.1",
            docs_url="https://docs.pingdom.com/api/", rate_limit_rpm=60,
            capabilities=["checks.list", "checks.create", "results.list", "actions.list", "teams.list", "maintenance.list"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="uptimerobot", name="UptimeRobot", category="Monitoring",
            description="Monitor website uptime, port, and keyword with alerting",
            auth_type="bearer", base_url="https://api.uptimerobot.com/v2",
            docs_url="https://uptimerobot.com/api/", rate_limit_rpm=60,
            capabilities=["monitors.list", "monitors.create", "monitors.reset", "alert_contacts.list", "mwindows.list", "psp.list"],
            config_fields=[{"key": "api_key", "label": "Main API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="statuspage", name="StatusPage", category="Monitoring",
            description="Create and manage status pages for service availability",
            auth_type="bearer", base_url="https://api.statuspage.io/v1",
            docs_url="https://developer.statuspage.io/", rate_limit_rpm=120,
            capabilities=["pages.list", "incidents.list", "incidents.create", "components.list", "subscribers.list", "metrics.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}, {"key": "page_id", "label": "Page ID", "secret": False}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Automation
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="zapier", name="Zapier", category="Automation",
            description="Connect apps and automate workflows with Zaps",
            auth_type="bearer", base_url="https://zapier.com/api/v2",
            docs_url="https://developer.zapier.com/api/v2/docs", rate_limit_rpm=60,
            capabilities=["zaps.list", "zaps.run", "connections.list", "actions.list", "triggers.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="make", name="Make (Integromat)", category="Automation",
            description="Create automated workflows connecting apps and services",
            auth_type="token", base_url="https://api.make.com/v2",
            docs_url="https://www.make.com/en/api-documentation", rate_limit_rpm=60,
            capabilities=["scenarios.list", "scenarios.run", "connections.list", "organizations.list", "teams.list"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="n8n", name="n8n", category="Automation",
            description="Self-hosted workflow automation with 400+ integrations",
            auth_type="bearer", base_url="",
            docs_url="https://docs.n8n.io/api/", rate_limit_rpm=120,
            capabilities=["workflows.list", "workflows.create", "workflows.execute", "executions.list", "credentials.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}, {"key": "url", "label": "n8n Instance URL", "secret": False}],
        ),
        ServiceMetadata(
            service_id="ifttt", name="IFTTT", category="Automation",
            description="Create applets that connect and automate IoT and web services",
            auth_type="bearer", base_url="https://connect.ifttt.com/v2",
            docs_url="https://platform.ifttt.com/docs/api_reference", rate_limit_rpm=60,
            capabilities=["applets.list", "triggers.fire", "actions.create", "services.list", "connections.list"],
            config_fields=[{"key": "api_key", "label": "Service Key", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Storage
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="dropbox", name="Dropbox", category="Storage",
            description="Store, sync, and share files via Dropbox cloud storage",
            auth_type="bearer", base_url="https://api.dropboxapi.com/2",
            docs_url="https://www.dropbox.com/developers/documentation/http/documentation", rate_limit_rpm=500,
            capabilities=["files.list", "files.upload", "files.download", "files.delete", "folders.create", "sharing.list"],
            config_fields=[{"key": "access_token", "label": "OAuth Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="aws_s3", name="AWS S3", category="Storage",
            description="Store and retrieve objects in Amazon S3 buckets",
            auth_type="bearer", base_url="https://s3.amazonaws.com",
            docs_url="https://docs.aws.amazon.com/AmazonS3/latest/API/", rate_limit_rpm=5500,
            capabilities=["buckets.list", "objects.list", "objects.get", "objects.put", "objects.delete", "presign.url"],
            config_fields=[{"key": "api_key", "label": "Access Key ID", "secret": False}, {"key": "api_secret", "label": "Secret Access Key", "secret": True}, {"key": "region", "label": "AWS Region", "secret": False}],
        ),
        ServiceMetadata(
            service_id="backblaze", name="Backblaze B2", category="Storage",
            description="Cloud object storage compatible with S3 API at lower cost",
            auth_type="basic", base_url="https://api.backblazeb2.com",
            docs_url="https://www.backblaze.com/b2/docs/", rate_limit_rpm=120,
            capabilities=["buckets.list", "buckets.create", "files.list", "files.upload", "files.download", "files.delete"],
            config_fields=[{"key": "api_key", "label": "Key ID", "secret": False}, {"key": "api_secret", "label": "Application Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="wasabi", name="Wasabi", category="Storage",
            description="S3-compatible hot cloud storage with no egress fees",
            auth_type="bearer", base_url="https://s3.wasabisys.com",
            docs_url="https://wasabi.com/help/docs/", rate_limit_rpm=300,
            capabilities=["buckets.list", "objects.list", "objects.get", "objects.put", "objects.delete", "buckets.create"],
            config_fields=[{"key": "api_key", "label": "Access Key ID", "secret": False}, {"key": "api_secret", "label": "Secret Access Key", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # Container & Infra
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="kubernetes", name="Kubernetes", category="Container & Infra",
            description="Manage pods, services, deployments, and namespaces in K8s clusters",
            auth_type="bearer", base_url="",
            docs_url="https://kubernetes.io/docs/reference/kubernetes-api/", rate_limit_rpm=300,
            capabilities=["namespaces.list", "pods.list", "services.list", "deployments.list", "pods.delete", "configmaps.list"],
            config_fields=[{"key": "api_key", "label": "Service Account Token", "secret": True}, {"key": "url", "label": "API Server URL", "secret": False}],
        ),
        ServiceMetadata(
            service_id="docker", name="Docker", category="Container & Infra",
            description="Manage local or remote Docker containers and images",
            auth_type="bearer", base_url="",
            docs_url="https://docs.docker.com/engine/api/", rate_limit_rpm=300,
            capabilities=["containers.list", "images.list", "container.create", "container.start", "container.stop", "image.pull"],
            config_fields=[{"key": "url", "label": "Docker Host URL", "secret": False}],
        ),
        ServiceMetadata(
            service_id="harbor", name="Harbor", category="Container & Infra",
            description="Manage container images, artifacts, and vulnerability scans",
            auth_type="basic", base_url="",
            docs_url="https://harbor.dev/docs/2.0.0/administration/installation/", rate_limit_rpm=120,
            capabilities=["repositories.list", "artifacts.list", "scan", "projects.list", "members.list", "retention.list"],
            config_fields=[{"key": "url", "label": "Harbor URL", "secret": False}, {"key": "api_key", "label": "Credentials", "secret": True}],
        ),
        ServiceMetadata(
            service_id="portainer", name="Portainer", category="Container & Infra",
            description="Web UI for Docker and Kubernetes management via Portainer",
            auth_type="bearer", base_url="",
            docs_url="https://docs.portainer.io/api", rate_limit_rpm=120,
            capabilities=["containers.list", "images.list", "volumes.list", "networks.list", "stacks.list", "endpoints.list"],
            config_fields=[{"key": "url", "label": "Portainer URL", "secret": False}, {"key": "api_key", "label": "Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="terraform", name="Terraform Cloud", category="Container & Infra",
            description="Manage infrastructure as code with workspaces, runs, and state",
            auth_type="bearer", base_url="https://app.terraform.io",
            docs_url="https://developer.hashicorp.com/terraform/cloud-docs/api-docs", rate_limit_rpm=30,
            capabilities=["workspaces.list", "runs.list", "run.create", "state.list", "variables.list", "organizations.list"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="ansible", name="Ansible (AWX/Tower)", category="Container & Infra",
            description="Run automation playbooks, manage inventories and job templates",
            auth_type="bearer", base_url="",
            docs_url="https://docs.ansible.com/automation-controller/latest/html/userguide/api.html", rate_limit_rpm=120,
            capabilities=["job_templates.list", "jobs.launch", "inventories.list", "credentials.list", "projects.list", "hosts.list"],
            config_fields=[{"key": "url", "label": "AWX/Tower URL", "secret": False}, {"key": "api_key", "label": "OAuth Token", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # AI & ML
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="openai", name="OpenAI API", category="AI & ML",
            description="Access GPT models, DALL-E, Whisper, and embeddings",
            auth_type="bearer", base_url="https://api.openai.com/v1",
            docs_url="https://platform.openai.com/docs/api-reference", rate_limit_rpm=60,
            capabilities=["chat.completions", "completions.create", "embeddings.create", "images.generate", "audio.transcribe", "models.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="huggingface", name="Hugging Face", category="AI & ML",
            description="Access models, datasets, and inference endpoints on HF Hub",
            auth_type="bearer", base_url="https://huggingface.co/api",
            docs_url="https://huggingface.co/docs/hub/api", rate_limit_rpm=120,
            capabilities=["models.list", "models.info", "inference.run", "datasets.list", "spaces.list", "endpoints.list"],
            config_fields=[{"key": "api_key", "label": "Access Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="replicate", name="Replicate", category="AI & ML",
            description="Run open-source ML models via API with automatic scaling",
            auth_type="bearer", base_url="https://api.replicate.com/v1",
            docs_url="https://replicate.com/docs/reference/http", rate_limit_rpm=60,
            capabilities=["predictions.create", "predictions.list", "predictions.get", "models.list", "collections.list", "trainings.create"],
            config_fields=[{"key": "api_key", "label": "API Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="stabilityai", name="Stability AI", category="AI & ML",
            description="Generate images with Stable Diffusion and other Stability models",
            auth_type="bearer", base_url="https://api.stability.ai/v1",
            docs_url="https://platform.stability.ai/docs/api-reference", rate_limit_rpm=150,
            capabilities=["image.generate", "image.upscale", "image.variations", "engines.list", "accounts.balance", "history.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="anthropic", name="Anthropic API", category="AI & ML",
            description="Access Claude models for text generation and analysis",
            auth_type="bearer", base_url="https://api.anthropic.com/v1",
            docs_url="https://docs.anthropic.com/en/api/", rate_limit_rpm=60,
            capabilities=["messages.create", "messages.list", "models.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="cohere", name="Cohere", category="AI & ML",
            description="Language models for text generation, embedding, and RAG",
            auth_type="bearer", base_url="https://api.cohere.ai/v1",
            docs_url="https://docs.cohere.com/reference", rate_limit_rpm=100,
            capabilities=["chat", "embed", "generate", "rerank", "classify", "models.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="google_ai", name="Google AI (Gemini)", category="AI & ML",
            description="Access Gemini models for multimodal AI via Google AI API",
            auth_type="api_key", base_url="https://generativelanguage.googleapis.com/v1beta",
            docs_url="https://ai.google.dev/docs/api_reference", rate_limit_rpm=60,
            capabilities=["generate", "chat", "embed", "models.list", "files.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        # ────────────────────────────────────────────────────────────────
        # E-commerce
        # ────────────────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="shopify", name="Shopify", category="E-commerce",
            description="Manage products, orders, customers, and inventory on Shopify",
            auth_type="bearer", base_url="",
            docs_url="https://shopify.dev/docs/api/admin-rest", rate_limit_rpm=40,
            capabilities=["products.list", "orders.list", "customers.list", "inventory.list", "fulfillments.list", "discounts.list"],
            config_fields=[{"key": "access_token", "label": "Admin API Access Token", "secret": True}, {"key": "shop_domain", "label": "Shop Domain", "secret": False}],
        ),
        ServiceMetadata(
            service_id="woocommerce", name="WooCommerce", category="E-commerce",
            description="Manage products, orders, coupons, and customers in WooCommerce",
            auth_type="basic", base_url="",
            docs_url="https://woocommerce.github.io/woocommerce-rest-api-docs/", rate_limit_rpm=120,
            capabilities=["products.list", "orders.list", "customers.list", "coupons.list", "reports.list", "categories.list"],
            config_fields=[{"key": "url", "label": "Store URL", "secret": False}, {"key": "api_key", "label": "Consumer Key", "secret": False}, {"key": "api_secret", "label": "Consumer Secret", "secret": True}],
        ),
        ServiceMetadata(
            service_id="bigcommerce", name="BigCommerce", category="E-commerce",
            description="Manage storefronts, products, orders, and catalogs on BigCommerce",
            auth_type="bearer", base_url="",
            docs_url="https://developer.bigcommerce.com/api-reference", rate_limit_rpm=120,
            capabilities=["products.list", "orders.list", "customers.list", "categories.list", "brands.list", "storefront.info"],
            config_fields=[{"key": "access_token", "label": "Store API Token", "secret": True}, {"key": "store_hash", "label": "Store Hash", "secret": False}],
        ),
        ServiceMetadata(
            service_id="youtube", name="YouTube", category="Media",
            description="Search videos, get channel info, manage playlists via YouTube Data API",
            auth_type="api_key", base_url="https://www.googleapis.com/youtube/v3",
            docs_url="https://developers.google.com/youtube/v3/docs", rate_limit_rpm=60,
            capabilities=["videos.list", "videos.get", "search.list", "channels.list",
                          "playlists.list", "playlistItems.list", "comments.list", "captions.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),

        # ── Triggers & Built-ins ───────────────────────────────────────
        ServiceMetadata(
            service_id="amplitude_event", name="Amplitude Event Trigger", category="Analytics",
            description="Amplitude analytics event trigger — fires on tracked events",
            auth_type="none", base_url="https://api.amplitude.com",
            docs_url="https://www.docs.developers.amplitude.com/",
            rate_limit_rpm=60,
            capabilities=["event.receive"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="discord_msg", name="Discord Message Trigger", category="Communication",
            description="Discord message trigger — fires on new messages",
            auth_type="none", base_url="https://discord.com/api",
            docs_url="https://discord.com/developers/docs",
            rate_limit_rpm=60,
            capabilities=["message.receive"],
            config_fields=[{"key": "bot_token", "label": "Bot Token", "secret": True}],
        ),
        ServiceMetadata(
            service_id="file_changed", name="File Changed Trigger", category="Storage",
            description="File change trigger — fires when files are modified",
            auth_type="none", base_url="https://www.googleapis.com/drive/v3",
            docs_url="https://developers.google.com/drive",
            rate_limit_rpm=60,
            capabilities=["file.watch"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="new_commit", name="New Commit Trigger", category="Developer Tools",
            description="Git commit trigger — fires on new commits pushed to a repository",
            auth_type="none", base_url="https://api.github.com",
            docs_url="https://docs.github.com/en/rest",
            rate_limit_rpm=60,
            capabilities=["commit.receive"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="new_email", name="New Email Trigger", category="Communication",
            description="Email trigger — fires when new email arrives",
            auth_type="none", base_url="https://gmail.googleapis.com",
            docs_url="https://developers.google.com/gmail/api",
            rate_limit_rpm=60,
            capabilities=["email.receive"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="new_issue", name="New Issue Trigger", category="Developer Tools",
            description="Issue trigger — fires when a new issue is opened",
            auth_type="none", base_url="https://api.github.com",
            docs_url="https://docs.github.com/en/rest",
            rate_limit_rpm=60,
            capabilities=["issue.receive"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="new_message", name="New Message Trigger", category="Communication",
            description="Message trigger — fires on new messages",
            auth_type="none", base_url="https://slack.com/api",
            docs_url="https://api.slack.com/methods",
            rate_limit_rpm=60,
            capabilities=["message.receive"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="posthog_event", name="PostHog Event Trigger", category="Analytics",
            description="PostHog event trigger — fires on tracked events",
            auth_type="none", base_url="https://app.posthog.com",
            docs_url="https://posthog.com/docs/api",
            rate_limit_rpm=60,
            capabilities=["event.receive"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        ServiceMetadata(
            service_id="heap", name="Heap Analytics", category="Analytics",
            description="Heap Analytics — auto-capture product analytics and event tracking",
            auth_type="bearer", base_url="https://heapanalytics.com/api",
            docs_url="https://developers.heap.io/",
            rate_limit_rpm=60,
            capabilities=["events.list", "events.track", "users.list"],
            config_fields=[{"key": "api_key", "label": "API Key", "secret": True}],
        ),
        # ── Schedule / Cron ───────────────────────────────────────────
        ServiceMetadata(
            service_id="schedule", name="Schedule / CRON", category="Automation",
            description="Time-based recurring schedule trigger",
            auth_type="none", base_url="http://localhost",
            docs_url="",
            rate_limit_rpm=0,
            capabilities=["schedule.trigger"],
            config_fields=[],
        ),
        ServiceMetadata(
            service_id="schedule_hr", name="Every Hour", category="Automation",
            description="Fires once every hour",
            auth_type="none", base_url="http://localhost",
            docs_url="",
            rate_limit_rpm=0,
            capabilities=["schedule.trigger"],
            config_fields=[],
        ),
        ServiceMetadata(
            service_id="schedule_daily", name="Daily at 9am", category="Automation",
            description="Fires every day at 9:00 AM",
            auth_type="none", base_url="http://localhost",
            docs_url="",
            rate_limit_rpm=0,
            capabilities=["schedule.trigger"],
            config_fields=[],
        ),
        ServiceMetadata(
            service_id="webhook", name="Webhook", category="Automation",
            description="HTTP webhook receiver — fires on incoming POST requests",
            auth_type="none", base_url="http://localhost",
            docs_url="",
            rate_limit_rpm=0,
            capabilities=["webhook.receive"],
            config_fields=[],
        ),
        # ── Communication (SMTP-based) ───────────────────────────────
        ServiceMetadata(
            service_id="email", name="Email / SMTP", category="Communication",
            description="Send and receive email via SMTP/IMAP",
            auth_type="none", base_url="",
            docs_url="",
            rate_limit_rpm=0,
            capabilities=["email.send", "email.receive"],
            config_fields=[
                {"key": "smtp_host", "label": "SMTP Host", "secret": False},
                {"key": "smtp_port", "label": "SMTP Port", "secret": False},
                {"key": "username", "label": "Username", "secret": False},
                {"key": "password", "label": "Password", "secret": True},
            ],
        ),
        # ── Distribution Platforms ───────────────────────────────────
        ServiceMetadata(
            service_id="appstore", name="App Store Connect", category="Distribution",
            description="Apple App Store Connect API — manage app metadata, releases, and analytics",
            auth_type="bearer", base_url="https://api.appstoreconnect.apple.com/v1",
            docs_url="https://developer.apple.com/documentation/appstoreconnectapi",
            rate_limit_rpm=60,
            capabilities=["apps.list", "apps.get", "builds.list", "releases.list"],
            config_fields=[
                {"key": "api_key", "label": "API Key", "secret": True},
                {"key": "issuer_id", "label": "Issuer ID", "secret": False},
            ],
        ),
        ServiceMetadata(
            service_id="googleplay", name="Google Play Console", category="Distribution",
            description="Google Play Console API — manage app releases, reviews, and analytics",
            auth_type="bearer", base_url="https://androidpublisher.googleapis.com/androidpublisher/v3",
            docs_url="https://developers.google.com/android-publisher",
            rate_limit_rpm=60,
            capabilities=["apps.list", "releases.list", "reviews.list", "orders.list"],
            config_fields=[
                {"key": "api_key", "label": "API Key", "secret": True},
                {"key": "package_name", "label": "Package Name", "secret": False},
            ],
        ),
        # ── SSL ────────────────────────────────────────────────────
        ServiceMetadata(
            service_id="zerossl", name="ZeroSSL", category="Domain & SSL",
            description="ZeroSSL API — issue, manage, and automate SSL certificates",
            auth_type="bearer", base_url="https://api.zerossl.com",
            docs_url="https://zerossl.com/documentation/api/",
            rate_limit_rpm=30,
            capabilities=["certificates.list", "certificates.create", "certificates.get", "certificates.delete", "certificates.renew"],
            config_fields=[
                {"key": "api_key", "label": "API Key", "secret": True},
                {"key": "email", "label": "Email", "secret": False},
            ],
        ),
    ]
    return {s.service_id: s for s in services}


_MARKETPLACE.update(_build_marketplace())


# ---------------------------------------------------------------------------
# Per-Service Rate Limiting
# ---------------------------------------------------------------------------

class _ServiceRateLimiter:
    """Sliding-window rate limiter per service.

    Tracks both self-imposed limits (from marketplace metadata) and
    remote-enforced limits (from HTTP 429 Retry-After headers).
    """

    def __init__(self) -> None:
        # service_id -> list of request timestamps
        self._windows: dict[str, list[float]] = {}
        # service_id -> float timestamp when the service is throttled until
        self._throttled_until: dict[str, float] = {}
        # service_id -> int requests remaining (from X-RateLimit-Remaining)
        self._remaining: dict[str, int] = {}
        # service_id -> int max requests per window
        self._limits: dict[str, int] = {}

    def _get_window(self, service_id: str) -> list[float]:
        if service_id not in self._windows:
            self._windows[service_id] = []
        return self._windows[service_id]

    def _evict(self, window: list[float], cutoff: float) -> list[float]:
        """Binary search to remove timestamps outside the window."""
        lo, hi = 0, len(window)
        while lo < hi:
            mid = (lo + hi) // 2
            if window[mid] <= cutoff:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0:
            return window[lo:]
        return window

    def record_request(self, service_id: str) -> None:
        """Record that a request was made."""
        now = time.time()
        window = self._get_window(service_id)
        window.append(now)
        window_60 = self._evict(window, now - 60.0)
        self._windows[service_id] = window_60

    def get_status(self, service_id: str) -> dict[str, Any]:
        """Get rate limit status for a service."""
        now = time.time()
        window = self._get_window(service_id)
        clean_window = self._evict(window, now - 60.0)
        self._windows[service_id] = clean_window
        throttled_until = self._throttled_until.get(service_id, 0.0)
        remaining = self._remaining.get(service_id)
        limit = self._limits.get(service_id)

        status: dict[str, Any] = {
            "requests_in_last_60s": len(clean_window),
            "throttled": now < throttled_until,
            "throttled_until": throttled_until if throttled_until > now else None,
        }
        if remaining is not None:
            status["remaining"] = remaining
        if limit is not None:
            status["limit"] = limit
        return status

    def should_throttle(self, service_id: str) -> tuple[bool, float]:
        """Check if we should throttle requests.

        Returns (should_throttle, wait_seconds).
        """
        now = time.time()
        # Check remote throttle
        throttled_until = self._throttled_until.get(service_id, 0.0)
        if now < throttled_until:
            return True, throttled_until - now

        # Check self-imposed limit from marketplace
        meta = _MARKETPLACE.get(service_id)
        if meta:
            window = self._get_window(service_id)
            clean_window = self._evict(window, now - 60.0)
            self._windows[service_id] = clean_window
            if len(clean_window) >= meta.rate_limit_rpm:
                return True, 60.0

        # Check X-RateLimit-Remaining
        remaining = self._remaining.get(service_id)
        if remaining is not None and remaining <= 1:
            return True, 5.0

        return False, 0.0

    def apply_response_headers(self, service_id: str, headers: dict[str, str]) -> None:
        """Update rate limit state from response headers."""
        remaining = headers.get("x-ratelimit-remaining")
        if remaining is not None:
            try:
                self._remaining[service_id] = int(remaining)
            except (ValueError, TypeError):
                pass

        limit = headers.get("x-ratelimit-limit")
        if limit is not None:
            try:
                self._limits[service_id] = int(limit)
            except (ValueError, TypeError):
                pass

        reset = headers.get("x-ratelimit-reset")
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                wait = float(retry_after)
                self._throttled_until[service_id] = time.time() + wait
            except (ValueError, TypeError):
                pass
        elif reset:
            try:
                self._throttled_until[service_id] = float(reset)
            except (ValueError, TypeError):
                pass


RATE_LIMITER = _ServiceRateLimiter()


# ---------------------------------------------------------------------------
# Data Transformation Pipeline
# ---------------------------------------------------------------------------

@dataclass
class FieldMapping:
    """Maps a source field path to a normalized output field."""

    source: str  # dot-separated path e.g. "data.items"
    target: str  # normalized field name
    transform: str = "identity"  # "identity" | "string" | "int" | "float" | "bool" | "iso_datetime"


@dataclass
class PaginationConfig:
    """Describes how a service paginates list endpoints."""

    next_cursor_field: str = ""  # e.g. "next_cursor"
    next_page_field: str = ""  # e.g. "next_page_token" or "next_page"
    page_param: str = "page"
    per_page_param: str = "per_page"
    default_per_page: int = 50
    max_pages: int = 20


# Field mappings per service for response normalization
_SERVICE_FIELD_MAPPINGS: dict[str, list[FieldMapping]] = {
    "slack": [
        FieldMapping("channels", "items"),
        FieldMapping("ok", "success", "bool"),
    ],
    "github": [
        FieldMapping("", "items"),  # Most GitHub endpoints return arrays directly
    ],
    "notion": [
        FieldMapping("results", "items"),
        FieldMapping("has_more", "has_more", "bool"),
        FieldMapping("next_cursor", "next_cursor"),
    ],
    "jira": [
        FieldMapping("issues", "items"),
        FieldMapping("total", "total", "int"),
    ],
    "linear": [
        FieldMapping("nodes", "items"),
        FieldMapping("pageInfo.hasNextPage", "has_more", "bool"),
        FieldMapping("pageInfo.endCursor", "next_cursor"),
    ],
    "trello": [],  # Trello returns flat arrays
    "cloudflare": [FieldMapping("result", "items"), FieldMapping("success", "success", "bool")],
    "stripe": [FieldMapping("data", "items"), FieldMapping("has_more", "has_more", "bool")],
    "sentry": [FieldMapping("", "items")],
    "gitlab": [FieldMapping("", "items")],
    "bitbucket": [FieldMapping("values", "items"), FieldMapping("next", "next_cursor")],
    "hubspot": [FieldMapping("results", "items"), FieldMapping("paging.next.after", "next_cursor")],
    "airtable": [FieldMapping("records", "items"), FieldMapping("offset", "next_cursor")],
    "asana": [FieldMapping("data", "items"), FieldMapping("next_page.offset", "next_cursor")],
    "digitalocean": [FieldMapping("droplets", "items")],
    "vercel": [FieldMapping("deployments", "items")],
    "shopify": [FieldMapping("products", "items")],
    "salesforce": [FieldMapping("records", "items"), FieldMapping("nextRecordsUrl", "next_cursor")],
}

_SERVICE_PAGINATION: dict[str, PaginationConfig] = {
    "slack": PaginationConfig(next_cursor_field="response_metadata.next_cursor", page_param="cursor", per_page_param="limit", default_per_page=100, max_pages=50),
    "github": PaginationConfig(page_param="page", per_page_param="per_page", default_per_page=100, max_pages=30),
    "notion": PaginationConfig(next_cursor_field="next_cursor", page_param="start_cursor", per_page_param="page_size", default_per_page=100, max_pages=100),
    "jira": PaginationConfig(page_param="startAt", per_page_param="maxResults", default_per_page=50, max_pages=50),
    "linear": PaginationConfig(next_cursor_field="pageInfo.endCursor", per_page_param="first", default_per_page=50, max_pages=50),
    "trello": PaginationConfig(page_param="page", per_page_param="limit", default_per_page=100, max_pages=100),
    "cloudflare": PaginationConfig(page_param="page", per_page_param="per_page", default_per_page=50, max_pages=50),
    "stripe": PaginationConfig(next_cursor_field="has_more", page_param="starting_after", per_page_param="limit", default_per_page=100, max_pages=100),
    "sentry": PaginationConfig(page_param="cursor", per_page_param="limit", default_per_page=100, max_pages=100),
    "gitlab": PaginationConfig(page_param="page", per_page_param="per_page", default_per_page=100, max_pages=100),
    "bitbucket": PaginationConfig(next_cursor_field="next", page_param="page", per_page_param="pagelen", default_per_page=50, max_pages=50),
    "hubspot": PaginationConfig(next_cursor_field="paging.next.after", page_param="after", per_page_param="limit", default_per_page=100, max_pages=500),
    "airtable": PaginationConfig(next_cursor_field="offset", page_param="offset", per_page_param="pageSize", default_per_page=100, max_pages=100),
    "asana": PaginationConfig(next_cursor_field="next_page.offset", page_param="offset", per_page_param="limit", default_per_page=100, max_pages=100),
    "digitalocean": PaginationConfig(page_param="page", per_page_param="per_page", default_per_page=40, max_pages=50),
    "vercel": PaginationConfig(next_cursor_field="pagination.next", page_param="until", per_page_param="limit", default_per_page=100, max_pages=100),
    "shopify": PaginationConfig(page_param="page_info", per_page_param="limit", default_per_page=50, max_pages=50),
    "sendgrid": PaginationConfig(page_param="offset", per_page_param="limit", default_per_page=100, max_pages=100),
}


def _get_nested(data: Any, path: str) -> Any:
    """Resolve a dot-separated path against a nested dict/list."""
    if not path:
        return data
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


def _apply_transform(value: Any, transform: str) -> Any:
    """Apply a transform function to a value."""
    if transform == "identity":
        return value
    if transform == "string":
        return str(value) if value is not None else ""
    if transform == "int":
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    if transform == "float":
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    if transform == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if transform == "iso_datetime":
        if isinstance(value, (int, float)):
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))
        return str(value)
    return value


def _transform_response(
    service_id: str,
    raw_data: Any,
    *,
    enrich: bool = True,
) -> dict[str, Any]:
    """Normalize a service response using field mappings.

    Returns a normalized dict with:
    - ``items``: the primary data list (or the raw data if unmapped)
    - ``has_more``: whether more pages exist
    - ``next_cursor``: cursor for next page (if applicable)
    - ``_meta``: enrichment metadata (timestamp, service, counts)
    """
    mappings = _SERVICE_FIELD_MAPPINGS.get(service_id, [])

    result: dict[str, Any] = {
        "service": service_id,
        "items": [],
        "has_more": False,
        "next_cursor": None,
        "raw_data": None,
    }

    if mappings and isinstance(raw_data, dict):
        for mapping in mappings:
            source_val = _get_nested(raw_data, mapping.source)
            if source_val is not None:
                result[mapping.target] = _apply_transform(source_val, mapping.transform)

        # If "items" wasn't mapped, use the raw data as items if it's a list
        if not result.get("items") and isinstance(raw_data, list):
            result["items"] = raw_data
        elif not result.get("items") and isinstance(raw_data, dict):
            # Try to find a list field
            for k, v in raw_data.items():
                if isinstance(v, list) and len(v) > 0:
                    result["items"] = v
                    break
            else:
                result["items"] = [raw_data]
    elif isinstance(raw_data, list):
        result["items"] = raw_data
    else:
        result["items"] = [raw_data] if raw_data is not None else []
        result["raw_data"] = raw_data

    # Enrichment
    if enrich:
        result["_meta"] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "service": service_id,
            "item_count": len(result.get("items", [])),
            "transformed": bool(mappings),
        }

    return result


def _truncate_output(data: Any, max_chars: int = _MAX_RESPONSE_CHARS) -> str:
    """Serialize data to JSON string, truncated to max_chars."""
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, total {len(text)} chars]"
    return text


# ---------------------------------------------------------------------------
# OAuth 2.0 Flow Manager
# ---------------------------------------------------------------------------

@dataclass
class OAuthState:
    """Stores in-flight OAuth 2.0 Authorization Code flow state."""

    service_id: str
    authorization_url: str
    code_verifier: str = ""
    redirect_uri: str = ""
    expires_at: float = 0.0
    scopes: list[str] = field(default_factory=list)


_OAUTH_CONFIGS: dict[str, dict[str, Any]] = {
    "google_drive": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
        "use_pkce": True,
    },
}


def _generate_code_verifier() -> str:
    """Generate a PKCE code verifier (43-128 chars)."""
    import secrets
    return secrets.token_urlsafe(32)


def _generate_code_challenge(verifier: str) -> str:
    """Generate PKCE S256 code challenge from verifier."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# ---------------------------------------------------------------------------
# Webhook System
# ---------------------------------------------------------------------------

@dataclass
class WebhookConfig:
    """Configuration for a registered webhook."""

    service_id: str
    webhook_id: str
    secret: str  # HMAC signing secret
    event_types: list[str] = field(default_factory=list)
    enabled: bool = True
    created_at: float = field(default_factory=time.time)


class WebhookManager:
    """Manages webhook registrations and signature verification."""

    def __init__(self) -> None:
        self._hooks: dict[str, WebhookConfig] = {}  # "service_id:webhook_id" -> config

    def register(
        self,
        service_id: str,
        webhook_id: str,
        secret: str,
        event_types: list[str] | None = None,
    ) -> WebhookConfig:
        """Register a new webhook."""
        key = f"{service_id}:{webhook_id}"
        config = WebhookConfig(
            service_id=service_id,
            webhook_id=webhook_id,
            secret=secret,
            event_types=event_types or [],
        )
        self._hooks[key] = config
        logger.info("Registered webhook %s for service %s", webhook_id, service_id)
        return config

    def unregister(self, service_id: str, webhook_id: str) -> bool:
        """Unregister a webhook. Returns True if it existed."""
        key = f"{service_id}:{webhook_id}"
        removed = self._hooks.pop(key, None)
        return removed is not None

    def get(self, service_id: str, webhook_id: str) -> WebhookConfig | None:
        """Get a webhook config."""
        return self._hooks.get(f"{service_id}:{webhook_id}")

    def verify_signature(
        self,
        service_id: str,
        webhook_id: str,
        payload: bytes,
        signature_header: str,
    ) -> bool:
        """Verify HMAC-SHA256 signature of a webhook payload.

        Supports multiple signature formats:
        - ``sha256=<hex>`` (GitHub, Slack style)
        - ``t=<timestamp>,v1=<hex>`` (Slack signing)
        - Raw hex digest
        """
        config = self.get(service_id, webhook_id)
        if not config:
            logger.warning("Webhook %s:%s not found", service_id, webhook_id)
            return False

        # Format 2: t=<timestamp>,v1=<hex> (Slack signing)
        # Slack's signing format requires HMAC over "{timestamp}:{payload}"
        if "," in signature_header:
            parts = signature_header.split(",")
            timestamp = ""
            received = ""
            for part in parts:
                if part.startswith("t="):
                    timestamp = part[2:]
                elif part.startswith("v1="):
                    received = part[3:]
            if timestamp and received:
                sig_basestring = f"{timestamp}:{payload.decode() if isinstance(payload, bytes) else payload}"
                expected = hmac.new(
                    config.secret.encode(),
                    sig_basestring.encode(),
                    hashlib.sha256,
                ).hexdigest()
                return hmac.compare_digest(received, expected)

        # Default HMAC computation (no timestamp)
        expected = hmac.new(
            config.secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        # Format 1: sha256=<hex>
        if signature_header.startswith("sha256="):
            received = signature_header[7:]
            return hmac.compare_digest(received, expected)

        # Format 3: raw hex
        return hmac.compare_digest(signature_header, expected)

    def list_webhooks(self, service_id: str | None = None) -> list[dict[str, Any]]:
        """List all webhooks, optionally filtered by service."""
        hooks = []
        for key, config in self._hooks.items():
            if service_id and config.service_id != service_id:
                continue
            hooks.append({
                "service_id": config.service_id,
                "webhook_id": config.webhook_id,
                "event_types": config.event_types,
                "enabled": config.enabled,
                "created_at": config.created_at,
            })
        return hooks


WEBHOOK_MANAGER = WebhookManager()


async def _handle_webhook_event(
    service_id: str,
    webhook_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Process a verified webhook event: store in DB and emit on event bus.

    Stores the event as a pending task so the agent can pick it up during
    the next reasoning cycle.
    """
    event_data = {
        "service_id": service_id,
        "webhook_id": webhook_id,
        "event_type": event_type,
        "payload": payload,
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Store as pending task in DB
    try:
        task_title = f"[Webhook] {service_id}: {event_type}"
        summary = json.dumps(event_data, ensure_ascii=False, default=str)[:500]
        task_id = await db.add_task(
            goal_id=None,
            title=task_title,
            scheduled_for=time.time(),
        )
        if task_id:
            await db.update_task(task_id, result=summary, status="pending")

        # Also store webhook event log in the journal
        await db.add_journal_entry(
            content=f"Webhook received from {service_id} ({event_type}): {summary}",
            mood="neutral",
            tags=["webhook", service_id, event_type],
            importance=0.6,
            permanent=False,
        )
    except Exception as exc:
        logger.error("Failed to persist webhook event: %s", exc)

    # Emit on event bus
    await _emit(
        "webhook_received",
        service_id=service_id,
        webhook_id=webhook_id,
        event_type=event_type,
        payload=payload,
    )

    logger.info(
        "Webhook event processed: %s/%s -> %s",
        service_id, webhook_id, event_type,
    )

    return {"status": "accepted", "task_created": True, "event_type": event_type}


# ---------------------------------------------------------------------------
# Connection Pooling
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


async def _get_shared_client() -> httpx.AsyncClient:
    """Get or create the shared httpx.AsyncClient for connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=30.0),
            limits=httpx.Limits(
                max_connections=_MAX_CONNECTOR_LIMIT,
                max_keepalive_connections=_MAX_KEEPALIVE,
                keepalive_expiry=30.0,
            ),
            http2=False,
            follow_redirects=True,
        )
    return _http_client


async def _close_shared_client() -> None:
    """Close the shared HTTP client (for cleanup)."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# Integration Health State
# ---------------------------------------------------------------------------

@dataclass
class IntegrationHealthState:
    """Tracks health for a single integration."""

    service_id: str
    healthy: bool = True
    last_check: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    avg_latency_ms: float = 0.0
    reason: str = ""
    reconnect_attempts: int = 0
    next_reconnect: float = 0.0


class IntegrationHub:
    """Centralized service mesh that manages all connected integrations.

    Features:
    - Health checking with periodic ping/verify
    - Auto-reconnect with exponential backoff
    - Connection pooling via shared httpx.AsyncClient
    - Per-service rate limiting
    - Webhook management
    - OAuth 2.0 flow state
    """

    _HEALTH_CHECK_INTERVAL: float = 300.0  # 5 minutes
    _MAX_RECONNECT_ATTEMPTS: int = 10
    _BASE_RECONNECT_DELAY: float = 2.0

    def __init__(self) -> None:
        self._health: dict[str, IntegrationHealthState] = {}
        self._oauth_states: dict[str, OAuthState] = {}
        self._last_health_check: float = 0.0

    # --- Health Management ---

    def _get_health(self, service_id: str) -> IntegrationHealthState:
        if service_id not in self._health:
            self._health[service_id] = IntegrationHealthState(service_id=service_id)
        return self._health[service_id]

    def record_success(self, service_id: str, latency_ms: float) -> None:
        """Record a successful API call."""
        health = self._get_health(service_id)
        health.healthy = True
        health.last_success = time.time()
        health.consecutive_failures = 0
        health.total_successes += 1
        health.reconnect_attempts = 0
        health.reason = ""

        # Running average latency
        n = health.total_successes + health.total_failures
        health.avg_latency_ms = health.avg_latency_ms * (1 - 1 / max(n, 1)) + latency_ms / max(n, 1)

    def record_failure(self, service_id: str, error: IntegrationError, detail: str = "") -> None:
        """Record a failed API call."""
        health = self._get_health(service_id)
        health.last_failure = time.time()
        health.consecutive_failures += 1
        health.total_failures += 1
        health.reason = f"{error.value}: {detail}"

        # If too many consecutive failures, mark unhealthy
        if health.consecutive_failures >= 5:
            health.healthy = False
            self._schedule_reconnect(service_id)

    def _schedule_reconnect(self, service_id: str) -> None:
        """Schedule auto-reconnect with exponential backoff."""
        health = self._get_health(service_id)
        health.reconnect_attempts += 1
        if health.reconnect_attempts > self._MAX_RECONNECT_ATTEMPTS:
            return
        delay = self._BASE_RECONNECT_DELAY * (2 ** (health.reconnect_attempts - 1))
        health.next_reconnect = time.time() + delay
        logger.info(
            "Scheduled reconnect for %s in %.1fs (attempt %d)",
            service_id, delay, health.reconnect_attempts,
        )

    def should_reconnect(self, service_id: str) -> bool:
        """Check if a reconnect attempt should be made."""
        health = self._get_health(service_id)
        if health.healthy:
            return False
        if health.reconnect_attempts > self._MAX_RECONNECT_ATTEMPTS:
            return False
        return time.time() >= health.next_reconnect

    def get_health_summary(self) -> list[dict[str, Any]]:
        """Get health summary for all integrations."""
        summaries = []
        for service_id, health in self._health.items():
            summaries.append({
                "service_id": service_id,
                "healthy": health.healthy,
                "consecutive_failures": health.consecutive_failures,
                "total_failures": health.total_failures,
                "total_successes": health.total_successes,
                "avg_latency_ms": round(health.avg_latency_ms, 1),
                "last_success": health.last_success,
                "last_failure": health.last_failure,
                "reason": health.reason,
                "reconnect_attempts": health.reconnect_attempts,
                "rate_limit": RATE_LIMITER.get_status(service_id),
            })
        return summaries

    async def get_health_for_service(self, service_id: str) -> dict[str, Any]:
        """Get detailed health for a specific service."""
        health = self._get_health(service_id)
        rate_status = RATE_LIMITER.get_status(service_id)
        connected = False
        try:
            row = await db.get_integration(service_id)
            connected = row is not None and row.get("connected", False)
        except Exception:
            pass

        return {
            "service_id": service_id,
            "connected": connected,
            "healthy": health.healthy,
            "consecutive_failures": health.consecutive_failures,
            "total_failures": health.total_failures,
            "total_successes": health.total_successes,
            "avg_latency_ms": round(health.avg_latency_ms, 1),
            "last_success": health.last_success or None,
            "last_failure": health.last_failure or None,
            "reason": health.reason,
            "rate_limit": rate_status,
        }

    # --- OAuth 2.0 ---

    def start_oauth_flow(
        self,
        service_id: str,
        redirect_uri: str = "",
        scopes: list[str] | None = None,
    ) -> OAuthState | None:
        """Start an OAuth 2.0 Authorization Code flow.

        Returns an OAuthState with the authorization URL, or None if
        the service doesn't support OAuth.
        """
        config = _OAUTH_CONFIGS.get(service_id)
        if not config:
            return None

        verifier = _generate_code_verifier() if config.get("use_pkce") else ""
        challenge = _generate_code_challenge(verifier) if verifier else ""

        params: dict[str, str] = {
            "client_id": "",  # Must be set by user via integration config
            "response_type": "code",
            "redirect_uri": redirect_uri,
        }
        if scopes:
            params["scope"] = " ".join(scopes)
        elif config.get("scopes"):
            params["scope"] = " ".join(config["scopes"])
        if verifier and challenge:
            params["code_challenge"] = challenge
            params["code_challenge_method"] = "S256"

        # Build URL from authorize_url + params
        from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
        parsed = urlparse(config["authorize_url"])
        existing = dict(parse_qsl(parsed.query))
        existing.update(params)
        auth_url = urlunparse(parsed._replace(query=urlencode(existing)))

        state = OAuthState(
            service_id=service_id,
            authorization_url=auth_url,
            code_verifier=verifier,
            redirect_uri=redirect_uri,
            expires_at=time.time() + 600,  # 10 minutes
            scopes=scopes or config.get("scopes", []),
        )
        self._oauth_states[service_id] = state
        return state

    async def complete_oauth_flow(
        self,
        service_id: str,
        code: str,
    ) -> dict[str, Any] | None:
        """Complete an OAuth 2.0 flow by exchanging the code for tokens.

        Returns the token response, or None if no flow was in progress.
        """
        state = self._oauth_states.get(service_id)
        if not state:
            return None
        if time.time() > state.expires_at:
            self._oauth_states.pop(service_id, None)
            return {"error": "OAuth flow expired"}

        config = _OAUTH_CONFIGS.get(service_id)
        if not config:
            return None

        # Read client_id/client_secret from integration config
        integration_row = None
        try:
            integration_row = await db.get_integration(service_id)
        except Exception:
            pass

        client_id = ""
        client_secret = ""
        if integration_row:
            integration_config = json.loads(integration_row.get("config", "{}") or "{}")
            client_id = integration_config.get("client_id", "")
            client_secret = integration_config.get("client_secret", "")

        token_data: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": state.redirect_uri,
            "client_id": client_id,
        }
        if client_secret:
            token_data["client_secret"] = client_secret
        if state.code_verifier:
            token_data["code_verifier"] = state.code_verifier

        try:
            client = await _get_shared_client()
            resp = await client.post(config["token_url"], data=token_data)
            resp.raise_for_status()
            tokens = resp.json()

            # Store tokens in integration config
            if integration_row:
                integration_config.update({
                    "access_token": tokens.get("access_token"),
                    "refresh_token": tokens.get("refresh_token"),
                    "token_expires_at": time.time() + tokens.get("expires_in", 3600),
                    "token_type": tokens.get("token_type", "Bearer"),
                })
                await db.update_integration_config(
                    service_id,
                    integration_config,
                )

            self._oauth_states.pop(service_id, None)
            return tokens
        except Exception as exc:
            logger.error("OAuth token exchange failed for %s: %s", service_id, exc)
            return {"error": str(exc)}

    # --- Health Check (async ping) ---

    async def health_check(self, service_id: str) -> dict[str, Any]:
        """Perform a live health check against an integration by making
        a lightweight API request."""
        health = self._get_health(service_id)
        health.last_check = time.time()

        try:
            headers, base_url = await _build_auth_headers(service_id)
            if not base_url:
                health.healthy = False
                health.reason = "no_base_url"
                return await self.get_health_for_service(service_id)

            # Use a known lightweight endpoint per service
            ping_paths: dict[str, tuple[str, str]] = {
                "slack": ("GET", "/auth.test"),
                "github": ("GET", "/user"),
                "notion": ("POST", "/users/me"),
                "linear": ("POST", "/graphql"),
                "trello": ("GET", "/members/me"),
                "discord": ("GET", "/users/@me"),
                "grafana": ("GET", "/api/org"),
                "google_drive": ("GET", "/about?fields=user"),
                "cloudflare": ("GET", "/user/tokens/verify"),
                "vercel": ("GET", "/v2/user"),
                "netlify": ("GET", "/user"),
                "supabase": ("GET", "/projects"),
                "sentry": ("GET", "/organizations/"),
                "stripe": ("GET", "/balance"),
                "hubspot": ("GET", "/oauth/v1/access-tokens/"),
                "airtable": ("GET", "/meta/whoami"),
                "twilio": ("GET", "/Accounts.json"),
                "telegram": ("GET", "/getMe"),
                "gitlab": ("GET", "/user"),
                "digitalocean": ("GET", "/account"),
                "shopify": ("GET", "/admin/api/2024-01/shop.json"),
                "sendgrid": ("GET", "/user/profile"),
                "asana": ("GET", "/users/me"),
                "clickup": ("GET", "/user"),
                "hostinger": ("GET", "/api/vps/v1/virtual-machines"),
                "heroku": ("GET", "/account"),
                "datadog": ("GET", "/validate"),
                "posthog": ("GET", "/api/users/@me/"),
                "dockerhub": ("GET", "/user/"),
                "openai": ("GET", "/models"),
                "anthropic": ("GET", "/models"),
                "salesforce": ("GET", "/sobjects"),
                "pagerduty": ("GET", "/users/me"),
                "dropbox": ("POST", "/users/get_current_account"),
                "microsoft_teams": ("GET", "/users/me"),
                "youtube": ("GET", "/videos?part=id&chart=mostPopular&maxResults=1"),

        "amplitude": ("GET", "/api/2/users?limit=1"),
        "ansible": ("GET", "/api/v2/me/"),
        "aws": ("GET", "/"),
        "aws_s3": ("GET", "/"),
        "azure": ("GET", "/subscriptions?api-version=2020-01-01"),
        "backblaze": ("GET", "/b2api/v2/b2_authorize_account"),
        "basecamp": ("GET", "/projects.json?per_page=1"),
        "bigcommerce": ("GET", "/api/v2/store"),
        "bitbucket": ("GET", "/2.0/user"),
        "circleci": ("GET", "/api/v2/me"),
        "close": ("GET", "/api/v1/me/"),
        "cloudways": ("GET", "/api/v1/server"),
        "cohere": ("GET", "/v1/models?page_size=1"),
        "confluence": ("GET", "/rest/api/space?limit=1"),
        "docker": ("GET", "/v1.41/version"),
        "firebase": ("GET", "/v1beta1/projects/-/availableLocations"),
        "flyio": ("GET", "/v1/apps"),
        "gcp": ("GET", "/"),
        "ghcr": ("GET", "/user"),
        "godaddy": ("GET", "/v1/domains?limit=1"),
        "google_ai": ("GET", "/v1beta/models?pageSize=1"),
        "google_sheets": ("GET", "/v4/spreadsheets"),
        "harbor": ("GET", "/api/v2.0/ping"),
        "hotjar": ("GET", "/v1/sites"),
        "huggingface": ("GET", "/api/models?limit=1"),
        "ifttt": ("GET", "/v2/status"),
        "jenkins": ("GET", "/api/json?tree=jobs[name]"),
        "jira": ("GET", "/rest/api/3/myself"),
        "kubernetes": ("GET", "/api/v1/namespaces"),
        "lemon_squeezy": ("GET", "/v1/products?page[size]=1"),
        "letsencrypt": ("GET", "/directory"),
        "mailgun": ("GET", "/v3/domains"),
        "make": ("GET", "/v2/users/me"),
        "mixpanel": ("GET", "/api/2.0/annotations?limit=1"),
        "monday": ("GET", "/v2/"),
        "n8n": ("GET", "/rest/healthz"),
        "namecheap": ("GET", "/xml.response"),
        "newrelic": ("GET", "/v2/accounts"),
        "paypal": ("GET", "/v1/identity/oauth2/userinfo?schema=openid"),
        "pingdom": ("GET", "/api/3.1/checks?limit=1"),
        "pipedrive": ("GET", "/v1/deals?limit=1"),
        "planetable": ("GET", "/v1/organizations"),
        "plausible": ("GET", "/api/v1/stats/aggregate"),
        "porkbun": ("POST", "/domain/listAll"),
        "portainer": ("GET", "/api/system/status"),
        "postmark": ("GET", "/server"),
        "railway": ("GET", "/"),
        "render": ("GET", "/v1/services?limit=1"),
        "replicate": ("GET", "/v1/models?page_size=1"),
        "route53": ("GET", "/2013-04-01/hostedzone"),
        "segment": ("GET", "/v1/destinations"),
        "stabilityai": ("GET", "/v1/user/account"),
        "statuspage": ("GET", "/v1/pages"),
        "terraform": ("GET", "/api/v2/organizations"),
        "uptimerobot": ("POST", "/v2/getMonitors"),
        "wasabi": ("GET", "/"),
        "woocommerce": ("GET", "/wp-json/wc/v3/"),
        "zapier": ("GET", "/api/v2/connected-accounts?limit=1"),
        "zoho": ("GET", "/crm/v2/settings/modules"),
            }
            method, path = ping_paths.get(service_id, ("GET", "/"))
            if method == "POST" and service_id == "linear":
                body = {"query": "{ viewer { id } }"}
            else:
                body = None

            client = await _get_shared_client()
            # v22: Handle query params already in base_url (e.g. Trello appends ?key=...&token=...)
            if "?" in base_url:
                _base, _query = base_url.split("?", 1)
                url = f"{_base}{path}?{_query}"
            else:
                url = f"{base_url}{path}"
            t0 = time.time()
            resp = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body,
                timeout=10.0,
            )
            latency_ms = (time.time() - t0) * 1000

            if resp.status_code < 500:
                self.record_success(service_id, latency_ms)
                health.last_check = time.time()
            else:
                self.record_failure(
                    service_id,
                    IntegrationError.RUNTIME,
                    f"Health check returned {resp.status_code}",
                )
        except Exception as exc:
            err_cat = _categorize_error(exc)
            self.record_failure(service_id, err_cat, str(exc)[:200])

        await _emit(
            "integration_health_check",
            service_id=service_id,
            healthy=health.healthy,
            reason=health.reason,
        )
        return await self.get_health_for_service(service_id)

    async def health_check_all(self) -> list[dict[str, Any]]:
        """Health check all connected integrations."""
        try:
            integrations = await db.list_connected_integrations(connected_only=True)
        except Exception:
            return []

        results = []
        for integration in integrations:
            sid = integration.get("id", "")
            if sid:
                try:
                    result = await self.health_check(sid)
                    results.append(result)
                except Exception as exc:
                    logger.error("Health check error for %s: %s", sid, exc)
        return results


HUB = IntegrationHub()


# ---------------------------------------------------------------------------
# Auth Header Builder (v17 bug-fixed)
# ---------------------------------------------------------------------------

async def _build_auth_headers(integration_id: str) -> tuple[dict[str, str], str]:
    """Build auth headers from stored integration config.

    v17 bug fix: Trello custom headers are no longer thrown away.
    Instead of resetting ``headers = {}`` for Trello, we now preserve
    custom headers and use query params for Trello auth.

    Returns (headers_dict, base_url).
    """
    row = await db.get_integration(integration_id)
    if not row or not row.get("connected"):
        # FIX: Provide actionable error instead of opaque ValueError.
        # Tell the agent and user exactly what they need to do.
        meta = _MARKETPLACE.get(integration_id)
        service_name = meta.name if meta else integration_id
        required_fields = []
        if meta and meta.config_fields:
            required_fields = [f"{f['key']} ({f['label']})" for f in meta.config_fields]
        raise ValueError(
            f"Integration '{service_name}' is not connected. "
            f"To use this integration, you first need to connect it by providing credentials. "
            f"Required: {', '.join(required_fields) if required_fields else 'API key or token'}. "
            f"Ask the user for their {service_name} credentials and use the integration connection API."
        )

    config: dict[str, Any] = {}
    try:
        config = json.loads(row.get("config", "{}") or "{}")
    except Exception:
        pass

    base_url = str(config.get("base_url", "")).rstrip("/")
    auth_type = str(config.get("auth_type", "Bearer")).strip().lower()
    api_key = config.get("api_key", "")
    if not api_key and integration_id == "hostinger":
        api_key = config.get("api_token", "")

    # Start with base headers — never overwrite these wholesale
    headers: dict[str, str] = {"Content-Type": "application/json"}

    # Apply custom headers from config FIRST (v17 fix: these are preserved)
    custom_headers_raw = config.get("headers", "")
    if custom_headers_raw:
        try:
            parsed = json.loads(custom_headers_raw) if isinstance(custom_headers_raw, str) else custom_headers_raw
            if isinstance(parsed, dict):
                headers.update({str(k): str(v) for k, v in parsed.items()})
        except Exception:
            pass

    # Determine base URL for known services if not explicitly set
    if not base_url:
        meta = _MARKETPLACE.get(integration_id)
        if meta and meta.base_url:
            base_url = meta.base_url
        else:
            _KNOWN_BASES: dict[str, str | None] = {
                "slack": "https://slack.com/api",
                "discord": "https://discord.com/api/v10",
                "notion": "https://api.notion.com/v1",
                "linear": "https://api.linear.app",
                "trello": "https://api.trello.com/1",
                "asana": "https://app.asana.com/api/1.0",
                "google_drive": "https://www.googleapis.com/drive/v3",
                "dropbox": "https://api.dropboxapi.com/2",
                "airtable": "https://api.airtable.com/v0",
                "hubspot": "https://api.hubapi.com",
                "pagerduty": "https://api.pagerduty.com",
                "zapier": "https://hooks.zapier.com",
                "vercel": "https://api.vercel.com",
                "netlify": "https://api.netlify.com/api/v1",
                "cloudflare": "https://api.cloudflare.com/client/v4",
                "supabase": "https://api.supabase.com/v1",
                "firebase": "https://firebase.googleapis.com/v1beta1",
                "aws": "https://api.aws.amazon.com",
                "gcp": "https://www.googleapis.com",
                "azure": "https://management.azure.com",
                "digitalocean": "https://api.digitalocean.com/v2",
                "heroku": "https://api.heroku.com",
                "hostinger": "https://developers.hostinger.com",
                "sentry": "https://sentry.io/api/0",
                "stripe": "https://api.stripe.com/v1",
                "twilio": "https://api.twilio.com/2010-04-01",
                "sendgrid": "https://api.sendgrid.com/v3",
                "mailgun": "https://api.mailgun.net/v3",
                "postmark": "https://api.postmarkapp.com",
                "telegram": "https://api.telegram.org",
                "gitlab": "https://gitlab.com/api/v4",
                "bitbucket": "https://api.bitbucket.org/2.0",
                "circleci": "https://circleci.com/api/v2",
                "dockerhub": "https://hub.docker.com/v2",
                "clickup": "https://api.clickup.com/api/v2",
                "monday": "https://api.monday.com/v2",
                "salesforce": "",
                "pipedrive": "https://api.pipedrive.com/v1",
                "zoho": "https://www.zohoapis.com/crm/v2",
                "paypal": "https://api-m.paypal.com",
                "lemon_squeezy": "https://api.lemonsqueezy.com/v1",
                "datadog": "https://api.datadoghq.com/api/v1",
                "newrelic": "https://api.newrelic.com/graphql",
                "pingdom": "https://api.pingdom.com/api/3.1",
                "uptimerobot": "https://api.uptimerobot.com/v2",
                "statuspage": "https://api.statuspage.io/v1",
                "make": "https://api.make.com/v2",
                "ifttt": "https://connect.ifttt.com/v2",
                "aws_s3": "https://s3.amazonaws.com",
                "backblaze": "https://api.backblazeb2.com",
                "wasabi": "https://s3.wasabisys.com",
                "terraform": "https://app.terraform.io",
                "openai": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com/v1",
                "cohere": "https://api.cohere.ai/v1",
                "google_ai": "https://generativelanguage.googleapis.com/v1beta",
                "huggingface": "https://huggingface.co/api",
                "replicate": "https://api.replicate.com/v1",
                "stabilityai": "https://api.stability.ai/v1",
                "shopify": "",
                "woocommerce": "",
                "bigcommerce": "",
                "posthog": "https://app.posthog.com",
                "amplitude": "https://amplitude.com/api/2",
                "mixpanel": "https://mixpanel.com/api/2.0",
                "segment": "https://api.segment.io/v1",
                "plausible": "https://plausible.io/api/v1",
                "hotjar": "https://api.hotjar.com/v1",
                "microsoft_teams": "https://graph.microsoft.com/v1.0",
                "godaddy": "https://api.godaddy.com",
                "namecheap": "https://api.namecheap.com/xml.response",
                "letsencrypt": "https://acme-v02.api.letsencrypt.org",
                "porkbun": "https://porkbun.com/api/json/v3",
                "planetable": "https://api.planetscale.com/v1",
                "railway": "https://backboard.railway.app/graphql/v2",
                "render": "https://api.render.com/v1",
                "flyio": "https://api.machines.dev/v1",
                "cloudways": "https://api.cloudways.com/api/v1",
                "google_sheets": "https://sheets.googleapis.com/v4",
                "ghcr": "https://api.github.com",
                "close": "https://api.close.com/api/v1",
                "basecamp": "https://3.basecampapi.com",
                "confluence": "",
                "n8n": "",
                "jenkins": "",
                "kubernetes": "",
                "docker": "",
                "harbor": "",
                "portainer": "",
                "ansible": "",
                "route53": "https://route53.amazonaws.com/2013-04-01",
            }
            base_url = _KNOWN_BASES.get(integration_id, "")

            # Services that need config-based URLs
            if not base_url:
                if integration_id == "jira":
                    domain = str(config.get("domain", "")).rstrip("/")
                    if domain:
                        base_url = f"{domain}/rest/api/3"
                elif integration_id == "salesforce":
                    instance_url = str(config.get("instance_url", "")).rstrip("/")
                    if instance_url:
                        base_url = f"{instance_url}/services/data/v58.0"
                elif integration_id == "grafana":
                    url = str(config.get("url", "")).rstrip("/")
                    if url:
                        base_url = url

    # --- Per-service auth (preserves custom headers) ---

    if integration_id == "slack":
        token = config.get("bot_token", api_key)
        headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "notion":
        token = config.get("token", api_key)
        headers["Authorization"] = f"Bearer {token}"
        headers["Notion-Version"] = "2022-06-28"

    elif integration_id == "trello":
        # v17 FIX: Use query params for auth, do NOT reset headers dict
        key = config.get("api_key", api_key)
        token = config.get("token", "")
        # Build base URL with query params for Trello auth
        if key and token:
            separator = "&" if "?" in base_url else "?"
            base_url = f"{base_url}{separator}key={key}&token={token}"
        # Headers are preserved (Content-Type + any custom headers)

    elif integration_id == "github":
        token = config.get("api_key", api_key)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"

    elif integration_id == "linear":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
        headers["Content-Type"] = "application/json"

    elif integration_id == "airtable":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "hubspot":
        token = config.get("access_token", api_key)
        headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "pagerduty":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Token token={token}"

    elif integration_id == "asana":
        token = config.get("access_token", api_key)
        headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "jira":
        email = config.get("email", "")
        token = config.get("api_key", api_key)
        if email and token:
            raw = f"{email}:{token}"
            headers["Authorization"] = f"Basic {base64.b64encode(raw.encode()).decode()}"
        elif token:
            headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "grafana":
        token = config.get("api_key", api_key)
        if token:
            headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "google_drive":
        token = config.get("access_token", api_key)
        if token:
            headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "discord":
        token = config.get("bot_token", api_key)
        headers["Authorization"] = f"Bot {token}"

    elif integration_id == "cloudflare":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "vercel":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "netlify":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "hostinger":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "supabase":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
        headers["apikey"] = token
    elif integration_id == "stripe":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "sentry":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "gitlab":
        token = config.get("api_key", api_key)
        headers["PRIVATE-TOKEN"] = token
    elif integration_id == "digitalocean":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "heroku":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "twilio":
        sid = config.get("api_key", "")
        token = config.get("api_secret", api_key)
        if sid and token:
            raw = f"{sid}:{token}"
            headers["Authorization"] = f"Basic {base64.b64encode(raw.encode()).decode()}"

    elif integration_id == "sendgrid":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "mailgun":
        token = config.get("api_key", api_key)
        domain = config.get("domain", "")
        raw = f"api:{token}"
        headers["Authorization"] = f"Basic {base64.b64encode(raw.encode()).decode()}"
        if domain:
            base_url = f"https://api.mailgun.net/v3/{domain}"

    elif integration_id == "postmark":
        token = config.get("api_key", api_key)
        headers["X-Postmark-Server-Token"] = token

    elif integration_id == "telegram":
        token = config.get("bot_token", api_key)
        if token:
            base_url = f"https://api.telegram.org/bot{token}"

    elif integration_id == "microsoft_teams":
        token = config.get("access_token", api_key)
        headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "bitbucket":
        token = config.get("api_key", api_key)
        username = config.get("username", "")
        if username and token:
            raw = f"{username}:{token}"
            headers["Authorization"] = f"Basic {base64.b64encode(raw.encode()).decode()}"
    elif token:
            headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "dockerhub":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"

    elif integration_id == "circleci":
        token = config.get("api_key", api_key)
        headers["Circle-Token"] = token

    elif integration_id == "jenkins":
        token = config.get("api_key", api_key)
        username = config.get("username", "")
        url = config.get("url", "")
        if url:
            base_url = url.rstrip("/")
        if username and token:
            raw = f"{username}:{token}"
            headers["Authorization"] = f"Basic {base64.b64encode(raw.encode()).decode()}"

    elif integration_id == "clickup":
        token = config.get("api_key", api_key)
        headers["Authorization"] = token
    elif integration_id == "monday":
        token = config.get("api_key", api_key)
        headers["Authorization"] = token
    elif integration_id == "confluence":
        token = config.get("api_key", api_key)
        email = config.get("email", "")
        domain = config.get("domain", "")
        if domain:
            base_url = f"{domain.rstrip('/')}/wiki/rest/api"
        if email and token:
            raw = f"{email}:{token}"
            headers["Authorization"] = f"Basic {base64.b64encode(raw.encode()).decode()}"
    elif integration_id == "basecamp":
        token = config.get("access_token", api_key)
        account_id = config.get("account_id", "")
        headers["Authorization"] = f"Bearer {token}"
        if account_id:
            base_url = f"https://3.basecampapi.com/{account_id}"
    elif integration_id == "hubspot":
        token = config.get("access_token", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "salesforce":
        token = config.get("access_token", api_key)
        instance_url = config.get("instance_url", "")
        headers["Authorization"] = f"Bearer {token}"
        if instance_url:
            base_url = f"{instance_url.rstrip('/')}/services/data/v58.0"
    elif integration_id == "pipedrive":
        token = config.get("api_key", api_key)
        if token:
            separator = "&" if "?" in base_url else "?"
            base_url = f"{base_url}{separator}api_token={token}"
    elif integration_id == "zoho":
        token = config.get("access_token", api_key)
        headers["Authorization"] = f"Zoho-oauthtoken {token}"
    elif integration_id == "close":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Basic {base64.b64encode(token.encode()).decode()}"
    elif integration_id == "shopify":
        token = config.get("access_token", api_key)
        shop_domain = config.get("shop_domain", "")
        headers["X-Shopify-Access-Token"] = token
        if shop_domain:
            base_url = f"https://{shop_domain}/admin/api/2024-01"
    elif integration_id == "woocommerce":
        token = config.get("api_key", api_key)
        secret = config.get("api_secret", "")
        url = config.get("url", "")
        if url:
            base_url = f"{url.rstrip('/')}/wp-json/wc/v3"
        if token and secret:
            raw = f"{token}:{secret}"
            headers["Authorization"] = f"Basic {base64.b64encode(raw.encode()).decode()}"
    elif integration_id == "bigcommerce":
        token = config.get("access_token", api_key)
        store_hash = config.get("store_hash", "")
        headers["X-Auth-Token"] = token
        if store_hash:
            base_url = f"https://api.bigcommerce.com/stores/{store_hash}/v3"
    elif integration_id == "datadog":
        key = config.get("api_key", "")
        app_key = config.get("application_key", api_key)
        headers["DD-API-KEY"] = key
        headers["DD-APPLICATION-KEY"] = app_key
    elif integration_id == "newrelic":
        token = config.get("api_key", api_key)
        headers["API-Key"] = token
    elif integration_id == "pingdom":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "uptimerobot":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "statuspage":
        token = config.get("api_key", api_key)
        page_id = config.get("page_id", "")
        headers["Authorization"] = f"OAuth {token}"
        if page_id:
            base_url = f"https://api.statuspage.io/v1/pages/{page_id}"
    elif integration_id == "openai":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "anthropic":
        token = config.get("api_key", api_key)
        headers["x-api-key"] = token
        headers["anthropic-version"] = "2023-06-01"
    elif integration_id == "cohere":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "google_ai":
        token = config.get("api_key", api_key)
        separator = "&" if "?" in base_url else "?"
        base_url = f"{base_url}{separator}key={token}"
    elif integration_id == "huggingface":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "replicate":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "stabilityai":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "paypal":
        token = config.get("access_token", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "lemon_squeezy":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "posthog":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "amplitude":
        key = config.get("api_key", "")
        secret = config.get("api_secret", api_key)
        if key and secret:
            raw = f"{key}:{secret}"
            headers["Authorization"] = f"Basic {base64.b64encode(raw.encode()).decode()}"
    elif integration_id == "mixpanel":
        secret = config.get("api_secret", api_key)
        headers["Authorization"] = f"Basic {base64.b64encode(secret.encode()).decode()}"
    elif integration_id == "segment":
        token = config.get("api_secret", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "plausible":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "hotjar":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "dropbox":
        token = config.get("access_token", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "zapier":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "make":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Token {token}"
    elif integration_id == "n8n":
        token = config.get("api_key", api_key)
        url = config.get("url", "")
        headers["X-N8N-API-KEY"] = token
        if url:
            base_url = url.rstrip("/") + "/api/v1"
    elif integration_id == "ifttt":
        token = config.get("api_key", api_key)
        separator = "&" if "?" in base_url else "?"
        base_url = f"{base_url}{separator}key={token}"
    elif integration_id == "godaddy":
        key = config.get("api_key", "")
        secret = config.get("api_secret", api_key)
        headers["Authorization"] = f"sso-key {key}:{secret}"
    elif integration_id == "namecheap":
        key = config.get("api_key", api_key)
        username = config.get("username", "")
        client_ip = config.get("client_ip", "")
        separator = "&" if "?" in base_url else "?"
        base_url = f"{base_url}{separator}ApiUser={username}&ApiKey={key}&UserName={username}&ClientIp={client_ip}"
    elif integration_id == "porkbun":
        key = config.get("api_key", "")
        secret = config.get("secret_key", api_key)
        headers["X-API-Key"] = key
        headers["X-Secret-Key"] = secret
    elif integration_id == "planetable":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "railway":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "render":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "flyio":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "cloudways":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "google_sheets":
        token = config.get("access_token", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "route53":
        key = config.get("api_key", "")
        secret = config.get("api_secret", api_key)
        headers["Authorization"] = f"AWS4-HMAC-SHA256 Credential={key}"
    elif integration_id == "pagerduty":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Token token={token}"
    elif integration_id == "terraform":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "ansible":
        token = config.get("api_key", api_key)
        url = config.get("url", "")
        headers["Authorization"] = f"Bearer {token}"
        if url:
            base_url = url.rstrip("/") + "/api/v2"
    elif integration_id == "harbor":
        token = config.get("api_key", api_key)
        url = config.get("url", "")
        headers["Authorization"] = f"Basic {base64.b64encode(token.encode()).decode()}"
        if url:
            base_url = url.rstrip("/") + "/api/v2.0"
    elif integration_id == "portainer":
        token = config.get("api_key", api_key)
        url = config.get("url", "")
        headers["Authorization"] = f"Bearer {token}"
        if url:
            base_url = url.rstrip("/") + "/api"
    elif integration_id == "kubernetes":
        token = config.get("api_key", api_key)
        url = config.get("url", "")
        headers["Authorization"] = f"Bearer {token}"
        if url:
            base_url = url.rstrip("/")
    elif integration_id == "docker":
        url = config.get("url", "http://localhost:2375")
        base_url = url.rstrip("/")
    elif integration_id == "firebase":
        token = config.get("access_token", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "aws":
        key = config.get("api_key", "")
        secret = config.get("api_secret", api_key)
        headers["Authorization"] = f"AWS4-HMAC-SHA256 Credential={key}"
    elif integration_id == "gcp":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "azure":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "letsencrypt":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
    elif integration_id == "aws_s3":
        key = config.get("api_key", "")
        secret = config.get("api_secret", api_key)
        region = config.get("region", "us-east-1")
        base_url = f"https://s3.{region}.amazonaws.com"
        headers["Authorization"] = f"AWS4-HMAC-SHA256 Credential={key}"
    elif integration_id == "backblaze":
        key = config.get("api_key", "")
        secret = config.get("api_secret", api_key)
        headers["Authorization"] = f"Basic {base64.b64encode(f'{key}:{secret}'.encode()).decode()}"
    elif integration_id == "wasabi":
        key = config.get("api_key", "")
        secret = config.get("api_secret", api_key)
        headers["Authorization"] = f"AWS4-HMAC-SHA256 Credential={key}"
    elif integration_id == "ghcr":
        token = config.get("api_key", api_key)
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/vnd.github+json"
    else:
        # Generic auth fallback
        if api_key:
            if auth_type in ("bearer", ""):
                headers["Authorization"] = f"Bearer {api_key}"
            elif auth_type == "basic":
                headers["Authorization"] = f"Basic {base64.b64encode(api_key.encode()).decode()}"
            elif auth_type in ("token", "api_key"):
                headers["X-API-Key"] = api_key

    return headers, base_url


# ---------------------------------------------------------------------------
# Token Auto-Refresh
# ---------------------------------------------------------------------------

async def _maybe_refresh_token(service_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Check if the access token is expired and refresh it.

    Handles OAuth-based services (e.g., google_drive) with refresh tokens.
    """
    expires_at = config.get("token_expires_at", 0)
    refresh_token = config.get("refresh_token", "")

    # Only refresh if we have a refresh token and the token is about to expire
    if not refresh_token or time.time() < (expires_at - 60):
        return config

    oauth_config = _OAUTH_CONFIGS.get(service_id)
    if not oauth_config:
        return config

    client_id = config.get("client_id", "")
    client_secret = config.get("client_secret", "")

    token_data: dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    "amplitude_event": ("GET", "/api/2/users?limit=1"),
    "discord_msg": ("GET", "/api/v10/users/@me"),
    "file_changed": ("GET", "/"),
    "new_commit": ("GET", "/"),
    "new_email": ("GET", "/"),
    "new_issue": ("GET", "/"),
    "new_message": ("GET", "/"),
    "posthog_event": ("GET", "/api/event?limit=1"),
    "heap": ("GET", "/api"),
    "schedule": ("GET", "/"),
    "schedule_hr": ("GET", "/"),
    "schedule_daily": ("GET", "/"),
    "webhook": ("GET", "/"),
    "email": ("GET", "/"),
    "appstore": ("GET", "/apps"),
    "googleplay": ("GET", "/androidpublisher/v3/applications"),
    "zerossl": ("GET", "/certificates"),

    }
    if client_id:
        token_data["client_id"] = client_id
    if client_secret:
        token_data["client_secret"] = client_secret

    try:
        client = await _get_shared_client()
        resp = await client.post(oauth_config["token_url"], data=token_data)
        resp.raise_for_status()
        tokens = resp.json()

        config["access_token"] = tokens.get("access_token", config.get("access_token"))
        new_refresh = tokens.get("refresh_token")
        if new_refresh:
            config["refresh_token"] = new_refresh
        config["token_expires_at"] = time.time() + tokens.get("expires_in", 3600)

        # Persist updated config
        try:
            await db.update_integration_config(service_id, config)
        except Exception as exc:
            logger.error("Failed to persist refreshed token for %s: %s", service_id, exc)

        logger.info("Auto-refreshed token for %s", service_id)
        await _emit("integration_token_refreshed", service_id=service_id)
    except Exception as exc:
        logger.error("Token refresh failed for %s: %s", service_id, exc)

    return config


# ---------------------------------------------------------------------------
# Pre-built Service Adapters
# ---------------------------------------------------------------------------

class ServiceAdapter(ABC):
    """Abstract base class for service-specific API adapters.

    Each adapter provides typed methods for common operations, handles
    pagination automatically, and normalizes responses.
    """

    service_id: str = ""

    @abstractmethod
    async def call(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> tuple[Any, IntegrationError | None]:
        """Make an API call to the service.

        Returns (response_data, error_or_none).
        """
        ...


class _BaseHttpAdapter(ServiceAdapter):
    """Base adapter for HTTP-based integrations."""

    service_id: str = ""

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> tuple[Any, IntegrationError | None]:
        """Execute an HTTP request with rate limiting, retry, and error handling."""
        # Rate limit check
        should_wait, wait_seconds = RATE_LIMITER.should_throttle(self.service_id)
        if should_wait:
            await asyncio.sleep(min(wait_seconds, 60.0))

        try:
            headers, base_url = await _build_auth_headers(self.service_id)
        except ValueError as exc:
            return str(exc), IntegrationError.NOT_CONNECTED

        if not base_url:
            return f"No base URL for '{self.service_id}'", IntegrationError.VALIDATION

        # Token auto-refresh for OAuth services
        if self.service_id in _OAUTH_CONFIGS:
            try:
                row = await db.get_integration(self.service_id)
                if row:
                    config = json.loads(row.get("config", "{}") or "{}")
                    config = await _maybe_refresh_token(self.service_id, config)
            except Exception:
                pass

        url = f"{base_url}{path}"
        t0 = time.time()

        try:
            client = await _get_shared_client()
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if body else None,
                params=query_params if query_params else None,
                timeout=timeout,
            )
            latency_ms = (time.time() - t0) * 1000

            # Update rate limiter from response headers
            RATE_LIMITER.apply_response_headers(
                self.service_id,
                dict(response.headers),
            )
            RATE_LIMITER.record_request(self.service_id)

            # Handle HTTP errors
            if response.status_code == 429:
                HUB.record_failure(self.service_id, IntegrationError.RATE_LIMITED, "HTTP 429")
                await _emit("integration_rate_limited", service_id=self.service_id, status_code=429)
                return {"error": "Rate limited", "status": 429}, IntegrationError.RATE_LIMITED

            if response.status_code == 401:
                HUB.record_failure(self.service_id, IntegrationError.AUTH, "HTTP 401")
                return {"error": "Unauthorized — check API key/token", "status": 401}, IntegrationError.AUTH

            if response.status_code == 403:
                HUB.record_failure(self.service_id, IntegrationError.AUTH, "HTTP 403")
                return {"error": "Forbidden — insufficient permissions", "status": 403}, IntegrationError.AUTH

            # Parse response
            try:
                data = response.json()
            except Exception:
                data = {"_raw": response.text[:_MAX_RESPONSE_CHARS]}

            # Handle service-specific error envelopes
            if isinstance(data, dict) and data.get("ok") is False:
                HUB.record_failure(self.service_id, IntegrationError.RUNTIME, data.get("error", "Unknown"))
                return {"error": data.get("error", "Unknown API error"), "raw": data}, IntegrationError.RUNTIME

            if isinstance(data, dict) and "error" in data and isinstance(data["error"], list):
                # Notion-style error array
                error_msg = data["error"][0].get("message", str(data["error"])) if data["error"] else "Unknown"
                HUB.record_failure(self.service_id, IntegrationError.RUNTIME, error_msg)
                return {"error": error_msg, "raw": data}, IntegrationError.RUNTIME

            if isinstance(data, dict) and "error" in data and response.status_code >= 400:
                HUB.record_failure(self.service_id, IntegrationError.RUNTIME, str(data["error"]))
                return {"error": str(data["error"]), "status": response.status_code}, IntegrationError.RUNTIME

            HUB.record_success(self.service_id, latency_ms)
            return data, None

        except httpx.TimeoutException as exc:
            HUB.record_failure(self.service_id, IntegrationError.TIMEOUT, str(exc)[:200])
            return f"Request timed out after {timeout}s", IntegrationError.TIMEOUT
        except httpx.ConnectError as exc:
            HUB.record_failure(self.service_id, IntegrationError.NETWORK, str(exc)[:200])
            return f"Connection error: {exc}", IntegrationError.NETWORK
        except Exception as exc:
            err_cat = _categorize_error(exc)
            HUB.record_failure(self.service_id, err_cat, str(exc)[:200])
            return f"API error ({type(exc).__name__}): {exc}", err_cat

    async def call(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> tuple[Any, IntegrationError | None]:
        """Execute a request. Subclasses can override for custom logic."""
        return await self._request(method, path, body=body, query_params=query_params, timeout=timeout)


# --- Slack Adapter ---

class SlackAdapter(_BaseHttpAdapter):
    """Slack API adapter with typed methods."""

    service_id = "slack"

    async def channels_list(
        self,
        *,
        limit: int = 100,
        cursor: str = "",
        types: str = "public_channel",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit, "types": types}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/conversations.list", query_params=params)

    async def messages_send(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"channel": channel, "text": text}
        if thread_ts:
            body["thread_ts"] = thread_ts
        return await self._request("POST", "/chat.postMessage", body=body)

    async def messages_list(
        self,
        *,
        channel: str,
        limit: int = 50,
        cursor: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"channel": channel, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/conversations.history", query_params=params)

    async def users_list(
        self,
        *,
        limit: int = 100,
        cursor: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/users.list", query_params=params)

    async def create_channel(
        self,
        *,
        name: str,
        description: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        """Create a new Slack channel."""
        body: dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        return await self._request("POST", "/conversations.create", body=body)

    async def invite_to_channel(
        self,
        *,
        channel_id: str,
        user_ids: list[str],
    ) -> tuple[Any, IntegrationError | None]:
        """Invite users to a Slack channel."""
        body: dict[str, Any] = {"channel": channel_id, "users": ",".join(user_ids)}
        return await self._request("POST", "/conversations.invite", body=body)

    async def update_message(
        self,
        *,
        channel_id: str,
        ts: str,
        text: str,
    ) -> tuple[Any, IntegrationError | None]:
        """Update an existing message in a channel."""
        body: dict[str, Any] = {"channel": channel_id, "ts": ts, "text": text}
        return await self._request("POST", "/chat.update", body=body)

    async def delete_message(
        self,
        *,
        channel_id: str,
        ts: str,
    ) -> tuple[Any, IntegrationError | None]:
        """Delete a message from a channel."""
        body: dict[str, Any] = {"channel": channel_id, "ts": ts}
        return await self._request("POST", "/chat.delete", body=body)


# --- GitHub Adapter ---

class GitHubAdapter(_BaseHttpAdapter):
    """GitHub API adapter with typed methods."""

    service_id = "github"

    async def repos_list(
        self,
        *,
        sort: str = "updated",
        per_page: int = 30,
        page: int = 1,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"sort": sort, "per_page": per_page, "page": page}
        return await self._request("GET", "/user/repos", query_params=params)

    async def issues_list(
        self,
        *,
        owner: str = "",
        repo: str = "",
        state: str = "open",
        per_page: int = 30,
        page: int = 1,
    ) -> tuple[Any, IntegrationError | None]:
        path = f"/repos/{owner}/{repo}/issues" if owner and repo else "/issues"
        params: dict[str, Any] = {"state": state, "per_page": per_page, "page": page}
        return await self._request("GET", path, query_params=params)

    async def issues_create(
        self,
        *,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        labels: list[str] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        issue_body: dict[str, Any] = {"title": title, "body": body}
        if labels:
            issue_body["labels"] = labels
        return await self._request("POST", f"/repos/{owner}/{repo}/issues", body=issue_body)

    async def prs_list(
        self,
        *,
        owner: str = "",
        repo: str = "",
        state: str = "open",
        per_page: int = 30,
        page: int = 1,
    ) -> tuple[Any, IntegrationError | None]:
        path = f"/repos/{owner}/{repo}/pulls" if owner and repo else "/pulls"
        params: dict[str, Any] = {"state": state, "per_page": per_page, "page": page}
        return await self._request("GET", path, query_params=params)

    async def prs_create(
        self,
        *,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        pr_body: dict[str, Any] = {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
        }
        return await self._request("POST", f"/repos/{owner}/{repo}/pulls", body=pr_body)

    async def files_read(
        self,
        *,
        owner: str,
        repo: str,
        path: str,
        ref: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        url_path = f"/repos/{owner}/{repo}/contents/{path}"
        params: dict[str, Any] = {}
        if ref:
            params["ref"] = ref
        return await self._request("GET", url_path, query_params=params)

    async def create_issue(
        self,
        *,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        assignees: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        """Create a new issue in a repository."""
        issue_body: dict[str, Any] = {"title": title, "body": body}
        if assignees:
            issue_body["assignees"] = assignees
        if labels:
            issue_body["labels"] = labels
        return await self._request("POST", f"/repos/{owner}/{repo}/issues", body=issue_body)

    async def update_issue(
        self,
        *,
        owner: str,
        repo: str,
        number: int,
        state: str | None = None,
        title: str | None = None,
        body: str | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        """Update an existing issue (state, title, body)."""
        patch_body: dict[str, Any] = {}
        if state is not None:
            patch_body["state"] = state
        if title is not None:
            patch_body["title"] = title
        if body is not None:
            patch_body["body"] = body
        return await self._request("PATCH", f"/repos/{owner}/{repo}/issues/{number}", body=patch_body)

    async def create_branch(
        self,
        *,
        owner: str,
        repo: str,
        branch: str,
        from_ref: str = "main",
    ) -> tuple[Any, IntegrationError | None]:
        """Create a new branch from an existing ref."""
        body: dict[str, Any] = {"ref": f"refs/heads/{branch}", "sha": from_ref}
        return await self._request("POST", f"/repos/{owner}/{repo}/git/refs", body=body)

    async def merge_pr(
        self,
        *,
        owner: str,
        repo: str,
        number: int,
        commit_title: str = "",
        merge_method: str = "merge",
    ) -> tuple[Any, IntegrationError | None]:
        """Merge a pull request."""
        body: dict[str, Any] = {"merge_method": merge_method}
        if commit_title:
            body["commit_title"] = commit_title
        return await self._request("PUT", f"/repos/{owner}/{repo}/pulls/{number}/merge", body=body)


# --- Notion Adapter ---

class NotionAdapter(_BaseHttpAdapter):
    """Notion API adapter with typed methods."""

    service_id = "notion"

    async def pages_list(
        self,
        *,
        page_size: int = 100,
        start_cursor: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"page_size": page_size}
        if start_cursor:
            body["start_cursor"] = start_cursor
        return await self._request("POST", "/search", body=body)

    async def databases_query(
        self,
        *,
        database_id: str,
        filter_obj: dict[str, Any] | None = None,
        sorts: list[dict[str, Any]] | None = None,
        page_size: int = 100,
        start_cursor: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"page_size": page_size}
        if filter_obj:
            body["filter"] = filter_obj
        if sorts:
            body["sorts"] = sorts
        if start_cursor:
            body["start_cursor"] = start_cursor
        return await self._request("POST", f"/databases/{database_id}/query", body=body)

    async def blocks_children(
        self,
        *,
        block_id: str,
        page_size: int = 100,
        start_cursor: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"page_size": page_size}
        if start_cursor:
            params["start_cursor"] = start_cursor
        return await self._request("GET", f"/blocks/{block_id}/children", query_params=params)

    async def create_page(
        self,
        *,
        parent_page_id: str,
        title: str,
        content: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        """Create a new page under a parent page."""
        children: list[dict[str, Any]] = []
        if content:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": content}}],
                },
            })
        body: dict[str, Any] = {
            "parent": {"type": "page_id", "page_id": parent_page_id},
            "properties": {
                "title": {
                    "title": [{"type": "text", "text": {"content": title}}],
                },
            },
        }
        if children:
            body["children"] = children
        return await self._request("POST", "/pages", body=body)

    async def update_page(
        self,
        *,
        page_id: str,
        properties: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        """Update properties and/or append content blocks to a page."""
        body: dict[str, Any] = {}
        if properties:
            body["properties"] = properties
        if content:
            body["children"] = [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": content}}],
                    },
                },
            ]
        return await self._request("PATCH", f"/pages/{page_id}", body=body)

    async def create_database(
        self,
        *,
        parent_page_id: str,
        title: str,
        properties: dict[str, Any],
    ) -> tuple[Any, IntegrationError | None]:
        """Create a new database under a parent page."""
        body: dict[str, Any] = {
            "parent": {"type": "page_id", "page_id": parent_page_id},
            "title": [
                {"type": "text", "text": {"content": title}},
            ],
            "properties": properties,
        }
        return await self._request("POST", "/databases", body=body)

    async def add_page_to_database(
        self,
        *,
        database_id: str,
        properties: dict[str, Any],
    ) -> tuple[Any, IntegrationError | None]:
        """Add a new page to a database with the given properties."""
        body: dict[str, Any] = {
            "parent": {"type": "database_id", "database_id": database_id},
            "properties": properties,
        }
        return await self._request("POST", "/pages", body=body)


# --- Jira Adapter ---

class JiraAdapter(_BaseHttpAdapter):
    """Jira Cloud API adapter with typed methods."""

    service_id = "jira"

    async def projects_list(
        self,
        *,
        start_at: int = 0,
        max_results: int = 50,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"startAt": start_at, "maxResults": max_results}
        return await self._request("GET", "/project", query_params=params)

    async def issues_search(
        self,
        *,
        jql: str,
        start_at: int = 0,
        max_results: int = 50,
        fields: list[str] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": max_results,
        }
        if fields:
            body["fields"] = fields
        return await self._request("POST", "/search", body=body)

    async def issues_create(
        self,
        *,
        project_key: str,
        summary: str,
        description: str = "",
        issue_type: str = "Task",
        fields_extra: dict[str, Any] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": {"type": "doc", "version": 1, "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": description}]}
                ]} if description else None,
                "issuetype": {"name": issue_type},
            }
        }
        if fields_extra:
            body["fields"].update(fields_extra)
        return await self._request("POST", "/issue", body=body)

    async def issues_transition(
        self,
        *,
        issue_key: str,
        transition_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"transition": {"id": transition_id}}
        return await self._request("POST", f"/issue/{issue_key}/transitions", body=body)

    async def add_comment(
        self,
        *,
        issue_key: str,
        comment: str,
    ) -> tuple[Any, IntegrationError | None]:
        """Add a comment to a Jira issue."""
        body: dict[str, Any] = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": comment}]}
                ],
            }
        }
        return await self._request("POST", f"/rest/api/3/issue/{issue_key}/comment", body=body)

    async def update_issue(
        self,
        *,
        issue_key: str,
        fields: dict[str, Any],
    ) -> tuple[Any, IntegrationError | None]:
        """Update fields on an existing Jira issue."""
        body: dict[str, Any] = {"fields": fields}
        return await self._request("PUT", f"/rest/api/3/issue/{issue_key}", body=body)

    async def transition_issue(
        self,
        *,
        issue_key: str,
        transition_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        """Transition a Jira issue to a new state by transition ID."""
        body: dict[str, Any] = {"transition": {"id": transition_id}}
        return await self._request("POST", f"/rest/api/3/issue/{issue_key}/transitions", body=body)

    async def assign_issue(
        self,
        *,
        issue_key: str,
        account_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        """Assign a Jira issue to a user by account ID."""
        body: dict[str, Any] = {
            "fields": {"assignee": {"accountId": account_id}},
        }
        return await self._request("PUT", f"/rest/api/3/issue/{issue_key}", body=body)


# --- Linear Adapter ---

class LinearAdapter(_BaseHttpAdapter):
    """Linear API adapter with typed methods (GraphQL)."""

    service_id = "linear"

    async def issues_list(
        self,
        *,
        team_id: str = "",
        first: int = 50,
        after: str = "",
        filter_str: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        filter_clause = f', filter: {{{filter_str}}}' if filter_str else ""
        query = f"""
        query {{
            issues(first: {first}, after: {json.dumps(after) if after else 'null'}{filter_clause}) {{
                nodes {{
                    id title description state {{ name }} assignee {{ name }} createdAt updatedAt
                }}
                pageInfo {{ hasNextPage endCursor }}
            }}
        }}
        """
        return await self._request("POST", "/graphql", body={"query": query})

    async def issues_create(
        self,
        *,
        team_id: str,
        title: str,
        description: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        query = """
        mutation($title: String!, $teamId: String!, $description: String) {
            issueCreate(input: {title: $title, teamId: $teamId, description: $description}) {
                issue { id title }
            }
        }
        """
        variables = {"title": title, "teamId": team_id, "description": description}
        return await self._request("POST", "/graphql", body={"query": query, "variables": variables})

    async def teams_list(self) -> tuple[Any, IntegrationError | None]:
        query = """
        query {
            teams {
                nodes { id name key }
            }
        }
        """
        return await self._request("POST", "/graphql", body={"query": query})


# --- Trello Adapter ---

class TrelloAdapter(_BaseHttpAdapter):
    """Trello API adapter with typed methods."""

    service_id = "trello"

    async def boards_list(
        self,
        *,
        filter_str: str = "open",
        fields: str = "id,name,desc",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"filter": filter_str, "fields": fields}
        return await self._request("GET", "/members/me/boards", query_params=params)

    async def cards_list(
        self,
        *,
        board_id: str,
        filter_str: str = "open",
        fields: str = "id,name,desc,due,idList",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"filter": filter_str, "fields": fields}
        return await self._request("GET", f"/boards/{board_id}/cards", query_params=params)

    async def cards_create(
        self,
        *,
        id_list: str,
        name: str,
        description: str = "",
        pos: str = "bottom",
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"name": name, "idList": id_list, "pos": pos}
        if description:
            body["desc"] = description
        return await self._request("POST", "/cards", body=body)

    async def lists_list(
        self,
        *,
        board_id: str,
        fields: str = "id,name",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"fields": fields}
        return await self._request("GET", f"/boards/{board_id}/lists", query_params=params)


# --- Grafana Adapter ---

class GrafanaAdapter(_BaseHttpAdapter):
    """Grafana API adapter with typed methods."""

    service_id = "grafana"

    async def dashboards_list(
        self,
        *,
        tags: list[str] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {}
        if tags:
            params["tags"] = ",".join(tags)
        return await self._request("GET", "/api/search", query_params=params)

    async def annotations_create(
        self,
        *,
        text: str,
        tags: list[str] | None = None,
        time_start: int | None = None,
        time_end: int | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"text": text}
        if tags:
            body["tags"] = tags
        if time_start:
            body["time"] = time_start
        if time_end:
            body["timeEnd"] = time_end
        return await self._request("POST", "/api/annotations", body=body)

    async def datasource_query(
        self,
        *,
        datasource_id: int,
        expr: str = "",
        start: int | None = None,
        end: int | None = None,
        step: str = "60s",
    ) -> tuple[Any, IntegrationError | None]:
        now_ms = int(time.time() * 1000)
        body: dict[str, Any] = {
            "queries": [{"refId": "A", "datasourceId": datasource_id, "expr": expr, "instant": False, "intervalMs": 60000}],
            "from": f"{(start or now_ms - 3600000)}",
            "to": f"{end or now_ms}",
        }
        return await self._request("POST", "/api/ds/query", body=body)


# --- Google Drive Adapter ---

class GoogleDriveAdapter(_BaseHttpAdapter):
    """Google Drive API adapter with typed methods."""

    service_id = "google_drive"

    async def files_list(
        self,
        *,
        page_size: int = 100,
        page_token: str = "",
        query: str = "",
        fields: str = "files(id,name,mimeType,size,modifiedTime),nextPageToken",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"pageSize": page_size, "fields": fields}
        if page_token:
            params["pageToken"] = page_token
        if query:
            params["q"] = query
        return await self._request("GET", "/files", query_params=params)

    async def files_upload(
        self,
        *,
        file_name: str,
        mime_type: str = "application/octet-stream",
        content: str = "",
        parents: list[str] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        metadata: dict[str, Any] = {"name": file_name, "mimeType": mime_type}
        if parents:
            metadata["parents"] = parents
        # Google Drive upload uses multipart
        body: dict[str, Any] = metadata
        return await self._request(
            "POST",
            "/files?uploadType=multipart",
            body=body,
        )

    async def files_read(
        self,
        *,
        file_id: str,
        fields: str = "id,name,mimeType,size,modifiedTime",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"fields": fields, "alt": "media"}
        return await self._request("GET", f"/files/{file_id}", query_params=params)


# --- Discord Adapter ---

class DiscordAdapter(_BaseHttpAdapter):
    """Discord API adapter with typed methods."""

    service_id = "discord"

    async def channels_list(
        self,
        *,
        guild_id: str,
        limit: int = 100,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        return await self._request("GET", f"/guilds/{guild_id}/channels", query_params=params)

    async def messages_send(
        self,
        *,
        channel_id: str,
        content: str,
        embed: dict[str, Any] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"content": content}
        if embed:
            body["embed"] = embed
        return await self._request("POST", f"/channels/{channel_id}/messages", body=body)

    async def guilds_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/users/@me/guilds")

    async def create_channel(
        self,
        *,
        guild_id: str,
        name: str,
        channel_type: int = 0,
    ) -> tuple[Any, IntegrationError | None]:
        """Create a new channel in a guild. channel_type: 0=text, 2=voice."""
        body: dict[str, Any] = {"name": name, "type": channel_type}
        return await self._request("POST", f"/guilds/{guild_id}/channels", body=body)

    async def delete_message(
        self,
        *,
        channel_id: str,
        message_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        """Delete a message from a channel."""
        return await self._request(
            "DELETE", f"/channels/{channel_id}/messages/{message_id}",
        )

    async def add_reaction(
        self,
        *,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> tuple[Any, IntegrationError | None]:
        """Add a reaction to a message (emoji should be URL-encoded)."""
        from urllib.parse import quote
        encoded = quote(emoji, safe="")
        return await self._request(
            "PUT",
            f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me",
        )


# --- PostHog Adapter ---

class PostHogAdapter(_BaseHttpAdapter):
    """PostHog API adapter for product analytics."""

    service_id = "posthog"

    async def capture(
        self,
        *,
        event: str,
        properties: dict[str, Any] | None = None,
        distinct_id: str,
        timestamp: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {
            "event": event,
            "properties": properties or {},
            "distinct_id": distinct_id,
        }
        if timestamp:
            body["timestamp"] = timestamp
        return await self._request("POST", "/capture", body=body)

    async def events_list(
        self,
        *,
        limit: int = 100,
        person_id: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        if person_id:
            params["person_id"] = person_id
        return await self._request("GET", "/api/events/", query_params=params)

    async def trends(
        self,
        *,
        date_from: str = "-7d",
        date_to: str = "now",
        event: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {
            "date_from": date_from,
            "date_to": date_to,
            "events": [{"id": event}] if event else [],
        }
        return await self._request("POST", "/api/trends/", body=body)


# --- Amplitude Adapter ---

class AmplitudeAdapter(_BaseHttpAdapter):
    """Amplitude API adapter for product analytics."""

    service_id = "amplitude"

    async def track(
        self,
        *,
        event_type: str,
        user_id: str,
        device_id: str = "",
        properties: dict[str, Any] | None = None,
        time: int = 0,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {
            "event_type": event_type,
            "user_id": user_id,
        }
        if device_id:
            body["device_id"] = device_id
        if properties:
            body["event_properties"] = properties
        if time:
            body["time"] = time
        return await self._request("POST", "/2/httpapi", body=body)

    async def users_list(
        self,
        *,
        limit: int = 100,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        return await self._request("GET", "/3/users", query_params=params)

    async def chart(
        self,
        *,
        chart_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/3/chart/{chart_id}")


# --- Mixpanel Adapter ---

class MixpanelAdapter(_BaseHttpAdapter):
    """Mixpanel API adapter for product analytics."""

    service_id = "mixpanel"

    async def track(
        self,
        *,
        event: str,
        distinct_id: str,
        properties: dict[str, Any] | None = None,
        time: int = 0,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = [
            {
                "event": event,
                "properties": {
                    "distinct_id": distinct_id,
                    "time": time,
                    **(properties or {}),
                }
            }
        ]
        return await self._request("POST", "/track", body=body)

    async def engage(
        self,
        *,
        distinct_id: str,
        profiles: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {
            "profiles": profiles or [{"$distinct_id": distinct_id}]
        }
        return await self._request("POST", "/engage", body=body)

    async def export(
        self,
        *,
        from_date: str,
        to_date: str,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"from_date": from_date, "to_date": to_date}
        return await self._request("GET", "/export", query_params=params)


# --- Hotjar Adapter ---

class HotjarAdapter(_BaseHttpAdapter):
    """Hotjar API adapter for heatmaps and session recordings."""

    service_id = "hotjar"

    async def sites_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/sites")

    async def heatmaps_list(
        self,
        *,
        site_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/sites/{site_id}/heatmaps")

    async def recordings_list(
        self,
        *,
        site_id: str,
        limit: int = 50,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        return await self._request("GET", f"/sites/{site_id}/recordings", query_params=params)

    async def feedback_list(
        self,
        *,
        site_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/sites/{site_id}/feedback")


# --- Cloudflare Adapter ---

class CloudflareAdapter(_BaseHttpAdapter):
    """Cloudflare API adapter for DNS, SSL, and CDN."""

    service_id = "cloudflare"

    async def zones_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/zones")

    async def dns_records_list(
        self,
        *,
        zone_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/zones/{zone_id}/dns_records")

    async def dns_record_create(
        self,
        *,
        zone_id: str,
        name: str,
        type: str,
        content: str,
        ttl: int = 3600,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {
            "name": name,
            "type": type,
            "content": content,
            "ttl": ttl,
        }
        return await self._request("POST", f"/zones/{zone_id}/dns_records", body=body)

    async def ssl_certificates_list(
        self,
        *,
        zone_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/zones/{zone_id}/ssl/tls")

    async def purge_cache(
        self,
        *,
        zone_id: str,
        files: list[str] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"purge_everything": True}
        if files:
            body["files"] = files
        return await self._request("POST", f"/zones/{zone_id}/purge_cache", body=body)


# --- Kubernetes Adapter ---

class KubernetesAdapter(_BaseHttpAdapter):
    """Kubernetes API adapter for container orchestration."""

    service_id = "kubernetes"

    async def namespaces_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/api/v1/namespaces")

    async def pods_list(
        self,
        *,
        namespace: str = "default",
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/api/v1/namespaces/{namespace}/pods")

    async def services_list(
        self,
        *,
        namespace: str = "default",
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/api/v1/namespaces/{namespace}/services")

    async def deployments_list(
        self,
        *,
        namespace: str = "default",
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/apis/apps/v1/namespaces/{namespace}/deployments")

    async def pods_delete(
        self,
        *,
        namespace: str,
        name: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("DELETE", f"/api/v1/namespaces/{namespace}/pods/{name}")


# --- Docker Adapter ---

class DockerAdapter(_BaseHttpAdapter):
    """Docker API adapter for container management."""

    service_id = "docker"

    async def containers_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/containers/json")

    async def images_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/images/json")

    async def container_create(
        self,
        *,
        name: str,
        image: str,
        ports: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {
            "name": name,
            "Image": image,
            "ExposedPorts": {p["containerPort"]: {} for p in (ports or [])},
        }
        return await self._request("POST", "/containers/create", body=body)

    async def container_start(
        self,
        *,
        container_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("POST", f"/containers/{container_id}/start")

    async def container_stop(
        self,
        *,
        container_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("POST", f"/containers/{container_id}/stop")

    async def image_pull(
        self,
        *,
        image: str,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"image": image}
        return await self._request("POST", "/images/create", query_params=params)


# --- Heap Adapter ---

class HeapAdapter(_BaseHttpAdapter):
    """Heap API adapter for digital experience analytics."""

    service_id = "heap"

    async def users_list(
        self,
        *,
        limit: int = 100,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        return await self._request("GET", "/users", query_params=params)

    async def events_list(
        self,
        *,
        user_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/users/{user_id}/events")

    async def sessions_list(
        self,
        *,
        user_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/users/{user_id}/sessions")


# --- Harbor Adapter ---

class HarborAdapter(_BaseHttpAdapter):
    """Harbor API adapter for container registry."""

    service_id = "harbor"

    async def repositories_list(
        self,
        *,
        project_name: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        params = {}
        if project_name:
            params["project_name"] = project_name
        return await self._request("GET", "/v2/_catalog", query_params=params)

    async def artifacts_list(
        self,
        *,
        project: str,
        repository: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/v2/{project}/{repository}/artifacts")

    async def scan(
        self,
        *,
        project: str,
        repository: str,
        reference: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("POST", f"/v2/{project}/{repository}/blobs/{reference}/scan")


# --- Portainer Adapter ---

class PortainerAdapter(_BaseHttpAdapter):
    """Portainer API adapter for container management UI."""

    service_id = "portainer"

    async def containers_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/api/docker/containers")

    async def images_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/api/docker/images")

    async def volumes_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/api/docker/volumes")

    async def networks_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/api/docker/networks")

    async def stacks_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/api/stacks")


# --- Terraform Adapter ---

class TerraformAdapter(_BaseHttpAdapter):
    """Terraform Cloud API adapter for IaC."""

    service_id = "terraform"

    async def workspaces_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/api/v2/workspaces")

    async def runs_list(
        self,
        *,
        workspace_name: str,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"workspace": workspace_name}
        return await self._request("GET", "/api/v2/runs", query_params=params)

    async def run_create(
        self,
        *,
        workspace: str,
        message: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {
            "data": {
                "attributes": {
                    "message": message,
                    "isDestroy": False,
                },
                "relationships": {
                    "workspace": {
                        "data": {"type": "workspaces", "id": workspace}
                    }
                },
                "type": "runs",
            }
        }
        return await self._request("POST", "/api/v2/runs", body=body)

    async def state_versions_list(
        self,
        *,
        workspace: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/api/v2/workspaces/{workspace}/state-versions")


# --- Route53 Adapter ---

class Route53Adapter(_BaseHttpAdapter):
    """AWS Route53 API adapter for DNS."""

    service_id = "route53"

    async def hosted_zones_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/2013-04-01/hostedzone")

    async def list_resource_record_sets(
        self,
        *,
        hosted_zone_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/2013-04-01/hostedzone/{hosted_zone_id}/rrset")

    async def change_resource_record_sets(
        self,
        *,
        hosted_zone_id: str,
        changes: list[dict[str, Any]],
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"Changes": changes}
        return await self._request("POST", f"/2013-04-01/hostedzone/{hosted_zone_id}/rrset/", body=body)


# --- GoDaddy Adapter ---

class GoDaddyAdapter(_BaseHttpAdapter):
    """GoDaddy API adapter for domains."""

    service_id = "godaddy"

    async def domains_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/v1/domains")

    async def dns_records(
        self,
        *,
        domain: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/v1/domains/{domain}/records")

    async def dns_record_create(
        self,
        *,
        domain: str,
        name: str,
        type: str,
        data: str,
        ttl: int = 3600,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = [{"name": name, "type": type, "data": data, "ttl": ttl}]
        return await self._request("PUT", f"/v1/domains/{domain}/records", body=body)


# --- Namecheap Adapter ---

class NamecheapAdapter(_BaseHttpAdapter):
    """Namecheap API adapter for domains."""

    service_id = "namecheap"

    async def domains_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/domains")

    async def dns_getHosts(
        self,
        *,
        domain: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/dns/getHosts?domain={domain}")

    async def dns_setHost(
        self,
        *,
        domain: str,
        hosts: list[dict[str, Any]],
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"hosts": hosts}
        return await self._request("POST", f"/dns/setHost?domain={domain}", body=body)


# --- LetsEncrypt Adapter ---

class LetsEncryptAdapter(_BaseHttpAdapter):
    """Let's Encrypt ACME adapter for SSL certificates."""

    service_id = "letsencrypt"

    async def new_order(
        self,
        *,
        identifiers: list[str],
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"identifiers": [{"type": "dns", "value": i} for i in identifiers]}
        return await self._request("POST", "/acme/new-order", body=body)

    async def new_authz(
        self,
        *,
        domain: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("POST", "/acme/new-authz", body={"identifier": {"type": "dns", "value": domain}})

    async def revoke(
        self,
        *,
        certificate: str,
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"certificate": certificate}
        return await self._request("POST", "/acme/revoke", body=body)


# --- ZeroSSL Adapter ---

class ZeroSSLAdapter(_BaseHttpAdapter):
    """ZeroSSL API adapter for SSL certificates.

    v22 FIX: Previously an empty shell — now implements all 4 actions
    referenced in _ADAPTER_ACTIONS["zerossl"].
    """

    service_id = "zerossl"

    async def domains_list(self) -> tuple[Any, IntegrationError | None]:
        """List all domains/certificates in the ZeroSSL account."""
        return await self._request("GET", "/certificates")

    async def certificate_create(
        self,
        *,
        domain: str = "",
        validation_method: str = "dns",
    ) -> tuple[Any, IntegrationError | None]:
        """Create a new SSL certificate."""
        body: dict[str, Any] = {
            "certificate_domains": domain,
            "certificate_validation_method": validation_method,
        }
        return await self._request("POST", "/certificates", body=body)

    async def certificate_verify(
        self,
        *,
        certificate_id: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        """Verify/validate an SSL certificate."""
        return await self._request("GET", f"/certificates/{certificate_id}")

    async def certificate_download(
        self,
        *,
        certificate_id: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        """Download an issued SSL certificate."""
        return await self._request("GET", f"/certificates/{certificate_id}/download")


# --- AWS Adapter ---

class AWSAdapter(_BaseHttpAdapter):
    """AWS API adapter for cloud infrastructure."""

    service_id = "aws"

    async def ec2_describe_instances(
        self,
        *,
        instance_ids: list[str] | None = None,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"Action": "DescribeInstances"}
        if instance_ids:
            params["InstanceId"] = instance_ids
        return await self._request("GET", "/", query_params=params)

    async def s3_list_buckets(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/")

    async def s3_list_objects(
        self,
        *,
        bucket: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/{bucket}?list-type=2")

    async def Lambda_list_functions(
        self,
        *,
        MaxItems: int = 50,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"MaxItems": MaxItems}
        return await self._request("GET", "/2015-03-31/functions/", query_params=params)

    async def CloudWatch_list_metrics(
        self,
        *,
        namespace: str = "AWS/EC2",
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"Namespace": namespace}
        return await self._request("GET", "/", query_params=params)


# --- GCP Adapter ---

class GCPAdapter(_BaseHttpAdapter):
    """Google Cloud Platform API adapter."""

    service_id = "gcp"

    async def compute_instances_list(
        self,
        *,
        project: str,
        zone: str = "-",
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/compute/v1/projects/{project}/zones/{zone}/instances")

    async def functions_list(
        self,
        *,
        project: str,
        location: str = "-",
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/cloudfunctions/v1/projects/{project}/locations/{location}/functions")

    async def run_services_list(
        self,
        *,
        project: str,
        location: str = "-",
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/apis/serving.knative.dev/v1/namespaces/{project}/services")

    async def storage_buckets_list(
        self,
        *,
        project: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/storage/v1/b?project={project}")


# --- Azure Adapter ---

class AzureAdapter(_BaseHttpAdapter):
    """Microsoft Azure API adapter."""

    service_id = "azure"

    async def resources_list(
        self,
        *,
        subscription_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/subscriptions/{subscription_id}/resources?api-version=2021-04-01")

    async def webapps_list(
        self,
        *,
        subscription_id: str,
        resource_group: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        path = f"/subscriptions/{subscription_id}/providers/Microsoft.Web/sites"
        if resource_group:
            path = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Web/sites"
        return await self._request("GET", f"{path}?api-version=2022-09-01")

    async def containers_list(
        self,
        *,
        subscription_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/subscriptions/{subscription_id}/providers/Microsoft.ContainerInstance/containerGroups?api-version=2023-05-01")

    async def functions_list(
        self,
        *,
        subscription_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/subscriptions/{subscription_id}/providers/Microsoft.Web/sites?api-version=2022-09-01")


# --- Vercel Adapter ---

class VercelAdapter(_BaseHttpAdapter):
    """Vercel API adapter for deployments."""

    service_id = "vercel"

    async def deployments_list(
        self,
        *,
        project_id: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        params = {}
        if project_id:
            params["projectId"] = project_id
        return await self._request("GET", "/v6/deployments", query_params=params)

    async def deployment_create(
        self,
        *,
        project_id: str,
        files: list[dict[str, Any]],
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"projectId": project_id, "files": files}
        return await self._request("POST", "/v13/deployments", body=body)

    async def projects_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/v6/projects")


# --- Netlify Adapter ---

class NetlifyAdapter(_BaseHttpAdapter):
    """Netlify API adapter for deployments."""

    service_id = "netlify"

    async def sites_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/api/v1/sites")

    async def deploys_list(
        self,
        *,
        site_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/api/v1/sites/{site_id}/deploys")

    async def deploy_trigger(
        self,
        *,
        site_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("POST", f"/api/v1/sites/{site_id}/deploys", body={})


# --- Supabase Adapter ---

class SupabaseAdapter(_BaseHttpAdapter):
    """Supabase API adapter for database and auth."""

    service_id = "supabase"

    async def table_select(
        self,
        *,
        table: str,
        select: str = "*",
        limit: int = 100,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"select": select, "limit": limit}
        return await self._request("GET", f"/rest/v1/{table}", query_params=params)

    async def table_insert(
        self,
        *,
        table: str,
        records: list[dict[str, Any]],
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("POST", f"/rest/v1/{table}", body=records)

    async def auth_users_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/auth/v1/admin/users")

    async def storage_buckets_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/storage/v1/bucket")


# --- Firebase Adapter ---

class FirebaseAdapter(_BaseHttpAdapter):
    """Firebase API adapter."""

    service_id = "firebase"

    async def projects_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/v1/projects")

    async def auth_users_list(
        self,
        *,
        project_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/identitytoolkit/v1/projects/{project_id}/accounts:list")

    async def firestore_collection_list(
        self,
        *,
        project_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/databases/{project_id}/documents")

    async def hosting_list(
        self,
        *,
        project_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/v1beta1/projects/{project_id}/sites")


# --- Docker Hub Adapter ---

class DockerHubAdapter(_BaseHttpAdapter):
    """Docker Hub API adapter."""

    service_id = "dockerhub"

    async def repositories_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/v2/repositories")

    async def repository_tags(
        self,
        *,
        namespace: str,
        repository: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/v2/repositories/{namespace}/{repository}/tags")

    async def repository_manifests(
        self,
        *,
        namespace: str,
        repository: str,
        tag: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/v2/repositories/{namespace}/{repository}/manifests/{tag}")


# --- GHCR Adapter ---

class GHCRAdapter(_BaseHttpAdapter):
    """GitHub Container Registry API adapter."""

    service_id = "ghcr"

    async def packages_list(
        self,
        *,
        owner: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/users/{owner}/packages")

    async def package_versions(
        self,
        *,
        owner: str,
        package: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/users/{owner}/packages/{package}/versions")

    async def container_list(
        self,
        *,
        org: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/orgs/{org}/container-registry")


# --- App Store Adapter ---

class AppStoreAdapter(_BaseHttpAdapter):
    """Apple App Store Connect API adapter."""

    service_id = "appstore"

    async def apps_list(
        self,
        *,
        limit: int = 50,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        return await self._request("GET", "/v1/apps", query_params=params)

    async def builds_list(
        self,
        *,
        app_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/v1/apps/{app_id}/builds")

    async def app_pricingtiers(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/v1/appPricingTiers")


# --- Google Play Adapter ---

class GooglePlayAdapter(_BaseHttpAdapter):
    """Google Play Console API adapter."""

    service_id = "googleplay"

    async def apps_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/playconsole/v3/apps")

    async def builds_list(
        self,
        *,
        app_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/playconsole/v3/applications/{app_id}/builds")

    async def releases_list(
        self,
        *,
        app_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/playconsole/v3/applications/{app_id}/releases")


# --- Sentry Adapter ---

class SentryAdapter(_BaseHttpAdapter):
    """Sentry API adapter for error tracking."""

    service_id = "sentry"

    async def issues_list(
        self,
        *,
        project: str = "",
        limit: int = 50,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        if project:
            params["project"] = project
        return await self._request("GET", "/api/0/issues/", query_params=params)

    async def events_list(
        self,
        *,
        issue_id: str,
    ) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", f"/api/0/issues/{issue_id}/events/")

    async def projects_list(self) -> tuple[Any, IntegrationError | None]:
        return await self._request("GET", "/api/0/projects/")


# --- Stripe Adapter ---

class StripeAdapter(_BaseHttpAdapter):
    """Stripe API adapter for payments."""

    service_id = "stripe"

    async def customers_list(
        self,
        *,
        limit: int = 50,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        return await self._request("GET", "/v1/customers", query_params=params)

    async def charges_list(
        self,
        *,
        limit: int = 50,
    ) -> tuple[Any, IntegrationError | None]:
        params: dict[str, Any] = {"limit": limit}
        return await self._request("GET", "/v1/charges", query_params=params)

    async def payments_create(
        self,
        *,
        amount: int,
        currency: str,
        customer: str = "",
    ) -> tuple[Any, IntegrationError | None]:
        body: dict[str, Any] = {"amount": amount, "currency": currency}
        if customer:
            body["customer"] = customer
        return await self._request("POST", "/v1/payment_intents", body=body)


# ---------------------------------------------------------------------------
# GenericEndpointAdapter — automatic REST routing for ALL services
# ---------------------------------------------------------------------------
# Instead of writing 37+ individual adapter classes, this adapter uses an
# endpoint route table to map action names → HTTP method + path + params.
# Typed adapters above still take priority; this is the fallback.
# ---------------------------------------------------------------------------

@dataclass
class _EndpointRoute:
    """A single endpoint route mapping: action → HTTP method + path."""
    method: str           # GET, POST, PUT, PATCH, DELETE
    path: str             # URL path template, e.g. "/v2/monitors"
    query_params: list[str] = field(default_factory=list)   # params → query string
    body_params: list[str] = field(default_factory=list)     # params → request body
    response_key: str = ""    # key to extract from response JSON (e.g. "data", "monitors")
    path_params: list[str] = field(default_factory=list)     # params interpolated into path


# ── Endpoint route table for ALL services without typed adapters ──
# Each service maps action names → _EndpointRoute definitions.
# This allows call_integration_api to work in adapter mode for ALL services.

_GENERIC_ENDPOINT_ROUTES: dict[str, dict[str, _EndpointRoute]] = {
    # ── Hosting & Deployment ──
    "hostinger": {
        # VPS Operations
        "vps.list":        _EndpointRoute("GET",  "/api/vps/v1/virtual-machines", response_key="data"),
        "vps.get":         _EndpointRoute("GET",  "/api/vps/v1/virtual-machines/{vps_id}", path_params=["vps_id"]),
        "vps.create":      _EndpointRoute("POST", "/api/vps/v1/virtual-machines", body_params=["name", "plan", "region"]),
        "vps.delete":      _EndpointRoute("DELETE", "/api/vps/v1/virtual-machines/{vps_id}", path_params=["vps_id"]),
        "vps.start":       _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/start", path_params=["vps_id"]),
        "vps.stop":        _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/stop", path_params=["vps_id"]),
        "vps.reboot":      _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/restart", path_params=["vps_id"]),
        "vps.recreate":    _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/recreate", path_params=["vps_id"]),
        "vps.metrics":     _EndpointRoute("GET",  "/api/vps/v1/virtual-machines/{vps_id}/metrics", path_params=["vps_id"]),
        "vps.backups":     _EndpointRoute("GET",  "/api/vps/v1/virtual-machines/{vps_id}/backups", path_params=["vps_id"]),
        "vps.backup.restore": _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/backups/{backup_id}/restore", path_params=["vps_id", "backup_id"]),
        "vps.firewall":    _EndpointRoute("GET",  "/api/vps/v1/virtual-machines/{vps_id}/firewall", path_params=["vps_id"]),
        "vps.firewall.update": _EndpointRoute("PUT",  "/api/vps/v1/virtual-machines/{vps_id}/firewall", path_params=["vps_id"]),
        # VPS Setup & Config
        "vps.setup":       _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/setup", path_params=["vps_id"]),
        "vps.set_hostname": _EndpointRoute("PUT",  "/api/vps/v1/virtual-machines/{vps_id}/hostname", path_params=["vps_id"]),
        "vps.set_root_password": _EndpointRoute("PUT", "/api/vps/v1/virtual-machines/{vps_id}/root-password", path_params=["vps_id"]),
        "vps.set_panel_password": _EndpointRoute("PUT", "/api/vps/v1/virtual-machines/{vps_id}/panel-password", path_params=["vps_id"]),
        # VPS Actions (history)
        "vps.actions":     _EndpointRoute("GET",  "/api/vps/v1/virtual-machines/{vps_id}/actions", path_params=["vps_id"]),
        "vps.action_details": _EndpointRoute("GET",  "/api/vps/v1/virtual-machines/{vps_id}/actions/{action_id}", path_params=["vps_id", "action_id"]),
        # Public Keys (SSH)
        "ssh_keys.list":   _EndpointRoute("GET",  "/api/vps/v1/public-keys", response_key="data"),
        "ssh_keys.attach": _EndpointRoute("POST", "/api/vps/v1/public-keys/attach/{vps_id}", path_params=["vps_id"]),
        "ssh_keys.detach": _EndpointRoute("POST", "/api/vps/v1/public-keys/detach/{vps_id}", path_params=["vps_id"]),
        # Data Centers (regions)
        "datacenters":     _EndpointRoute("GET",  "/api/vps/v1/data-centers", response_key="data"),
        # Plans & Pricing
        "plans.list":      _EndpointRoute("GET",  "/api/vps/v1/plans", response_key="data"),
        # Firewall Management (full CRUD)
        "firewall.list":   _EndpointRoute("GET",  "/api/vps/v1/firewall", response_key="data"),
        "firewall.create": _EndpointRoute("POST", "/api/vps/v1/firewall"),
        "firewall.get":    _EndpointRoute("GET",  "/api/vps/v1/firewall/{firewall_id}", path_params=["firewall_id"]),
        "firewall.delete": _EndpointRoute("DELETE", "/api/vps/v1/firewall/{firewall_id}", path_params=["firewall_id"]),
        # Firewall Rules
        "firewall.rule.create": _EndpointRoute("POST", "/api/vps/v1/firewall/{firewall_id}/rules", path_params=["firewall_id"]),
        "firewall.rule.update": _EndpointRoute("PUT",  "/api/vps/v1/firewall/{firewall_id}/rules/{rule_id}", path_params=["firewall_id", "rule_id"]),
        "firewall.rule.delete": _EndpointRoute("DELETE", "/api/vps/v1/firewall/{firewall_id}/rules/{rule_id}", path_params=["firewall_id", "rule_id"]),
        # Firewall VPS Association
        "firewall.activate": _EndpointRoute("POST", "/api/vps/v1/firewall/{firewall_id}/activate/{vps_id}", path_params=["firewall_id", "vps_id"]),
        "firewall.deactivate": _EndpointRoute("POST", "/api/vps/v1/firewall/{firewall_id}/deactivate/{vps_id}", path_params=["firewall_id", "vps_id"]),
        "firewall.sync":   _EndpointRoute("POST", "/api/vps/v1/firewall/{firewall_id}/sync/{vps_id}", path_params=["firewall_id", "vps_id"]),
        # Docker Management
        "docker.projects": _EndpointRoute("GET",  "/api/vps/v1/virtual-machines/{vps_id}/docker", path_params=["vps_id"]),
        "docker.create_project": _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/docker", path_params=["vps_id"]),
        "docker.delete_project": _EndpointRoute("DELETE", "/api/vps/v1/virtual-machines/{vps_id}/docker/{project_name}", path_params=["vps_id", "project_name"]),
        "docker.start_project": _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/docker/{project_name}/start", path_params=["vps_id", "project_name"]),
        "docker.stop_project": _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/docker/{project_name}/stop", path_params=["vps_id", "project_name"]),
        "docker.restart_project": _EndpointRoute("POST", "/api/vps/v1/virtual-machines/{vps_id}/docker/{project_name}/restart", path_params=["vps_id", "project_name"]),
        # Domains
        "domains.list":    _EndpointRoute("GET",  "/api/domains/v1/portfolio", response_key="data"),
        "domains.get":     _EndpointRoute("GET",  "/api/domains/v1/portfolio/{domain}", path_params=["domain"]),
        "domains.check":   _EndpointRoute("POST", "/api/domains/v1/availability", body_params=["domain"]),
        "domains.forward": _EndpointRoute("POST", "/api/domains/v1/forwarding", body_params=["domain", "forward_to"]),
        # DNS Records
        "dns.list":        _EndpointRoute("GET",  "/api/dns/v1/zones/{domain}", path_params=["domain"]),
        "dns.update":      _EndpointRoute("PUT",  "/api/dns/v1/zones/{domain}", path_params=["domain"]),
        "dns.delete":      _EndpointRoute("DELETE", "/api/dns/v1/zones/{domain}", path_params=["domain"]),
        "dns.reset":       _EndpointRoute("POST", "/api/dns/v1/zones/{domain}/reset", path_params=["domain"]),
        "dns.validate":    _EndpointRoute("POST", "/api/dns/v1/zones/{domain}/validate", path_params=["domain"]),
        # DNS Snapshots
        "dns.snapshots":   _EndpointRoute("GET",  "/api/dns/v1/snapshots/{domain}", path_params=["domain"]),
        "dns.snapshot.restore": _EndpointRoute("POST", "/api/dns/v1/snapshots/{domain}/restore", path_params=["domain"]),
        # Billing
        "billing.catalog": _EndpointRoute("GET",  "/api/billing/v1/catalog", response_key="data"),
        "billing.catalog_vps": _EndpointRoute("GET", "/api/billing/v1/catalog?category=VPS", response_key="data"),
        "billing.orders":  _EndpointRoute("GET",  "/api/billing/v1/orders", response_key="data"),
        "billing.orders.create": _EndpointRoute("POST", "/api/billing/v1/orders"),
        "billing.subscriptions": _EndpointRoute("GET", "/api/billing/v1/subscriptions", response_key="data"),
        "billing.payment_methods": _EndpointRoute("GET", "/api/billing/v1/payment-methods", response_key="data"),
    },
    "railway": {
        "projects.list":   _EndpointRoute("POST", "/graphql/v2", body_params=["query"], response_key="data"),
        "deployments.list": _EndpointRoute("POST", "/graphql/v2", body_params=["query"], response_key="data"),
        "services.list":   _EndpointRoute("POST", "/graphql/v2", body_params=["query"], response_key="data"),
        "variables.list":  _EndpointRoute("POST", "/graphql/v2", body_params=["query"], response_key="data"),
        "triggers.deploy": _EndpointRoute("POST", "/graphql/v2", body_params=["query"], response_key="data"),
    },
    "render": {
        "services.list":   _EndpointRoute("GET",  "/v1/services", response_key="data"),
        "deploys.list":    _EndpointRoute("GET",  "/v1/services/{service_id}/deploys", path_params=["service_id"], response_key="data"),
        "deploy.create":   _EndpointRoute("POST", "/v1/services/{service_id}/deploys", path_params=["service_id"]),
        "custom_domains.list": _EndpointRoute("GET", "/v1/services/{service_id}/customdomains", path_params=["service_id"], response_key="data"),
        "envvars.list":    _EndpointRoute("GET",  "/v1/services/{service_id}/envvars", path_params=["service_id"], response_key="data"),
    },
    "flyio": {
        "apps.list":       _EndpointRoute("GET",  "/v1/apps", response_key="data"),
        "machines.list":   _EndpointRoute("GET",  "/v1/apps/{app_id}/machines", path_params=["app_id"], response_key="data"),
        "machines.create": _EndpointRoute("POST", "/v1/apps/{app_id}/machines", path_params=["app_id"], body_params=["name", "region", "config"]),
        "machines.start":  _EndpointRoute("POST", "/v1/apps/{app_id}/machines/{machine_id}/start", path_params=["app_id", "machine_id"]),
        "machines.stop":   _EndpointRoute("POST", "/v1/apps/{app_id}/machines/{machine_id}/stop", path_params=["app_id", "machine_id"]),
        "volumes.list":    _EndpointRoute("GET",  "/v1/apps/{app_id}/volumes", path_params=["app_id"], response_key="data"),
    },
    "heroku": {
        "apps.list":       _EndpointRoute("GET",  "/apps", response_key="data"),
        "apps.create":     _EndpointRoute("POST", "/apps", body_params=["name", "region", "stack"]),
        "dynos.list":      _EndpointRoute("GET",  "/apps/{app_id}/dynos", path_params=["app_id"], response_key="data"),
        "dynos.restart":   _EndpointRoute("DELETE", "/apps/{app_id}/dynos", path_params=["app_id"]),
        "releases.list":   _EndpointRoute("GET",  "/apps/{app_id}/releases", path_params=["app_id"], response_key="data"),
        "configvars.list": _EndpointRoute("GET",  "/apps/{app_id}/config-vars", path_params=["app_id"]),
    },
    "digitalocean": {
        "droplets.list":    _EndpointRoute("GET", "/v2/droplets", response_key="droplets"),
        "droplets.create":  _EndpointRoute("POST", "/v2/droplets", body_params=["name", "region", "size", "image"]),
        "kubernetes.list":  _EndpointRoute("GET", "/v2/kubernetes/clusters", response_key="kubernetes_clusters"),
        "databases.list":   _EndpointRoute("GET", "/v2/databases", response_key="databases"),
        "domains.list":     _EndpointRoute("GET", "/v2/domains", response_key="domains"),
        "volumes.list":     _EndpointRoute("GET", "/v2/volumes", response_key="volumes"),
    },
    "cloudways": {
        "servers.list":    _EndpointRoute("GET",  "/api/v1/servers", response_key="data"),
        "apps.list":       _EndpointRoute("GET",  "/api/v1/apps", response_key="data"),
        "apps.create":     _EndpointRoute("POST", "/api/v1/apps", body_params=["server_id", "app_label", "app_type"]),
        "services.list":   _EndpointRoute("GET",  "/api/v1/services", response_key="data"),
        "domains.manage":  _EndpointRoute("POST", "/api/v1/app/manage", body_params=["server_id", "app_id", "domain"]),
    },
    # ── Cloud & Infrastructure (missing) ──
    "planetable": {
        "databases.list":      _EndpointRoute("GET",  "/v1/organizations/{org}/databases", path_params=["org"], response_key="data"),
        "databases.create":    _EndpointRoute("POST", "/v1/organizations/{org}/databases", path_params=["org"], body_params=["name", "region"]),
        "branches.list":       _EndpointRoute("GET",  "/v1/organizations/{org}/databases/{db}/branches", path_params=["org", "db"], response_key="data"),
        "branches.create":     _EndpointRoute("POST", "/v1/organizations/{org}/databases/{db}/branches", path_params=["org", "db"], body_params=["name"]),
        "backups.list":        _EndpointRoute("GET",  "/v1/organizations/{org}/databases/{db}/backups", path_params=["org", "db"], response_key="data"),
        "deploy_requests.list": _EndpointRoute("GET", "/v1/organizations/{org}/databases/{db}/deploy-requests", path_params=["org", "db"], response_key="data"),
    },
    # ── Domain & DNS (missing) ──
    "porkbun": {
        "domains.list":    _EndpointRoute("GET",  "/json/v3/domains/listAll", response_key="domains"),
        "dns.list":        _EndpointRoute("GET",  "/json/v3/dns/retrieve/{domain}", path_params=["domain"], response_key="records"),
        "dns.create":      _EndpointRoute("POST", "/json/v3/dns/create/{domain}", path_params=["domain"], body_params=["name", "type", "content", "ttl"]),
        "dns.edit":        _EndpointRoute("POST", "/json/v3/dns/edit/{domain}", path_params=["domain"], body_params=["id", "name", "type", "content"]),
        "dns.delete":      _EndpointRoute("POST", "/json/v3/dns/delete/{domain}", path_params=["domain"], body_params=["id"]),
        "domains.pricing": _EndpointRoute("GET",  "/json/v3/domains/pricing", response_key="pricing"),
    },
    # ── Communication (missing) ──
    "twilio": {
        "sms.send":        _EndpointRoute("POST", "/2010-04-01/Accounts/{account_sid}/Messages.json", path_params=["account_sid"], body_params=["To", "From", "Body"]),
        "calls.make":      _EndpointRoute("POST", "/2010-04-01/Accounts/{account_sid}/Calls.json", path_params=["account_sid"], body_params=["To", "From", "Url"]),
        "numbers.list":    _EndpointRoute("GET",  "/2010-04-01/Accounts/{account_sid}/IncomingPhoneNumbers.json", path_params=["account_sid"], response_key="incoming_phone_numbers"),
        "messages.list":   _EndpointRoute("GET",  "/2010-04-01/Accounts/{account_sid}/Messages.json", path_params=["account_sid"], response_key="messages"),
        "accounts.list":   _EndpointRoute("GET",  "/2010-04-01/Accounts.json", response_key="accounts"),
        "recordings.list": _EndpointRoute("GET",  "/2010-04-01/Accounts/{account_sid}/Recordings.json", path_params=["account_sid"], response_key="recordings"),
    },
    "sendgrid": {
        "mail.send":       _EndpointRoute("POST", "/v3/mail/send", body_params=["personalizations", "from", "subject", "content"]),
        "contacts.list":   _EndpointRoute("GET",  "/v3/marketing/contacts", response_key="result"),
        "templates.list":  _EndpointRoute("GET",  "/v3/templates", response_key="templates"),
        "stats.list":      _EndpointRoute("GET",  "/v3/stats", query_params=["aggregated_by", "start_date", "end_date"], response_key="stats"),
        "suppressions.list": _EndpointRoute("GET", "/v3/suppressions", response_key="suppressions"),
        "api_keys.list":   _EndpointRoute("GET",  "/v3/api_keys", response_key="result"),
    },
    "mailgun": {
        "messages.send":   _EndpointRoute("POST", "/v3/{domain}/messages", path_params=["domain"], body_params=["to", "from", "subject", "text"]),
        "domains.list":    _EndpointRoute("GET",  "/v3/domains", response_key="items"),
        "events.list":     _EndpointRoute("GET",  "/v3/{domain}/events", path_params=["domain"], query_params=["event", "limit"], response_key="items"),
        "routes.list":     _EndpointRoute("GET",  "/v3/routes", response_key="items"),
        "webhooks.list":   _EndpointRoute("GET",  "/v3/domains/{domain}/webhooks", path_params=["domain"], response_key="webhooks"),
        "stats.list":      _EndpointRoute("GET",  "/v3/{domain}/stats/total", path_params=["domain"], query_params=["event", "duration"], response_key="stats"),
    },
    "postmark": {
        "email.send":      _EndpointRoute("POST", "/email", body_params=["From", "To", "Subject", "TextBody"]),
        "email.batch":     _EndpointRoute("POST", "/email/batch", body_params=["Messages"]),
        "bounces.list":    _EndpointRoute("GET",  "/bounces", query_params=["count", "offset"], response_key="Bounces"),
        "templates.list":  _EndpointRoute("GET",  "/templates", query_params=["count", "offset"], response_key="Templates"),
        "stats.list":      _EndpointRoute("GET",  "/stats/outbound", query_params=["fromdate", "todate"], response_key="Days"),
        "domains.list":    _EndpointRoute("GET",  "/domains", response_key="Domains"),
    },
    "telegram": {
        "messages.send":   _EndpointRoute("POST", "/bot{token}/sendMessage", path_params=["token"], body_params=["chat_id", "text"]),
        "messages.edit":   _EndpointRoute("POST", "/bot{token}/editMessageText", path_params=["token"], body_params=["chat_id", "message_id", "text"]),
        "chats.list":      _EndpointRoute("GET",  "/bot{token}/getUpdates", path_params=["token"], response_key="result"),
        "updates.get":     _EndpointRoute("GET",  "/bot{token}/getUpdates", path_params=["token"], response_key="result"),
        "stickers.list":   _EndpointRoute("GET",  "/bot{token}/getStickerSet", path_params=["token"], query_params=["name"], response_key="result"),
        "files.send":      _EndpointRoute("POST", "/bot{token}/sendDocument", path_params=["token"], body_params=["chat_id", "document"]),
    },
    "microsoft_teams": {
        "channels.list":   _EndpointRoute("GET",  "/teams/{team_id}/channels", path_params=["team_id"], response_key="value"),
        "messages.send":   _EndpointRoute("POST", "/teams/{team_id}/channels/{channel_id}/messages", path_params=["team_id", "channel_id"], body_params=["body"]),
        "messages.list":   _EndpointRoute("GET",  "/teams/{team_id}/channels/{channel_id}/messages", path_params=["team_id", "channel_id"], response_key="value"),
        "teams.list":      _EndpointRoute("GET",  "/me/joinedTeams", response_key="value"),
        "files.list":      _EndpointRoute("GET",  "/groups/{group_id}/drive/root/children", path_params=["group_id"], response_key="value"),
        "meetings.list":   _EndpointRoute("GET",  "/me/onlineMeetings", response_key="value"),
    },
    # ── Developer Tools (missing) ──
    "gitlab": {
        "projects.list":   _EndpointRoute("GET",  "/api/v4/projects", query_params=["membership", "search", "per_page"], response_key="data"),
        "issues.list":     _EndpointRoute("GET",  "/api/v4/issues", query_params=["state", "labels", "per_page"], response_key="data"),
        "issues.create":   _EndpointRoute("POST", "/api/v4/projects/{id}/issues", path_params=["id"], body_params=["title", "description", "labels"]),
        "merges.list":     _EndpointRoute("GET",  "/api/v4/projects/{id}/merge_requests", path_params=["id"], query_params=["state"], response_key="data"),
        "pipelines.list":  _EndpointRoute("GET",  "/api/v4/projects/{id}/pipelines", path_params=["id"], response_key="data"),
        "runners.list":    _EndpointRoute("GET",  "/api/v4/runners", response_key="data"),
    },
    "bitbucket": {
        "repositories.list": _EndpointRoute("GET", "/2.0/repositories/{workspace}", path_params=["workspace"], query_params=["q", "sort"], response_key="values"),
        "pullrequests.list": _EndpointRoute("GET", "/2.0/repositories/{workspace}/{repo_slug}/pullrequests", path_params=["workspace", "repo_slug"], query_params=["state"], response_key="values"),
        "pipelines.list":    _EndpointRoute("GET", "/2.0/repositories/{workspace}/{repo_slug}/pipelines/", path_params=["workspace", "repo_slug"], response_key="values"),
        "deployments.list":  _EndpointRoute("GET", "/2.0/repositories/{workspace}/{repo_slug}/deployments/", path_params=["workspace", "repo_slug"], response_key="values"),
        "issues.list":       _EndpointRoute("GET", "/2.0/repositories/{workspace}/{repo_slug}/issues", path_params=["workspace", "repo_slug"], query_params=["state"], response_key="values"),
        "webhooks.list":     _EndpointRoute("GET", "/2.0/repositories/{workspace}/{repo_slug}/hooks", path_params=["workspace", "repo_slug"], response_key="values"),
    },
    "circleci": {
        "pipelines.list":    _EndpointRoute("GET", "/v2/project/{project_slug}/pipeline", path_params=["project_slug"], query_params=["branch"], response_key="items"),
        "pipeline.trigger":  _EndpointRoute("POST", "/v2/project/{project_slug}/pipeline", path_params=["project_slug"], body_params=["branch", "parameters"]),
        "workflows.list":    _EndpointRoute("GET", "/v2/pipeline/{pipeline_id}/workflow", path_params=["pipeline_id"], response_key="items"),
        "jobs.list":         _EndpointRoute("GET", "/v2/workflow/{workflow_id}/job", path_params=["workflow_id"], response_key="items"),
        "artifacts.list":    _EndpointRoute("GET", "/v2/project/{project_slug}/{pipeline_number}/artifacts", path_params=["project_slug", "pipeline_number"], response_key="items"),
        "contexts.list":     _EndpointRoute("GET", "/v2/context", response_key="items"),
    },
    "jenkins": {
        "jobs.list":         _EndpointRoute("GET",  "/api/json", query_params=["tree", "pretty"], response_key="jobs"),
        "build.trigger":     _EndpointRoute("POST", "/job/{job_name}/build", path_params=["job_name"]),
        "build.status":      _EndpointRoute("GET",  "/job/{job_name}/lastBuild/api/json", path_params=["job_name"]),
        "queue.list":        _EndpointRoute("GET",  "/queue/api/json", response_key="items"),
        "nodes.list":        _EndpointRoute("GET",  "/computer/api/json", response_key="computer"),
        "credentials.list":  _EndpointRoute("GET",  "/credential-store/domain/_/api/json", response_key="credentials"),
    },
    "pagerduty": {
        "incidents.list":    _EndpointRoute("GET",  "/incidents", query_params=["statuses[]", "service_ids[]", "limit"], response_key="incidents"),
        "incidents.create":  _EndpointRoute("POST", "/incidents", body_params=["incident"]),
        "oncalls.list":      _EndpointRoute("GET",  "/oncalls", query_params=["limit"], response_key="oncalls"),
        "services.list":     _EndpointRoute("GET",  "/services", query_params=["limit"], response_key="services"),
        "schedules.list":    _EndpointRoute("GET",  "/schedules", query_params=["limit"], response_key="schedules"),
        "escalation.list":   _EndpointRoute("GET",  "/escalation_policies", query_params=["limit"], response_key="escalation_policies"),
    },
    # ── Productivity (missing) ──
    "asana": {
        "projects.list":    _EndpointRoute("GET",  "/api/1.0/projects", query_params=["workspace", "archived"], response_key="data"),
        "tasks.list":       _EndpointRoute("GET",  "/api/1.0/tasks", query_params=["project", "assignee", "completed_since"], response_key="data"),
        "tasks.create":     _EndpointRoute("POST", "/api/1.0/tasks", body_params=["name", "notes", "projects", "assignee", "due_on"]),
        "tasks.update":     _EndpointRoute("PUT",  "/api/1.0/tasks/{task_gid}", path_params=["task_gid"], body_params=["name", "notes", "completed"]),
        "sections.list":    _EndpointRoute("GET",  "/api/1.0/projects/{project_gid}/sections", path_params=["project_gid"], response_key="data"),
        "tags.list":        _EndpointRoute("GET",  "/api/1.0/tags", query_params=["workspace"], response_key="data"),
        "workspaces.list":  _EndpointRoute("GET",  "/api/1.0/workspaces", response_key="data"),
    },
    "clickup": {
        "spaces.list":      _EndpointRoute("GET",  "/api/v2/team/{team_id}/space", path_params=["team_id"], response_key="spaces"),
        "lists.list":       _EndpointRoute("GET",  "/api/v2/space/{space_id}/list", path_params=["space_id"], response_key="lists"),
        "tasks.list":       _EndpointRoute("GET",  "/api/v2/list/{list_id}/task", path_params=["list_id"], query_params=["statuses[]", "page"], response_key="tasks"),
        "tasks.create":     _EndpointRoute("POST", "/api/v2/list/{list_id}/task", path_params=["list_id"], body_params=["name", "description", "status", "priority"]),
        "tasks.update":     _EndpointRoute("PUT",  "/api/v2/task/{task_id}", path_params=["task_id"], body_params=["name", "description", "status"]),
        "time.list":        _EndpointRoute("GET",  "/api/v2/team/{team_id}/time_entries", path_params=["team_id"], response_key="data"),
        "goals.list":       _EndpointRoute("GET",  "/api/v2/team/{team_id}/goal", path_params=["team_id"], response_key="goals"),
    },
    "monday": {
        "boards.list":      _EndpointRoute("POST", "/v2", body_params=["query"], response_key="data"),
        "items.list":       _EndpointRoute("POST", "/v2", body_params=["query"], response_key="data"),
        "items.create":     _EndpointRoute("POST", "/v2", body_params=["query"], response_key="data"),
        "updates.create":   _EndpointRoute("POST", "/v2", body_params=["query"], response_key="data"),
        "groups.list":      _EndpointRoute("POST", "/v2", body_params=["query"], response_key="data"),
        "workspaces.list":  _EndpointRoute("POST", "/v2", body_params=["query"], response_key="data"),
    },
    "airtable": {
        "records.list":     _EndpointRoute("GET",  "/v0/{base_id}/{table_name}", path_params=["base_id", "table_name"], query_params=["view", "filterByFormula", "maxRecords"], response_key="records"),
        "records.create":   _EndpointRoute("POST", "/v0/{base_id}/{table_name}", path_params=["base_id", "table_name"], body_params=["fields"]),
        "records.update":   _EndpointRoute("PATCH", "/v0/{base_id}/{table_name}/{record_id}", path_params=["base_id", "table_name", "record_id"], body_params=["fields"]),
        "records.delete":   _EndpointRoute("DELETE", "/v0/{base_id}/{table_name}/{record_id}", path_params=["base_id", "table_name", "record_id"]),
        "bases.list":       _EndpointRoute("GET",  "/v0/meta/bases", response_key="bases"),
        "tables.list":      _EndpointRoute("GET",  "/v0/meta/bases/{base_id}/tables", path_params=["base_id"], response_key="tables"),
    },
    "confluence": {
        "pages.list":       _EndpointRoute("GET",  "/wiki/rest/api/content", query_params=["spaceKey", "type", "limit"], response_key="results"),
        "pages.create":     _EndpointRoute("POST", "/wiki/rest/api/content", body_params=["type", "title", "space", "body"]),
        "pages.update":     _EndpointRoute("PUT",  "/wiki/rest/api/content/{id}", path_params=["id"], body_params=["type", "title", "version", "body"]),
        "spaces.list":      _EndpointRoute("GET",  "/wiki/rest/api/space", query_params=["limit"], response_key="results"),
        "attachments.list": _EndpointRoute("GET",  "/wiki/rest/api/content/{id}/child/attachment", path_params=["id"], response_key="results"),
        "comments.list":    _EndpointRoute("GET",  "/wiki/rest/api/content/{id}/child/comment", path_params=["id"], response_key="results"),
    },
    "basecamp": {
        "projects.list":    _EndpointRoute("GET",  "/{account_id}/projects.json", path_params=["account_id"], response_key="data"),
        "todos.list":       _EndpointRoute("GET",  "/{account_id}/buckets/{project_id}/todolists/{todolist_id}/todos.json", path_params=["account_id", "project_id", "todolist_id"], response_key="data"),
        "todos.create":     _EndpointRoute("POST", "/{account_id}/buckets/{project_id}/todolists/{todolist_id}/todos.json", path_params=["account_id", "project_id", "todolist_id"], body_params=["content", "description", "due_on"]),
        "messages.list":    _EndpointRoute("GET",  "/{account_id}/buckets/{project_id}/messages.json", path_params=["account_id", "project_id"], response_key="data"),
        "schedules.list":   _EndpointRoute("GET",  "/{account_id}/buckets/{project_id}/schedules.json", path_params=["account_id", "project_id"], response_key="data"),
        "campfires.list":   _EndpointRoute("GET",  "/{account_id}/buckets/{project_id}/chats.json", path_params=["account_id", "project_id"], response_key="data"),
    },
    # ── Data & Analytics (missing) ──
    "google_sheets": {
        "spreadsheets.get":     _EndpointRoute("GET", "/v4/spreadsheets/{spreadsheetId}", path_params=["spreadsheetId"], response_key="sheets"),
        "spreadsheets.create":  _EndpointRoute("POST", "/v4/spreadsheets", body_params=["properties", "sheets"]),
        "values.get":           _EndpointRoute("GET", "/v4/spreadsheets/{spreadsheetId}/values/{range}", path_params=["spreadsheetId", "range"], response_key="values"),
        "values.update":        _EndpointRoute("PUT", "/v4/spreadsheets/{spreadsheetId}/values/{range}", path_params=["spreadsheetId", "range"], body_params=["values", "valueInputOption"], query_params=["valueInputOption"]),
        "values.append":        _EndpointRoute("POST", "/v4/spreadsheets/{spreadsheetId}/values/{range}:append", path_params=["spreadsheetId", "range"], body_params=["values"], query_params=["valueInputOption"]),
        "sheets.list":          _EndpointRoute("GET", "/v4/spreadsheets/{spreadsheetId}", path_params=["spreadsheetId"], response_key="sheets"),
    },
    "segment": {
        "track":            _EndpointRoute("POST", "/v1/track", body_params=["userId", "event", "properties"]),
        "identify":         _EndpointRoute("POST", "/v1/identify", body_params=["userId", "traits"]),
        "page":             _EndpointRoute("POST", "/v1/page", body_params=["userId", "name", "properties"]),
        "group":            _EndpointRoute("POST", "/v1/group", body_params=["userId", "groupId", "traits"]),
        "sources.list":     _EndpointRoute("GET",  "/v1/sources", response_key="sources"),
        "destinations.list": _EndpointRoute("GET", "/v1/destinations", response_key="destinations"),
    },
    "plausible": {
        "stats.aggregate":   _EndpointRoute("GET", "/api/v1/stats/aggregate", query_params=["site_id", "period", "date", "metrics"], response_key="results"),
        "stats.timeseries":  _EndpointRoute("GET", "/api/v1/stats/timeseries", query_params=["site_id", "period", "date", "metrics"], response_key="results"),
        "stats.breakdown":   _EndpointRoute("GET", "/api/v1/stats/breakdown", query_params=["site_id", "period", "date", "property", "metrics"], response_key="results"),
        "sites.list":        _EndpointRoute("GET", "/api/v1/sites", response_key="sites"),
        "goals.list":        _EndpointRoute("GET", "/api/v1/sites/{site_id}/goals", path_params=["site_id"], response_key="goals"),
        "events.list":       _EndpointRoute("GET", "/api/v1/sites/{site_id}/events", path_params=["site_id"], response_key="events"),
    },
    # ── CRM & Sales (missing) ──
    "hubspot": {
        "contacts.list":    _EndpointRoute("GET",  "/crm/v3/objects/contacts", query_params=["limit", "after", "properties"], response_key="results"),
        "contacts.create":  _EndpointRoute("POST", "/crm/v3/objects/contacts", body_params=["properties"]),
        "deals.list":       _EndpointRoute("GET",  "/crm/v3/objects/deals", query_params=["limit", "after", "properties"], response_key="results"),
        "companies.list":   _EndpointRoute("GET",  "/crm/v3/objects/companies", query_params=["limit", "after"], response_key="results"),
        "tickets.list":     _EndpointRoute("GET",  "/crm/v3/objects/tickets", query_params=["limit", "after"], response_key="results"),
        "pipelines.list":   _EndpointRoute("GET",  "/crm/v3/pipelines", query_params=["object_type"], response_key="results"),
    },
    "salesforce": {
        "contacts.list":    _EndpointRoute("GET",  "/services/data/v59.0/sobjects/Contact", query_params=["q"], response_key="recentItems"),
        "leads.list":       _EndpointRoute("GET",  "/services/data/v59.0/sobjects/Lead", query_params=["q"], response_key="recentItems"),
        "opportunities.list": _EndpointRoute("GET", "/services/data/v59.0/sobjects/Opportunity", response_key="recentItems"),
        "cases.list":       _EndpointRoute("GET",  "/services/data/v59.0/sobjects/Case", response_key="recentItems"),
        "accounts.list":    _EndpointRoute("GET",  "/services/data/v59.0/sobjects/Account", response_key="recentItems"),
        "soql.query":       _EndpointRoute("GET",  "/services/data/v59.0/query", query_params=["q"], response_key="records"),
    },
    "pipedrive": {
        "deals.list":       _EndpointRoute("GET",  "/v1/deals", query_params=["status", "limit", "start"], response_key="data"),
        "persons.list":     _EndpointRoute("GET",  "/v1/persons", query_params=["limit", "start"], response_key="data"),
        "organizations.list": _EndpointRoute("GET", "/v1/organizations", query_params=["limit", "start"], response_key="data"),
        "activities.list":  _EndpointRoute("GET",  "/v1/activities", query_params=["limit", "start"], response_key="data"),
        "pipelines.list":   _EndpointRoute("GET",  "/v1/pipelines", response_key="data"),
        "products.list":    _EndpointRoute("GET",  "/v1/products", query_params=["limit", "start"], response_key="data"),
    },
    "zoho": {
        "leads.list":       _EndpointRoute("GET",  "/crm/v2/Leads", query_params=["page", "per_page"], response_key="data"),
        "contacts.list":    _EndpointRoute("GET",  "/crm/v2/Contacts", query_params=["page", "per_page"], response_key="data"),
        "deals.list":       _EndpointRoute("GET",  "/crm/v2/Deals", query_params=["page", "per_page"], response_key="data"),
        "accounts.list":    _EndpointRoute("GET",  "/crm/v2/Accounts", query_params=["page", "per_page"], response_key="data"),
        "tasks.list":       _EndpointRoute("GET",  "/crm/v2/Tasks", query_params=["page", "per_page"], response_key="data"),
        "modules.list":     _EndpointRoute("GET",  "/crm/v2/settings/modules", response_key="modules"),
    },
    "close": {
        "leads.list":        _EndpointRoute("GET", "/api/v1/lead/", query_params=["query", "_skip", "_limit"], response_key="data"),
        "contacts.list":     _EndpointRoute("GET", "/api/v1/contact/", query_params=["lead_id", "_limit"], response_key="data"),
        "opportunities.list": _EndpointRoute("GET", "/api/v1/opportunity/", query_params=["lead_id", "_limit"], response_key="data"),
        "activities.list":   _EndpointRoute("GET", "/api/v1/activity/", query_params=["lead_id", "_type", "_limit"], response_key="data"),
        "tasks.list":        _EndpointRoute("GET", "/api/v1/task/", query_params=["_limit", "is_complete"], response_key="data"),
        "sequences.list":    _EndpointRoute("GET", "/api/v1/sequence/", response_key="data"),
    },
    # ── Payments (missing) ──
    "paypal": {
        "orders.create":    _EndpointRoute("POST", "/v2/checkout/orders", body_params=["intent", "purchase_units"]),
        "orders.list":      _EndpointRoute("GET",  "/v2/checkout/orders", query_params=["page", "page_size"], response_key="items"),
        "payments.list":    _EndpointRoute("GET",  "/v2/payments", query_params=["page", "page_size"], response_key="items"),
        "payouts.create":   _EndpointRoute("POST", "/v1/payments/payouts", body_params=["sender_batch_header", "items"]),
        "subscriptions.list": _EndpointRoute("GET", "/v1/billing/subscriptions", query_params=["page", "page_size"], response_key="items"),
        "invoices.list":    _EndpointRoute("GET",  "/v2/invoicing/invoices", query_params=["page", "page_size"], response_key="items"),
    },
    "lemon_squeezy": {
        "products.list":    _EndpointRoute("GET",  "/v1/products", query_params=["page"], response_key="data"),
        "orders.list":      _EndpointRoute("GET",  "/v1/orders", query_params=["page"], response_key="data"),
        "customers.list":   _EndpointRoute("GET",  "/v1/customers", query_params=["page"], response_key="data"),
        "subscriptions.list": _EndpointRoute("GET", "/v1/subscriptions", query_params=["page"], response_key="data"),
        "licenses.list":    _EndpointRoute("GET",  "/v1/licenses", query_params=["page"], response_key="data"),
        "discounts.list":   _EndpointRoute("GET",  "/v1/discounts", query_params=["page"], response_key="data"),
    },
    # ── Monitoring (missing) ──
    "datadog": {
        "metrics.list":     _EndpointRoute("GET",  "/api/v1/metrics", response_key="metrics"),
        "metrics.query":    _EndpointRoute("GET",  "/api/v1/query", query_params=["from", "to", "query"], response_key="series"),
        "dashboards.list":  _EndpointRoute("GET",  "/api/v1/dashboard", response_key="dashboards"),
        "monitors.list":    _EndpointRoute("GET",  "/api/v1/monitor", response_key="data"),
        "events.list":      _EndpointRoute("GET",  "/api/v1/events", query_params=["start", "end"], response_key="events"),
        "hosts.list":       _EndpointRoute("GET",  "/api/v1/hosts", response_key="host_list"),
    },
    "newrelic": {
        "accounts.list":    _EndpointRoute("POST", "/graphql", body_params=["query"], response_key="data"),
        "apps.list":        _EndpointRoute("POST", "/graphql", body_params=["query"], response_key="data"),
        "alerts.list":      _EndpointRoute("POST", "/graphql", body_params=["query"], response_key="data"),
        "deployments.list": _EndpointRoute("POST", "/graphql", body_params=["query"], response_key="data"),
        "nrql.query":       _EndpointRoute("POST", "/graphql", body_params=["query"], response_key="data"),
        "dashboards.list":  _EndpointRoute("POST", "/graphql", body_params=["query"], response_key="data"),
    },
    "pingdom": {
        "checks.list":      _EndpointRoute("GET",  "/api/3.1/checks", response_key="checks"),
        "checks.create":    _EndpointRoute("POST", "/api/3.1/checks", body_params=["name", "host", "type", "resolution"]),
        "results.list":     _EndpointRoute("GET",  "/api/3.1/checks/{check_id}/results", path_params=["check_id"], response_key="results"),
        "actions.list":     _EndpointRoute("GET",  "/api/3.1/actions", response_key="actions"),
        "teams.list":       _EndpointRoute("GET",  "/api/3.1/teams", response_key="teams"),
        "maintenance.list": _EndpointRoute("GET",  "/api/3.1/maintenance", response_key="maintenance"),
    },
    "uptimerobot": {
        "monitors.list":    _EndpointRoute("POST", "/v2/getMonitors", body_params=["custom_uptime_ratios"], response_key="monitors"),
        "monitors.create":  _EndpointRoute("POST", "/v2/newMonitor", body_params=["friendly_name", "url", "type"]),
        "monitors.reset":   _EndpointRoute("POST", "/v2/resetMonitor", body_params=["id"]),
        "alert_contacts.list": _EndpointRoute("POST", "/v2/getAlertContacts", response_key="alert_contacts"),
        "mwindows.list":    _EndpointRoute("POST", "/v2/getMWindows", response_key="mwindows"),
        "psp.list":         _EndpointRoute("POST", "/v2/getPSPs", response_key="psps"),
    },
    "statuspage": {
        "pages.list":       _EndpointRoute("GET",  "/v1/pages", response_key="data"),
        "incidents.list":   _EndpointRoute("GET",  "/v1/pages/{page_id}/incidents.json", path_params=["page_id"], response_key="data"),
        "incidents.create": _EndpointRoute("POST", "/v1/pages/{page_id}/incidents.json", path_params=["page_id"], body_params=["incident"]),
        "components.list":  _EndpointRoute("GET",  "/v1/pages/{page_id}/components.json", path_params=["page_id"], response_key="data"),
        "subscribers.list": _EndpointRoute("GET",  "/v1/pages/{page_id}/subscribers.json", path_params=["page_id"], response_key="data"),
        "metrics.list":     _EndpointRoute("GET",  "/v1/pages/{page_id}/metrics.json", path_params=["page_id"], response_key="data"),
    },
    # ── Automation (missing) ──
    "zapier": {
        "zaps.list":        _EndpointRoute("GET",  "/api/v2/zaps", response_key="data"),
        "zaps.run":         _EndpointRoute("POST", "/api/v2/zaps/{zap_id}/run", path_params=["zap_id"], body_params=["input"]),
        "connections.list": _EndpointRoute("GET",  "/api/v2/connections", response_key="data"),
        "actions.list":     _EndpointRoute("GET",  "/api/v2/actions", response_key="data"),
        "triggers.list":    _EndpointRoute("GET",  "/api/v2/triggers", response_key="data"),
    },
    "make": {
        "scenarios.list":   _EndpointRoute("GET",  "/api/v2/scenarios", response_key="scenarios"),
        "scenarios.run":    _EndpointRoute("POST", "/api/v2/scenarios/{scenario_id}/run", path_params=["scenario_id"]),
        "connections.list": _EndpointRoute("GET",  "/api/v2/connections", response_key="connections"),
        "organizations.list": _EndpointRoute("GET", "/api/v2/organizations", response_key="organizations"),
        "teams.list":       _EndpointRoute("GET",  "/api/v2/teams", response_key="teams"),
    },
    "n8n": {
        "workflows.list":   _EndpointRoute("GET",  "/api/v1/workflows", response_key="data"),
        "workflows.create": _EndpointRoute("POST", "/api/v1/workflows", body_params=["name", "nodes", "connections"]),
        "workflows.execute": _EndpointRoute("POST", "/api/v1/workflows/{id}/execute", path_params=["id"], body_params=["startNodes", "runData"]),
        "executions.list":  _EndpointRoute("GET",  "/api/v1/executions", response_key="data"),
        "credentials.list": _EndpointRoute("GET",  "/api/v1/credentials", response_key="data"),
    },
    "ifttt": {
        "applets.list":     _EndpointRoute("GET",  "/api/v2/applets", response_key="data"),
        "triggers.fire":    _EndpointRoute("POST", "/api/v2/triggers/{trigger_slug}/fire", path_params=["trigger_slug"], body_params=["value1", "value2", "value3"]),
        "actions.create":   _EndpointRoute("POST", "/api/v2/actions/{action_slug}", path_params=["action_slug"], body_params=["value1", "value2", "value3"]),
        "services.list":    _EndpointRoute("GET",  "/api/v2/services", response_key="data"),
        "connections.list": _EndpointRoute("GET",  "/api/v2/connections", response_key="data"),
    },
    # ── Storage (missing) ──
    "dropbox": {
        "files.list":       _EndpointRoute("POST", "/2/files/list_folder", body_params=["path", "recursive"]),
        "files.upload":     _EndpointRoute("POST", "/2/files/upload", body_params=["path", "mode", "autorename"]),
        "files.download":   _EndpointRoute("POST", "/2/files/download", body_params=["path"]),
        "files.delete":     _EndpointRoute("POST", "/2/files/delete_v2", body_params=["path"]),
        "folders.create":   _EndpointRoute("POST", "/2/files/create_folder_v2", body_params=["path", "autorename"]),
        "sharing.list":     _EndpointRoute("POST", "/2/sharing/list_shared_links", body_params=["path"]),
    },
    "aws_s3": {
        "buckets.list":     _EndpointRoute("GET",  "/", response_key="Buckets"),
        "objects.list":     _EndpointRoute("GET",  "/{bucket}", path_params=["bucket"], query_params=["prefix", "max-keys"], response_key="Contents"),
        "objects.get":      _EndpointRoute("GET",  "/{bucket}/{key}", path_params=["bucket", "key"]),
        "objects.put":      _EndpointRoute("PUT",  "/{bucket}/{key}", path_params=["bucket", "key"], body_params=["body"]),
        "objects.delete":   _EndpointRoute("DELETE", "/{bucket}/{key}", path_params=["bucket", "key"]),
        "presign.url":      _EndpointRoute("GET",  "/{bucket}/{key}", path_params=["bucket", "key"], query_params=["presign", "expires"]),
    },
    "backblaze": {
        "buckets.list":     _EndpointRoute("GET",  "/b2api/v2/b2_list_buckets", query_params=["accountId"], response_key="buckets"),
        "buckets.create":   _EndpointRoute("POST", "/b2api/v2/b2_create_bucket", body_params=["accountId", "bucketName", "bucketType"]),
        "files.list":       _EndpointRoute("GET",  "/b2api/v2/b2_list_file_names", query_params=["bucketId", "maxFileCount"], response_key="files"),
        "files.upload":     _EndpointRoute("POST", "/b2api/v2/b2_get_upload_url", body_params=["bucketId"]),
        "files.download":   _EndpointRoute("GET",  "/b2api/v2/b2_download_file_by_id", query_params=["fileId"]),
        "files.delete":     _EndpointRoute("POST", "/b2api/v2/b2_delete_file_version", body_params=["fileName", "fileId"]),
    },
    "wasabi": {
        "buckets.list":     _EndpointRoute("GET",  "/", response_key="Buckets"),
        "objects.list":     _EndpointRoute("GET",  "/{bucket}", path_params=["bucket"], query_params=["prefix", "max-keys"], response_key="Contents"),
        "objects.get":      _EndpointRoute("GET",  "/{bucket}/{key}", path_params=["bucket", "key"]),
        "objects.put":      _EndpointRoute("PUT",  "/{bucket}/{key}", path_params=["bucket", "key"], body_params=["body"]),
        "objects.delete":   _EndpointRoute("DELETE", "/{bucket}/{key}", path_params=["bucket", "key"]),
        "buckets.create":   _EndpointRoute("PUT",  "/{bucket}", path_params=["bucket"]),
    },
    # ── Container & Infra (missing) ──
    "ansible": {
        "job_templates.list": _EndpointRoute("GET", "/api/v2/job_templates/", query_params=["page_size", "order_by"], response_key="results"),
        "jobs.launch":        _EndpointRoute("POST", "/api/v2/job_templates/{id}/launch/", path_params=["id"], body_params=["extra_vars"]),
        "inventories.list":   _EndpointRoute("GET", "/api/v2/inventories/", query_params=["page_size"], response_key="results"),
        "credentials.list":   _EndpointRoute("GET", "/api/v2/credentials/", query_params=["page_size"], response_key="results"),
        "projects.list":      _EndpointRoute("GET", "/api/v2/projects/", query_params=["page_size"], response_key="results"),
        "hosts.list":         _EndpointRoute("GET", "/api/v2/hosts/", query_params=["page_size"], response_key="results"),
    },
    # ── AI & ML (missing) ──
    "openai": {
        "chat.completions":  _EndpointRoute("POST", "/v1/chat/completions", body_params=["model", "messages", "temperature", "max_tokens"]),
        "completions.create": _EndpointRoute("POST", "/v1/completions", body_params=["model", "prompt", "max_tokens"]),
        "embeddings.create": _EndpointRoute("POST", "/v1/embeddings", body_params=["model", "input"]),
        "images.generate":   _EndpointRoute("POST", "/v1/images/generations", body_params=["model", "prompt", "n", "size"]),
        "audio.transcribe":  _EndpointRoute("POST", "/v1/audio/transcriptions", body_params=["model", "file"]),
        "models.list":       _EndpointRoute("GET",  "/v1/models", response_key="data"),
    },
    "anthropic": {
        "messages.create":   _EndpointRoute("POST", "/v1/messages", body_params=["model", "messages", "max_tokens", "system"]),
        "messages.list":     _EndpointRoute("GET",  "/v1/messages", query_params=["conversation_id"], response_key="data"),
        "models.list":       _EndpointRoute("GET",  "/v1/models", response_key="data"),
    },
    "cohere": {
        "chat":              _EndpointRoute("POST", "/v1/chat", body_params=["message", "model", "temperature"]),
        "embed":             _EndpointRoute("POST", "/v1/embed", body_params=["texts", "model", "input_type"]),
        "generate":          _EndpointRoute("POST", "/v1/generate", body_params=["prompt", "model", "max_tokens"]),
        "rerank":            _EndpointRoute("POST", "/v1/rerank", body_params=["query", "documents", "model"]),
        "classify":          _EndpointRoute("POST", "/v1/classify", body_params=["inputs", "model"]),
        "models.list":       _EndpointRoute("GET",  "/v1/models", response_key="models"),
    },
    "google_ai": {
        "generate":          _EndpointRoute("POST", "/v1beta/models/{model}:generateContent", path_params=["model"], body_params=["contents", "generationConfig"]),
        "chat":              _EndpointRoute("POST", "/v1beta/models/{model}:generateContent", path_params=["model"], body_params=["contents", "generationConfig"]),
        "embed":             _EndpointRoute("POST", "/v1beta/models/{model}:embedContent", path_params=["model"], body_params=["content"]),
        "models.list":       _EndpointRoute("GET",  "/v1beta/models", response_key="models"),
        "files.list":        _EndpointRoute("GET",  "/v1beta/files", response_key="files"),
    },
    "huggingface": {
        "models.list":      _EndpointRoute("GET",  "/models", query_params=["search", "limit", "sort"], response_key="data"),
        "models.info":      _EndpointRoute("GET",  "/models/{model_id}", path_params=["model_id"], response_key="data"),
        "inference.run":    _EndpointRoute("POST", "/models/{model_id}", path_params=["model_id"], body_params=["inputs", "parameters"]),
        "datasets.list":    _EndpointRoute("GET",  "/datasets", query_params=["search", "limit"], response_key="data"),
        "spaces.list":      _EndpointRoute("GET",  "/spaces", query_params=["search", "limit"], response_key="data"),
        "endpoints.list":   _EndpointRoute("GET",  "/api/endpoints", response_key="data"),
    },
    "replicate": {
        "predictions.create": _EndpointRoute("POST", "/v1/predictions", body_params=["model", "input", "webhook"]),
        "predictions.list":  _EndpointRoute("GET",  "/v1/predictions", query_params=["limit"], response_key="results"),
        "predictions.get":   _EndpointRoute("GET",  "/v1/predictions/{id}", path_params=["id"]),
        "models.list":       _EndpointRoute("GET",  "/v1/models", query_params=["limit"], response_key="results"),
        "collections.list":  _EndpointRoute("GET",  "/v1/collections", response_key="results"),
        "trainings.create":  _EndpointRoute("POST", "/v1/trainings", body_params=["model", "input", "webhook"]),
    },
    "stabilityai": {
        "image.generate":   _EndpointRoute("POST", "/v1/generation/{engine_id}/text-to-image", path_params=["engine_id"], body_params=["text_prompts", "cfg_scale", "steps"]),
        "image.upscale":    _EndpointRoute("POST", "/v1/generation/{engine_id}/image-to-image/upscale", path_params=["engine_id"], body_params=["image", "width", "height"]),
        "image.variations": _EndpointRoute("POST", "/v1/generation/{engine_id}/image-to-image", path_params=["engine_id"], body_params=["image", "text_prompts"]),
        "engines.list":     _EndpointRoute("GET",  "/v1/engines/list", response_key="data"),
        "accounts.balance": _EndpointRoute("GET",  "/v1/user/account", response_key="data"),
        "history.list":     _EndpointRoute("GET",  "/v1/history", query_params=["limit"], response_key="data"),
    },
    # ── E-commerce (missing) ──
    "shopify": {
        "products.list":    _EndpointRoute("GET",  "/admin/api/2024-01/products.json", query_params=["limit", "status", "vendor"], response_key="products"),
        "orders.list":      _EndpointRoute("GET",  "/admin/api/2024-01/orders.json", query_params=["limit", "status"], response_key="orders"),
        "customers.list":   _EndpointRoute("GET",  "/admin/api/2024-01/customers.json", query_params=["limit"], response_key="customers"),
        "inventory.list":   _EndpointRoute("GET",  "/admin/api/2024-01/inventory_items.json", query_params=["limit"], response_key="inventory_items"),
        "fulfillments.list": _EndpointRoute("GET", "/admin/api/2024-01/fulfillment_orders.json", query_params=["limit"], response_key="fulfillment_orders"),
        "discounts.list":   _EndpointRoute("GET",  "/admin/api/2024-01/price_rules.json", query_params=["limit"], response_key="price_rules"),
    },
    "woocommerce": {
        "products.list":    _EndpointRoute("GET",  "/wp-json/wc/v3/products", query_params=["per_page", "status", "category"], response_key="data"),
        "orders.list":      _EndpointRoute("GET",  "/wp-json/wc/v3/orders", query_params=["per_page", "status"], response_key="data"),
        "customers.list":   _EndpointRoute("GET",  "/wp-json/wc/v3/customers", query_params=["per_page"], response_key="data"),
        "coupons.list":     _EndpointRoute("GET",  "/wp-json/wc/v3/coupons", query_params=["per_page"], response_key="data"),
        "reports.list":     _EndpointRoute("GET",  "/wp-json/wc/v3/reports", response_key="data"),
        "categories.list":  _EndpointRoute("GET",  "/wp-json/wc/v3/products/categories", query_params=["per_page"], response_key="data"),
    },
    "youtube": {
        "videos.list":       _EndpointRoute(method="GET", path="/videos", query_params=["part", "chart", "myRating", "maxResults", "pageToken", "regionCode", "videoCategoryId"]),
        "videos.get":        _EndpointRoute(method="GET", path="/videos", query_params=["part", "id"]),
        "videos.rate":       _EndpointRoute(method="POST", path="/videos/rate", query_params=["id", "rating"]),
        "search.list":       _EndpointRoute(method="GET", path="/search", query_params=["part", "q", "type", "maxResults", "pageToken", "order", "channelId", "publishedAfter", "publishedBefore", "regionCode", "relevanceLanguage", "safeSearch", "videoCaption", "videoCategoryId", "videoDefinition", "videoDimension", "videoDuration", "videoEmbeddable", "videoLicense", "videoSyndicated", "videoType"]),
        "channels.list":     _EndpointRoute(method="GET", path="/channels", query_params=["part", "id", "forHandle", "forUsername", "mine", "maxResults", "pageToken", "categoryId"]),
        "playlists.list":    _EndpointRoute(method="GET", path="/playlists", query_params=["part", "id", "channelId", "mine", "maxResults", "pageToken"]),
        "playlistItems.list": _EndpointRoute(method="GET", path="/playlistItems", query_params=["part", "playlistId", "id", "maxResults", "pageToken", "videoId"]),
        "comments.list":     _EndpointRoute(method="GET", path="/commentThreads", query_params=["part", "id", "videoId", "channelId", "allThreadsRelatedToChannelId", "maxResults", "pageToken", "order", "searchTerms", "textFormat"]),
        "captions.list":     _EndpointRoute(method="GET", path="/captions", query_params=["part", "videoId", "id", "onBehalfOf", "tlang"]),
    },

    "bigcommerce": {
        "products.list":    _EndpointRoute("GET",  "/stores/{store_hash}/v3/catalog/products", path_params=["store_hash"], query_params=["limit", "page"], response_key="data"),
        "orders.list":      _EndpointRoute("GET",  "/stores/{store_hash}/v2/orders", path_params=["store_hash"], query_params=["limit", "page"], response_key="data"),
        "customers.list":   _EndpointRoute("GET",  "/stores/{store_hash}/v3/customers", path_params=["store_hash"], query_params=["limit", "page"], response_key="data"),
        "categories.list":  _EndpointRoute("GET",  "/stores/{store_hash}/v3/catalog/categories", path_params=["store_hash"], query_params=["limit", "page"], response_key="data"),
        "brands.list":      _EndpointRoute("GET",  "/stores/{store_hash}/v3/catalog/brands", path_params=["store_hash"], query_params=["limit", "page"], response_key="data"),
        "storefront.info":  _EndpointRoute("GET",  "/stores/{store_hash}/v2/store", path_params=["store_hash"]),
    },
}


class GenericEndpointAdapter(_BaseHttpAdapter):
    """Universal adapter that uses endpoint route maps for ANY service.

    For services without a typed adapter class, this adapter translates
    action names into proper HTTP method + path + parameter placement
    using the _GENERIC_ENDPOINT_ROUTES table. This means ALL marketplace
    services can be called in adapter mode (with `action` parameter),
    not just the 38 with typed adapters.
    """

    service_id: str = ""

    def __init__(self, sid: str = ""):
        self.service_id = sid or self.service_id

    async def _execute_route(
        self,
        route: _EndpointRoute,
        params: dict[str, Any],
    ) -> tuple[Any, IntegrationError | None]:
        """Execute an endpoint route with parameter placement."""
        # Build path with path parameters interpolated
        path = route.path
        for pp in route.path_params:
            val = params.pop(pp, "")
            if not val:
                return f"Missing required path parameter '{pp}' for {self.service_id}", IntegrationError.VALIDATION
            path = path.replace(f"{{{pp}}}", str(val))

        # Separate query vs body params
        query_params = {}
        for qp in route.query_params:
            if qp in params:
                query_params[qp] = params.pop(qp)

        body = {}
        for bp in route.body_params:
            if bp in params:
                body[bp] = params.pop(bp)

        # Any remaining params go into body for POST or query for GET
        remaining = {k: v for k, v in params.items() if k not in ("service", "action", "timeout")}
        if route.method in ("GET", "DELETE"):
            query_params.update(remaining)
        else:
            body.update(remaining)

        # Make the request
        result, error = await self._request(
            method=route.method,
            path=path,
            body=body if body else None,
            query_params=query_params if query_params else None,
        )

        if error:
            return result, error

        # Extract data from response_key if specified
        if route.response_key and isinstance(result, dict):
            extracted = result.get(route.response_key, result)
            return extracted, None

        return result, None

    async def _call_action(self, action: str, params: dict[str, Any]) -> tuple[Any, IntegrationError | None]:
        """Dynamic action dispatch using endpoint routes."""
        routes = _GENERIC_ENDPOINT_ROUTES.get(self.service_id, {})
        route = routes.get(action)
        if not route:
            available = ", ".join(routes.keys()) if routes else "none (use raw HTTP mode)"
            return f"Unknown action '{action}' for '{self.service_id}'. Available: {available}", IntegrationError.VALIDATION
        return await self._execute_route(route, dict(params))


# --- Adapter Registry ---

# Build a GenericEndpointAdapter for each service that doesn't have a typed adapter
_GENERIC_ADAPTER_INSTANCES: dict[str, GenericEndpointAdapter] = {}
for _sid in _GENERIC_ENDPOINT_ROUTES:
    _GENERIC_ADAPTER_INSTANCES[_sid] = GenericEndpointAdapter(sid=_sid)

_ADAPTERS: dict[str, type[_BaseHttpAdapter]] = {
    "slack": SlackAdapter,
    "github": GitHubAdapter,
    "notion": NotionAdapter,
    "jira": JiraAdapter,
    "linear": LinearAdapter,
    "trello": TrelloAdapter,
    "grafana": GrafanaAdapter,
    "google_drive": GoogleDriveAdapter,
    "discord": DiscordAdapter,
    "posthog": PostHogAdapter,
    "amplitude": AmplitudeAdapter,
    "mixpanel": MixpanelAdapter,
    "hotjar": HotjarAdapter,
    "cloudflare": CloudflareAdapter,
    "kubernetes": KubernetesAdapter,
    "docker": DockerAdapter,
    "heap": HeapAdapter,
    "harbor": HarborAdapter,
    "portainer": PortainerAdapter,
    "terraform": TerraformAdapter,
    "route53": Route53Adapter,
    "godaddy": GoDaddyAdapter,
    "namecheap": NamecheapAdapter,
    "letsencrypt": LetsEncryptAdapter,
    "zerossl": ZeroSSLAdapter,
    "aws": AWSAdapter,
    "gcp": GCPAdapter,
    "azure": AzureAdapter,
    "vercel": VercelAdapter,
    "netlify": NetlifyAdapter,
    "supabase": SupabaseAdapter,
    "firebase": FirebaseAdapter,
    "dockerhub": DockerHubAdapter,
    "ghcr": GHCRAdapter,
    "appstore": AppStoreAdapter,
    "googleplay": GooglePlayAdapter,
    "sentry": SentryAdapter,
    "stripe": StripeAdapter,
}

_adapter_instances: dict[str, _BaseHttpAdapter] = {}


def _get_adapter(service_id: str) -> _BaseHttpAdapter | None:
    """Get or create an adapter instance for a service.

    Priority:
    1. Typed adapter (from _ADAPTERS) — has custom logic
    2. Generic adapter (from _GENERIC_ADAPTER_INSTANCES) — uses endpoint routes
    """
    adapter_cls = _ADAPTERS.get(service_id)
    if not adapter_cls:
        # v19: Fall back to generic adapter for services without typed adapters
        generic = _GENERIC_ADAPTER_INSTANCES.get(service_id)
        if generic:
            return generic
        return None
    if service_id not in _adapter_instances:
        _adapter_instances[service_id] = adapter_cls()
    return _adapter_instances[service_id]


# ---------------------------------------------------------------------------
# API Versioning Support
# ---------------------------------------------------------------------------

_SERVICE_API_VERSIONS: dict[str, dict[str, str]] = {
    "github": {
        "v2022-11-28": "2022-11-28",
        "v2022-08-16": "2022-08-16",
    },
    "notion": {
        "2022-06-28": "2022-06-28",
        "2021-08-16": "2021-08-16",
    },
    "discord": {
        "v10": "10",
        "v9": "9",
        "v8": "8",
    },
}


def _negotiate_api_version(service_id: str, preferred: str = "") -> str | None:
    """Negotiate the best available API version for a service.

    Args:
        service_id: The service identifier.
        preferred: The preferred version. If empty, uses the latest.

    Returns:
        The negotiated version string, or None if versioning is not supported.
    """
    versions = _SERVICE_API_VERSIONS.get(service_id)
    if not versions:
        return None

    if preferred and preferred in versions:
        return versions[preferred]

    # Return the latest version
    latest_key = next(iter(versions))
    return versions[latest_key]


def _build_versioned_headers(service_id: str, preferred_version: str = "") -> dict[str, str] | None:
    """Build version-specific headers for a service.

    Returns additional headers, or None if no versioning headers are needed.
    """
    version = _negotiate_api_version(service_id, preferred_version)
    if not version:
        return None

    extra: dict[str, str] = {}
    if service_id == "github":
        extra["X-GitHub-Api-Version"] = version
    elif service_id == "notion":
        extra["Notion-Version"] = version
    return extra


# ---------------------------------------------------------------------------
# Adapter Action Dispatch
# ---------------------------------------------------------------------------

_ADAPTER_ACTIONS: dict[str, dict[str, str]] = {
    "slack": {
        "channels.list": "channels_list",
        "messages.send": "messages_send",
        "messages.list": "messages_list",
        "users.list": "users_list",
    },
    "github": {
        "repos.list": "repos_list",
        "issues.list": "issues_list",
        "issues.create": "issues_create",
        "prs.list": "prs_list",
        "prs.create": "prs_create",
        "files.read": "files_read",
    },
    "notion": {
        "pages.list": "pages_list",
        "databases.query": "databases_query",
        "blocks.children": "blocks_children",
    },
    "jira": {
        "projects.list": "projects_list",
        "issues.search": "issues_search",
        "issues.create": "issues_create",
        "issues.transition": "issues_transition",
    },
    "linear": {
        "issues.list": "issues_list",
        "issues.create": "issues_create",
        "teams.list": "teams_list",
    },
    "trello": {
        "boards.list": "boards_list",
        "cards.list": "cards_list",
        "cards.create": "cards_create",
        "lists.list": "lists_list",
    },
    "grafana": {
        "dashboards.list": "dashboards_list",
        "annotations.create": "annotations_create",
        "datasource.query": "datasource_query",
    },
    "google_drive": {
        "files.list": "files_list",
        "files.upload": "files_upload",
        "files.read": "files_read",
    },
    "discord": {
        "channels.list": "channels_list",
        "messages.send": "messages_send",
        "guilds.list": "guilds_list",
    },
    "posthog": {
        "events.capture": "capture",
        "events.list": "events_list",
        "trends": "trends",
    },
    "amplitude": {
        "track": "track",
        "users.list": "users_list",
        "chart": "chart",
    },
    "mixpanel": {
        "track": "track",
        "engage": "engage",
        "export": "export",
    },
    "hotjar": {
        "sites.list": "sites_list",
        "heatmaps.list": "heatmaps_list",
        "recordings.list": "recordings_list",
        "feedback.list": "feedback_list",
    },
    "cloudflare": {
        "zones.list": "zones_list",
        "dns.list": "dns_records_list",
        "dns.create": "dns_record_create",
        "ssl.list": "ssl_certificates_list",
        "cache.purge": "purge_cache",
    },
    "kubernetes": {
        "namespaces.list": "namespaces_list",
        "pods.list": "pods_list",
        "services.list": "services_list",
        "deployments.list": "deployments_list",
        "pods.delete": "pods_delete",
    },
    "docker": {
        "containers.list": "containers_list",
        "images.list": "images_list",
        "container.create": "container_create",
        "container.start": "container_start",
        "container.stop": "container_stop",
        "image.pull": "image_pull",
    },
    "heap": {
        "users.list": "users_list",
        "events.list": "events_list",
        "sessions.list": "sessions_list",
    },
    "harbor": {
        "repositories.list": "repositories_list",
        "artifacts.list": "artifacts_list",
        "scan": "scan",
    },
    "portainer": {
        "containers.list": "containers_list",
        "images.list": "images_list",
        "volumes.list": "volumes_list",
        "networks.list": "networks_list",
        "stacks.list": "stacks_list",
    },
    "terraform": {
        "workspaces.list": "workspaces_list",
        "runs.list": "runs_list",
        "run.create": "run_create",
        "state.list": "state_versions_list",
    },
    "route53": {
        "zones.list": "hosted_zones_list",
        "records.list": "list_resource_record_sets",
        "records.change": "change_resource_record_sets",
    },
    "godaddy": {
        "domains.list": "domains_list",
        "dns.records": "dns_records",
        "dns.create": "dns_record_create",
    },
    "namecheap": {
        "domains.list": "domains_list",
        "dns.get": "dns_getHosts",
        "dns.set": "dns_setHost",
    },
    "letsencrypt": {
        "order.new": "new_order",
        "authz.new": "new_authz",
        "revoke": "revoke",
    },
    "zerossl": {
        "domains.list": "domains_list",
        "cert.create": "certificate_create",
        "cert.verify": "certificate_verify",
        "cert.download": "certificate_download",
    },
    "aws": {
        "ec2.list": "ec2_describe_instances",
        "s3.buckets": "s3_list_buckets",
        "s3.objects": "s3_list_objects",
        "lambda.list": "Lambda_list_functions",
        "cloudwatch.metrics": "CloudWatch_list_metrics",
    },
    "gcp": {
        "compute.list": "compute_instances_list",
        "functions.list": "functions_list",
        "run.list": "run_services_list",
        "storage.buckets": "storage_buckets_list",
    },
    "azure": {
        "resources.list": "resources_list",
        "webapps.list": "webapps_list",
        "containers.list": "containers_list",
        "functions.list": "functions_list",
    },
    "vercel": {
        "deployments.list": "deployments_list",
        "deployment.create": "deployment_create",
        "projects.list": "projects_list",
    },
    "netlify": {
        "sites.list": "sites_list",
        "deploys.list": "deploys_list",
        "deploy.trigger": "deploy_trigger",
    },
    "supabase": {
        "table.select": "table_select",
        "table.insert": "table_insert",
        "auth.users": "auth_users_list",
        "storage.buckets": "storage_buckets_list",
    },
    "firebase": {
        "projects.list": "projects_list",
        "auth.users": "auth_users_list",
        "firestore.collections": "firestore_collection_list",
        "hosting.list": "hosting_list",
    },
    "dockerhub": {
        "repositories.list": "repositories_list",
        "repository.tags": "repository_tags",
        "repository.manifests": "repository_manifests",
    },
    "ghcr": {
        "packages.list": "packages_list",
        "package.versions": "package_versions",
        "container.list": "container_list",
    },
    "appstore": {
        "apps.list": "apps_list",
        "builds.list": "builds_list",
        "pricingtiers": "app_pricingtiers",
    },
    "googleplay": {
        "apps.list": "apps_list",
        "builds.list": "builds_list",
        "releases.list": "releases_list",
    },
    "sentry": {
        "issues.list": "issues_list",
        "events.list": "events_list",
        "projects.list": "projects_list",
    },
    "stripe": {
        "customers.list": "customers_list",
        "charges.list": "charges_list",
        "payment.create": "payments_create",
    },
}


async def _dispatch_adapter_action(
    service_id: str,
    action: str,
    params: dict[str, Any],
) -> tuple[Any, IntegrationError | None]:
    """Dispatch an action to a service adapter.

    FIX: Now checks whether the integration is actually connected in the DB
    before attempting to call the adapter. If not connected, returns a
    clear, actionable error telling the agent to inform the user they need
    to connect the integration first — instead of an opaque auth error.

    Args:
        service_id: The integration service identifier.
        action: The action name (e.g. "channels.list").
        params: Parameters to pass to the adapter method.

    Returns:
        (response_data, error_or_none).
    """
    # FIX: Check connection status first, before even trying the adapter.
    # Previously, auth errors from unconnected integrations were opaque.
    # Now we give a clear "you need to connect this first" message.
    try:
        integration_row = await db.get_integration(service_id)
        if not integration_row or not integration_row.get("connected"):
            meta = _MARKETPLACE.get(service_id)
            service_name = meta.name if meta else service_id
            config_fields = []
            if meta and meta.config_fields:
                config_fields = [f["key"] for f in meta.config_fields]
            return (
                f"Integration '{service_name}' is not connected. "
                f"You need to connect it first before using it. "
                f"Required config: {', '.join(config_fields) if config_fields else 'API key or token'}. "
                f"Ask the user to provide their {service_name} credentials so you can connect the integration.",
                IntegrationError.NOT_CONNECTED,
            )
    except Exception:
        # DB lookup failed — proceed with adapter dispatch (may still work
        # if the adapter has its own connection logic)
        pass

    adapter = _get_adapter(service_id)
    if not adapter:
        # FIX: If no adapter exists (not even generic), suggest connecting
        meta = _MARKETPLACE.get(service_id)
        if meta:
            return (
                f"No adapter for '{service_id}' ({meta.name}). "
                f"Try using the raw HTTP mode with 'method' and 'path' parameters, "
                f"or connect the integration first with the required credentials: "
                f"{', '.join(f['key'] for f in meta.config_fields)}. ",
                IntegrationError.VALIDATION,
            )
        return f"No adapter for service '{service_id}'", IntegrationError.VALIDATION

    # v19: If using a GenericEndpointAdapter, dispatch through its route table
    if isinstance(adapter, GenericEndpointAdapter):
        try:
            result = await adapter._call_action(action, params)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            return result, None
        except TypeError as exc:
            return f"Invalid parameters for {service_id}.{action}: {exc}", IntegrationError.VALIDATION
        except Exception as exc:
            err_cat = _categorize_error(exc)
            return f"Adapter error ({type(exc).__name__}): {exc}", err_cat

    # Typed adapter path: look up method name from _ADAPTER_ACTIONS
    actions = _ADAPTER_ACTIONS.get(service_id, {})
    method_name = actions.get(action)
    if not method_name:
        return (
            f"Unknown action '{action}' for '{service_id}'. "
            f"Available: {', '.join(actions.keys())}",
            IntegrationError.VALIDATION,
        )

    method = getattr(adapter, method_name, None)
    if not method or not callable(method):
        return f"Adapter method '{method_name}' not found on {type(adapter).__name__}", IntegrationError.RUNTIME

    try:
        result = await method(**params)
        # Adapter methods already return (data, error_or_none) tuples,
        # so pass through directly instead of wrapping in another tuple.
        if isinstance(result, tuple) and len(result) == 2:
            return result
        # Fallback: if the method returns a bare value, wrap it
        return result, None
    except TypeError as exc:
        return f"Invalid parameters for {service_id}.{action}: {exc}", IntegrationError.VALIDATION
    except Exception as exc:
        err_cat = _categorize_error(exc)
        return f"Adapter error ({type(exc).__name__}): {exc}", err_cat


# ---------------------------------------------------------------------------
# Tool 1: call_integration_api
# ---------------------------------------------------------------------------

@tool(
    name="call_integration_api",
    description=(
        "Make an authenticated API call to any connected integration. "
        "Supports two calling modes:\n"
        "1. **Raw mode**: Provide `method`, `path`, and optional `body`/`query_params` "
        "to make a direct API call.\n"
        "2. **Adapter mode**: Provide `action` (e.g. 'channels.list') to use a typed "
        "service adapter with automatic parameter handling.\n\n"
        "ALL 75+ marketplace services support adapter mode with automatic endpoint routing. "
        "This includes: Slack, GitHub, Notion, Jira, Linear, Trello, Grafana, Google Drive, "
        "Discord, GitLab, Bitbucket, Twilio, SendGrid, Mailgun, Postmark, Telegram, MS Teams, "
        "Asana, ClickUp, Monday.com, Airtable, Confluence, Basecamp, HubSpot, Salesforce, "
        "Pipedrive, Zoho, Close.io, Stripe, PayPal, Lemon Squeezy, Datadog, New Relic, "
        "Pingdom, UptimeRobot, StatusPage, Zapier, Make, n8n, IFTTT, Dropbox, AWS S3, "
        "Backblaze, Wasabi, Hostinger, Railway, Render, Fly.io, Heroku, DigitalOcean, "
        "Cloudways, PlanetScale, Porkbun, Kubernetes, Docker, Harbor, Portainer, Terraform, "
        "Ansible, OpenAI, Anthropic, Cohere, Google AI, Hugging Face, Replicate, Stability AI, "
        "Shopify, WooCommerce, BigCommerce, and more.\n"
        "For GitHub use github_* tools; for Email use email_* tools.\n"
        "Supports GET, POST, PUT, PATCH, DELETE methods.\n"
        "Includes per-service rate limiting, auto-reconnect, and error categorization."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "description": "The integration service ID (e.g. 'slack', 'github', 'notion', 'jira', 'twilio', 'gitlab', 'hubspot', 'shopify', etc.)"
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "description": "HTTP method (raw mode)"
            },
            "path": {
                "type": "string",
                "description": "API path, e.g. '/conversations.list' for Slack, '/v1/databases' for Notion (raw mode)"
            },
            "action": {
                "type": "string",
                "description": "Typed adapter action, e.g. 'channels.list', 'messages.send', 'issues.create', 'repos.list' (adapter mode)"
            },
            "body": {
                "type": "object",
                "description": "Request body JSON for POST/PUT/PATCH (raw mode, or params for adapter mode)"
            },
            "query_params": {
                "type": "object",
                "description": "Query parameters as key-value pairs (raw mode)"
            },
            "timeout": {
                "type": "number",
                "description": "Request timeout in seconds (default: 30, max: 120)"
            },
        },
        "required": ["service"],
    },
    risk="medium",
    category="integrations",
)
async def call_integration_api(params: dict[str, Any]) -> str:
    """Execute an authenticated API call against a connected integration.

    Supports raw HTTP mode and typed adapter mode.
    Handles rate limiting, auto-reconnect, token refresh, and error categorization.
    """
    service_id = params.get("service", "")
    action = params.get("action", "")
    method = params.get("method", "GET").upper()
    path = params.get("path", "")
    body = params.get("body")
    query_params = params.get("query_params")
    timeout = min(float(params.get("timeout", _DEFAULT_TIMEOUT)), 120.0)

    if not service_id:
        return "Error: 'service' is required."

    # Adapter mode
    if action:
        adapter_params = body if isinstance(body, dict) else {}
        data, error = await _dispatch_adapter_action(service_id, action, adapter_params)
        if error:
            await _emit(
                "integration_call_failed",
                service_id=service_id,
                action=action,
                error=error.value,
                detail=str(data)[:200],
            )
            # Trigger self-improvement analysis on failures
            try:
                await db.log_improvement(
                    category="tool_failure",
                    tool_name=f"integration:{service_id}",
                    detail=f"Action '{action}' failed: {error.value} — {str(data)[:200]}",
                )
            except Exception:
                pass

            # FIX: For NOT_CONNECTED errors, return the data directly — it already
            # contains the user-friendly "you need to connect" message from
            # _dispatch_adapter_action. No need to wrap it further.
            if error == IntegrationError.NOT_CONNECTED:
                return str(data)
            return f"Error [{error.value}]: {data}"

        # Transform and return
        normalized = _transform_response(service_id, data)
        await _emit(
            "integration_call_succeeded",
            service_id=service_id,
            action=action,
            item_count=len(normalized.get("items", [])),
        )
        return _truncate_output(normalized)

    # Raw HTTP mode
    if not path:
        return "Error: Either 'action' or 'path' is required."

    # Rate limit pre-check
    should_wait, wait_seconds = RATE_LIMITER.should_throttle(service_id)
    if should_wait:
        await _emit(
            "integration_rate_limited",
            service_id=service_id,
            wait_seconds=wait_seconds,
        )
        return f"Rate limited for '{service_id}'. Please wait {wait_seconds:.1f}s before retrying."

    try:
        headers, base_url = await _build_auth_headers(service_id)
    except ValueError as exc:
        return str(exc)

    if not base_url:
        return f"Error: No base URL configured for '{service_id}'. Set base_url in integration config."

    # Token auto-refresh for OAuth services
    if service_id in _OAUTH_CONFIGS:
        try:
            row = await db.get_integration(service_id)
            if row:
                config = json.loads(row.get("config", "{}") or "{}")
                config = await _maybe_refresh_token(service_id, config)
        except Exception:
            pass

    # API versioning
    version_headers = _build_versioned_headers(service_id)
    if version_headers:
        headers.update(version_headers)

    url = f"{base_url}{path}"
    t0 = time.time()

    try:
        client = await _get_shared_client()
        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            json=body if body else None,
            params=query_params if query_params else None,
            timeout=timeout,
        )
        latency_ms = (time.time() - t0) * 1000

        # Update rate limiter from response headers
        RATE_LIMITER.apply_response_headers(service_id, dict(response.headers))
        RATE_LIMITER.record_request(service_id)

        # Handle rate limiting
        if response.status_code == 429:
            HUB.record_failure(service_id, IntegrationError.RATE_LIMITED, "HTTP 429")
            retry_after = response.headers.get("Retry-After", "unknown")
            await _emit(
                "integration_rate_limited",
                service_id=service_id,
                status_code=429,
                retry_after=retry_after,
            )
            return f"Rate limited by {service_id}. Retry-After: {retry_after}s."

        # Parse response
        try:
            data = response.json()
        except Exception:
            data = {"_status": response.status_code, "_raw": response.text[:_MAX_RESPONSE_CHARS]}

        # Handle service-specific error envelopes
        if isinstance(data, dict) and data.get("ok") is False:
            error_msg = data.get("error", "Unknown error")
            HUB.record_failure(service_id, IntegrationError.RUNTIME, error_msg)
            await _emit(
                "integration_call_failed",
                service_id=service_id,
                path=path,
                error="api_error",
                detail=error_msg,
            )
            return f"API error: {error_msg}"

        HUB.record_success(service_id, latency_ms)

        # Transform the response
        normalized = _transform_response(service_id, data)
        await _emit(
            "integration_call_succeeded",
            service_id=service_id,
            path=path,
            method=method,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 1),
        )
        return _truncate_output(normalized)

    except httpx.TimeoutException:
        HUB.record_failure(service_id, IntegrationError.TIMEOUT, f"{timeout}s")
        await _emit(
            "integration_call_failed",
            service_id=service_id,
            path=path,
            error="timeout",
            detail=f"timeout_after_{timeout}s",
        )
        return f"Request timed out calling {service_id} at {url} (timeout: {timeout}s)"
    except httpx.ConnectError as exc:
        HUB.record_failure(service_id, IntegrationError.NETWORK, str(exc)[:200])
        await _emit(
            "integration_call_failed",
            service_id=service_id,
            path=path,
            error="network",
            detail=str(exc)[:200],
        )
        return f"Connection error to {service_id}: {exc}"
    except Exception as exc:
        err_cat = _categorize_error(exc)
        HUB.record_failure(service_id, err_cat, str(exc)[:200])
        await _emit(
            "integration_call_failed",
            service_id=service_id,
            path=path,
            error=err_cat.value,
            detail=str(exc)[:200],
        )
        try:
            await db.log_improvement(
                category="tool_failure",
                tool_name=f"integration:{service_id}",
                detail=f"Raw API call failed: {err_cat.value} — {str(exc)[:200]}",
            )
        except Exception:
            pass
        return f"API call error [{err_cat.value}]: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Tool 2: list_connected_integrations
# ---------------------------------------------------------------------------

@tool(
    name="list_connected_integrations",
    description=(
        "List all connected integrations with their IDs, descriptions, and health status. "
        "Use this to discover which integrations are available before calling call_integration_api. "
        "Shows connection health, rate limit status, and available adapter actions."
    ),
    parameters_schema={
        "type": "object",
        "properties": {},
    },
    category="integrations",
)
async def list_connected_integrations(params: dict[str, Any]) -> str:
    """List all connected integrations with health and rate limit info."""
    integrations = await db.list_connected_integrations(connected_only=True)
    if not integrations:
        return (
            "No integrations are currently connected.\n\n"
            "Use `list_available_integrations` to see what can be connected, "
            "or `configure_integration` to set up a new connection.\n"
            "Connect integrations in the Integrations panel of the UI."
        )

    lines = [f"## Connected Integrations ({len(integrations)})\n"]

    for integration in integrations:
        iid = integration.get("id", "")
        iname = integration.get("name", "")
        idesc = integration.get("description", "")
        icat = integration.get("category", "")

        # Get health info
        health_info = await HUB.get_health_for_service(iid)
        rate_info = RATE_LIMITER.get_status(iid)

        # Get available actions
        actions = _ADAPTER_ACTIONS.get(iid, {})
        action_str = ", ".join(actions.keys()) if actions else "raw HTTP only"

        status_emoji = "✅" if health_info.get("healthy", True) else "❌"
        lines.append(f"### {status_emoji} **{iname}** (ID: `{iid}`)")
        lines.append(f"  - Category: {icat}")
        lines.append(f"  - Description: {idesc}")
        lines.append(f"  - Health: {'healthy' if health_info.get('healthy') else 'degraded — ' + health_info.get('reason', '')}")
        lines.append(f"  - Avg Latency: {health_info.get('avg_latency_ms', 0):.0f}ms")
        lines.append(f"  - Successes: {health_info.get('total_successes', 0)} | Failures: {health_info.get('total_failures', 0)}")
        lines.append(f"  - Requests (60s): {rate_info.get('requests_in_last_60s', 0)}")
        if rate_info.get("remaining") is not None:
            lines.append(f"  - Rate Limit: {rate_info['remaining']} remaining")
        lines.append(f"  - Actions: {action_str}")
        lines.append("")

    lines.append("---\n")
    lines.append("Use `call_integration_api` with `service` and `action` (or `method`+`path`) to interact.")
    lines.append("For GitHub, prefer the dedicated `github_*` tools. For Email, use `email_*` tools.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: list_available_integrations
# ---------------------------------------------------------------------------

@tool(
    name="list_available_integrations",
    description=(
        "Browse the integration marketplace — shows all integrations the agent can connect to. "
        "Includes capabilities, auth type, rate limits, pricing tier, and setup instructions. "
        "Use this to discover new services to connect."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Filter by category (e.g., 'Communication', 'Developer Tools', 'Project Management', 'Productivity', 'Monitoring', 'Cloud Storage')"
            },
        },
    },
    category="integrations",
)
async def list_available_integrations(params: dict[str, Any]) -> str:
    """List all available integrations from the marketplace."""
    category_filter = params.get("category", "").strip().lower()

    services = sorted(_MARKETPLACE.values(), key=lambda s: (s.category, s.name))
    if category_filter:
        services = [s for s in services if s.category.lower() == category_filter]

    if not services:
        if category_filter:
            available = sorted({s.category for s in _MARKETPLACE.values()})
            return f"No integrations found in category '{category_filter}'.\nAvailable categories: {', '.join(available)}"
        return "No integrations available in the marketplace."

    # Get already-connected IDs
    connected_ids: set[str] = set()
    try:
        connected = await db.list_connected_integrations(connected_only=True)
        connected_ids = {c.get("id", "") for c in connected}
    except Exception:
        pass

    lines = [f"## Integration Marketplace ({len(services)} services)\n"]

    current_category = ""
    for svc in services:
        if svc.category != current_category:
            current_category = svc.category
            lines.append(f"### {current_category}\n")

        is_connected = svc.service_id in connected_ids
        conn_str = "✅ Connected" if is_connected else "⚪ Not connected"

        lines.append(f"#### {svc.name} `{svc.service_id}` — {conn_str}")
        lines.append(f"  - {svc.description}")
        lines.append(f"  - Auth: {svc.auth_type} | Rate limit: {svc.rate_limit_rpm} req/min")
        lines.append(f"  - Capabilities: {', '.join(svc.capabilities)}")
        if svc.config_fields:
            fields_desc = ", ".join(
                f"{f['label']} ({'secret' if f['secret'] else 'visible'})"
                for f in svc.config_fields
            )
            lines.append(f"  - Setup fields: {fields_desc}")
        lines.append(f"  - Docs: {svc.docs_url}")
        lines.append("")

    lines.append("---\n")
    lines.append("To connect a service, use `configure_integration` with the service ID.")
    lines.append("To use a connected service, use `call_integration_api` with the service ID and action.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: integration_health_check
# ---------------------------------------------------------------------------

@tool(
    name="integration_health_check",
    description=(
        "Check the health of one or all integrations by making a lightweight API ping. "
        "Returns health status, latency, error details, and rate limit state. "
        "Use this to diagnose connectivity issues before making API calls."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "description": "Specific service to check (e.g., 'slack', 'notion'). If omitted, checks all connected integrations."
            },
        },
    },
    category="integrations",
)
async def integration_health_check(params: dict[str, Any]) -> str:
    """Perform health check on integrations."""
    service_id = params.get("service", "").strip()

    if service_id:
        result = await HUB.health_check(service_id)
        health = result.get("healthy", False)
        emoji = "✅" if health else "❌"
        lines = [f"## Health Check: {emoji} `{service_id}`\n"]
        lines.append(f"  - Connected: {'Yes' if result.get('connected') else 'No'}")
        lines.append(f"  - Healthy: {'Yes' if health else 'No'}")
        if not health:
            lines.append(f"  - Reason: {result.get('reason', 'unknown')}")
        lines.append(f"  - Successes: {result.get('total_successes', 0)}")
        lines.append(f"  - Failures: {result.get('total_failures', 0)}")
        lines.append(f"  - Avg Latency: {result.get('avg_latency_ms', 0):.1f}ms")
        if result.get("last_success"):
            lines.append(f"  - Last Success: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(result['last_success']))}")
        if result.get("last_failure"):
            lines.append(f"  - Last Failure: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(result['last_failure']))}")
        rate = result.get("rate_limit", {})
        lines.append(f"  - Requests (60s): {rate.get('requests_in_last_60s', 0)}")
        if rate.get("remaining") is not None:
            lines.append(f"  - Rate Limit Remaining: {rate['remaining']}")
        if rate.get("throttled"):
            lines.append(f"  - ⚠️ Currently throttled until: {time.strftime('%H:%M:%S', time.gmtime(rate['throttled_until']))}")
        return "\n".join(lines)
    else:
        results = await HUB.health_check_all()
        if not results:
            return "No connected integrations to check."

        healthy = sum(1 for r in results if r.get("healthy"))
        unhealthy = len(results) - healthy
        lines = [f"## Integration Health Summary: {healthy} healthy, {unhealthy} unhealthy\n"]

        for result in results:
            sid = result.get("service_id", "?")
            h = result.get("healthy", False)
            emoji = "✅" if h else "❌"
            reason = ""
            if not h:
                reason = f" — {result.get('reason', 'unknown')}"
            avg_lat = result.get("avg_latency_ms", 0)
            lines.append(f"  {emoji} `{sid}`: {avg_lat:.0f}ms avg latency{reason}")

        lines.append(f"\n---\nChecked {len(results)} integration(s) at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: configure_integration
# ---------------------------------------------------------------------------

@tool(
    name="configure_integration",
    description=(
        "Set up a new integration connection or update an existing one. "
        "Provide the service ID and the required configuration fields (API key, tokens, URLs, etc.). "
        "Use `list_available_integrations` to see what fields each service needs."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "description": "The integration service ID (e.g., 'slack', 'notion', 'jira')"
            },
            "config": {
                "type": "object",
                "description": "Configuration object with auth credentials and settings. Required fields vary by service."
            },
        },
        "required": ["service", "config"],
    },
    risk="medium",
    category="integrations",
)
async def configure_integration(params: dict[str, Any]) -> str:
    """Configure (create or update) an integration connection."""
    service_id = params.get("service", "").strip()
    config_obj = params.get("config")

    if not service_id:
        return "Error: 'service' is required."
    if not config_obj or not isinstance(config_obj, dict):
        return "Error: 'config' must be a JSON object with the required fields."

    # Validate service exists in marketplace
    meta = _MARKETPLACE.get(service_id)
    if not meta:
        return f"Error: Unknown service '{service_id}'. Use `list_available_integrations` to see available services."

    # Check for required config fields
    missing_fields = []
    for field_def in meta.config_fields:
        key = field_def["key"]
        if field_def.get("required", True) and key not in config_obj:
            missing_fields.append(key)
    # v22 FIX: Warn about missing required fields but don't block
    # (user might have partial config or credentials from environment)
    missing_warning = ""
    if missing_fields:
        missing_warning = f"\n⚠️ Warning: Missing recommended fields: {', '.join(missing_fields)}. The integration may not work until these are provided."

    # Build the full config
    full_config: dict[str, Any] = dict(config_obj)
    full_config.setdefault("auth_type", meta.auth_type)
    full_config.setdefault("base_url", meta.base_url)

    # Check if already connected
    existing = None
    try:
        existing = await db.get_integration(service_id)
    except Exception:
        pass

    if existing:
        # Merge with existing config
        try:
            existing_config = json.loads(existing.get("config", "{}") or "{}")
            existing_config.update(full_config)
            full_config = existing_config
        except Exception:
            pass
        try:
            await db.update_integration_config(service_id, full_config)
        except Exception as exc:
            return f"Error updating integration '{service_id}': {exc}"
        await _emit("integration_configured", service_id=service_id, updated=True)
        return f"✅ Updated existing integration '{service_id}' ({meta.name}).{missing_warning}\n\nIt is now ready to use with `call_integration_api`."
    else:
        # Create new integration
        try:
            await db.add_integration(
                id=service_id,
                name=meta.name,
                category=meta.category,
                description=meta.description,
                config=json.dumps(full_config),
                connected=1,
            )
        except Exception as exc:
            return f"Error creating integration '{service_id}': {exc}"
        await _emit("integration_configured", service_id=service_id, updated=False)
        return (
            f"✅ Connected new integration '{service_id}' ({meta.name}).{missing_warning}\n\n"
            f"Use `call_integration_api` with service='{service_id}' and action='{meta.capabilities[0]}' to get started.\n"
            f"Available actions: {', '.join(meta.capabilities)}"
        )


# ---------------------------------------------------------------------------
# Tool 6: integration_webhook_info
# ---------------------------------------------------------------------------

@tool(
    name="integration_webhook_info",
    description=(
        "Get webhook configuration information for a connected integration. "
        "Shows the webhook URL, setup instructions, and currently registered webhooks. "
        "Use this to set up event-driven integrations (e.g., receive Slack events, GitHub webhooks)."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "description": "The integration service ID (e.g., 'slack', 'github')"
            },
            "action": {
                "type": "string",
                "enum": ["info", "register", "list"],
                "description": "Action: 'info' (get setup info), 'register' (register a new webhook), 'list' (list registered webhooks)"
            },
            "webhook_id": {
                "type": "string",
                "description": "Webhook ID for 'register' action"
            },
            "secret": {
                "type": "string",
                "description": "HMAC signing secret for 'register' action"
            },
            "event_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Event types to listen for (e.g., ['push', 'pull_request'])"
            },
        },
        "required": ["service"],
    },
    risk="medium",
    category="integrations",
)
async def integration_webhook_info(params: dict[str, Any]) -> str:
    """Get or manage webhook configuration for an integration."""
    service_id = params.get("service", "").strip()
    action = params.get("action", "info").strip().lower()

    if not service_id:
        return "Error: 'service' is required."

    if action == "info":
        # Provide setup instructions
        webhook_base = "/api/integrations/webhook"
        lines = [f"## Webhook Setup for `{service_id}`\n"]
        lines.append(f"**Webhook endpoint**: `{webhook_base}/{service_id}/{{webhook_id}}`")
        lines.append("")

        meta = _MARKETPLACE.get(service_id)
        if meta:
            lines.append(f"**Service**: {meta.name} ({meta.category})")
            lines.append(f"**Docs**: {meta.docs_url}")
            lines.append("")

        lines.append("### Setup Steps:")
        lines.append(f"1. Choose a unique `webhook_id` (e.g., 'my-slack-events')")
        lines.append(f"2. Generate a signing secret (use a cryptographically secure random string)")
        lines.append(f"3. Configure the webhook in {service_id}'s dashboard, pointing to:")
        lines.append(f"   `{webhook_base}/{service_id}/{{webhook_id}}`")
        lines.append(f"4. Use the `register` action with the webhook_id and secret")
        lines.append(f"5. Incoming events will be stored as pending tasks for the agent")
        lines.append("")

        lines.append("### Security:")
        lines.append("- All webhooks are verified using HMAC-SHA256 signature validation")
        lines.append("- The signing secret is used to verify that events come from the real service")
        lines.append("- Events are stored in the database as pending tasks")
        lines.append("")

        lines.append("### Signature Formats Supported:")
        lines.append("- `sha256=<hex>` (GitHub, generic)")
        lines.append("- `t=<timestamp>,v1=<hex>` (Slack signing)")
        lines.append("- Raw HMAC-SHA256 hex digest")
        lines.append("")

        lines.append("To register a webhook, use action='register' with webhook_id and secret.")

        return "\n".join(lines)

    elif action == "register":
        webhook_id = params.get("webhook_id", "").strip()
        secret = params.get("secret", "").strip()
        event_types = params.get("event_types")

        if not webhook_id:
            return "Error: 'webhook_id' is required for register action."
        if not secret:
            return "Error: 'secret' is required for register action."

        config = WEBHOOK_MANAGER.register(
            service_id=service_id,
            webhook_id=webhook_id,
            secret=secret,
            event_types=event_types if isinstance(event_types, list) else [],
        )

        await _emit(
            "integration_webhook_registered",
            service_id=service_id,
            webhook_id=webhook_id,
            event_types=event_types,
        )

        webhook_url = f"/api/integrations/webhook/{service_id}/{webhook_id}"
        lines = [f"✅ Webhook registered successfully!\n"]
        lines.append(f"  - Webhook ID: `{webhook_id}`")
        lines.append(f"  - Service: `{service_id}`")
        lines.append(f"  - Endpoint: `{webhook_url}`")
        if event_types:
            lines.append(f"  - Event types: {', '.join(str(e) for e in event_types)}")
        lines.append("")
        lines.append("Configure your service to send events to this endpoint.")
        lines.append("Incoming events will be verified and stored as pending tasks.")

        return "\n".join(lines)

    elif action == "list":
        webhooks = WEBHOOK_MANAGER.list_webhooks(service_id)
        if not webhooks:
            return f"No webhooks registered for '{service_id}'. Use action='register' to add one."

        lines = [f"## Registered Webhooks for `{service_id}` ({len(webhooks)})\n"]
        for wh in webhooks:
            lines.append(f"  - **{wh['webhook_id']}**")
            lines.append(f"    - Events: {', '.join(wh['event_types']) if wh['event_types'] else 'all'}")
            lines.append(f"    - Enabled: {'Yes' if wh['enabled'] else 'No'}")
            lines.append(f"    - Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(wh['created_at']))}")
        return "\n".join(lines)

    else:
        return f"Error: Unknown action '{action}'. Use 'info', 'register', or 'list'."


# ---------------------------------------------------------------------------
# System Prompt Context Builder
# ---------------------------------------------------------------------------

async def get_integration_context_for_prompt() -> str:
    """Build a compact summary of connected integrations for the system prompt.

    Includes rate limit status and available actions so the agent can make
    informed decisions about which integrations to use.
    """
    try:
        integrations = await db.list_connected_integrations(connected_only=True)
    except Exception:
        return ""

    if not integrations:
        return ""

    lines = ["## Connected Integrations"]
    for integration in integrations:
        sid = integration.get("id", "")
        iname = integration.get("name", "")
        health = await HUB.get_health_for_service(sid)
        rate = RATE_LIMITER.get_status(sid)

        # v19: Check typed adapter actions first, then generic endpoint routes
        actions = _ADAPTER_ACTIONS.get(sid, {})
        if not actions:
            generic_routes = _GENERIC_ENDPOINT_ROUTES.get(sid, {})
            actions = {action: route.path for action, route in generic_routes.items()}

        healthy_str = "healthy" if health.get("healthy") else f"degraded({health.get('reason', '?')})"
        rpm = rate.get("requests_in_last_60s", 0)

        line = f"- {iname}({sid}): {healthy_str}, {rpm}req/min"
        if actions:
            line += f", actions: {', '.join(actions.keys())}"
        lines.append(line)

    return "\n".join(lines)
