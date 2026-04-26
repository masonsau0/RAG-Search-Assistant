"""Generate the demo company-knowledge-base corpus.

Writes ~35 short policy/FAQ-style documents into `corpus/` so the RAG
engine has something realistic to retrieve over without depending on any
external dataset. Each doc is one `.txt` file: first line is the
headline, the rest is body text.

Themes covered: HR & benefits, engineering practices, security &
incident response, onboarding, office logistics, travel & expenses,
internal tools.
"""

from __future__ import annotations

from pathlib import Path

DOCS: list[tuple[str, str, str]] = [
    # ------------------------------------------------------------- HR / time off
    ("hr_001", "Paid Time Off (PTO) Policy",
     "Full-time employees accrue 1.67 days of PTO per month, totalling 20 days "
     "per year. PTO requests must be submitted in the HRIS (Workday) at least "
     "2 weeks in advance for vacations longer than 3 days. Manager approval is "
     "required. Unused PTO carries over up to a maximum of 10 days into the "
     "next calendar year; anything beyond is forfeited on January 1. PTO is "
     "paid out at the prorated salary rate upon resignation."),

    ("hr_002", "Parental Leave Policy",
     "All employees who have completed 12 months of service are eligible for "
     "16 weeks of paid parental leave at 100% of base salary, regardless of "
     "gender or how the child enters the family (birth, adoption, surrogacy). "
     "Leave can be taken in up to two blocks within the first 18 months. "
     "Notify your manager and HR at least 60 days before the planned start "
     "date when possible. Health benefits continue uninterrupted during leave."),

    ("hr_003", "Sick Leave",
     "Sick leave is unlimited and paid. Employees are expected to use it "
     "responsibly. For absences longer than 3 consecutive working days, a "
     "doctor's note must be submitted to HR. Mental health days fall under "
     "the same policy. Notify your manager by 9 AM on the day of absence "
     "via Slack or email."),

    ("hr_004", "Remote Work Policy",
     "The default arrangement is hybrid: 3 days in-office (Tuesday, Wednesday, "
     "Thursday) and 2 days remote. Fully-remote arrangements are available on "
     "a case-by-case basis with VP approval. Cross-border remote work requires "
     "Legal review and may have tax implications. International remote work "
     "for more than 30 days/year requires a formal agreement."),

    ("hr_005", "Performance Reviews",
     "Performance reviews happen twice a year — in April and October. Each "
     "review includes a self-assessment, manager review, and peer feedback "
     "from at least 3 colleagues. Compensation adjustments are decided in the "
     "April cycle. The performance rating scale is: Below Expectations, "
     "Meets Expectations, Exceeds Expectations, and Outstanding. Promotion "
     "nominations are submitted by managers in early March."),

    ("hr_006", "Internal Transfers",
     "Employees in good standing with at least 12 months in their current "
     "role may apply for internal transfers. Apply through the internal job "
     "board; your current manager will be notified automatically. Standard "
     "transition is 4 weeks but can extend to 8 weeks for critical roles. "
     "Compensation is benchmarked to the new role's band."),

    # ------------------------------------------------------------- Benefits
    ("ben_001", "Health Insurance & Benefits Start Date",
     "Health, dental, and vision insurance start on the first day of "
     "employment — no waiting period. Coverage extends to spouse, "
     "common-law partner, and dependent children. The company covers 100% "
     "of the employee premium and 80% of dependent premiums. Open enrolment "
     "for the following year happens each November."),

    ("ben_002", "Retirement Plan (RRSP / 401k)",
     "Eligible employees can contribute to the company-matched retirement "
     "plan starting on day 1. The company matches 100% of the first 5% of "
     "salary contributed. Vesting is immediate. Contribution changes can be "
     "made monthly through the benefits portal."),

    ("ben_003", "Equity & Stock Options",
     "All full-time employees receive a stock option grant at hire, with a "
     "4-year vesting schedule and a 1-year cliff. Annual refresh grants are "
     "performance-based and granted in the April review cycle. The exercise "
     "window after departure is 90 days unless the employee has 2+ years of "
     "tenure, in which case it extends to 7 years."),

    ("ben_004", "Wellness Stipend",
     "Each employee receives $1,200 per year for wellness expenses: gym "
     "memberships, fitness classes, mental health apps, ergonomic equipment "
     "for home office, and similar. Submit receipts in Expensify under the "
     "Wellness category. Funds reset each January 1 and do not roll over."),

    ("ben_005", "Learning & Development Budget",
     "Each employee has $2,500 per year for professional development: "
     "courses, conferences, certifications, books. Pre-approval from your "
     "manager is required for items above $500. Time off for conferences "
     "(up to 5 days/year) does not count against PTO."),

    # ------------------------------------------------------------- Engineering
    ("eng_001", "Code Review Process",
     "All non-trivial changes go through pull request review. At minimum "
     "one reviewer from the relevant team must approve. For changes "
     "touching shared infrastructure, two reviewers are required, with at "
     "least one from the platform team. PRs should be under 400 lines "
     "where practical; larger PRs need a design doc linked. Reviewers are "
     "expected to respond within 1 business day."),

    ("eng_002", "Deployment Process",
     "Deployments happen via the internal CI/CD pipeline. Every merge to "
     "`main` automatically deploys to staging. Production deployments are "
     "gated by a manual approval and a green canary check (5% traffic for "
     "30 minutes). Friday after 3 PM and during code-freeze periods (last "
     "week of December, week before major launches) are blocked."),

    ("eng_003", "On-Call Rotation",
     "Engineers in the on-call rotation are paged via PagerDuty. Each "
     "rotation is 1 week, Monday morning to Monday morning, with primary "
     "and secondary on-call. SLA: ack within 5 minutes for SEV-1, 15 "
     "minutes for SEV-2. On-call gets a $400 weekly stipend plus comp "
     "time. New engineers shadow at least 2 rotations before going live."),

    ("eng_004", "Incident Response",
     "When a production incident is detected, the on-call engineer "
     "creates an incident channel in Slack named `#inc-YYYYMMDD-short-desc`. "
     "Severity is assigned (SEV-1 / SEV-2 / SEV-3). For SEV-1, the "
     "engineering manager and a comms lead must be paged. After resolution, "
     "a blameless post-mortem is written within 5 business days and shared "
     "company-wide."),

    ("eng_005", "Testing Standards",
     "All new code requires tests. Unit tests for business logic, "
     "integration tests for API endpoints. Coverage target is 80% on "
     "changed files (enforced in CI). End-to-end tests are owned by QA "
     "and run on every release-candidate build. Flaky tests are quarantined "
     "within 48 hours and fixed or removed within 1 week."),

    ("eng_006", "Monitoring & Observability",
     "Production services emit metrics to Prometheus (scraped every 15s), "
     "logs to Loki, and traces to Tempo. Dashboards live in Grafana. "
     "Standard SLOs: 99.9% availability and p95 latency under 300ms for "
     "user-facing APIs. Alerts route to the relevant on-call rotation via "
     "PagerDuty. Runbooks are linked from each alert."),

    ("eng_007", "Source Control & Branching",
     "We use trunk-based development. Feature branches branch from `main`, "
     "stay alive less than 1 week, and merge back via squash-and-merge. "
     "Long-lived feature branches are discouraged — use feature flags "
     "instead. Commit messages follow the Conventional Commits format "
     "(feat:, fix:, chore:, docs:)."),

    # ------------------------------------------------------------- Security
    ("sec_001", "Password Policy",
     "Passwords must be at least 16 characters and unique per service. "
     "Use the company password manager (1Password) — sharing credentials "
     "via Slack, email, or chat is prohibited. Passwords are rotated only "
     "when there is reason to believe they are compromised; routine "
     "rotation is no longer required by policy."),

    ("sec_002", "Multi-Factor Authentication (MFA)",
     "MFA is mandatory for all corporate accounts: email, SSO, code "
     "repositories, cloud consoles, and the password manager. Hardware "
     "keys (YubiKey) are issued to engineers and security-sensitive roles "
     "and are the required method for accessing production systems. "
     "TOTP via authenticator apps is acceptable for non-production access."),

    ("sec_003", "Data Classification",
     "Data is classified as Public, Internal, Confidential, or Restricted. "
     "Customer PII is Restricted and must only be accessed via the "
     "approved data-access tooling, which logs every query. Restricted "
     "data may not be exported, screenshotted, or stored on local disks. "
     "Confidential data (internal metrics, business plans) requires a "
     "VPN connection to access."),

    ("sec_004", "Reporting Security Incidents",
     "If you suspect a security incident — phishing, lost device, "
     "compromised credentials, suspicious activity — report immediately "
     "to security@company.com or via the Slack channel `#security-incidents`. "
     "Do not investigate on your own. The Security team will triage within "
     "30 minutes and coordinate response. Reports are non-punitive."),

    ("sec_005", "VPN & Network Access",
     "Engineering access to production systems requires the corporate "
     "VPN (Tailscale) plus a hardware key. The VPN is also required for "
     "accessing internal staging environments and certain SaaS tools "
     "(billing, customer data). Public Wi-Fi without VPN is allowed only "
     "for browsing public web; never for company work."),

    ("sec_006", "Acceptable Use of AI Tools",
     "Approved AI tools (corporate ChatGPT Enterprise, GitHub Copilot, "
     "internal LLM gateway) may be used for company work. Do not paste "
     "Confidential or Restricted data into consumer AI tools. All AI-"
     "generated code must be reviewed and tested before merging — the "
     "submitting engineer is responsible for correctness regardless of "
     "the source."),

    # ------------------------------------------------------------- Onboarding
    ("onb_001", "First Day Checklist",
     "Your manager will meet you at 9:30 AM in the lobby. By end of day "
     "1 you should have: laptop set up, SSO logins working, calendar "
     "connected, Slack joined, business cards ordered. IT runs an "
     "onboarding clinic at 10 AM in the Atrium for any setup issues. "
     "Don't worry about productivity in the first week — focus on meeting "
     "your team and reading the onboarding wiki."),

    ("onb_002", "IT Help on Day 1",
     "For any IT or laptop issues during onboarding, contact the IT desk "
     "via Slack (`#it-help`) or email it@company.com. The IT walk-up "
     "clinic is in Room 4-12 from 9 AM to 5 PM Monday through Friday. "
     "After hours, file a ticket in ServiceNow — non-urgent tickets are "
     "answered within 1 business day."),

    ("onb_003", "Equipment Setup",
     "New employees receive: laptop (M-series MacBook Pro by default; "
     "Linux/Windows by request), external monitor, keyboard, mouse, "
     "headset, and a YubiKey hardware-security key. Home-office stipend "
     "of $500 is available in the first 90 days for additional ergonomic "
     "items (chair, desk lamp, etc.) — submit receipts via Expensify."),

    ("onb_004", "Onboarding Buddy Program",
     "Every new hire is paired with an onboarding buddy from a different "
     "team for the first 90 days. The buddy meets you weekly for a 30-min "
     "coffee chat and is a no-judgment go-to for any 'is this normal?' "
     "questions. Buddies are not a replacement for your manager — they're "
     "a sounding board on culture and process."),

    # ------------------------------------------------------------- Office
    ("off_001", "Office Hours & Access",
     "Offices are accessible from 7 AM to 9 PM on weekdays via badge. "
     "After hours and weekend access requires badge activation through "
     "the facilities portal. The office is closed on statutory holidays "
     "and during the company-wide year-end shutdown (Dec 24 - Jan 2)."),

    ("off_002", "Parking & Transit",
     "On-site parking is first-come, first-served. Underground parking "
     "is reserved for visitors and accessibility needs. The company "
     "subsidises monthly transit passes (50% reimbursement up to $150 / "
     "month). Bike storage is available on the ground floor with shower "
     "access on the 2nd floor."),

    ("off_003", "Meals & Kitchen",
     "Lunch is provided in-office on Tuesday, Wednesday, and Thursday "
     "(catered, 12 PM - 1:30 PM). Snacks, coffee, and drinks are stocked "
     "all day in the kitchens. Dietary preferences (vegetarian, vegan, "
     "gluten-free, halal, kosher) are accommodated; flag your needs in "
     "the onboarding form so they're available from day 1."),

    ("off_004", "Dress Code",
     "Smart casual is the default. No suits required. T-shirts and jeans "
     "are fine; visible hygiene matters more than formality. For client "
     "meetings or external events, business casual is expected. Dress "
     "codes can be relaxed further on Fridays and during heat waves."),

    ("off_005", "Visitor Policy",
     "Visitors must be pre-registered in the visitor portal at least 24 "
     "hours in advance. They check in at reception, sign an NDA if they "
     "will see Confidential material, and are escorted at all times by "
     "their host. Family-and-friends visits are welcome but follow the "
     "same registration process."),

    # ------------------------------------------------------------- Travel
    ("trv_001", "Business Travel & Booking",
     "All business travel is booked through Navan (formerly TripActions). "
     "Approved class: economy for flights under 6 hours, premium economy "
     "for 6-10 hours, business for 10+ hours. Book at least 14 days "
     "ahead when possible. Loyalty programs are personal — keep your "
     "miles. Ground transport: Uber/Lyft, transit, or rental car as "
     "needed."),

    ("trv_002", "Expense Reports",
     "Expenses are submitted in Expensify within 30 days. Receipts are "
     "required for any item above $25. Per diem for meals while travelling: "
     "$75 / day domestic, $100 / day international. Reimbursement runs "
     "weekly; expect funds within 5-7 business days of approval."),

    # ------------------------------------------------------------- Internal tools
    ("tools_001", "Slack Etiquette",
     "Public channels are preferred — DMs are for personal or sensitive "
     "topics. Use threads for replies to keep main channels readable. "
     "Set your status to indicate availability (focus, lunch, OOO). For "
     "urgent issues outside business hours, page via PagerDuty rather "
     "than Slack. Do not @channel unless it's actionable for everyone."),

    ("tools_002", "Jira Workflow",
     "Engineering work is tracked in Jira. Issue types: Story, Bug, Task, "
     "Spike. Stories require acceptance criteria and a story-point estimate. "
     "Sprint planning is biweekly on Monday mornings; standup is daily at "
     "10 AM. Roadmap items live in epics and roll up to quarterly "
     "objectives in the OKRs board."),
]


def main():
    out_dir = Path("corpus")
    out_dir.mkdir(exist_ok=True)
    for docno, headline, body in DOCS:
        path = out_dir / f"{docno}.txt"
        path.write_text(f"{headline}\n\n{body}\n", encoding="utf-8")
    print(f"Wrote {len(DOCS)} documents to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
