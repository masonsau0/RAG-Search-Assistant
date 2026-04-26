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

    # ------------------------------------------------------------- Everyday — cooking
    ("life_001", "How to Cook Pasta",
     "Bring a large pot of water to a rolling boil. Add about 1 tablespoon "
     "of salt per litre of water. Add the pasta and stir to prevent sticking. "
     "Cook according to the package time, usually 8-12 minutes for dry pasta. "
     "Test by tasting — pasta should be al dente, firm to the bite but not "
     "crunchy. Reserve 1 cup of pasta water before draining; the starch helps "
     "sauces cling. Drain in a colander but do not rinse unless using cold."),

    ("life_002", "How to Boil an Egg",
     "Place eggs in a single layer in a saucepan and cover with cold water "
     "by 1 inch. Bring to a rolling boil over high heat, then remove from "
     "heat, cover, and let stand. For soft-boiled eggs (runny yolk), let "
     "stand 4-5 minutes. For medium (jammy yolk), 7-8 minutes. For hard-"
     "boiled (fully set yolk), 10-12 minutes. Transfer immediately to ice "
     "water for 5 minutes — this stops cooking and makes peeling easier."),

    ("life_003", "Basic Knife Skills",
     "Use a sharp chef's knife — a dull knife is more dangerous because it "
     "slips. Curl the fingertips of your guiding hand under so the blade "
     "rides against your knuckles, never your fingertips. For onions: cut "
     "in half through the root, peel, place flat side down, make horizontal "
     "and vertical cuts, then cut across. Wash and dry knives by hand — "
     "dishwashers dull and damage them."),

    # ------------------------------------------------------------- Everyday — household
    ("home_001", "How to Unclog a Drain",
     "First try a plunger — fill the sink with enough water to cover the "
     "plunger cup, then push down sharply 5-6 times. If that fails, try a "
     "mixture of 1/2 cup baking soda followed by 1/2 cup white vinegar; let "
     "fizz for 15 minutes, then flush with boiling water. For stubborn "
     "clogs, a hand snake (drum auger) is more reliable than chemical "
     "drain cleaners, which can damage pipes and are hazardous to skin."),

    ("home_002", "Replacing a Light Bulb Safely",
     "Turn off the light switch and let the bulb cool for at least 5 "
     "minutes — incandescent and halogen bulbs get extremely hot. Stand on "
     "a stable surface, never a chair with wheels. Remove the old bulb by "
     "turning counter-clockwise. Match the replacement type and wattage to "
     "the fixture's rating (printed inside or near the socket). LED bulbs "
     "last 10-25x longer than incandescent and use 75% less energy."),

    ("home_003", "Doing Laundry — Basics",
     "Sort clothes by colour (whites, lights, darks) and by fabric care "
     "(delicates separately). Read care labels before the first wash — some "
     "items are dry-clean only or hand-wash only. Use cold water for most "
     "loads — it saves energy and prevents shrinking and fading. Use about "
     "2 tablespoons of detergent per load (less than the cap typically "
     "suggests). Don't overload the machine — clothes need room to tumble."),

    # ------------------------------------------------------------- Everyday — health
    ("health_001", "Tips for Better Sleep",
     "Aim for 7-9 hours per night. Keep a consistent sleep schedule, even "
     "on weekends. Avoid caffeine after 2 PM and large meals within 3 hours "
     "of bedtime. Keep the bedroom cool (around 18-20°C / 65-68°F), dark, "
     "and quiet. Avoid screens for 30-60 minutes before bed, or use blue-"
     "light filters. If you can't fall asleep within 20 minutes, get up "
     "and do something quiet in dim light until you feel drowsy."),

    ("health_002", "How Much Water to Drink",
     "A common rule of thumb is 8 glasses (about 2 litres) per day, but "
     "individual needs vary with body size, activity level, climate, and "
     "diet. A simple test: urine should be pale yellow, not dark. Most "
     "fruits, vegetables, and other beverages count toward total fluid "
     "intake. Increase water during exercise, hot weather, or illness "
     "with fever. You usually don't need to drink before you feel thirsty."),

    ("health_003", "Starting an Exercise Habit",
     "Start small — 10-15 minutes of walking daily for the first 2 weeks "
     "is more sustainable than an ambitious 1-hour gym session you'll "
     "abandon. The CDC recommends 150 minutes of moderate aerobic activity "
     "per week (about 30 min 5 days a week) plus 2 sessions of strength "
     "training. Build the habit first, then add intensity. Consistency "
     "beats peak effort — three short workouts a week for a year beats "
     "one perfect month and zero after that."),

    # ------------------------------------------------------------- Everyday — finance
    ("money_001", "Building a Basic Budget",
     "The 50/30/20 rule is a simple starting point: 50% of after-tax "
     "income to needs (rent, utilities, groceries, transit, insurance), "
     "30% to wants (dining out, entertainment, hobbies), 20% to savings "
     "and debt payoff. Track spending for one month before budgeting — "
     "most people underestimate how much they spend on food and "
     "subscriptions. Apps like YNAB, Mint, or a simple spreadsheet all "
     "work; consistency matters more than the tool."),

    ("money_002", "Emergency Fund Basics",
     "An emergency fund is money set aside for unexpected expenses — job "
     "loss, medical bills, urgent repairs. Target 3-6 months of essential "
     "living expenses (not full salary) in a high-interest savings account "
     "that's separate from your everyday account but accessible within a "
     "day. Build it before investing in higher-risk assets. If you have "
     "high-interest debt (credit cards), pay that down first while "
     "saving a small starter fund of $1,000-2,000."),

    ("money_003", "How Compound Interest Works",
     "Compound interest is interest earned on both the original amount and "
     "previously-earned interest — it grows faster than simple interest. "
     "Example: $1,000 invested at 7% annual return becomes about $1,967 "
     "after 10 years and $7,612 after 30 years. The earlier you start, the "
     "more dramatic the effect. A 25-year-old saving $200/month at 7% "
     "until age 65 ends up with about $525,000; starting at 35 with the "
     "same amount yields about $245,000."),

    # ------------------------------------------------------------- Everyday — tech help
    ("tech_001", "Resetting Your Wi-Fi Router",
     "If the internet stops working, restart the router first: unplug the "
     "power cable, wait 30 seconds (so capacitors drain), then plug back "
     "in. Wait 2-3 minutes for it to fully reconnect. If that doesn't fix "
     "it, restart your modem the same way. A factory reset (small button "
     "you press with a paperclip for 10+ seconds) wipes all settings "
     "including Wi-Fi password — only do this if you're prepared to set "
     "it up from scratch."),

    ("tech_002", "Why Use a Password Manager",
     "A password manager stores all your passwords encrypted, so you only "
     "need to remember one master password. Benefits: every site can have "
     "a unique strong password (so a breach at one site can't cascade), "
     "auto-fill prevents phishing on lookalike domains, and built-in "
     "generators create secure passwords automatically. Popular options: "
     "1Password, Bitwarden (free tier), and Apple Keychain or Google "
     "Password Manager (free, built into the OS)."),

    # ------------------------------------------------------------- Everyday — travel
    ("travel_001", "How to Pack for a Trip",
     "Roll clothes instead of folding to save space and reduce wrinkles. "
     "Pack heavier items (shoes, books) at the bottom near the wheels for "
     "stability. Carry on essentials: medications, a change of clothes, "
     "phone charger, and any irreplaceable items. Wear your bulkiest "
     "shoes and jacket on the plane. Check airline liquids rules — most "
     "limit carry-on liquids to containers under 100 ml, all in a clear "
     "1-litre bag. Photograph your packed luggage before checking it."),

    ("travel_002", "What to Do if Your Flight is Delayed",
     "Check the airline app for the latest status — gate agents often "
     "know less than the app. If the delay is significant, call the "
     "airline's customer service line while standing in the rebooking "
     "line — whichever responds first wins. For long delays in airports, "
     "many airlines provide meal vouchers (ask). For overnight delays, "
     "ask if a hotel room is provided. Travel insurance and many credit "
     "cards reimburse delay-related expenses; keep all receipts."),
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
